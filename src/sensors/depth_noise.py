import torch
import torch.nn.functional as F


@torch.no_grad()
def sim_to_real_depth_noise(
    depth: torch.Tensor,
    max_depth: float = 6.0,
    invalid_threshold: float = 1e-4,
    geometric_noise: bool = True,
    output_normalized: bool = True,
) -> torch.Tensor:
    """
    将 MuJoCo 仿真深度扰动成更接近真实 filtered_depth 的形式。

    Args:
        depth:
            米制仿真深度。
            支持 [H,W]、[B,H,W]、[B,1,H,W]。
        max_depth:
            最大有效深度，当前真实数据使用 6 m。
        invalid_threshold:
            小于该值的仿真深度视为无效值。
        geometric_noise:
            是否添加小幅相机外参扰动。
        output_normalized:
            True 返回 [0,1]，用于网络；
            False 返回米制深度。

    Returns:
        与输入 shape 相同的深度张量。
    """
    original_shape = depth.shape
    original_ndim = depth.ndim

    if original_ndim == 2:
        x = depth[None, None]
    elif original_ndim == 3:
        x = depth[:, None]
    elif original_ndim == 4 and depth.shape[1] == 1:
        x = depth
    else:
        raise ValueError(
            f"Expected [H,W], [B,H,W] or [B,1,H,W], got {depth.shape}"
        )

    x = x.float().clone()
    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype

    # 1. 与真实 filtered_depth 对齐无效深度定义：
    #    sim 中的 0 转成真实管线中的 6 m。
    invalid = (
        ~torch.isfinite(x)
        | (x <= invalid_threshold)
        | (x >= max_depth)
    )
    x = torch.where(
        invalid,
        torch.full_like(x, max_depth),
        x,
    )

    # 2. 小幅外参误差：
    #    平移约 ±1 像素，尺度约 ±2%。
    if geometric_noise:
        _, _, height, width = x.shape

        tx_pixels = torch.empty(
            batch_size, device=device, dtype=dtype
        ).uniform_(-1.0, 1.0)

        ty_pixels = torch.empty(
            batch_size, device=device, dtype=dtype
        ).uniform_(-1.0, 1.0)

        scale = torch.empty(
            batch_size, device=device, dtype=dtype
        ).uniform_(0.98, 1.02)

        theta = torch.zeros(
            batch_size, 2, 3, device=device, dtype=dtype
        )
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = 2.0 * tx_pixels / max(width - 1, 1)
        theta[:, 1, 2] = 2.0 * ty_pixels / max(height - 1, 1)

        grid = F.affine_grid(
            theta,
            x.shape,
            align_corners=False,
        )

        # 使用 max_depth - x：
        # grid_sample 边界默认补 0，转换回来后边界就是 6 m。
        inverse_depth = max_depth - x

        inverse_depth = F.grid_sample(
            inverse_depth,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        x = max_depth - inverse_depth

    # 3. 每帧轻微尺度误差和深度偏置。
    depth_scale = torch.empty(
        batch_size, 1, 1, 1,
        device=device,
        dtype=dtype,
    ).uniform_(0.98, 1.02)

    depth_bias = torch.empty(
        batch_size, 1, 1, 1,
        device=device,
        dtype=dtype,
    ).uniform_(-0.02, 0.02)

    valid = x < max_depth
    calibrated = x * depth_scale + depth_bias
    x = torch.where(valid, calibrated, x)

    # 4. 深度相关噪声：
    #    距离越远，深度相机噪声通常越大。
    noise_sigma = 0.003 + 0.0015 * x.square()
    gaussian_noise = torch.randn_like(x) * noise_sigma

    x = torch.where(
        valid,
        x + gaussian_noise,
        x,
    )

    # 5. 模拟 filtered_depth 的轻微空间平滑。
    #    只随机对一部分 batch 使用，保留训练多样性。
    blurred = F.avg_pool2d(
        x,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    blur_mask = (
        torch.rand(
            batch_size, 1, 1, 1,
            device=device,
        ) < 0.5
    )

    x = torch.where(blur_mask, blurred, x)

    # 6. 少量无效测量；真实数据中无效值应写成 6，而不是 0。
    dropout_mask = (
        torch.rand_like(x) < 0.003
    ) & (x < max_depth)

    x = torch.where(
        dropout_mask,
        torch.full_like(x, max_depth),
        x,
    )

    # 7. 限制值域并模拟真实数据约 1 mm 的量化精度。
    x = x.clamp(0.001, max_depth)
    x = torch.round(x * 1000.0) / 1000.0

    if output_normalized:
        x = x / max_depth

    if original_ndim == 2:
        return x[0, 0]

    if original_ndim == 3:
        return x[:, 0]

    return x.reshape(original_shape)