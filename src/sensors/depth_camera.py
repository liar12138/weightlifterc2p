import torch
import random
import torchvision.transforms as T
from torchvision.transforms import functional as F

from mjlab.sensor import CameraSensorCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensor
from mjlab.envs import ManagerBasedRlEnv, mdp as envs_mdp

class AddGaussianNoise:
    """给张量添加高斯噪声，sigma 在给定范围内随机采样。"""
    def __init__(self, sigma_range=(0.0, 0.02), p=0.5):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            noise = torch.randn_like(img) * sigma
            return img + noise
        return img


class RandomTranslateScale:
    """随机平移和缩放（保持输出尺寸，用 0 填充边界）。"""
    def __init__(self, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5):
        self.translate = translate
        self.scale = scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 随机平移比例（相对于尺寸）
            dx = random.uniform(-self.translate[0], self.translate[0]) * img.shape[-1]
            dy = random.uniform(-self.translate[1], self.translate[1]) * img.shape[-2]
            # 随机缩放因子
            scale = random.uniform(*self.scale)
            # 使用仿射变换（旋转角度为0）
            return F.affine(img, angle=0, translate=(dx, dy), scale=scale, shear=0, fill=0)
        return img


def get_depth_transform():
    """
    返回一个 Composed transform，适用于形状为 (1, 51, 29) 的深度图，
    输出形状为 (1, 48, 27)。
    """
    transform = T.Compose([
        # 1. 随机旋转 ±10 度，填充 0（深度图中通常 0 表示无效值）
        T.RandomRotation(degrees=10, fill=0),
        
        # 2. 中心裁剪到目标尺寸 (48, 27) —— 也可以改用 RandomCrop 增加变化
        T.CenterCrop((48, 27)),
        # 若希望随机裁剪，可使用 T.RandomCrop((48, 27)) 替换上面一行

        # 4. 添加高斯噪声（概率 0.5，sigma 在 0~0.02 均匀采样，适配深度图归一化值域 [0,1]）
        AddGaussianNoise(sigma_range=(0.0, 0.02), p=0.5),

        # 5. 随机平移（<10% 尺寸）和缩放（0.9~1.1），保持尺寸
        RandomTranslateScale(translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5),

        # 6. 随机擦除（模拟遮挡），概率 0.2，擦除区域 2%~10% 面积
        T.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0),

        # 7. 可选：高斯模糊（模拟传感器模糊），核大小 3，sigma 随机
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    return transform


def depth_camera(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  normalization_max_depth: float | None = 6.0,
) -> torch.Tensor:
    """Extract depth image from a camera sensor.

    Args:
    env: The manager-based RL environment.
    sensor_name: Name of the camera sensor.

    Returns:
    Depth image tensor of shape ``[num_envs, height, width, 1]`` (float32).
    """
    sensor: CameraSensor = env.scene[sensor_name]
    depth = sensor.data.depth
    # return depth
    if len(depth.shape) == 5:
        return torch.clamp(depth, 0.0, normalization_max_depth).div(normalization_max_depth).squeeze_(-1)
    elif len(depth.shape) == 4:
        return torch.clamp(depth, 0.0, normalization_max_depth).div(normalization_max_depth).squeeze_(-1)

Depth_Camera_Transform = None

@torch.no_grad()
def depth_camera_transform(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
    """Crop and perturb depth, optionally restoring the native resolution."""
    return depth_camera(env, sensor_name, 6.0)
    global Depth_Camera_Transform
    if Depth_Camera_Transform == None:
        Depth_Camera_Transform = get_depth_transform()
    return Depth_Camera_Transform(depth)

def deploy_depth_camera_to(cfg,depth_history_length = 1):

    camera_cfg = CameraSensorCfg(
        name="depth_camera",
        parent_body="robot/torso_link",
        pos=(0.0576235, 0.01753, 0.41987),
        quat=(0.915, 0, -0.404, 0),
        fovy=55.2,
        width=27,
        height=48,
        data_types=("depth",),
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (camera_cfg,)

    cfg.observations["depth_camera"] = ObservationGroupCfg(
        terms={
            "depth_camera": ObservationTermCfg(
                func=depth_camera_transform,
                params={
                    "sensor_name": "depth_camera",
                },
                history_length=depth_history_length,
                flatten_history_dim=False,
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    return cfg