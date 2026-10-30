import math
import torch

from mjlab.utils.lab_api.math import (
    quat_mul,
    quat_from_euler_xyz,
    normalize,
)

from mjlab.envs import ManagerBasedRlEnv

@torch.no_grad()
def set_pose(
    self,
    pos: torch.Tensor,
    quat: torch.Tensor = None,
    env_ids=None,
):
    if self._warp_model is None:
        raise RuntimeError(
            "Camera has not been initialized."
        )

    if env_ids is None:
        env_ids = torch.arange(
            self._warp_model.cam_pos.shape[0],
            device=self._warp_model.cam_pos.device,
        )

    self._warp_model.cam_pos[env_ids,self.camera_idx,:] = pos

    if quat:
        self._warp_model.cam_quat[env_ids,self.camera_idx,:] = quat

@torch.no_grad()
def randomize_depth_camera_pose(
    env: ManagerBasedRlEnv,
    env_ids=None,
    sensor_name: str = "real_depth_camera",
    pos_range=None,
    rot_range_deg=None,
):

    if pos_range is None:
        pos_range = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
        }

    if rot_range_deg is None:
        rot_range_deg = {
            "roll": (-0.01, 0.01),
            "pitch": (-0.01, 0.01),
            "yaw": (-0.01, 0.01),
        }

    sensor = env.scene[sensor_name]

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
        )

    num_envs = len(env_ids)

    base_pos = torch.tensor(
        sensor.cfg.pos,
        device=env.device,
        dtype=torch.float32,
    ).repeat(num_envs, 1)

    base_quat = torch.tensor(
        sensor.cfg.quat,
        device=env.device,
        dtype=torch.float32,
    ).repeat(num_envs, 1)

    base_quat = normalize(base_quat)

    pos_noise = torch.empty(num_envs,3,device=env.device)

    pos_noise[:, 0].uniform_(*pos_range["x"])

    pos_noise[:, 1].uniform_(*pos_range["y"])

    pos_noise[:, 2].uniform_(*pos_range["z"])

    new_pos = base_pos + pos_noise

    # =========================
    # rotation randomization
    # =========================

    # roll = torch.empty(num_envs,device=env.device).uniform_(
    #     math.radians(rot_range_deg["roll"][0]),
    #     math.radians(rot_range_deg["roll"][1]),
    # )

    # pitch =  torch.empty(num_envs,device=env.device).uniform_(
    #     math.radians(rot_range_deg["pitch"][0]),
    #     math.radians(rot_range_deg["pitch"][1]),
    # )

    # yaw =  torch.empty(num_envs,device=env.device).uniform_(
    #     math.radians(rot_range_deg["yaw"][0]),
    #     math.radians(rot_range_deg["yaw"][1]),
    # )

    # noise_quat = quat_from_euler_xyz(
    #     roll,
    #     pitch,
    #     yaw,
    # )

    # new_quat = quat_mul(
    #     base_quat,
    #     noise_quat,
    # )

    # new_quat = normalize(new_quat)

    # =========================
    # 写进 MJWarp
    # =========================

    set_pose(
        sensor,
        pos=new_pos,
        env_ids=env_ids,
    )

import torchvision.transforms.v2 as v2

class DepthNoiseMaskTransform:

    def __init__(self):
        self.transforms = v2.Compose([
            # 1. 先做几何增强（裁剪翻转等），避免模糊破坏边缘结构
            # v2.RandomHorizontalFlip(0.001),
            v2.RandomRotation(degrees=6),
            v2.RandomResizedCrop((25,35),scale=(0.9,1)),  # 先裁剪
            v2.Resize((27,48)),             # 再resize回原尺寸
            
            # 2. 再添加噪声（模拟传感器扰动）
            v2.GaussianNoise(mean=0.0, sigma=0.05, clip=False),
            v2.RandomErasing(p=0.05, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0),
            
            # 3. 【关键新增】用 3x3 高斯核轻微模糊，抹掉高频噪声，sigma 建议 0.5~1.0
            v2.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)),
        ])

    def __call__(self,x):
        return self.transforms(x)