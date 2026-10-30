import math,torch
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_from_euler_xyz

def add_yaw_noise(pose: torch.Tensor, max_yaw_deg: float) -> torch.Tensor:
  """
  在 pose 四元数 [w,x,y,z] 上叠加绕 z 轴 ±max_yaw_deg 的随机偏航，位置不变。
  """
  # 四元数 -> 三维欧拉角 (roll, pitch, yaw)
  roll, pitch, yaw = euler_xyz_from_quat(pose[:, 3:7])

  # z 轴(yaw)上加随机噪声
  max_yaw = math.radians(max_yaw_deg)
  yaw = yaw + torch.empty_like(yaw).uniform_(-max_yaw, max_yaw)

  # 转回四元数
  pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
  return pose