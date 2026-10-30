from __future__ import annotations
from mjlab.tasks.tracking.mdp.commands import *

import math
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_from_euler_xyz
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers import CommandTerm, CommandTermCfg
from collections.abc import Callable

if TYPE_CHECKING:
  import viser
  from mjlab.envs import ManagerBasedRlEnv

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


def get_forward_xy(pose: torch.Tensor, distance: float) -> torch.Tensor:
    """
    计算机器人前方 distance 距离处的 XY 坐标（世界坐标系）。

    Args:
        pose (torch.Tensor): 形状为 [env_num, 7] 的张量，每行包含
            [x, y, z, qw, qx, qy, qz]（位置和四元数，w 在前）。
        distance (float): 前向移动的距离（标量，自动广播）。

    Returns:
        torch.Tensor: 形状为 [env_num, 2]，包含每个环境对应的
            (forward_x, forward_y) 世界坐标。
    """
    # 提取位置
    x = pose[:, 0]
    y = pose[:, 1]
    # 提取四元数分量
    qw, qx, qy, qz = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]

    # 计算旋转矩阵的第一列（即局部 X 轴在世界坐标系中的方向）
    # 该列对应于将 (1,0,0) 旋转后的向量
    fx = 1 - 2 * (qy**2 + qz**2)          # X 分量
    fy = 2 * (qx * qy + qw * qz)          # Y 分量
    # fz = 2 * (qx * qz - qw * qy)        # Z 分量（此处不需要）

    # 沿前向方向移动 distance，得到新的 XY 坐标
    new_x = x + distance * fx
    new_y = y + distance * fy

    return torch.stack([new_x, new_y], dim=1)


class MotionBarbellCommand(MotionCommand):

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.box: Entity = env.scene["barbell"]
    self.candidate_frames = torch.tensor([0,130,130,130], device=self.device)
    self.keep_range = (160,500)
    self.graping_range = (130,160)
    self.hands_index = [self.robot.find_joints(name)[0][0] for name in self.robot.joint_names if "hand" in name]
    self.plates_geom_index,_ = self.box.find_geoms(["plate_left","plate_right"])
    # hands :
    #   0  1     1   2
    #   / /       \  \
    #  / /  2  0   \  \
    # / /  /    \   \  \
    # ■   /      \     ■
    # left          right
    self.left_hands, _  = self.robot.find_bodies(['left_hand_index_1_link','left_hand_middle_1_link'])
    self.right_hands, _ = self.robot.find_bodies(['right_hand_index_1_link','right_hand_middle_1_link'])

    self.garp_point = {
      "left_hand" :self.robot.find_bodies(["left_hand_thumb_2_link","left_hand_index_0_link","left_hand_middle_0_link"])[0],
      "right_hand":self.robot.find_bodies(["right_hand_thumb_2_link","right_hand_index_0_link","right_hand_middle_0_link"])[0],
    }

    self.finger_grap = {
      "left":self.robot.find_bodies(["left_hand_thumb_2_link","left_hand_index_1_link","left_hand_middle_1_link"])[0],
      "right":self.robot.find_bodies(["right_hand_thumb_2_link","right_hand_index_1_link","right_hand_middle_1_link"])[0],
    }

    
  @property
  def graping(self):
    return (self.graping_range[0] < self.time_steps) & ( self.time_steps < self.keep_range[1])

  @property
  def graped(self):
    return (self.keep_range[0] < self.time_steps) & ( self.time_steps < self.keep_range[1])

  def _uniform_sampling(self, env_ids: torch.Tensor):
    # 假设 candidate 是一个 Python list（或其他可迭代对象）
    rand_indices = torch.randint(0, len(self.candidate_frames), (len(env_ids),), device=self.device)
    self.time_steps[env_ids] = self.candidate_frames[rand_indices]


  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    pose:torch.Tensor = self.robot._data.data.qpos[env_ids][:,self.robot.data.indexing.free_joint_q_adr].clone()

    # time_step = self.time_steps[env_ids]
    # greping = (time_step > self.keep_range[0]) & ( time_step < self.keep_range[1])

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    low, high = self.cfg.box_xy_noise_range

    xy_noise = torch.rand(len(env_ids), 2, device=pose.device) * (high - low) + low

    if len(pose.shape) == 1:
        pose.unsqueeze_(0)

    pose[:,:2] = get_forward_xy(pose, 0.4) + xy_noise * 0.1
    pose[:,2] = pose[:,2] - 0.44
    add_yaw_noise(pose, max_yaw_deg=30.0)

    self.box.write_root_link_pose_to_sim(pose, env_ids=env_ids)

  def _debug_vis_impl(self, visualizer):
    super()._debug_vis_impl(visualizer)
    # 假设你的类中有如下属性（示例）
    env_indices = visualizer.get_env_indices(self.num_envs)  # 或 range(self.num_envs)
    if not env_indices:
        return

    # 获取几何位置（假设形状为 [num_envs, num_geoms, 3]）
    geom_pos = self.box.data.geom_pos_w.cpu().numpy()  # 转为 numpy
    joit_pos = self.robot.data.body_link_pos_w.cpu().numpy()  # 转为 numpy

    z_offset = 1.0  # 你的偏移量（可调）

    for batch in env_indices:
      # 取第 batch 个环境的两个几何体位置
      pos0 = geom_pos[batch, self.plates_geom_index[0]] + np.array([0, 0, z_offset])
      # pos1 = geom_pos[batch, self.plates_geom_index[1]] + np.array([0, 0, z_offset])
      vec = self.bellbar_vector.detach().cpu().numpy()[batch]
      
      visualizer.add_arrow(
          pos0, pos0 + vec,
          color=(0.2, 0.2, 0.6, 0.6),
          width=0.015
      )
      
      pos0 = joit_pos[batch, self.left_hands[1],:3] + np.array([0,0,0.1])
      pos1 = joit_pos[batch, self.left_hands[0],:3] + np.array([0,0,0.1])
      
      visualizer.add_arrow(
        pos0, pos1,
        color=(0.2, 0.2, 0.6, 0.6),
        width=0.003
      )

      pos0 = joit_pos[batch, self.right_hands[0],:3] + np.array([0,0,0.1])
      pos1 = joit_pos[batch, self.right_hands[1],:3] + np.array([0,0,0.1])
      
      visualizer.add_arrow(
        pos0, pos1,
        color=(0.2, 0.2, 0.6, 0.6),
        width=0.003
      )

      # grap point 
      pos0 = joit_pos[batch, self.garp_point['left_hand'],:3].mean(-2)
      
      visualizer.add_sphere(
        pos0, 0.005,
        color=(0.2, 0.2, 0.6, 0.6),
      )

      # grap point 
      pos0 = joit_pos[batch, self.garp_point['right_hand'],:3].mean(-2)
      
      visualizer.add_sphere(
        pos0, 0.005,
        color=(0.2, 0.2, 0.6, 0.6),
      )

      # 杠铃片和手

      pos0 = geom_pos[batch, self.plates_geom_index[0]]
      pos1 = joit_pos[batch, self.garp_point['right_hand'],:3].mean(-2)
      
      visualizer.add_arrow(
          pos0, pos1,
          color=(0.2, 0.2, 0.6, 0.6),
          width=0.015
      )

      pos1 = geom_pos[batch, self.plates_geom_index[1]]
      pos0 = joit_pos[batch, self.garp_point['left_hand'],:3].mean(-2)

      visualizer.add_arrow(
          pos0, pos1,
          color=(0.2, 0.2, 0.6, 0.6),
          width=0.015
      )

  @property
  def bellbar_vector(self):
    with torch.no_grad():
      geom_pos = self.box.data.geom_pos_w  
      pos0 = geom_pos[:, self.plates_geom_index[0]]
      pos1 = geom_pos[:, self.plates_geom_index[1]]
      return pos1 - pos0   # [batch, 3]

  @property
  def right_hand_plate_vector(self):
    with torch.no_grad():
      geom_pos = self.box.data.geom_pos_w  
      joit_pos = self.robot.data.body_link_pos_w
      pos0 = geom_pos[:, self.plates_geom_index[0]]
      pos1 = joit_pos[:, self.garp_point['right_hand'],:3].mean(-2)
      return pos1 - pos0

  @property
  def left_hand_plate_vector(self):
    with torch.no_grad():
      geom_pos = self.box.data.geom_pos_w
      joit_pos = self.robot.data.body_link_pos_w
      pos0 = joit_pos[:, self.garp_point['left_hand'],:3].mean(-2)
      pos1 = geom_pos[:, self.plates_geom_index[1]]
      return pos1 - pos0

  @property
  def left_fingers(self):
    with torch.no_grad():
      return self.robot.data.body_link_pos_w[:,self.finger_grap['left'], :3]

  @property
  def right_fingers(self):
    with torch.no_grad():
      return self.robot.data.body_link_pos_w[:,self.finger_grap['right'], :3]

@dataclass(kw_only=True)
class MotionBarbellCommandCfg(MotionCommandCfg):
  
  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)
  box_xy_noise_range: tuple[float, float] = (0.0, 0.0)

  def build(self, env) -> MotionBarbellCommand:
    return MotionBarbellCommand(self, env)



