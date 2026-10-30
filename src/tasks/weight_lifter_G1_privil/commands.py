from __future__ import annotations
from mjlab.tasks.tracking.mdp.commands import *

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import yaw_quat
from collections.abc import Callable

if TYPE_CHECKING:
  import viser
  from mjlab.envs import ManagerBasedRlEnv

# 片半径
_BARBELL_PLATE_RADIUS = 0.225

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
    # v0.2.12: rebalance starts — more frame-0 tracking, less grasp-window shortcut
    # v0.2.13: 0,0,0,130,130,130,220 -> 0,0,130,130,130,220
    self.candidate_frames = torch.tensor([0, 0, 0, 130, 130, 130, 220], device=self.device)
    self.keep_range = (220,500)
    # o6 无 hand 子节点
    finger_keys = ("thumb", "index", "middle", "ring", "pinky")
    self.hands_index = [
      self.robot.find_joints(name)[0][0]
        for name in self.robot.joint_names
          if any(k in name for k in finger_keys)
    ]
    self.plates_geom_index,_ = self.box.find_geoms(["plate_left","plate_right"])
    # G1+O6 actuated joints: 51 total = 29 body + 22 hand (11/side).
    # Hand joints (per side; prefix left_/right_; names have no "hand"):
    #   thumb:  *_thumb_cmc_yaw, *_thumb_cmc_pitch, *_thumb_ip
    #   index:  *_index_mcp_pitch, *_index_dip
    #   middle: *_middle_mcp_pitch, *_middle_dip
    #   ring:   *_ring_mcp_pitch, *_ring_dip
    #   pinky:  *_pinky_mcp_pitch, *_pinky_dip
    # Hand bodies (chain tip-ward; prefix left_/right_):
    #   *_hand_base_link
    #   thumb:  *_thumb_metacarpals_base2 -> *_thumb_metacarpals -> *_thumb_distal
    #   index:  *_index_proximal -> *_index_distal
    #   middle: *_middle_proximal -> *_middle_distal
    #   ring:   *_ring_proximal -> *_ring_distal
    #   pinky:  *_pinky_proximal -> *_pinky_distal
    # Current usage:
    #   finger span  = index_distal ---- pinky_distal
    #   grasp point  = mean(thumb_distal + 4x finger proximal)
    self.left_hands, _  = self.robot.find_bodies(['left_index_distal', 'left_pinky_distal'])
    self.right_hands, _ = self.robot.find_bodies(['right_index_distal', 'right_pinky_distal'])

    # self.garp_point = {
    #   "left_hand": self.robot.find_bodies([
    #     "left_thumb_distal",
    #     "left_index_proximal",
    #     "left_middle_proximal",
    #     "left_ring_proximal",
    #     "left_pinky_proximal",
    #   ])[0],
    #   "right_hand": self.robot.find_bodies([
    #     "right_thumb_distal",
    #     "right_index_proximal",
    #     "right_middle_proximal",
    #     "right_ring_proximal",
    #     "right_pinky_proximal",
    #   ])[0],
    # }
    self.garp_point = {
      "left_hand": self.robot.find_bodies([
        "left_thumb_metacarpals",
        "left_index_distal",
        "left_index_proximal",
      ])[0],
      "right_hand": self.robot.find_bodies([
        "right_thumb_metacarpals",
        "right_index_distal",
        "right_index_proximal",
      ])[0],
    }  
    # Five fingertip bodies for enclosure / fingers_grap (thumb + 4 fingers).
    self.finger_grap = {
      "left": self.robot.find_bodies([
        "left_thumb_distal",
        "left_index_distal",
        "left_middle_distal",
        "left_ring_distal",
        "left_pinky_distal",
      ])[0],
      "right": self.robot.find_bodies([
        "right_thumb_distal",
        "right_index_distal",
        "right_middle_distal",
        "right_ring_distal",
        "right_pinky_distal",
      ])[0],
    }

  @property
  def graping(self):
    return (self.time_steps > self.keep_range[0]) & ( self.time_steps < self.keep_range[1])

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

    pose[:, :2] = get_forward_xy(pose, 1.0) + xy_noise
    pose[:, 3:7] = yaw_quat(pose[:, 3:7])
    env_z = self._env.scene.env_origins[env_ids, 2]
    pose[:, 2] = env_z + _BARBELL_PLATE_RADIUS + 1e-3
    # 调整高度和速度
    self.box.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.box.write_root_link_velocity_to_sim(
      torch.zeros(len(env_ids), 6, device=pose.device, dtype=pose.dtype),
      env_ids=env_ids,
    )

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
      # left-》right
      pos0 = geom_pos[batch, self.plates_geom_index[0]] + np.array([0, 0, z_offset])  # plate_left
      pos1 = geom_pos[batch, self.plates_geom_index[1]] + np.array([0, 0, z_offset])  # plate_right
      
      visualizer.add_arrow(
          pos0, pos1,
          color=(0.2, 0.2, 0.6, 0.6),
          width=0.015
      )
      
      pos0 = joit_pos[batch, self.left_hands[0],:3] + np.array([0,0,0.1])  # index
      pos1 = joit_pos[batch, self.left_hands[1],:3] + np.array([0,0,0.1])  # pinky
      
      visualizer.add_arrow(
        pos0, pos1,
        color=(0.2, 0.2, 0.6, 0.6),
        width=0.003
      )

      pos0 = joit_pos[batch, self.right_hands[1],:3] + np.array([0,0,0.1])  # pinky
      pos1 = joit_pos[batch, self.right_hands[0],:3] + np.array([0,0,0.1])  # index
      
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

      pos0 = geom_pos[batch, self.plates_geom_index[1]]
      pos1 = joit_pos[batch, self.garp_point['left_hand'],:3].mean(-2)
      
      visualizer.add_arrow(
          pos1, pos0,
          color=(0.2, 0.2, 0.6, 0.6),
          width=0.015
      )

      pos0 = joit_pos[batch, self.garp_point['right_hand'],:3].mean(-2)
      pos1 = geom_pos[batch, self.plates_geom_index[0]]

      visualizer.add_arrow(
          pos1, pos0,
          color=(0.2, 0.2, 0.6, 0.6),
          width=0.015
      )

  @property
  def bellbar_vector(self):
    with torch.no_grad():
      geom_pos = self.box.data.geom_pos_w
      plate_left = geom_pos[:, self.plates_geom_index[0]]
      plate_right = geom_pos[:, self.plates_geom_index[1]]
      return plate_right - plate_left  # [batch, 3]

  @property
  def right_hand_plate_vector(self):
    with torch.no_grad():
      geom_pos = self.box.data.geom_pos_w
      joit_pos = self.robot.data.body_link_pos_w
      grasp = joit_pos[:, self.garp_point["right_hand"], :3].mean(-2)
      plate_left = geom_pos[:, self.plates_geom_index[0]]
      return grasp - plate_left

  @property
  def left_hand_plate_vector(self):
    with torch.no_grad():
      geom_pos = self.box.data.geom_pos_w
      joit_pos = self.robot.data.body_link_pos_w
      plate_right = geom_pos[:, self.plates_geom_index[1]]
      grasp = joit_pos[:, self.garp_point["left_hand"], :3].mean(-2)
      return plate_right - grasp

  @property
  def left_fingers(self):
    """Left fingertip positions, shape [B, 5, 3] (thumb, index, middle, ring, pinky)."""
    with torch.no_grad():
      return self.robot.data.body_link_pos_w[:, self.finger_grap["left"], :3]

  @property
  def right_fingers(self):
    """Right fingertip positions, shape [B, 5, 3] (thumb, index, middle, ring, pinky)."""
    with torch.no_grad():
      return self.robot.data.body_link_pos_w[:, self.finger_grap["right"], :3]


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



