from __future__ import annotations
from mjlab.tasks.tracking.mdp.commands import *

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.envs.mdp.dr.geom import geom_size
from mjlab.managers.scene_entity_config import SceneEntityCfg
from collections.abc import Callable

if TYPE_CHECKING:
  import viser
  from mjlab.envs import ManagerBasedRlEnv

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
    self.candidate_frames = torch.tensor([0,0,153,153,153], device=self.device)
    self.keep_range = (295,785)
    # o6 无 hand 子节点
    finger_keys = ("thumb", "index", "middle", "ring", "pinky")
    self.hands_index = [
      self.robot.find_joints(name)[0][0]
        for name in self.robot.joint_names
          if any(k in name for k in finger_keys)
    ]
    self.plates_geom_index,_ = self.box.find_geoms(["plate_left","plate_right"])
    # 4 个片 geom（左右 × 视觉/碰撞），用于运行时随机半径与读回
    self._plate_geom_ids, _ = self.box.find_geoms([r"plate_.*"])
    self._plate_global_ids = self.box.indexing.geom_ids[self._plate_geom_ids]
    self._plate_cfg = SceneEntityCfg("barbell", geom_names=[r"plate_.*"])
    # 让 geom_size/geom_rbound/geom_aabb 按环境分维（供 dr.geom_size 使用）
    env.sim.expand_model_fields(("geom_size", "geom_rbound", "geom_aabb"))
    # H2+O6 actuated joints: 53 total = 31 body + 22 hand (11/side).
    # Body (name patterns): *_hip_{pitch,roll,yaw}_joint, *_knee_joint,
    #   *_ankle_{pitch,roll}_joint, waist_{yaw,roll,pitch}_joint,
    #   head_{pitch,yaw}_joint, *_shoulder_{pitch,roll,yaw}_joint,
    #   *_elbow_joint, *_wrist_{roll,pitch,yaw}_joint.
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

    self.garp_point = {
      "left_hand": self.robot.find_bodies([
        "left_thumb_distal",
        "left_index_proximal",
        "left_middle_proximal",
        "left_ring_proximal",
        "left_pinky_proximal",
      ])[0],
      "right_hand": self.robot.find_bodies([
        "right_thumb_distal",
        "right_index_proximal",
        "right_middle_proximal",
        "right_ring_proximal",
        "right_pinky_proximal",
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
    pose: torch.Tensor = self.robot._data.data.qpos[env_ids][:, self.robot.data.indexing.free_joint_q_adr].clone()

    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int)

    time_step = self.time_steps[env_ids]
    greping = (time_step > self.keep_range[0]) & (time_step < self.keep_range[1])
    n = len(env_ids)
    p = min(1.0, self._env.common_step_counter / self.cfg.curriculum_total_steps)

    # 课程范围：距离中间值向两侧扩、片半径 15→5 递减、侧向 0→max
    fwd_lo = self.cfg.forward_mid - self.cfg.forward_half_range * p
    fwd_hi = self.cfg.forward_mid + self.cfg.forward_half_range * p
    r_hi = self.cfg.plate_radius_max - self.cfg.plate_radius_span * p
    r_lo = max(self.cfg.plate_radius_min, r_hi - self.cfg.plate_radius_window)
    lat = self.cfg.lateral_max * p

    # 1) 杠铃片半径（shared_random 保证 4 个 geom 同值）
    geom_size(self._env, env_ids, ranges=(r_lo, r_hi),
              asset_cfg=self._plate_cfg, axes=[0], operation="abs", shared_random=True)
    # 2) 读回实际半径（左/右/视觉/碰撞同值）
    radius = self._env.sim.model.geom_size[:, self._plate_global_ids[0], 0]
    r = radius[env_ids]

    # 3) 采样 forward / lateral
    fwd = torch.rand(n, device=pose.device) * (fwd_hi - fwd_lo) + fwd_lo
    xy_noise = (torch.rand(n, 2, device=pose.device) * 2.0 - 1.0) * lat

    if len(pose.shape) == 1:
      pose.unsqueeze_(0)

    orig = pose.clone()
    env_z = self._env.scene.env_origins[:, 2]
    pose[~greping, :2] = get_forward_xy(orig[~greping], fwd[~greping]) + xy_noise[~greping]
    pose[~greping, 2] = env_z[env_ids][~greping] + r[~greping] + 1e-3

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
      pos1 = geom_pos[batch, self.plates_geom_index[1]] + np.array([0, 0, z_offset])
      
      visualizer.add_arrow(
          pos0, pos1,
          color=(0.2, 0.2, 0.6, 0.6),
          width=0.015
      )
      
      pos0 = joit_pos[batch, self.left_hands[1],:3] + np.array([0,0,0.1])
      pos1 = joit_pos[batch, self.left_hands[0],:3] + np.array([0,0,0.1])
      
      visualizer.add_arrow(
        pos1, pos0,
        color=(0.2, 0.2, 0.6, 0.6),
        width=0.003
      )

      pos0 = joit_pos[batch, self.right_hands[0],:3] + np.array([0,0,0.1])
      pos1 = joit_pos[batch, self.right_hands[1],:3] + np.array([0,0,0.1])
      
      visualizer.add_arrow(
        pos1, pos0,
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
      pos0 = geom_pos[:, self.plates_geom_index[0]]
      pos1 = geom_pos[:, self.plates_geom_index[1]]
      return pos0 - pos1   # [batch, 3]

  @property
  def right_hand_plate_vector(self):
    with torch.no_grad():
      geom_pos = self.box.data.geom_pos_w  
      joit_pos = self.robot.data.body_link_pos_w
      pos0 = joit_pos[:,self.garp_point['right_hand'],:3].mean(-2)
      pos1 = geom_pos[:,self.plates_geom_index[0]]
      return pos0 - pos1

  @property
  def left_hand_plate_vector(self):
    with torch.no_grad():
      geom_pos = self.box.data.geom_pos_w
      joit_pos = self.robot.data.body_link_pos_w
      pos0 = geom_pos[:, self.plates_geom_index[1]]
      pos1 = joit_pos[:, self.garp_point['left_hand'],:3].mean(-2)
      return pos0 - pos1

  
@dataclass(kw_only=True)
class MotionBarbellCommandCfg(MotionCommandCfg):
  
  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)
  box_xy_noise_range: tuple[float, float] = (0.0, 0.0)
  # 课程学习（前 2/3 训练完成）
  curriculum_total_steps: float = 1.0e7
  forward_mid: float = 0.30
  forward_half_range: float = 0.10
  plate_radius_max: float = 0.15
  plate_radius_min: float = 0.05
  plate_radius_span: float = 0.10
  plate_radius_window: float = 0.04
  lateral_max: float = 0.05

  def build(self, env) -> MotionBarbellCommand:
    return MotionBarbellCommand(self, env)



