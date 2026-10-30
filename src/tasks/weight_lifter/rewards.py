from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from typing import cast
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.tasks.tracking.mdp.rewards import _get_body_indexes
from mjlab.sensor import ContactSensor

from .commands import MotionBarbellCommand

@torch.no_grad()
def px(
    v1: torch.Tensor,
    v2: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
):
  # 归一化，得到 cos θ
  norm1 = torch.norm(v1, dim=dim, keepdim=True)
  norm2 = torch.norm(v2, dim=dim, keepdim=True)
  cos_theta = (v1 / (norm1 + eps) * v2 / (norm2 + eps)).sum(dim=dim)

  # 数值裁剪，防止 acos 输入越界产生 nan
  cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

  return 1 / (cos_theta + 0.5) - 1

@torch.no_grad()
def cos_angle_minus_abs_angle(
    v1: torch.Tensor,
    v2: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """返回 cos θ - |θ|，其中 θ 为 v1 与 v2 的夹角（弧度）。

    Args:
        v1 (torch.Tensor): 形状 [batch, N]
        v2 (torch.Tensor): 形状 [batch, N]
        dim (int): 向量所在维度，默认 -1（即最后一维为 N）
        eps (float): 防止除零

    Returns:
        torch.Tensor: 形状 [batch]，值域约 [cos π - π, 1] = [-π-1, 1]
    """
    # 归一化，得到 cos θ
    norm1 = torch.norm(v1, dim=dim, keepdim=True)
    norm2 = torch.norm(v2, dim=dim, keepdim=True)
    cos_theta = (v1 / (norm1 + eps) * v2 / (norm2 + eps)).sum(dim=dim)

    # 数值裁剪，防止 acos 输入越界产生 nan
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # 夹角 θ ∈ [0, π]
    theta = torch.acos(cos_theta)

    return cos_theta - theta.abs()

@torch.no_grad()
def barbell_reward_hands_height(
  env: ManagerBasedRlEnv,
  joint_names = ['left_hand_index_1_joint', 'right_hand_index_1_joint'],

) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]
  indexes,_ = robot_asset.find_joints(joint_names)

  barbell_hands_distant = - (robot_asset.data.body_link_pos_w[:,indexes, 2] - barbell_asset.data.root_link_pos_w[:, 2].unsqueeze(0).T).norm(dim = -1)
  
  return barbell_hands_distant.exp() 

@torch.no_grad()
def barbell_reward_hands_height(
  env: ManagerBasedRlEnv,
  joint_names = ['left_hand_index_1_joint', 'right_hand_index_1_joint'],

) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]
  indexes,_ = robot_asset.find_joints(joint_names)

  barbell_hands_distant = - (robot_asset.data.body_link_pos_w[:,indexes, 2] - barbell_asset.data.root_link_pos_w[:, 2].unsqueeze(0).T).norm(dim = -1)
  
  return barbell_hands_distant.exp()

@torch.no_grad()
def pently_hands_parrel(
    env: ManagerBasedRlEnv,
    command_name: str,
):
  robot_asset: Entity = env.scene["robot"]
  joit_pos = robot_asset.data.body_link_pos_w
  command = env.command_manager.get_term(command_name)

  bellbar = command.bellbar_vector  # [B, 3]

  # 与 _debug_vis_impl 的箭头方向对齐：
  #   左手箭头：left_hands[1](middle) -> left_hands[0](index)，即 index_1 - middle_1
  #   右手箭头：right_hands[0](index) -> right_hands[1](middle)，即 middle_1 - index_1
  left = joit_pos[:, command.left_hands[0], :3] - joit_pos[:, command.left_hands[1], :3]
  right = joit_pos[:, command.right_hands[1], :3] - joit_pos[:, command.right_hands[0], :3]

  return cos_angle_minus_abs_angle(left,bellbar) + cos_angle_minus_abs_angle(right,bellbar)  # [B] ∈ [0, 2]

@torch.no_grad()
def left_hand_grap_bar(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  bv = command.bellbar_vector
  lv = command.left_hand_plate_vector

  return px(bv,lv) * command.graping.float()


@torch.no_grad()
def right_hand_grap_bar(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  bv = command.bellbar_vector
  rv = command.right_hand_plate_vector

  return px(bv,rv) * command.graping.float()

@torch.no_grad()
def fingers_grap(
  env: ManagerBasedRlEnv,
  command_name:MotionBarbellCommand
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  def tringle(a):
    # a 形状 (batch, 3, 3)
    diff1 = a[:, 0, :] - a[:, 1, :]   # (batch, 3)
    diff2 = a[:, 0, :] - a[:, 2, :]   # (batch, 3)
    return diff1.norm(dim=-1) + diff2.norm(dim=-1)  # (batch,)

  return (tringle(command.right_fingers) + tringle(command.left_fingers)) * command.graped.float()