from typing import TYPE_CHECKING, cast


import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionBarbellCommand
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

@torch.no_grad()
def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w - command.robot_anchor_pos_w)
    > threshold
  ).any(-1)


@torch.no_grad()
def barbell_droped_from_hands(
  env: ManagerBasedRlEnv,
  command_name: str,
  limit_distant: float,
  joint_names = ['left_hand_index_1_joint', 'right_hand_index_1_joint'],
):
  """Terminate when the asset's orientation exceeds the limit angle."""

  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]

  indexes,_ = robot_asset.find_joints(joint_names)
  barbell_hands_distant = torch.abs(robot_asset.data.body_link_pos_w[:,indexes, 2] - barbell_asset.data.root_link_pos_w[:, 2].unsqueeze(0).T)
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  
  return (barbell_hands_distant > limit_distant).any(-1) & command.graping

@torch.no_grad()
def far_away_root_pos(
  env: ManagerBasedRlEnv,
  limit_distant: float,
):
  """Terminate when the asset's orientation exceeds the limit angle."""

  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]

  barbell_hands_distant = torch.abs(robot_asset.data.root_com_pos_w[:,:2] - barbell_asset.data.root_link_pos_w[:,:2]).mean(-1)
  
  return barbell_hands_distant > limit_distant 


def bad_body_dof(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:

  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  error = torch.abs(command.joint_pos - command.robot_joint_pos)
  result = torch.any(error > threshold, dim=-1)

  return result

def bad_hands_dof(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:

  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  error = torch.abs(command.joint_pos[:,command.hands_index] - command.robot_joint_pos[:,command.hands_index])
  result = torch.any(error > threshold, dim=-1)

  return result

def bad_anchor_ori_relative(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))

    ref_anchor_quat_w = command.body_quat_relative_w[
        :, command.motion_anchor_body_index
    ]

    robot_anchor_quat_w = command.robot_anchor_quat_w

    motion_projected_gravity_b = quat_apply_inverse(
        ref_anchor_quat_w,
        asset.data.gravity_vec_w,
    )

    robot_projected_gravity_b = quat_apply_inverse(
        robot_anchor_quat_w,
        asset.data.gravity_vec_w,
    )

    return (
        motion_projected_gravity_b[:, 2]
        - robot_projected_gravity_b[:, 2]
    ).abs() > threshold

def not_on_the_bar(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:

  command : MotionBarbellCommand = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  command.bellbar_vector
  command.right_hand_plate_vector
  command.left_hand_plate_vector

  return ((torch.cosine_similarity(command.bellbar_vector, command.right_hand_plate_vector, dim=1) < threshold) | (torch.cosine_similarity(command.bellbar_vector, command.left_hand_plate_vector, dim=1) < threshold) ) & command.graped 