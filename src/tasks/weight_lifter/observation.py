from __future__ import annotations
from typing import cast

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    subtract_frame_transforms,
)
from mjlab.sensor import CameraSensor
import torch.nn.functional as F

from mjlab.entity import Entity
from .commands import MotionBarbellCommand

@torch.no_grad()
def barbell_qpos(env: ManagerBasedRlEnv):
    barbell_asset: Entity = env.scene["barbell"]

    return barbell_asset.data.root_com_pose_w

@torch.no_grad()
def bellbar_vector(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  return command.bellbar_vector

@torch.no_grad()
def right_hand_plate_vector(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  return command.right_hand_plate_vector

@torch.no_grad()
def left_hand_plate_vector(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  return command.left_hand_plate_vector

def motion_anchor_pos_b_ref(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    ref_anchor_pos_w = command.body_pos_relative_w[
        :, command.motion_anchor_body_index
    ]

    ref_anchor_quat_w = command.body_quat_relative_w[
        :, command.motion_anchor_body_index
    ]

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        ref_anchor_pos_w,
        ref_anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b_ref(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    ref_anchor_pos_w = command.body_pos_relative_w[
        :, command.motion_anchor_body_index
    ]

    ref_anchor_quat_w = command.body_quat_relative_w[
        :, command.motion_anchor_body_index
    ]

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        ref_anchor_pos_w,
        ref_anchor_quat_w,
    )

    mat = matrix_from_quat(ori)

    return mat[..., :2].reshape(mat.shape[0], -1)

def graping(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command : MotionBarbellCommand = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
    return command.graping.unsqueeze(1)