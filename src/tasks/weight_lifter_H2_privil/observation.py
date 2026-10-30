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

@torch.no_grad()
def barbell_pos_b(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:

    robot: Entity = env.scene["robot"]
    barbell: Entity = env.scene["barbell"]

    # robot root pose
    robot_pos_w = robot.data.root_link_pos_w
    robot_quat_w = robot.data.root_link_quat_w

    # barbell root pose
    barbell_pos_w = barbell.data.root_link_pos_w
    barbell_quat_w = barbell.data.root_link_quat_w

    # barbell pose expressed in robot frame
    pos_b, _ = subtract_frame_transforms(
        robot_pos_w,
        robot_quat_w,
        barbell_pos_w,
        barbell_quat_w,
    )

    return pos_b

@torch.no_grad()
def bellbar_vector_b(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    robot: Entity = env.scene["robot"]

    # [N, 3], world frame
    vector_w = command.bellbar_vector

    # robot rotation matrix: body -> world
    rot_wb = matrix_from_quat(
        robot.data.root_link_quat_w
    )

    # world -> body = R^T
    vector_b = torch.bmm(
        rot_wb.transpose(1, 2),
        vector_w.unsqueeze(-1),
    ).squeeze(-1)

    # bellbar_vector 主要表达方向，所以建议归一化
    vector_b = F.normalize(
        vector_b,
        dim=-1,
        eps=1e-6,
    )

    return vector_b

@torch.no_grad()
def left_hand_to_barbell_b(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    robot: Entity = env.scene["robot"]
    barbell: Entity = env.scene["barbell"]

    hand_pos_w = (
        robot.data.body_link_pos_w[
            :, command.garp_point["left_hand"], :3
        ]
        .mean(dim=1)
    )

    geom_pos_w = barbell.data.geom_pos_w

    target_pos_w = geom_pos_w[
        :, command.plates_geom_index[1], :
    ]
    vector_w = target_pos_w - hand_pos_w

    rot_wb = matrix_from_quat(
        robot.data.root_link_quat_w
    )

    vector_b = torch.bmm(
        rot_wb.transpose(1, 2),
        vector_w.unsqueeze(-1),
    ).squeeze(-1)

    return vector_b

@torch.no_grad()
def right_hand_to_barbell_b(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    robot: Entity = env.scene["robot"]
    barbell: Entity = env.scene["barbell"]

    hand_pos_w = (
        robot.data.body_link_pos_w[
            :, command.garp_point["right_hand"], :3
        ]
        .mean(dim=1)
    )

    geom_pos_w = barbell.data.geom_pos_w

    target_pos_w = geom_pos_w[
        :, command.plates_geom_index[0], :
    ]

    # target - hand
    vector_w = target_pos_w - hand_pos_w

    rot_wb = matrix_from_quat(
        robot.data.root_link_quat_w
    )

    vector_b = torch.bmm(
        rot_wb.transpose(1, 2),
        vector_w.unsqueeze(-1),
    ).squeeze(-1)

    return vector_b