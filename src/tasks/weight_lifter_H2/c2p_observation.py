"""Ground-truth C2P labels kept separate from estimator inputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import cast
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms
from mjlab.entity import Entity
from .commands import MotionBarbellCommand


@torch.no_grad()
def barbell_quat_b(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    barbell: Entity = env.scene["barbell"]
    _, quat_b = subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        barbell.data.root_link_pos_w,
        barbell.data.root_link_quat_w,
    )
    return quat_b


@torch.no_grad()
def barbell_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    barbell: Entity = env.scene["barbell"]
    pos, _ = subtract_frame_transforms(
        robot.data.root_link_pos_w, robot.data.root_link_quat_w,
        barbell.data.root_link_pos_w, barbell.data.root_link_quat_w,
    )
    return pos


@torch.no_grad()
def bellbar_vector_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
    rot_wb = matrix_from_quat(env.scene["robot"].data.root_link_quat_w)
    vector = torch.bmm(rot_wb.transpose(1, 2), command.bellbar_vector.unsqueeze(-1)).squeeze(-1)
    return F.normalize(vector, dim=-1, eps=1e-6)


@torch.no_grad()
def _hand_to_plate_b(env: ManagerBasedRlEnv, command_name: str, side: str) -> torch.Tensor:
    command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
    robot: Entity = env.scene["robot"]
    barbell: Entity = env.scene["barbell"]
    hand_pos = robot.data.body_link_pos_w[:, command.garp_point[f"{side}_hand"], :3].mean(dim=1)
    plate_index = command.plates_geom_index[1 if side == "left" else 0]
    vector_w = barbell.data.geom_pos_w[:, plate_index] - hand_pos
    rot_wb = matrix_from_quat(robot.data.root_link_quat_w)
    return torch.bmm(rot_wb.transpose(1, 2), vector_w.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def left_hand_to_barbell_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _hand_to_plate_b(env, command_name, "left")


@torch.no_grad()
def right_hand_to_barbell_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _hand_to_plate_b(env, command_name, "right")


@torch.no_grad()
def c2p_target_16d(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    del command_name
    return torch.cat(
        (
            barbell_pos_b(env),
            barbell_quat_b(env),
            bellbar_vector_b(env, "motion"),
            left_hand_to_barbell_b(env, "motion"),
            right_hand_to_barbell_b(env, "motion"),
        ),
        dim=-1,
    )


__all__ = ["c2p_target_16d", "barbell_quat_b"]
