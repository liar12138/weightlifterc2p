from typing import TYPE_CHECKING, cast


import torch

from mjlab.utils.lab_api.math import quat_apply_inverse,quat_apply

from .commands import MotionCommand
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor

@torch.no_grad()
def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w - command.robot_anchor_pos_w)
    > threshold
  ).any(-1)


@torch.no_grad()
def barbell_droped_from_hands(
  env: ManagerBasedRlEnv,
  command_name: str,
  limit_distant: float,
  joint_names = ['left_hand_base_link', 'right_hand_base_link'],
):
  """Terminate when the asset's orientation exceeds the limit angle."""

  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]

  indexes,_ = robot_asset.find_bodies(joint_names)
  barbell_hands_distant = torch.abs(robot_asset.data.body_link_pos_w[:,indexes, 2] - barbell_asset.data.root_link_pos_w[:, 2].unsqueeze(0).T)
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  result=(barbell_hands_distant > limit_distant).any(-1) & command.graping
#   if result :
#      print("barbell_droped_from_hands")
  return result
# @torch.no_grad()
# def barbell_dropped_from_hands(
#     env: ManagerBasedRlEnv,
#     command_name: str,
#     limit_distance: float = 0.12,
#     check_start_frame: int = 220,
#     grip_offset: float = 0.25,
#     hand_body_names=(
#         "left_hand_base_link",
#         "right_hand_base_link",
#     ),
# ):
#     robot: Entity = env.scene["robot"]
#     barbell: Entity = env.scene["barbell"]

#     hand_ids, _ = robot.find_bodies(hand_body_names)

#     hand_pos_w = robot.data.body_link_pos_w[:, hand_ids, :]

#     bar_pos_w = barbell.data.root_link_pos_w
#     bar_quat_w = barbell.data.root_link_quat_w

#     # 杠铃局部 x 轴
#     bar_axis_local = torch.zeros_like(bar_pos_w)
#     bar_axis_local[:, 0] = 1.0

#     bar_axis_w = quat_apply(bar_quat_w, bar_axis_local)

#     left_target_w = bar_pos_w + grip_offset * bar_axis_w
#     right_target_w = bar_pos_w - grip_offset * bar_axis_w

#     left_dist = torch.linalg.vector_norm(
#         hand_pos_w[:, 0] - left_target_w,
#         dim=-1,
#     )

#     right_dist = torch.linalg.vector_norm(
#         hand_pos_w[:, 1] - right_target_w,
#         dim=-1,
#     )

#     command = cast(
#         MotionCommand,
#         env.command_manager.get_term(command_name),
#     )

#     after_check_frame = command.time_step >= check_start_frame

#     hands_too_far = (
#         (left_dist > limit_distance)
#         | (right_dist > limit_distance)
#     )

#     return after_check_frame & hands_too_far

@torch.no_grad()
def far_away_root_pos(
  env: ManagerBasedRlEnv,
  limit_distant: float,
):
  """Terminate when the asset's orientation exceeds the limit angle."""

  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]

  barbell_hands_distant = torch.abs(robot_asset.data.root_com_pos_w[:,:2] - barbell_asset.data.root_link_pos_w[:,:2]).mean(-1)
  result =barbell_hands_distant > limit_distant 
#   if result :
#      print("far_away_root_pos")
  return result

def bad_hands_dof(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
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

    command = cast(MotionCommand, env.command_manager.get_term(command_name))

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


from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.contact_sensor import ContactSensor


@torch.no_grad()
def knee_ground_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Terminate when either knee or shin touches the terrain."""

    sensor = cast(ContactSensor, env.scene[sensor_name])

    found = sensor.data.found
    print("force_history",found)
    if found is None:
        return torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
    return found.reshape(env.num_envs, -1).any(dim=1)

@torch.no_grad()
def bad_lower_body_dof(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """下肢关节与参考动作偏差过大时终止。"""

    command = cast(
        MotionCommand,
        env.command_manager.get_term(command_name),
    )

    robot: Entity = env.scene["robot"]

    joint_names = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
    )

    joint_ids, matched_names = robot.find_joints(joint_names)

    # 调试时确认确实匹配到了 8 个关节
    if len(joint_ids) != len(joint_names):
        raise RuntimeError(
            f"下肢关节匹配失败：\n"
            f"requested={joint_names}\n"
            f"matched={matched_names}"
        )

    reference_pos = command.joint_pos[:, joint_ids]
    current_pos = command.robot_joint_pos[:, joint_ids]

    error = torch.abs(reference_pos - current_pos)

    # 任意一个关节偏差超过阈值，就终止
    return torch.any(error > threshold, dim=-1)

@torch.no_grad()
def bad_finger_dof(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:

  command = cast(
      MotionCommand,
      env.command_manager.get_term(command_name),
  )

  finger_ids = command.hands_index

  # [num_envs, 22]
  error = torch.abs(
      command.joint_pos[:, finger_ids]
      - command.robot_joint_pos[:, finger_ids]
  )

  # [num_envs]
  bad_tracking = torch.any(error > threshold, dim=-1)
  # 只在抓握阶段触发终止
  return bad_tracking & command.graping

@torch.no_grad()
def knee_barbell_contact_termination(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:

    sensor = cast(
        ContactSensor,
        env.scene[sensor_name],
    )

    force = sensor.data.force

    if force is None:
        return torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    # [N, 2, 3] -> [N, 2]
    force_mag = torch.linalg.vector_norm(
        force,
        dim=-1,
    )

    # 任意一个膝盖产生接触力就终止
    return (force_mag > 1e-6).any(dim=-1)


@torch.no_grad()
def non_foot_ground_contact_termination(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1e-6,
) -> torch.Tensor:
    """除双脚外的任何身体部位触地即终止。"""
    sensor = cast(ContactSensor, env.scene[sensor_name])
    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    mag = torch.norm(force.reshape(env.num_envs, -1, 3), dim=-1)
    return (mag > force_threshold).any(dim=-1)


@torch.no_grad()
def bar_body_contact_termination(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """杠铃与非手部身体部位发生强力碰撞即终止。"""
    sensor = cast(ContactSensor, env.scene[sensor_name])
    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    mag = torch.norm(force.reshape(env.num_envs, -1, 3), dim=-1)
    return (mag > force_threshold).any(dim=-1)