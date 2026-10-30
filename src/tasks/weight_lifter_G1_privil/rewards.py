from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from .commands import MotionBarbellCommand


@torch.no_grad()
def cos_theta_minus_theta(a, b, dim=-1, eps=1e-8):
  """
  计算 cos(θ) - θ，其中 θ 是向量 a 和 b 之间的夹角（弧度制）。

  Args:
      a (torch.Tensor): 形状为 [batch, 3] 或更通用的 [..., dim]
      b (torch.Tensor): 形状与 a 相同
      dim (int): 特征维度，对于 [batch, 3] 可设为 1 或保留默认 -1
      eps (float): 防止分母为零的小常数

  Returns:
      torch.Tensor: 形状为 [batch, ...]，值为 cos(θ) - θ（弧度）
  """
  norm_a = torch.norm(a, dim=dim, keepdim=True)
  norm_b = torch.norm(b, dim=dim, keepdim=True)

  a_norm = a / (norm_a + eps)
  b_norm = b / (norm_b + eps)
  cos_theta = (a_norm * b_norm).sum(dim=dim)

  cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
  theta = torch.acos(cos_theta).clip(0, torch.pi / 2)
  return -torch.sin(theta)


@torch.no_grad()
def barbell_reward_hands_height(
  env: ManagerBasedRlEnv,
  body_names=("left_index_distal", "right_index_distal"),
) -> torch.Tensor:
  """Reward hands staying near barbell height."""
  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]
  indexes, _ = robot_asset.find_bodies(list(body_names))

  barbell_hands_distant = -(
    robot_asset.data.body_link_pos_w[:, indexes, 2]
    - barbell_asset.data.root_link_pos_w[:, 2].unsqueeze(0).T
  ).norm(dim=-1)

  return barbell_hands_distant.exp()


@torch.no_grad()
def pently_hands_parrel(
  env: ManagerBasedRlEnv,
  command_name: str,
):
  robot_asset: Entity = env.scene["robot"]
  joit_pos = robot_asset.data.body_link_pos_w
  command = env.command_manager.get_term(command_name)

  bellbar = command.bellbar_vector

  pos0 = joit_pos[:, command.left_hands[1], :3]
  pos1 = joit_pos[:, command.left_hands[0], :3]
  left = pos0 - pos1

  pos0 = joit_pos[:, command.right_hands[0], :3]
  pos1 = joit_pos[:, command.right_hands[1], :3]
  right = pos0 - pos1

  dot_left = torch.sum(left * bellbar, dim=-1)
  dot_right = torch.sum(right * bellbar, dim=-1)

  norm_left = torch.norm(left, dim=-1)
  norm_right = torch.norm(right, dim=-1)
  norm_bell = torch.norm(bellbar, dim=-1)

  eps = 1e-8
  cos_left = torch.clamp(dot_left / (norm_left * norm_bell + eps), -1.0, 1.0)
  cos_right = torch.clamp(dot_right / (norm_right * norm_bell + eps), -1.0, 1.0)

  return torch.acos(cos_left) + torch.acos(cos_right)


@torch.no_grad()
def left_hand_grap_bar(
  env: ManagerBasedRlEnv,
  command_name: str,
):
  command: MotionBarbellCommand = env.command_manager.get_term(command_name)
  return (
    cos_theta_minus_theta(command.bellbar_vector, command.left_hand_plate_vector)
    * command.graping.float()
  )


@torch.no_grad()
def right_hand_grap_bar(
  env: ManagerBasedRlEnv,
  command_name: str,
):
  command: MotionBarbellCommand = env.command_manager.get_term(command_name)
  return (
    cos_theta_minus_theta(command.bellbar_vector, command.right_hand_plate_vector)
    * command.graping.float()
  )


@torch.no_grad()
def fingers_grap(
  env: ManagerBasedRlEnv,
  command_name: str,
):
  """Penalize open grasp: sum of thumb-to-fingertip distances (5-finger enclosure)."""
  command: MotionBarbellCommand = env.command_manager.get_term(command_name)

  def enclosure_span(tips: torch.Tensor) -> torch.Tensor:
    # tips: [B, 5, 3] — index 0 is thumb, 1..4 are other fingertips.
    thumb = tips[:, 0:1, :]
    others = tips[:, 1:, :]
    return (others - thumb).norm(dim=-1).sum(dim=-1)

  span = enclosure_span(command.left_fingers) + enclosure_span(command.right_fingers)
  return span * command.graping.float()* command.graping.float()


@torch.no_grad()
def fingers_barbell_contact_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    left_sensor_name: str,
    right_sensor_name: str,
    start_frame: int = 160,
    end_frame: int = 528,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    left_sensor = cast(
        ContactSensor,
        env.scene[left_sensor_name],
    )

    right_sensor = cast(
        ContactSensor,
        env.scene[right_sensor_name],
    )

    left_found = left_sensor.data.found
    right_found = right_sensor.data.found

    if left_found is None or right_found is None:
        return torch.zeros(
            env.num_envs,
            device=env.device,
        )

    left_score = (
        left_found.reshape(env.num_envs, -1) > 0
    ).float().mean(dim=-1)

    right_score = (
        right_found.reshape(env.num_envs, -1) > 0
    ).float().mean(dim=-1)

    # 关键修改：
    # 奖励由接触更差的那只手决定
    reward = torch.minimum(
        left_score,
        right_score,
    )

    active = (
        (command.time_steps >= start_frame)
        & (command.time_steps <= end_frame)
    )

    return reward * active.float()

@torch.no_grad()
def fingers_barbell_contact_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  left_sensor_name: str,
  right_sensor_name: str,
  start_frame: int = 150,
  end_frame: int = 528,
  target_contact_ratio: float = 0.25,
) -> torch.Tensor:
  """Penalize insufficient finger-shaft contact (ported from H2)."""
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  left_sensor = cast(ContactSensor, env.scene[left_sensor_name])
  right_sensor = cast(ContactSensor, env.scene[right_sensor_name])

  left_found = left_sensor.data.found
  right_found = right_sensor.data.found
  if left_found is None or right_found is None:
    return torch.zeros(env.num_envs, device=env.device)

  left_ratio = (left_found.reshape(env.num_envs, -1) > 0).float().mean(dim=-1)
  right_ratio = (right_found.reshape(env.num_envs, -1) > 0).float().mean(dim=-1)
  penalty = torch.relu(target_contact_ratio - left_ratio) + torch.relu(
    target_contact_ratio - right_ratio
  )

  active = (command.time_steps >= start_frame) & (command.time_steps <= end_frame)
  return penalty * active.float()


@torch.no_grad()
def both_hands_grip_bar_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names=("left_hand_base_link", "right_hand_base_link"),
  grip_offset: float = 0.25,
  std: float = 0.10,
) -> torch.Tensor:
  """Reward both palms near target grip points along the bar axis (H2)."""
  robot: Entity = env.scene["robot"]
  barbell: Entity = env.scene["barbell"]
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  hand_ids, _ = robot.find_bodies(list(body_names))
  hand_pos_w = robot.data.body_link_pos_w[:, hand_ids, :]
  bar_pos_w = barbell.data.root_link_pos_w
  bar_quat_w = barbell.data.root_link_quat_w

  # Olympic barbell axis is local +Y (see olympic_barbell.xml). Athlete left hand
  # sits on +Y (XML right_grip), athlete right on -Y (XML left_grip).
  bar_axis_local = torch.zeros_like(bar_pos_w)
  bar_axis_local[:, 1] = 1.0
  bar_axis_w = quat_apply(bar_quat_w, bar_axis_local)

  left_target_w = bar_pos_w + grip_offset * bar_axis_w
  right_target_w = bar_pos_w - grip_offset * bar_axis_w

  left_dist = torch.linalg.vector_norm(hand_pos_w[:, 0] - left_target_w, dim=-1)
  right_dist = torch.linalg.vector_norm(hand_pos_w[:, 1] - right_target_w, dim=-1)
  worst_dist = torch.maximum(left_dist, right_dist)
  return torch.exp(-torch.square(worst_dist / std)) * command.graping.float()


@torch.no_grad()
def hands_barbell_height_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  start_frame: int = 150,
  body_names=("left_hand_base_link", "right_hand_base_link"),
) -> torch.Tensor:
  """Penalize hand-barbell height mismatch after start_frame (H2)."""
  robot: Entity = env.scene["robot"]
  barbell: Entity = env.scene["barbell"]
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))

  hand_ids, _ = robot.find_bodies(list(body_names))
  hand_z = robot.data.body_link_pos_w[:, hand_ids, 2]
  barbell_z = barbell.data.root_link_pos_w[:, 2].unsqueeze(-1)
  worst_error = torch.max(torch.abs(hand_z - barbell_z), dim=-1).values
  active = command.time_steps >= start_frame
  return torch.square(worst_error) * active.float()


@torch.no_grad()
def barbell_height_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  start_frame: int = 220,
  end_frame: int = 400,
) -> torch.Tensor:
  """Reward barbell height during the lift window (H2)."""
  barbell: Entity = env.scene["barbell"]
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  barbell_z = barbell.data.root_link_pos_w[:, 2]
  active = (command.time_steps >= start_frame) & (command.time_steps <= end_frame)
  return barbell_z * active.float()


@torch.no_grad()
def finger_dof_tracking_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  start_frame: int = 150,
  end_frame: int = 512,
) -> torch.Tensor:
  """Penalize finger joint tracking error in the grasp window (H2)."""
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  finger_ids = command.hands_index
  error = command.joint_pos[:, finger_ids] - command.robot_joint_pos[:, finger_ids]
  error = torch.mean(torch.square(error), dim=-1)
  active = (command.time_steps >= start_frame) & (command.time_steps <= end_frame)
  return error * active.float()


@torch.no_grad()
def lower_body_dof_tracking_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  start_frame: int = 200,
) -> torch.Tensor:
  """Penalize lower-body joint tracking error after start_frame (H2)."""
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  robot: Entity = env.scene["robot"]
  joint_names = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
  )
  joint_ids, _ = robot.find_joints(joint_names)
  error = command.joint_pos[:, joint_ids] - command.robot_joint_pos[:, joint_ids]
  error = torch.mean(torch.square(error), dim=-1)
  active = command.time_steps >= start_frame
  return error * active.float()


@torch.no_grad()
def root_z_tracking_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  start_frame: int = 150,
) -> torch.Tensor:
  """Penalize torso/anchor height tracking error after start_frame (H2)."""
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
  active = command.time_steps >= start_frame
  return error * active.float()


@torch.no_grad()
def fingers_grip_bar_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  start_frame: int = 150,
  end_frame: int = 512,
  std: float = 0.05,
  finger_body_names=(
    "left_index_proximal",
    "left_index_distal",
    "left_middle_proximal",
    "left_middle_distal",
    "left_ring_proximal",
    "left_ring_distal",
    "left_pinky_proximal",
    "left_pinky_distal",
    "right_index_proximal",
    "right_index_distal",
    "right_middle_proximal",
    "right_middle_distal",
    "right_ring_proximal",
    "right_ring_distal",
    "right_pinky_proximal",
    "right_pinky_distal",
  ),
) -> torch.Tensor:
  """Reward finger links staying close to the bar axis (H2)."""
  robot: Entity = env.scene["robot"]
  barbell: Entity = env.scene["barbell"]
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))

  finger_ids, _ = robot.find_bodies(list(finger_body_names))
  finger_pos_w = robot.data.body_link_pos_w[:, finger_ids, :]
  bar_pos_w = barbell.data.root_link_pos_w
  bar_quat_w = barbell.data.root_link_quat_w

  # Olympic barbell shaft axis is local +Y, not +X.
  bar_axis_local = torch.zeros_like(bar_pos_w)
  bar_axis_local[:, 1] = 1.0
  bar_axis_w = quat_apply(bar_quat_w, bar_axis_local)
  bar_axis_w = bar_axis_w / (torch.norm(bar_axis_w, dim=-1, keepdim=True) + 1e-8)

  vec = finger_pos_w - bar_pos_w[:, None, :]
  projection = torch.sum(vec * bar_axis_w[:, None, :], dim=-1)
  closest_point = bar_pos_w[:, None, :] + projection[..., None] * bar_axis_w[:, None, :]
  finger_dist = torch.linalg.vector_norm(finger_pos_w - closest_point, dim=-1)
  # 8 fingers x (proximal, distal)
  finger_dist = finger_dist.reshape(finger_dist.shape[0], 8, 2)
  finger_worst_dist = torch.max(finger_dist, dim=-1).values
  reward = torch.exp(-torch.square(finger_worst_dist / std)).mean(dim=-1)

  active = (command.time_steps >= start_frame) & (command.time_steps <= end_frame)
  return reward * active.float()



@torch.no_grad()
def knee_barbell_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:

    sensor = cast(
        ContactSensor,
        env.scene[sensor_name],
    )

    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs,device=env.device)

    force_mag = torch.linalg.vector_norm(force,dim=-1)

    force_mag = force_mag.reshape(env.num_envs,-1)

    contact = (force_mag > 1e-6).any(dim=-1)

    return contact.float()


@torch.no_grad()
def lower_body_dof_tracking_error_no_ankle(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 200,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
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

    joint_ids, _ = robot.find_joints(joint_names)

    error = (
        command.joint_pos[:, joint_ids]
        - command.robot_joint_pos[:, joint_ids]
    )

    error = torch.mean(torch.square(error), dim=-1)

    active = command.time_steps >= start_frame

    return error * active.float()


@torch.no_grad()
def knee_barbell_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:

    sensor = cast(
        ContactSensor,
        env.scene[sensor_name],
    )

    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs,device=env.device)

    force_mag = torch.linalg.vector_norm(force,dim=-1)

    force_mag = force_mag.reshape(env.num_envs,-1)

    contact = (force_mag > 1e-6).any(dim=-1)

    return contact.float()

@torch.no_grad()
def knee_dof_tracking_error(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    robot: Entity = env.scene["robot"]

    # 找左右膝关节
    knee_ids, _ = robot.find_joints(
        (
            "left_knee_joint",
            "right_knee_joint",
        )
    )

    # reference - current
    error = (
        command.joint_pos[:, knee_ids]
        - command.robot_joint_pos[:, knee_ids]
    )

    # 两个膝关节的 MSE
    error = torch.mean(
        torch.square(error),
        dim=-1,
    )

    return error

@torch.no_grad()
def fingers_opposition_grasp_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 160,
    end_frame: int = 528,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    barbell: Entity = env.scene["barbell"]

    bar_pos = barbell.data.root_link_pos_w
    bar_quat = barbell.data.root_link_quat_w

    # 杠铃轴：local +Y
    bar_axis_local = torch.zeros_like(bar_pos)
    bar_axis_local[:, 1] = 1.0

    bar_axis = quat_apply(
        bar_quat,
        bar_axis_local,
    )

    bar_axis = bar_axis / (
        torch.norm(bar_axis, dim=-1, keepdim=True) + 1e-8
    )

    def hand_opposition(
        finger_pos: torch.Tensor,
    ):
        # finger_pos: [N, 5, 3]
        # 0 = thumb
        # 1:5 = index/middle/ring/pinky

        thumb = finger_pos[:, 0, :]
        others = finger_pos[:, 1:, :].mean(dim=1)

        # --------------------------------
        # thumb 到杠铃轴的径向向量
        # --------------------------------
        thumb_vec = thumb - bar_pos

        thumb_proj = torch.sum(
            thumb_vec * bar_axis,
            dim=-1,
            keepdim=True,
        )

        thumb_radial = (
            thumb_vec
            - thumb_proj * bar_axis
        )

        # --------------------------------
        # 其他四指到杠铃轴的径向向量
        # --------------------------------
        other_vec = others - bar_pos

        other_proj = torch.sum(
            other_vec * bar_axis,
            dim=-1,
            keepdim=True,
        )

        other_radial = (
            other_vec
            - other_proj * bar_axis
        )

        # normalize
        thumb_radial = thumb_radial / (
            torch.norm(
                thumb_radial,
                dim=-1,
                keepdim=True,
            ) + 1e-8
        )

        other_radial = other_radial / (
            torch.norm(
                other_radial,
                dim=-1,
                keepdim=True,
            ) + 1e-8
        )

        # cos = -1 最好
        # 说明 thumb 和其他四指在杆子的两侧
        cos = torch.sum(
            thumb_radial * other_radial,
            dim=-1,
        )

        # cos = -1 -> reward = 1
        # cos =  1 -> reward = 0
        reward = (1.0 - cos) / 2.0

        return reward

    left_reward = hand_opposition(
        command.left_fingers
    )

    right_reward = hand_opposition(
        command.right_fingers
    )

    # 仍然使用更差的那只手决定奖励
    # 防止一只手很好、一只手很差
    reward = torch.minimum(
        left_reward,
        right_reward,
    )

    active = (
        (command.time_steps >= start_frame)
        & (command.time_steps <= end_frame)
    )

    return reward * active.float()

@torch.no_grad()
def fingers_opposition_grasp_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 160,
    end_frame: int = 528,
    target_cos: float = -0.5,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    barbell: Entity = env.scene["barbell"]

    bar_pos = barbell.data.root_link_pos_w
    bar_quat = barbell.data.root_link_quat_w

    # 杠铃轴 local +Y
    bar_axis_local = torch.zeros_like(bar_pos)
    bar_axis_local[:, 1] = 1.0

    bar_axis = quat_apply(
        bar_quat,
        bar_axis_local,
    )

    bar_axis = bar_axis / (
        torch.norm(bar_axis, dim=-1, keepdim=True) + 1e-8
    )

    def hand_opposition_cos(
        finger_pos: torch.Tensor,
    ):
        # 0 = thumb
        # 1:5 = 其他四指
        thumb = finger_pos[:, 0, :]
        others = finger_pos[:, 1:, :].mean(dim=1)

        # thumb 到 bar axis 的径向向量
        thumb_vec = thumb - bar_pos

        thumb_proj = torch.sum(
            thumb_vec * bar_axis,
            dim=-1,
            keepdim=True,
        )

        thumb_radial = (
            thumb_vec
            - thumb_proj * bar_axis
        )

        # 其他四指到 bar axis 的径向向量
        other_vec = others - bar_pos

        other_proj = torch.sum(
            other_vec * bar_axis,
            dim=-1,
            keepdim=True,
        )

        other_radial = (
            other_vec
            - other_proj * bar_axis
        )

        thumb_radial = thumb_radial / (
            torch.norm(
                thumb_radial,
                dim=-1,
                keepdim=True,
            ) + 1e-8
        )

        other_radial = other_radial / (
            torch.norm(
                other_radial,
                dim=-1,
                keepdim=True,
            ) + 1e-8
        )

        cos = torch.sum(
            thumb_radial * other_radial,
            dim=-1,
        )

        return cos

    left_cos = hand_opposition_cos(
        command.left_fingers
    )

    right_cos = hand_opposition_cos(
        command.right_fingers
    )

    # cos 越大越差
    # target_cos 例如 -0.5
    left_penalty = torch.relu(
        left_cos - target_cos
    )

    right_penalty = torch.relu(
        right_cos - target_cos
    )

    # 左右手分别处罚
    penalty = (
        left_penalty
        + right_penalty
    )

    active = (
        (command.time_steps >= start_frame)
        & (command.time_steps <= end_frame)
    )

    return penalty * active.float()