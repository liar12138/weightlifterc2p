from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from typing import cast
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.tasks.tracking.mdp.rewards import _get_body_indexes
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    quat_mul,
)
from .commands import MotionBarbellCommand
from mjlab.managers.scene_entity_config import SceneEntityCfg
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
    # 1. 计算 L2 范数 (保持维度以便广播)
    norm_a = torch.norm(a, dim=dim, keepdim=True)
    norm_b = torch.norm(b, dim=dim, keepdim=True)
    
    # 2. 归一化并计算点积，得到 cos(θ) (形状为 [batch])
    a_norm = a / (norm_a + eps)
    b_norm = b / (norm_b + eps)
    cos_theta = (a_norm * b_norm).sum(dim=dim)
    
    # 3. 数值裁剪，防止因浮点误差导致 acos 输入超出 [-1, 1] 而出现 nan
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # 4. 计算夹角 θ（弧度）
    theta = torch.acos(cos_theta).clip(0,torch.pi/2)
    
    # 5. 返回 cos(θ) - θ
    return -torch.sin(theta)

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
  body_names = ['left_hand_base_link', 'right_hand_base_link'],

) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]
  body_ids, _ = robot_asset.find_bodies(body_names)

  barbell_hands_distant = - (robot_asset.data.body_link_pos_w[:,body_ids, 2] - barbell_asset.data.root_link_pos_w[:, 2].unsqueeze(0).T).norm(dim = -1)
  
  return barbell_hands_distant.exp() 

import torch

@torch.no_grad()
def pently_hands_parrel(
    env: ManagerBasedRlEnv,
    command_name 
):
    robot_asset: Entity = env.scene["robot"]
    joit_pos = robot_asset.data.body_link_pos_w  
    command = env.command_manager.get_term(command_name)

    bellbar = command.bellbar_vector

    pos0 = joit_pos[:, command.left_hands[1], :3]
    pos1 = joit_pos[:, command.left_hands[0], :3]
    left = pos1 - pos0      # [batch, 3]
      
    pos0 = joit_pos[:, command.right_hands[0], :3]
    pos1 = joit_pos[:, command.right_hands[1], :3]
    right = pos1 - pos0     # [batch, 3]

    # ---- 计算夹角和 ----
    # 点积
    dot_left = torch.sum(left * bellbar, dim=-1)
    dot_right = torch.sum(right * bellbar, dim=-1)
    
    # 模长
    norm_left = torch.norm(left, dim=-1)
    norm_right = torch.norm(right, dim=-1)
    norm_bell = torch.norm(bellbar, dim=-1)
    
    eps = 1e-8  # 防止除零
    cos_left = dot_left / (norm_left * norm_bell + eps)
    cos_right = dot_right / (norm_right * norm_bell + eps)
    
    # 截断到 [-1, 1] 避免 acos 输入越界
    cos_left = torch.clamp(cos_left, -1.0, 1.0)
    cos_right = torch.clamp(cos_right, -1.0, 1.0)
    
    angle_left = torch.acos(cos_left)   # 弧度
    angle_right = torch.acos(cos_right)
    
    total_angle = angle_left + angle_right   # [batch]
    return total_angle

@torch.no_grad()
def left_hand_grap_bar(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  bv = command.bellbar_vector
  lv = command.left_hand_plate_vector

  return cos_theta_minus_theta(bv,lv) * command.graping.float()


@torch.no_grad()
def right_hand_grap_bar(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  bv = command.bellbar_vector
  rv = command.right_hand_plate_vector

  return cos_theta_minus_theta(bv,rv) * command.graping.float()

@torch.no_grad()
def both_hands_grip_bar_reward(
  env: ManagerBasedRlEnv,
  body_names=("left_hand_base_link", "right_hand_base_link"),
  grip_offset: float = 0.25,
  std: float = 0.10,
) -> torch.Tensor:

  robot: Entity = env.scene["robot"]
  barbell: Entity = env.scene["barbell"]

  hand_ids, _ = robot.find_bodies(body_names)

  # [num_envs, 2, 3]
  hand_pos_w = robot.data.body_link_pos_w[:, hand_ids, :]

  # [num_envs, 3]
  bar_pos_w = barbell.data.root_link_pos_w
  bar_quat_w = barbell.data.root_link_quat_w

  # 假设杠铃杆沿 barbell 的局部 x 轴
  bar_axis_local = torch.zeros_like(bar_pos_w)
  bar_axis_local[:, 0] = 1.0

  # 杠铃轴在世界坐标系中的方向
  bar_axis_w = quat_apply(bar_quat_w, bar_axis_local)

  left_target_w = bar_pos_w + grip_offset * bar_axis_w
  right_target_w = bar_pos_w - grip_offset * bar_axis_w

  left_dist = torch.linalg.vector_norm(
    hand_pos_w[:, 0] - left_target_w,
    dim=-1,
  )

  right_dist = torch.linalg.vector_norm(
    hand_pos_w[:, 1] - right_target_w,
    dim=-1,
  )

  # 奖励由距离更远的那只手决定，防止只伸一只手
  worst_dist = torch.maximum(left_dist, right_dist)

  return torch.exp(-torch.square(worst_dist / std))


@torch.no_grad()
def torso_height_below_reference_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  margin: float = 0.05,
):
  robot: Entity = env.scene["robot"]
  command: MotionBarbellCommand = env.command_manager.get_term(command_name)

  torso_ids, _ = robot.find_bodies(("torso_link",))
  torso_id = torso_ids[0]

  current_z = robot.data.body_link_pos_w[:, torso_id, 2]

  reference_z = command.body_pos_w[:, torso_id, 2]

  excessive_drop = torch.relu(reference_z - current_z - margin)

  return torch.square(excessive_drop / 0.10)

@torch.no_grad()
def finger_dof_tracking_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int =150,
    end_frame = 512,
) -> torch.Tensor:

  command = cast(
      MotionBarbellCommand,
      env.command_manager.get_term(command_name),
  )

  finger_ids = command.hands_index

  error = (
      command.joint_pos[:, finger_ids]
      - command.robot_joint_pos[:, finger_ids]
  )

  error = torch.mean(torch.square(error), dim=-1)

  active = ((command.time_steps >= start_frame) & (command.time_steps <= end_frame))

  return error * active.float()

@torch.no_grad()
def hands_barbell_height_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 150,
    body_names=(
        "left_hand_base_link",
        "right_hand_base_link",
    ),
    end_frame = 512,
) -> torch.Tensor:

    robot: Entity = env.scene["robot"]
    barbell: Entity = env.scene["barbell"]

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    hand_ids, _ = robot.find_bodies(body_names)

    # [num_envs, 2]
    hand_z = robot.data.body_link_pos_w[:, hand_ids, 2]

    # [num_envs, 1]
    barbell_z = barbell.data.root_link_pos_w[:, 2].unsqueeze(-1)

    # [num_envs, 2]
    height_error = torch.abs(hand_z - barbell_z)

    # 取左右手中高度误差更大的那只
    worst_error = torch.max(height_error, dim=-1).values

    penalty = torch.square(worst_error)

    # start_frame 之后才生效
    active = ((command.time_steps >= start_frame)  & (command.time_steps <= end_frame))

    return penalty * active.float()

@torch.no_grad()
def lower_body_dof_tracking_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    #start_frame: int = 200,
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

    error = (
        command.joint_pos[:, joint_ids]
        - command.robot_joint_pos[:, joint_ids]
    )

    error = torch.mean(torch.square(error), dim=-1)

    #active = command.time_steps >= start_frame 

    #return error * active.float()
    return error



@torch.no_grad()
def lower_body_dof_tracking_error_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 200,
    std: float = 0.10,
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

    error = (
        command.joint_pos[:, joint_ids]
        - command.robot_joint_pos[:, joint_ids]
    )

    error = torch.mean(torch.square(error), dim=-1)

    active = command.time_steps >= start_frame 
    reward = torch.exp(
            -torch.square(error / std)
        )
    return reward * active.float()

@torch.no_grad()
def ankle_dof_tracking_error(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    robot: Entity = env.scene["robot"]

    joint_names = (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )

    joint_ids, _ = robot.find_joints(joint_names)

    error = (
        command.joint_pos[:, joint_ids]
        - command.robot_joint_pos[:, joint_ids]
    )

    error = torch.mean(torch.square(error), dim=-1)

    return error 


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
def barbell_lift_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 200,
    target_height: float = 0.8,
) -> torch.Tensor:

    barbell: Entity = env.scene["barbell"]

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    barbell_z = barbell.data.root_link_pos_w[:, 2]

    active = command.time_step >= start_frame

    reward = torch.clamp(
        barbell_z / target_height,
        min=0.0,
        max=1.0,
    )

    return reward * active.float()

@torch.no_grad()
def barbell_height_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 220,
    end_frame: int = 400,
) -> torch.Tensor:

    barbell: Entity = env.scene["barbell"]

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    barbell_z = barbell.data.root_link_pos_w[:, 2]

    active = (command.time_steps >= start_frame) & (command.time_steps <= end_frame)

    return barbell_z * active.float()

@torch.no_grad()
def root_z_tracking_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 150,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    reference_z = command.anchor_pos_w[:, 2]

    robot_z = command.robot_anchor_pos_w[:, 2]

    error = torch.square(reference_z - robot_z)

    active = command.time_steps >= start_frame

    return error * active.float()

@torch.no_grad()
def root_z_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 220,
    std: float = 0.05,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    reference_z = command.anchor_pos_w[:, 2]
    robot_z = command.robot_anchor_pos_w[:, 2]

    error = reference_z - robot_z

    reward = torch.exp(
        -torch.square(error / std)
    )

    active = command.time_steps >= start_frame

    return reward * active.float()

@torch.no_grad()
def head_z_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 220,
    std: float = 0.05,
    head_body_name: str = "head_yaw_link",
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    robot: Entity = env.scene["robot"]

    head_body_ids, _ = robot.find_bodies(
        (head_body_name,)
    )

    head_id = head_body_ids[0]

    # 当前机器人头部世界坐标 z
    robot_head_z = robot.data.body_link_pos_w[:, head_id, 2]

    # 参考 motion 中对应 body 的 z
    reference_head_z = command.body_pos_w[:, head_id, 2]

    error = reference_head_z - robot_head_z

    reward = torch.exp(
        -torch.square(error / std)
    )

    active = command.time_steps >= start_frame

    return reward * active.float()

@torch.no_grad()
def fingers_grip_bar_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 70,
    end_frame: int = 483,
    target_radius: float = 0.025,
    std: float = 0.015,
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

    robot: Entity = env.scene["robot"]
    barbell: Entity = env.scene["barbell"]

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    finger_ids, _ = robot.find_bodies(finger_body_names)

    finger_pos_w = robot.data.body_link_pos_w[:, finger_ids, :]

    bar_pos_w = barbell.data.root_link_pos_w
    bar_quat_w = barbell.data.root_link_quat_w

    # 杠铃轴
    bar_axis_local = torch.zeros_like(bar_pos_w)
    bar_axis_local[:, 0] = 1.0

    bar_axis_w = quat_apply(
        bar_quat_w,
        bar_axis_local,
    )

    bar_axis_w = bar_axis_w / (
        torch.norm(bar_axis_w, dim=-1, keepdim=True) + 1e-8
    )

    # finger -> bar center
    vec = finger_pos_w - bar_pos_w[:, None, :]

    # 投影到 bar axis
    projection = torch.sum(
        vec * bar_axis_w[:, None, :],
        dim=-1,
    )

    closest_point = (
        bar_pos_w[:, None, :]
        + projection[..., None] * bar_axis_w[:, None, :]
    )

    # 到杠铃中心轴的径向距离
    radial_dist = torch.linalg.vector_norm(
        finger_pos_w - closest_point,
        dim=-1,
    )

    #
    # 关键修改：
    # 不再要求 radial_dist -> 0
    # 而是 radial_dist -> target_radius
    #
    radial_error = torch.abs(
        radial_dist - target_radius
    )

    radial_error = radial_error.reshape(
        radial_error.shape[0],
        8,
        2,
    )

    finger_error = torch.max(
        radial_error,
        dim=-1,
    ).values

    finger_reward = torch.exp(
        -torch.square(finger_error / std)
    )

    reward = finger_reward.mean(dim=-1)

    active = (
        (command.time_steps >= start_frame)
        & (command.time_steps <= end_frame)
    )

    return reward * active.float()


@torch.no_grad()
def fingers_barbell_contact_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    left_sensor_name: str,
    right_sensor_name: str,
    start_frame: int = 150,
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


    left_contact = (
        left_found.reshape(env.num_envs, -1) > 0
    ).float()

    right_contact = (
        right_found.reshape(env.num_envs, -1) > 0
    ).float()


    left_score = left_contact.mean(dim=-1)
    right_score = right_contact.mean(dim=-1)

    contact_reward = ( left_score + right_score) / 2.0

    both_contact = ( (left_score > 0) & (right_score > 0)).float()

    reward = contact_reward + 0.5 * both_contact

    # 150 ~ 528 帧生效
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

    command = cast(MotionBarbellCommand,env.command_manager.get_term(command_name))

    left_sensor = cast(ContactSensor,env.scene[left_sensor_name])

    right_sensor = cast(ContactSensor,env.scene[right_sensor_name])

    left_found = left_sensor.data.found
    right_found = right_sensor.data.found

    if left_found is None or right_found is None:
        return torch.zeros(env.num_envs,device=env.device)

    left_contact = (left_found.reshape(env.num_envs, -1) > 0).float()

    right_contact = (right_found.reshape(env.num_envs, -1) > 0).float()


    left_ratio = left_contact.mean(dim=-1)
    right_ratio = right_contact.mean(dim=-1)

    left_penalty = torch.relu(target_contact_ratio - left_ratio)

    right_penalty = torch.relu(target_contact_ratio - right_ratio)

    # 左右手分别计算，防止只抓一只手
    penalty = left_penalty + right_penalty

    active = (
        (command.time_steps >= start_frame)
        & (command.time_steps <= end_frame)
    )

    return penalty * active.float()

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
def head_dof_tracking_error_pen(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:

    command = cast(MotionBarbellCommand,env.command_manager.get_term(command_name))

    robot: Entity = env.scene["robot"]

    head_joint_ids, _ = robot.find_joints(("head_pitch_joint","head_yaw_joint"))

    error = (command.joint_pos[:, head_joint_ids]- command.robot_joint_pos[:, head_joint_ids])

    error = torch.mean(torch.square(error),dim=-1)

    return error

@torch.no_grad()
def head_dof_tracking_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.05,
) -> torch.Tensor:

    command = cast(MotionBarbellCommand,env.command_manager.get_term(command_name))

    robot: Entity = env.scene["robot"]

    head_joint_ids, _ = robot.find_joints(("head_pitch_joint","head_yaw_joint"))

    error = (command.joint_pos[:, head_joint_ids]- command.robot_joint_pos[:, head_joint_ids])

    result = torch.mean(torch.square(error),dim=-1)

    return torch.exp( -torch.square(result / std))

@torch.no_grad()
def hands_height_tracking_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 350,
    body_names=(
        "left_hand_base_link",
        "right_hand_base_link",
    ),
    std: float = 0.05,
) -> torch.Tensor:

    robot: Entity = env.scene["robot"]

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    hand_ids, _ = robot.find_bodies(body_names)

    current_hand_z = robot.data.body_link_pos_w[:, hand_ids, 2]


    target_hand_z = command.body_pos_w[:, hand_ids, 2]

    height_error = current_hand_z - target_hand_z

    result = torch.mean(
        torch.square(height_error),
        dim=-1,
    )
    active = command.time_steps >= start_frame

    return torch.exp( -torch.square(result / std)) * active.float()



@torch.no_grad()
def knee_height_tracking_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 170,
    body_names=(
        "left_knee_link",
        "right_knee_link",
    ),
    std: float =0.05,
) -> torch.Tensor:

    robot: Entity = env.scene["robot"]

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    # 找左右膝盖 body
    knee_ids, _ = robot.find_bodies(body_names)

    # 当前机器人膝盖高度
    # [num_envs, 2]
    current_knee_z = robot.data.body_link_pos_w[:, knee_ids, 2]

    # reference motion 中膝盖高度
    # [num_envs, 2]
    target_knee_z = command.body_pos_w[:, knee_ids, 2]

    # 高度误差
    height_error = current_knee_z - target_knee_z

    # 左右膝盖 MSE
    penalty = torch.mean(
        torch.square(height_error),
        dim=-1,
    )

    # 指定帧之后生效
    active = command.time_steps >= start_frame
    return torch.exp( -torch.square(penalty / std)) * active.float()
@torch.no_grad()
def knee_height_tracking_error_pen(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 170,
    body_names=(
        "left_knee_link",
        "right_knee_link",
    ),
) -> torch.Tensor:

    robot: Entity = env.scene["robot"]

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    # 找左右膝盖 body
    knee_ids, _ = robot.find_bodies(body_names)

    # 当前机器人膝盖高度
    # [num_envs, 2]
    current_knee_z = robot.data.body_link_pos_w[:, knee_ids, 2]

    # reference motion 中膝盖高度
    # [num_envs, 2]
    target_knee_z = command.body_pos_w[:, knee_ids, 2]

    # 高度误差
    height_error = current_knee_z - target_knee_z

    # 左右膝盖 MSE
    penalty = torch.mean(
        torch.square(height_error),
        dim=-1,
    )

    # 指定帧之后生效
    active = command.time_steps >= start_frame

    return penalty * active.float()


@torch.no_grad()
def feet_air_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    start_frame: int = 300,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    sensor = cast(
        ContactSensor,
        env.scene[sensor_name],
    )

    found = sensor.data.found

    if found is None:
        return torch.zeros(
            env.num_envs,
            device=env.device,
        )

    # [num_envs, num_contacts]
    contact = (
        found.reshape(env.num_envs, -1) > 0
    )

    # 只要完全没有脚接触地面，就认为双脚腾空
    any_foot_contact = contact.any(dim=-1)

    penalty = (~any_foot_contact).float()

    active = command.time_steps >= start_frame

    return penalty * active.float()

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
def reward_center_of_mass(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma_com: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:

  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]

  feet_indexes,_ = asset.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link",))
 
  com_xy = asset.data.root_com_pos_w[:, :2]

  z_l = command.robot_body_pos_w[:, feet_indexes[0], 2]
  z_r = command.robot_body_pos_w[:, feet_indexes[1], 2]

  single_support = torch.abs(z_l - z_r) > 0.05

  foot_l_xy = command.robot_body_pos_w[:, feet_indexes[0], :2]
  foot_r_xy = command.robot_body_pos_w[:, feet_indexes[1], :2]

  cond = z_l > z_r
  lower_foot_xy = torch.where(cond.unsqueeze(-1), foot_r_xy, foot_l_xy)

  error_sq = torch.sum((com_xy - lower_foot_xy) ** 2, dim=-1)

  reward = torch.exp(-error_sq / (sigma_com ** 2))

  return reward * single_support

def camera_hands_alignment_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    sensor_name: str = "h2_depth_camera",
    sigma: float = 0.05,
):
    robot = env.scene["robot"]
    sensor = env.scene[sensor_name]

    command: MotionBarbellCommand = (env.command_manager.get_term(command_name))

    # head world pose
    head_idx = robot.find_bodies(["head_yaw_link"])[0][0]

    head_pos_w = robot.data.body_link_pos_w[:, head_idx, :3]

    head_quat_w = robot.data.body_link_quat_w[:, head_idx]

    # camera local pose
    camera_local_pos = sensor._local_pos.expand( head_pos_w.shape[0], -1)

    camera_local_quat = sensor._local_quat.expand(head_quat_w.shape[0], -1)

    # camera world position
    camera_pos_w = head_pos_w + quat_apply(head_quat_w, camera_local_pos)

    # camera world orientation
    camera_quat_w = quat_mul(head_quat_w,camera_local_quat)

    # camera forward = local -Z
    camera_forward_local = torch.tensor(
        [0.0, 0.0, -1.0],
        device=head_pos_w.device,
        dtype=head_pos_w.dtype,
    ).expand(head_pos_w.shape[0], -1)

    camera_forward_w = quat_apply(camera_quat_w,camera_forward_local)

    # hands midpoint
    hands_mid_w = command.hands_midpoint_w

    camera_to_hands = hands_mid_w - camera_pos_w

    camera_to_hands = torch.nn.functional.normalize(camera_to_hands,dim=-1,eps=1e-6)

    camera_forward_w = torch.nn.functional.normalize(camera_forward_w,dim=-1,eps=1e-6)

    cos_sim = torch.sum(camera_forward_w * camera_to_hands,dim=-1).clamp(-1.0, 1.0)
    return torch.exp(-(1.0 - cos_sim) / sigma)