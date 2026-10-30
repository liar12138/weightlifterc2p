"""Reward terms specific to the H2 velocity-tracking tasks."""
from __future__ import annotations

import torch
import re

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import variable_posture,_DEFAULT_ASSET_CFG

class PakPose(variable_posture):

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    robot: Entity = env.scene['robot']

    left_leg_pattern = re.compile(r'^left_(hip|knee|ankle)_.*')
    right_leg_pattern = re.compile(r'^right_(hip|knee|ankle)_.*')
    left_arm_pattern = re.compile(r'^left_(shoulder|elbow|wrist)_.*')
    right_arm_pattern = re.compile(r'^right_(shoulder|elbow|wrist)_.*')

    # 迈左腿时保留惩罚的关节：去掉左腿（摆动腿）+ 去掉右手（对侧手臂随摆）
    _penalize_left_active = [
      s for s in robot.joint_names
      if not left_leg_pattern.match(s) and not right_arm_pattern.match(s)
    ]
    # 迈右腿时保留惩罚的关节：去掉右腿 + 去掉左手
    _penalize_right_active = [
      s for s in robot.joint_names
      if not right_leg_pattern.match(s) and not left_arm_pattern.match(s)
    ]

    self._penalize_left_active_idx = robot.find_joints(_penalize_left_active)[0]
    self._penalize_right_active_idx = robot.find_joints(_penalize_right_active)[0]
    self.ankle_idx = robot.find_bodies(
      ['left_ankle_pitch_link', 'right_ankle_pitch_link']
    )[0]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg,
    command_name: str,
    walking_threshold: float = 0.1,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = total_speed < walking_threshold   # bool [env]
    moving_mask = ~standing_mask

    # 主动脚：速度较大的脚（相等时选左脚）
    robot: Entity = env.scene['robot']
    body_v = robot.data.body_link_lin_vel_w[:, self.ankle_idx]
    left_speed = torch.norm(body_v[:, 0], dim=1)
    right_speed = torch.norm(body_v[:, 1], dim=1)
    active_left = left_speed >= right_speed           # bool [env]
    active_right = ~active_left

    # 每个关节的绝对偏差, [env, 31]
    current_joint_pos = asset.data.joint_pos
    desired_joint_pos = self.default_joint_pos[0]
    err = (current_joint_pos - desired_joint_pos).abs()

    # 惩罚 = 选中关节中的最大绝对偏差（越大越差）
    penalty = torch.zeros(current_joint_pos.shape[0], device=current_joint_pos.device)

    # standing：惩罚所有关节（等价于 penalty_standing_pose）
    if standing_mask.any():
      penalty[standing_mask] = err[standing_mask].min(dim=-1).values

    # moving：迈左腿 -> 不惩罚左腿 + 右手；迈右腿 -> 不惩罚右腿 + 左手
    if (moving_mask & active_left).any():
      sel = moving_mask & active_left
      penalty[sel] = err[sel][:, self._penalize_left_active_idx].min(dim=-1).values
    if (moving_mask & active_right).any():
      sel = moving_mask & active_right
      penalty[sel] = err[sel][:, self._penalize_right_active_idx].min(dim=-1).values

    return torch.cos(penalty)

from src.tasks.weight_lifter.rewards import cos_angle_minus_abs_angle

@torch.no_grad()
def penalty_standing_pose(env: ManagerBasedRlEnv):
  command = env.command_manager.get_command('twist')
  robot: Entity = env.scene['robot']

  linear_speed = torch.norm(command[:, :2], dim=1)
  angular_speed = torch.abs(command[:, 2])
  total_speed = linear_speed + angular_speed

  standing_mask = (total_speed < 0.1)

  error = torch.max((robot.data.default_joint_pos- robot.data.joint_pos).abs(),dim = -1).values

  return torch.where(standing_mask,error,0.0)

@torch.no_grad()
def penalty_electrical_power_cost(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    deadband_w: float = 150.0,
    normalization_w: float = 500.0,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]

    joint_ids, _ = robot.find_joints(asset_cfg.joint_names)
    actuator_ids, _ = robot.find_actuators(asset_cfg.joint_names)

    torque = robot.data.actuator_force[:, actuator_ids]
    joint_velocity = robot.data.joint_vel[:, joint_ids]

    output_power = -torque * joint_velocity
    excessive_power = torch.relu(output_power - deadband_w)

    return torch.sum(
        torch.square(excessive_power / normalization_w),
        dim=1,
    )

def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  value = torch.exp(-lin_vel_error / std**2) - 0.01 * lin_vel_error
  return torch.where((command[:, :2].norm(2,dim=-1) > 0.1) , value, - 10 * xy_error)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  value = torch.exp(-ang_vel_error / std**2) - z_error
  return value

def stand_action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel.

  Operates on raw policy output (before per-term scale/offset).
  """
  command = env.command_manager.get_command('twist')
  linear_speed = torch.norm(command[:, :2], dim=1)

  angular_speed = torch.abs(command[:, 2])
  total_speed = linear_speed

  standing_mask = ((total_speed < 0.1) & (angular_speed < 0.1)).float()
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  ) * standing_mask
