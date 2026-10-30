from typing import cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.sensor.contact_sensor import ContactSensor

from .commands import MotionBarbellCommand, MotionCommand
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


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
  body_names=("left_index_distal", "right_index_distal"),
):
  """Terminate when hands drift too far from the barbell height while grasping."""
  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]

  indexes, _ = robot_asset.find_bodies(list(body_names))
  barbell_hands_distant = torch.abs(
    robot_asset.data.body_link_pos_w[:, indexes, 2]
    - barbell_asset.data.root_link_pos_w[:, 2].unsqueeze(0).T
  )
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))

  return (barbell_hands_distant > limit_distant).any(-1) & command.graping


@torch.no_grad()
def far_away_root_pos(
  env: ManagerBasedRlEnv,
  limit_distant: float,
):
  """Terminate when the robot root is too far from the barbell in XY."""
  robot_asset: Entity = env.scene["robot"]
  barbell_asset: Entity = env.scene["barbell"]

  barbell_hands_distant = torch.abs(
    robot_asset.data.root_com_pos_w[:, :2] - barbell_asset.data.root_link_pos_w[:, :2]
  ).mean(-1)

  return barbell_hands_distant > limit_distant


def bad_body_dof(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.abs(command.joint_pos - command.robot_joint_pos)
  return torch.any(error > threshold, dim=-1)


def bad_hands_dof(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.abs(
    command.joint_pos[:, command.hands_index]
    - command.robot_joint_pos[:, command.hands_index]
  )
  return torch.any(error > threshold, dim=-1)


def bad_anchor_ori_relative(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  ref_anchor_quat_w = command.body_quat_relative_w[:, command.motion_anchor_body_index]
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
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


@torch.no_grad()
def not_on_the_bar(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  """Terminate in grasp window when either hand is poorly aligned with the bar."""
  command = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
  left_bad = (
    torch.cosine_similarity(
      command.bellbar_vector, command.left_hand_plate_vector, dim=1
    )
    < threshold
  )
  right_bad = (
    torch.cosine_similarity(
      command.bellbar_vector, command.right_hand_plate_vector, dim=1
    )
    < threshold
  )
  return (left_bad | right_bad) & command.graping


@torch.no_grad()
def shin_barbell_contact_termination(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor = cast(ContactSensor, env.scene[sensor_name])
  force = sensor.data.force
  if force is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  force_mag = torch.linalg.vector_norm(force, dim=-1)
  return (force_mag > 1e-6).any(dim=-1)

def pack_timer_safe(func):
  class safe_timer_termi:

    def __init__(self, cfg, env : ManagerBasedRlEnv):
      self.wait_count = torch.zeros(env.num_envs,device=env.device)

    def __call__(
      self,
      env: ManagerBasedRlEnv,
      **kws
    ):
      """Terminate when the asset's orientation exceeds the limit angle."""
      termi = func(**kws)
      self.wait_count[termi] = self.wait_count[termi] + 1
      result = self.wait_count > (0.8 / env.step_dt)
      self.wait_count[result] = 0
      return result

  return safe_timer_termi


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