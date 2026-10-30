"""Event terms specific to the H2 velocity-tracking tasks."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def push_when_standing(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  command_threshold: float,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Apply a mild velocity impulse only to environments with a zero twist."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  standing_ids = env_ids[
    torch.all(torch.abs(command[env_ids, :3]) <= command_threshold, dim=1)
  ]
  if len(standing_ids) == 0:
    return

  asset = env.scene[asset_cfg.name]
  velocity = asset.data.root_link_vel_w[standing_ids].clone()
  components = ("x", "y", "z", "roll", "pitch", "yaw")
  for index, name in enumerate(components):
    low, high = velocity_range[name]
    velocity[:, index].add_(
      torch.empty(len(standing_ids), device=env.device).uniform_(low, high)
    )
  asset.write_root_link_velocity_to_sim(velocity, env_ids=standing_ids)
