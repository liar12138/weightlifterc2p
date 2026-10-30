"""mjlab 1.5 ManagerBasedRlEnv wrapper for holosoma FastSAC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv

DEFAULT_OBS_MAP: dict[str, str] = {
  "actor": "actor_obs",
  "critic": "critic_obs",
}


@dataclass
class _InitState:
  default_joint_angles: dict[str, float] = field(default_factory=dict)


@dataclass
class _Control:
  action_scale: float


@dataclass
class MjlabRobotConfig:
  dof_names: list[str]
  dof_pos_lower_limit_list: list[float]
  dof_pos_upper_limit_list: list[float]
  init_state: _InitState
  control: _Control
  actions_dim: int


@dataclass
class _ObsGroupCfgShim:
  terms: dict[str, Any] = field(default_factory=dict)
  history_length: int = 1


@dataclass
class _ObsManagerCfgShim:
  groups: dict[str, _ObsGroupCfgShim]


class _ObservationManagerShim:
  def __init__(self, mj_env: ManagerBasedRlEnv, obs_map: dict[str, str]) -> None:
    self._mj = mj_env
    self._obs_map = obs_map
    holosoma_groups: dict[str, _ObsGroupCfgShim] = {}
    for mj_key, holo_key in obs_map.items():
      if mj_key not in mj_env.observation_manager.active_terms:
        continue
      term_names = mj_env.observation_manager.active_terms[mj_key]
      holosoma_groups[holo_key] = _ObsGroupCfgShim(
        terms={name: None for name in term_names},
        history_length=1,
      )
    self.cfg = _ObsManagerCfgShim(groups=holosoma_groups)

  def get_obs_dims(self) -> dict[str, int]:
    dims: dict[str, int] = {}
    for mj_key, holo_key in self._obs_map.items():
      group_dim = self._mj.observation_manager.group_obs_dim.get(mj_key)
      if group_dim is None:
        continue
      if isinstance(group_dim, tuple):
        dims[holo_key] = int(group_dim[0])
      else:
        dims[holo_key] = int(sum(d[0] for d in group_dim))
    return dims


def build_robot_config_from_mjlab_env(env: ManagerBasedRlEnv) -> MjlabRobotConfig:
  robot = env.scene["robot"]
  joint_names = list(robot.joint_names)
  default_pos = robot.data.default_joint_pos[0].detach().cpu().tolist()
  soft_limits = robot.data.soft_joint_pos_limits[0].detach().cpu()
  lower = soft_limits[:, 0].tolist()
  upper = soft_limits[:, 1].tolist()
  default_angles = {
    name: float(default_pos[i]) for i, name in enumerate(joint_names)
  }

  action_term = env.action_manager.get_term("joint_pos")
  scale_val = 0.25
  if hasattr(action_term, "scale"):
    sc = action_term.scale
    if isinstance(sc, dict):
      scale_val = float(next(iter(sc.values())))
    elif isinstance(sc, (int, float)):
      scale_val = float(sc)
    elif isinstance(sc, torch.Tensor):
      scale_val = float(sc.flatten()[0].item())

  return MjlabRobotConfig(
    dof_names=joint_names,
    dof_pos_lower_limit_list=lower,
    dof_pos_upper_limit_list=upper,
    init_state=_InitState(default_joint_angles=default_angles),
    control=_Control(action_scale=scale_val),
    actions_dim=env.action_manager.total_action_dim,
  )


class MjlabFastSACEnv:
  """Holosoma BaseTask-compatible wrapper using mjlab 1.5 env.step() / reset()."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    obs_map: dict[str, str] | None = None,
  ) -> None:
    self._env = env
    self._obs_map = obs_map or dict(DEFAULT_OBS_MAP)
    self.device = env.device
    self.num_envs = env.num_envs
    self.robot_config = build_robot_config_from_mjlab_env(env)
    self.observation_manager = _ObservationManagerShim(env, self._obs_map)
    self.log_dict: dict[str, Any] = {}

    self.rew_buf = torch.zeros(self.num_envs, device=self.device)
    self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.extras: dict[str, Any] = {}
    self.obs_buf_dict: dict[str, torch.Tensor] = {}

    if self._env.cfg.auto_reset:
      raise ValueError(
        "FastSAC requires env.cfg.auto_reset=False; set it before creating the env."
      )
    self._env.reset()

  def reset_all(self) -> dict[str, torch.Tensor]:
    self._env.reset()
    actions = torch.zeros(
      self.num_envs,
      self.robot_config.actions_dim,
      device=self.device,
    )
    return self.step({"actions": actions})[0]

  def step(
    self, actor_state: dict[str, torch.Tensor]
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
    actions = actor_state["actions"].to(self.device)
    obs_raw, rew, terminated, truncated, _extras = self._env.step(actions)

    terminal_mapped = self._map_obs(obs_raw)
    done = terminated | truncated
    reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

    current_mapped = {k: v.clone() for k, v in terminal_mapped.items()}
    if reset_env_ids.numel() > 0:
      if reset_env_ids.dim() == 0:
        reset_env_ids = reset_env_ids.unsqueeze(0)
      self._env.reset(env_ids=reset_env_ids)
      fresh_obs = self._map_obs(self._env.get_observations())
      for key in current_mapped:
        current_mapped[key][reset_env_ids] = fresh_obs[key][reset_env_ids]

    self.obs_buf_dict = current_mapped
    self.rew_buf = rew
    self.reset_buf = done.to(dtype=torch.long)
    self.time_out_buf = truncated

    episode_stats = self._collect_episode_stats(reset_env_ids)
    info_dict = self._build_info_dict(terminal_mapped, episode_stats)

    return self.obs_buf_dict, self.rew_buf, self.reset_buf, info_dict

  def _map_obs(self, raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for mj_key, holo_key in self._obs_map.items():
      if mj_key in raw:
        out[holo_key] = raw[mj_key]
    return out

  def _collect_episode_stats(
    self, reset_env_ids: torch.Tensor
  ) -> dict[str, dict[str, torch.Tensor]]:
    episode: dict[str, torch.Tensor] = {}
    episode_all: dict[str, torch.Tensor] = {}
    if reset_env_ids.numel() == 0:
      return {"episode": episode, "episode_all": episode_all}

    for name, buf in self._env.reward_manager._episode_sums.items():
      vals = buf[reset_env_ids].clone()
      episode[f"rew_{name}"] = vals
      episode_all[f"rew_{name}"] = buf.clone()
    episode["r"] = self.rew_buf[reset_env_ids].clone()
    episode_all["r"] = self.rew_buf.clone()
    return {"episode": episode, "episode_all": episode_all}

  def _build_info_dict(
    self,
    terminal_obs_dict: dict[str, torch.Tensor],
    episode_stats: dict[str, Any],
  ) -> dict[str, Any]:
    info: dict[str, Any] = {
      "time_outs": self.time_out_buf.clone(),
      "episode": episode_stats.get("episode", {}),
      "episode_all": episode_stats.get("episode_all", {}),
      "raw_episode": {},
      "raw_episode_all": {},
      "to_log": self._tensor_log_dict(),
    }
    if terminal_obs_dict:
      store: dict[str, torch.Tensor] = {}
      reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
      for key, terminal_val in terminal_obs_dict.items():
        store[key] = self.obs_buf_dict[key].clone()
        if reset_ids.numel() > 0:
          if reset_ids.dim() == 0:
            reset_ids = reset_ids.unsqueeze(0)
          store[key][reset_ids] = terminal_val[reset_ids]
      info["final_observations"] = store
    return info

  def _tensor_log_dict(self) -> dict[str, torch.Tensor]:
    raw = self._env.extras.get("log", {})
    out: dict[str, torch.Tensor] = {}
    for key, val in raw.items():
      if isinstance(val, torch.Tensor):
        out[key] = val
      elif isinstance(val, (int, float)):
        out[key] = torch.tensor(float(val), device=self.device, dtype=torch.float32)
    return out

  def get_checkpoint_state(self) -> dict[str, Any] | None:
    return None

  def synchronize_curriculum_state(self, device: str, world_size: int) -> None:
    del device, world_size

  def get_step_reward_breakdown(
    self, env_ids: torch.Tensor
  ) -> tuple[list[str], torch.Tensor]:
    if env_ids.dim() == 0:
      env_ids = env_ids.unsqueeze(0)
    rm = self._env.reward_manager
    dt = self._env.step_dt
    names = list(rm.active_terms)
    scaled = rm._step_reward[env_ids] * dt
    return names, scaled

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self._env

  @property
  def command_manager(self):
    return self._env.command_manager
