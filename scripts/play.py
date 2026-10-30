"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

sys.path.append('./')
from src import *  # noqa: E402, F403

from src.rl.fastsac.log_paths import resolve_experiment_log_root
from src.rl.fastsac.registry_ext import load_fastsac_rl_cfg
from src.rl.fastsac.runners import FastSACRunner
from src.rl.wrapper import BodyOnlyRslRlVecEnvWrapper


def _map_fastsac_play_obs(obs) -> dict[str, torch.Tensor]:
  """Map mjlab play observations to FastSAC inference keys.

  RslRlVecEnvWrapper returns a TensorDict with ``actor`` / ``critic`` groups.
  TensorDict is not a ``dict`` subclass, so avoid ``isinstance(obs, dict)``.
  """
  if isinstance(obs, torch.Tensor):
    return {"actor_obs": obs}
  if "actor_obs" in obs.keys():
    return {"actor_obs": obs["actor_obs"]}
  if "actor" in obs.keys():
    return {"actor_obs": obs["actor"]}
  raise KeyError(
    f"FastSAC play expected 'actor_obs' or 'actor' in observations, got {list(obs.keys())}"
  )


class _FastSACPlayEnvWrapper:
  """Reset done envs before step when ``auto_reset=False`` (FastSAC training mode)."""

  def __init__(self, env: RslRlVecEnvWrapper) -> None:
    self.env = env

  def __getattr__(self, name: str):
    return getattr(self.env, name)

  @property
  def unwrapped(self):
    return self.env.unwrapped

  def get_observations(self):
    return self.env.get_observations()

  def reset(self, *args, **kwargs):
    return self.env.reset(*args, **kwargs)

  def close(self) -> None:
    self.env.close()

  def step(self, actions: torch.Tensor):
    mj_env = self.env.unwrapped
    if (
      not mj_env.cfg.auto_reset
      and hasattr(mj_env, "_manual_reset_pending")
      and torch.any(mj_env._manual_reset_pending)
    ):
      pending_ids = mj_env._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
      if pending_ids.numel() > 0:
        mj_env.reset(env_ids=pending_ids)
    return self.env.step(actions)


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


@dataclass(frozen=True)
class PlayConfig:
  algo: Literal["ppo", "fast_sac"] = "ppo"
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  motion_choose: int | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  log_root: str = "logs"
  """Legacy field; FastSAC checkpoints live under logs/{experiment_name}/fastsac/."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False
  record_depth_path: str | None = None



def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")


  if cfg.motion_choose is not None:
    if "skill" not in env_cfg.commands:
      raise ValueError(
        "`--motion_choose` was provided, but env_cfg.commands has no 'skill' command."
      )

    skill_cmd = env_cfg.commands["skill"]

    if cfg.motion_choose not in (0, 1):
      raise ValueError(
        f"Invalid motion_choose={cfg.motion_choose}. Expected 0=pick or 1=place."
      )

    skill_cmd.motion_choose = cfg.motion_choose

    print(
      f"[INFO]: Using motion_choose={cfg.motion_choose} "
      f"({'pick' if cfg.motion_choose == 0 else 'place'})"
    )

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    if cfg.algo == "fast_sac":
      exp_name = load_fastsac_rl_cfg(task_id).experiment_name
      log_root_path = resolve_experiment_log_root(exp_name, "fastsac")
    else:
      log_root_path = (Path("logs") / agent_cfg.experiment_name / "rsl_rl").resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  if cfg.algo == "fast_sac":
    from dataclasses import replace
    env_cfg = replace(env_cfg, auto_reset=False)

  if cfg.record_depth_path is not None:
      grp = env_cfg.observations.get("depth_camera")
      if grp is not None:
          grp.terms["depth_camera"].params["save_path"] = cfg.record_depth_path

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  play_env: RslRlVecEnvWrapper | _FastSACPlayEnvWrapper
  if DUMMY_MODE:
    play_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    action_shape: tuple[int, ...] = play_env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=play_env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=play_env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    if cfg.algo == "fast_sac":
      assert resume_path is not None
      sac_cfg = load_fastsac_rl_cfg(task_id)
      runner = FastSACRunner(env, sac_cfg, str(resume_path.parent), device)
      runner.load(str(resume_path))
      fast_sac_policy = runner.get_inference_policy(device=device)

      def policy(obs):
        return fast_sac_policy(_map_fastsac_play_obs(obs))

      play_env = _FastSACPlayEnvWrapper(
        RslRlVecEnvWrapper(env, clip_actions=None)
      )
    else:
      play_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
      #BodyOnlyRslRlVecEnvWrapper
      #play_env = BodyOnlyRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
      runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
      runner = runner_cls(play_env, asdict(agent_cfg), device=device)
      runner.load(
        str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
      )
      policy = runner.get_inference_policy(device=device)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None and cfg.algo != "fast_sac":
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(play_env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(play_env, policy, checkpoint_manager=ckpt_manager).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  play_env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
