"""Script to train RL agent with PPO (rsl_rl) or FastSAC (holosoma)."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

import mjlab
import tyro

sys.path.append("./")

from src import *  # noqa: E402, F403

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from src.rl.wrapper import BodyOnlyRslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  algo: Literal["ppo", "fast_sac"] = "ppo"
  agent: RslRlBaseRunnerCfg | None = None
  fast_sac_agent: HolosomaFastSACRunnerCfg | None = None
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  log_root: str = "logs"
  """Legacy field; runs now use logs/{experiment_name}/{algo}/{timestamp}/."""
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str, algo: Literal["ppo", "fast_sac"] = "ppo") -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    ppo_cfg = load_rl_cfg(task_id)
    fast_sac_cfg: HolosomaFastSACRunnerCfg | None = None
    try:
      fast_sac_cfg = load_fastsac_rl_cfg(task_id)
    except ValueError:
      if algo == "fast_sac":
        raise
    return TrainConfig(
      env=env_cfg,
      algo=algo,
      agent=ppo_cfg,
      fast_sac_agent=fast_sac_cfg,
    )


def _setup_train_log_file(log_dir: Path, rank: int) -> None:
  if rank != 0:
    return
  log_dir.mkdir(parents=True, exist_ok=True)
  train_log = log_dir / "train.log"
  root = logging.getLogger()
  if any(
    isinstance(h, logging.FileHandler)
    and getattr(h, "baseFilename", None) == str(train_log.resolve())
    for h in root.handlers
  ):
    return
  handler = logging.FileHandler(train_log, encoding="utf-8")
  handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
  )
  root.addHandler(handler)
  if root.level == logging.NOTSET:
    root.setLevel(logging.INFO)


def _experiment_name(cfg: TrainConfig) -> str:
  if cfg.algo == "fast_sac":
    assert cfg.fast_sac_agent is not None
    return cfg.fast_sac_agent.experiment_name
  assert cfg.agent is not None
  return cfg.agent.experiment_name


def _run_name(cfg: TrainConfig) -> str:
  if cfg.algo == "fast_sac":
    assert cfg.fast_sac_agent is not None
    return cfg.fast_sac_agent.run_name
  assert cfg.agent is not None
  return cfg.agent.run_name


def _resolve_motion(cfg: TrainConfig, rank: int) -> str | None:
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )
  if not is_tracking_task:
    return None
  motion_cmd = cfg.env.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
    if rank == 0:
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    return motion_cmd.motion_file
  if cfg.registry_name:
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    if rank == 0:
      print(f"[INFO] Using motion from W&B registry: {motion_cmd.motion_file}")
    return motion_cmd.motion_file
  raise ValueError(
    "For tracking tasks, provide either:\n"
    "  --registry-name your-org/motions/motion-name (download from WandB)\n"
    "  --env.commands.motion.motion-file /path/to/motion.npz (local file)"
  )


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    base_seed = (
      cfg.fast_sac_agent.seed
      if cfg.algo == "fast_sac" and cfg.fast_sac_agent
      else cfg.agent.seed  # type: ignore[union-attr]
    )
    seed = base_seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    base_seed = (
      cfg.fast_sac_agent.seed
      if cfg.algo == "fast_sac" and cfg.fast_sac_agent
      else cfg.agent.seed  # type: ignore[union-attr]
    )
    seed = base_seed + rank

  configure_torch_backends()
  os.environ["MUJOCO_GL"] = "egl"

  env_cfg = replace(cfg.env, seed=seed, auto_reset=(cfg.algo == "ppo"))
  if cfg.algo == "fast_sac" and cfg.fast_sac_agent is not None:
    cfg = replace(cfg, env=env_cfg, fast_sac_agent=replace(cfg.fast_sac_agent, seed=seed))
  elif cfg.agent is not None:
    cfg = replace(cfg, env=env_cfg, agent=replace(cfg.agent, seed=seed))
  else:
    cfg = replace(cfg, env=env_cfg)

  print(
    f"[INFO] Training with: algo={cfg.algo}, device={device}, seed={seed}, rank={rank}"
  )

  registry_name = _resolve_motion(cfg, rank)

  if cfg.enable_nan_guard:
    env_cfg = replace(env_cfg, sim=replace(env_cfg.sim, nan_guard=replace(
      env_cfg.sim.nan_guard, enabled=True
    )))
    cfg = replace(cfg, env=env_cfg)
    if rank == 0:
      print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    _setup_train_log_file(log_dir, rank)

  env = NanSafeRlEnvFinalObs(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent
  resume_path: Path | None = None
  if cfg.algo == "fast_sac":
    sac_cfg = cfg.fast_sac_agent
    assert sac_cfg is not None
    if sac_cfg.resume:
      resume_path = get_checkpoint_path(
        log_root_path, sac_cfg.load_run, sac_cfg.load_checkpoint
      )
  else:
    ppo_cfg = cfg.agent
    assert ppo_cfg is not None
    if ppo_cfg.resume:
      if cfg.wandb_run_path is not None:
        resume_path, was_cached = get_wandb_checkpoint_path(
          log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
        )
        if rank == 0:
          print(
            f"[INFO]: Loading checkpoint from W&B: {resume_path.name} "
            f"({'cached' if was_cached else 'downloaded'})"
          )
      else:
        resume_path = get_checkpoint_path(
          log_root_path, ppo_cfg.load_run, ppo_cfg.load_checkpoint
        )

  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env_cfg_dict = asdict(cfg.env)

  if cfg.algo == "fast_sac":
    assert cfg.fast_sac_agent is not None
    runner_cls = load_fastsac_runner_cls(task_id) or FastSACRunner
    runner = runner_cls(env, cfg.fast_sac_agent, str(log_dir), device)
    agent_cfg_dict = asdict(cfg.fast_sac_agent)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
      print(f"[INFO]: Loading FastSAC checkpoint from: {resume_path}")
      runner.load(str(resume_path))
    if rank == 0:
      dump_yaml(log_dir / "params" / "env.yaml", env_cfg_dict)
      dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dict)
    runner.learn(num_learning_iterations=cfg.fast_sac_agent.max_iterations)
  else:
    assert cfg.agent is not None
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    #BodyOnlyRslRlVecEnvWrapper
    #env = BodyOnlyRslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    agent_cfg_dict = asdict(cfg.agent)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner_kwargs = {}
    if registry_name is not None:
      runner_kwargs["registry_name"] = registry_name
    if rank == 0:
      dump_yaml(log_dir / "params" / "env.yaml", env_cfg_dict)
      dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dict)
    runner = runner_cls(env, agent_cfg_dict, str(log_dir), device, **runner_kwargs)
    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
      print(f"[INFO]: Loading model checkpoint from: {resume_path}")
      runner.load(str(resume_path))
    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)
  if args.algo == "fast_sac":
    assert_fastsac_task(task_id)
    if args.fast_sac_agent is None:
      args = replace(
        args,
        fast_sac_agent=load_fastsac_rl_cfg(task_id),
      )

  log_subdir = "fastsac" if args.algo == "fast_sac" else "rsl_rl"
  log_dir = resolve_run_log_dir(_experiment_name(args), log_subdir, _run_name(args))

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  if args.algo == "fast_sac" and num_gpus > 1:
    print(
      "[WARN] FastSAC supports single-GPU training only; using the first GPU.",
      flush=True,
    )
    if selected_gpus is not None:
      selected_gpus = [selected_gpus[0]]
    num_gpus = 1

  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  if num_gpus <= 1:
    run_train(task_id, args, log_dir)
    return

  import torchrunx

  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if args.torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
    else:
      os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

  print(f"[INFO] Launching PPO with {num_gpus} GPUs (torchrunx)", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_train, task_id, args, log_dir)


def main():
  maybe_print_top_level_help("train")
  import mjlab.tasks  # noqa: F401
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
