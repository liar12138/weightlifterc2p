#!/usr/bin/env python3
"""Train the H2 barbell C2P estimator online in MJLab.

Example:
  python scripts/train_c2p.py --num-envs 64 --history-length 10 --iterations 10001
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_src_namespace() -> None:
    """Import project subpackages without executing optional FastSAC imports."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    src_path = REPO_ROOT / "src"
    tasks_path = src_path / "tasks"
    if "src" not in sys.modules:
        module = types.ModuleType("src")
        module.__path__ = [str(src_path)]  # type: ignore[attr-defined]
        sys.modules["src"] = module
    if "src.tasks" not in sys.modules:
        module = types.ModuleType("src.tasks")
        module.__path__ = [str(tasks_path)]  # type: ignore[attr-defined]
        sys.modules["src.tasks"] = module
    h2_path = tasks_path / "weight_lifter_H2"
    if "src.tasks.weight_lifter_H2" not in sys.modules:
        module = types.ModuleType("src.tasks.weight_lifter_H2")
        module.__path__ = [str(h2_path)]  # type: ignore[attr-defined]
        sys.modules["src.tasks.weight_lifter_H2"] = module
    configs_path = h2_path / "configs"
    if "src.tasks.weight_lifter_H2.configs" not in sys.modules:
        module = types.ModuleType("src.tasks.weight_lifter_H2.configs")
        module.__path__ = [str(configs_path)]  # type: ignore[attr-defined]
        sys.modules["src.tasks.weight_lifter_H2.configs"] = module


@dataclass
class C2PTrainConfig:
    num_envs: int = 64
    history_length: int = 10
    iterations: int = 10001
    num_steps_per_env: int = 24
    c2p_epochs: int = 5
    c2p_mini_batches: int = 4
    c2p_learning_rate: float = 1.0e-4
    c2p_weight_decay: float = 1.0e-5
    c2p_max_grad_norm: float = 1.0
    save_interval: int = 100
    teacher_checkpoint: str = "src/assets/model_28300.pt"
    device: str | None = None
    log_dir: str = "logs/h2_barbell_c2p"


def main() -> None:
    args = tyro.cli(C2PTrainConfig, config=mjlab.TYRO_FLAGS)
    if args.history_length < 1:
        raise ValueError("--history-length must be positive")
    os.environ["C2P_HISTORY_LENGTH"] = str(args.history_length)
    _ensure_src_namespace()
    from src.tasks.weight_lifter_H2.c2p_runner import OnlineC2PRunner
    from src.tasks.weight_lifter_H2.configs.c2p_cfg import (
        unitree_h2_pull_up_barbell_c2p_env_cfg,
    )

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = unitree_h2_pull_up_barbell_c2p_env_cfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(env)
    runner_cfg = vars(args).copy()
    runner_cfg["num_steps_per_env"] = args.num_steps_per_env
    runner = OnlineC2PRunner(
        wrapped_env,
        runner_cfg,
        log_dir=args.log_dir,
        device=device,
    )
    try:
        runner.learn(args.iterations)
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    main()
