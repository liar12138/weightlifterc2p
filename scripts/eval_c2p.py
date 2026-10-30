#!/usr/bin/env python3
"""Evaluate an online C2P checkpoint against MJLab ground-truth labels."""

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
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    src_path = REPO_ROOT / "src"
    tasks_path = src_path / "tasks"
    h2_path = tasks_path / "weight_lifter_H2"
    configs_path = h2_path / "configs"
    for name, path in (
        ("src", src_path),
        ("src.tasks", tasks_path),
        ("src.tasks.weight_lifter_H2", h2_path),
        ("src.tasks.weight_lifter_H2.configs", configs_path),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]  # type: ignore[attr-defined]
            sys.modules[name] = module


@dataclass
class EvalConfig:
    checkpoint: str
    num_envs: int = 64
    steps: int = 500
    history_length: int = 10
    device: str | None = None
    teacher_checkpoint: str = "src/assets/model_28300.pt"


def main() -> None:
    args = tyro.cli(EvalConfig, config=mjlab.TYRO_FLAGS)
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
    runner = OnlineC2PRunner(
        wrapped_env,
        {
            "teacher_checkpoint": args.teacher_checkpoint,
            "num_steps_per_env": 1,
            "c2p_epochs": 1,
            "c2p_mini_batches": 1,
        },
        device=device,
    )
    runner.load(args.checkpoint)
    obs = wrapped_env.get_observations().to(device)
    pos_errors: list[torch.Tensor] = []
    quat_errors: list[torch.Tensor] = []
    vector_errors: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for _ in range(args.steps):
                prediction = runner.model(obs["actor"], obs["depth_camera"])
                target = obs["c2p_target"]
                pos_errors.append(torch.linalg.vector_norm(prediction[:, :3] - target[:, :3], dim=-1))
                quat_errors.append(torch.linalg.vector_norm(prediction[:, 3:7] - target[:, 3:7], dim=-1))
                vector_errors.append(torch.linalg.vector_norm(prediction[:, 7:] - target[:, 7:], dim=-1))
                actions = runner.teacher(runner._teacher_obs(obs))
                obs, _, _, _ = wrapped_env.step(actions.to(wrapped_env.device))
                obs = obs.to(device)
    finally:
        wrapped_env.close()
    pos = torch.cat(pos_errors)
    quat = torch.cat(quat_errors)
    vectors = torch.cat(vector_errors)
    print(f"position_rmse_m={torch.sqrt(torch.mean(pos.square())).item():.6f}")
    print(f"position_mae_m={torch.mean(pos).item():.6f}")
    print(f"position_p95_m={torch.quantile(pos, 0.95).item():.6f}")
    print(f"position_within_5cm={torch.mean((pos < 0.05).float()).item():.4f}")
    print(f"quaternion_l2_rmse={torch.sqrt(torch.mean(quat.square())).item():.6f}")
    print(f"vector_l2_rmse={torch.sqrt(torch.mean(vectors.square())).item():.6f}")


if __name__ == "__main__":
    main()
