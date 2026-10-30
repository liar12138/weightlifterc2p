"""Unified log directory layout aligned with PPO."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

AlgoLogDir = Literal["rsl_rl", "fastsac"]


def resolve_run_log_dir(
  experiment_name: str,
  algo: AlgoLogDir,
  run_name: str = "",
) -> Path:
  """Return ``logs/{experiment_name}/{algo}/{timestamp}[_run_name]``."""
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if run_name:
    stamp = f"{stamp}_{run_name}"
  return (Path("logs") / experiment_name / algo / stamp).resolve()


def resolve_experiment_log_root(experiment_name: str, algo: AlgoLogDir) -> Path:
  """Return ``logs/{experiment_name}/{algo}`` for checkpoint lookup."""
  return (Path("logs") / experiment_name / algo).resolve()
