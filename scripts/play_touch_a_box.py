#!/usr/bin/env python3
"""Play G1+Revo2 hand ContactSensor check.

Does **not** edit ``scripts/play.py`` or ``src/__init__.py``.

Bypasses a broken ``src/__init__.py`` (missing optional task packages) by
installing lightweight namespace stubs, then registers this task and launches
mjlab's Viser play loop (same viewer stack as ``scripts/play.py``).

Usage (conda: capylab), from ``weightlifter/``:

  python scripts/play_touch_a_box.py --agent zero --viewer viser --num_envs 1
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TASK = "Unitree-G1-Revo2-Touch-A-Box"


@dataclass(frozen=True)
class TouchPlayConfig:
  agent: Literal["zero", "random"] = "zero"
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["viser"] = "viser"
  no_terminations: bool = True


def _ensure_src_namespace() -> None:
  """Allow ``import src.tasks.touch_a_box`` without executing broken ``src/__init__.py``."""
  if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

  src_path = _REPO_ROOT / "src"
  tasks_path = src_path / "tasks"

  if "src" not in sys.modules:
    mod = types.ModuleType("src")
    mod.__path__ = [str(src_path)]  # type: ignore[attr-defined]
    sys.modules["src"] = mod

  if "src.tasks" not in sys.modules:
    mod = types.ModuleType("src.tasks")
    mod.__path__ = [str(tasks_path)]  # type: ignore[attr-defined]
    sys.modules["src.tasks"] = mod


def _register_task() -> None:
  _ensure_src_namespace()
  import src.tasks.touch_a_box.configs  # noqa: F401


def main() -> None:
  _register_task()
  import mjlab.tasks  # noqa: F401

  if _DEFAULT_TASK not in list_tasks():
    raise RuntimeError(f"Task {_DEFAULT_TASK!r} failed to register")

  argv = sys.argv[1:]
  remaining = argv
  if argv and not argv[0].startswith("-") and argv[0] == _DEFAULT_TASK:
    remaining = argv[1:]

  args = tyro.cli(
    TouchPlayConfig,
    args=remaining,
    default=TouchPlayConfig(),
    prog=f"{sys.argv[0]} {_DEFAULT_TASK}",
    config=mjlab.TYRO_FLAGS,
  )

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(_DEFAULT_TASK, play=True)
  agent_cfg = load_rl_cfg(_DEFAULT_TASK)
  if args.no_terminations:
    env_cfg.terminations = {}
  env_cfg.scene.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  play_env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
  action_shape: tuple[int, ...] = play_env.unwrapped.action_space.shape

  if args.agent == "zero":

    class PolicyZero:
      def __call__(self, obs):
        del obs
        return torch.zeros(action_shape, device=play_env.unwrapped.device)

    policy = PolicyZero()
  else:

    class PolicyRandom:
      def __call__(self, obs):
        del obs
        return 2 * torch.rand(action_shape, device=play_env.unwrapped.device) - 1

    policy = PolicyRandom()

  sensor = env.scene["hand_tactile"]
  print(f"[hand_tactile] primary_names ({len(sensor.primary_names)}):")
  for i, name in enumerate(sensor.primary_names):
    print(f"  [{i:02d}] {name}")
  print("[INFO] Opening viser. Watch console for [hand_tactile] contact lines every ~0.5s.")

  if args.viewer != "viser":
    raise RuntimeError("This script only supports --viewer viser (no desktop display).")

  ViserPlayViewer(play_env, policy).run()
  play_env.close()


if __name__ == "__main__":
  main()
