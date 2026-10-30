"""Temporary smoke test for touch_a_box (safe to delete)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
src_path = root / "src"
tasks_path = src_path / "tasks"
m = types.ModuleType("src")
m.__path__ = [str(src_path)]  # type: ignore[attr-defined]
sys.modules["src"] = m
m = types.ModuleType("src.tasks")
m.__path__ = [str(tasks_path)]  # type: ignore[attr-defined]
sys.modules["src.tasks"] = m

import src.tasks.touch_a_box.configs  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends

assert "Unitree-G1-Revo2-Touch-A-Box" in list_tasks()
cfg = load_env_cfg("Unitree-G1-Revo2-Touch-A-Box", play=True)
print("entities", list(cfg.scene.entities.keys()))
print("sensors", [s.name for s in (cfg.scene.sensors or ())])
print("print event", "print_hand_tactile" in cfg.events)

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device", device)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device=device)
sensor = env.scene["hand_tactile"]
print("primaries", len(sensor.primary_names))
assert len(sensor.primary_names) == 34, sensor.primary_names
env.reset()

max_active = 0
max_force = 0.0
for _ in range(80):
  env.step(torch.zeros(env.action_space.shape, device=device))
  data = sensor.data
  found = data.found[0]
  force = data.force[0]
  active = int((found > 0).sum().item())
  max_active = max(max_active, active)
  max_force = max(max_force, float(force.norm(dim=-1).max()))

print("peak active", max_active, "peak max|F|", max_force)
assert max_active > 0, "expected at least one hand contact during rollout"
assert max_force > 1e-3, "expected non-trivial contact force"
env.close()
print("OK")
