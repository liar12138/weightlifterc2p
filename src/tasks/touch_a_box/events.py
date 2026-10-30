"""Runtime helpers for hand tactile verification."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor


def print_hand_tactile(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  sensor_name: str = "hand_tactile",
  force_eps: float = 1e-3,
) -> None:
  """Print non-zero hand ContactSensor readings for env 0."""
  del env_ids
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.found is None or data.force is None:
    print(f"[{sensor_name}] found/force is None (check fields=...)")
    return

  found = data.found[0]
  force = data.force[0]
  names = sensor.primary_names
  active: list[str] = []
  for i, name in enumerate(names):
    if float(found[i]) <= 0:
      continue
    mag = float(torch.linalg.norm(force[i]))
    if mag < force_eps:
      active.append(f"{name}: found={float(found[i]):.0f} |F|~0")
    else:
      fx, fy, fz = (float(force[i, 0]), float(force[i, 1]), float(force[i, 2]))
      active.append(f"{name}: |F|={mag:.3f} F=({fx:.2f},{fy:.2f},{fz:.2f})")

  n = len(names)
  print(
    f"[{sensor_name}] primaries={n} active={len(active)} "
    + ("| " + " | ".join(active) if active else "| (no contact)")
  )
