#!/usr/bin/env python3
"""Verify get_h2_revo2_robot_cfg() returns a usable EntityCfg.

Usage (with capylab activated):
  conda activate capylab
  python scripts/verify_h2_revo2_cfg.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Allow importing local robot package without installing the repo.
_ROBOTS_ROOT = Path(__file__).resolve().parents[1] / "src" / "assets" / "robots"
if str(_ROBOTS_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROBOTS_ROOT))


def main() -> int:
  print("==> importing get_XX_robot_cfg ...")
  try:
    from unitree_h2_revo2.h2_constants import get_h2_revo2_robot_cfg
    from unitree_g1_revo2.g1_constants import get_g1_revo2_robot_cfg
    from unitree_g1_dex3.g1_constants import get_g1_dex3_robot_cfg
    from unitree_h2_o6.h2_o6_constants import get_h2_o6_robot_cfg 
  except Exception:
    print("FAIL: import failed")
    traceback.print_exc()
    return 1

  print("==> calling get_XX_robot_cfg() ...")
  try:
    cfg1 = get_h2_revo2_robot_cfg()
    cfg2 = get_g1_revo2_robot_cfg()
    cfg3 = get_g1_dex3_robot_cfg()
    cfg4 = get_h2_o6_robot_cfg()
  except Exception:
    print("FAIL: get_h2_revo2_robot_cfg() raised")
    traceback.print_exc()
    return 1

  from mjlab.entity import EntityCfg

  for name, cfg in (
    ("h2_revo2", cfg1),
    ("g1_revo2", cfg2),
    ("g1_dex3", cfg3),
    ("h2_o6", cfg4),
  ):
    print(f"\n==> {name}")
    print(f"type(cfg)        = {type(cfg)}")
    print(f"isinstance       = {isinstance(cfg, EntityCfg)}")
    print(f"spec_fn          = {getattr(cfg, 'spec_fn', None)}")
    print(f"collisions       = {getattr(cfg, 'collisions', None)}")
    print(f"articulation     = {type(getattr(cfg, 'articulation', None)).__name__}")
    init_state = getattr(cfg, "init_state", None)
    if init_state is not None:
      print(f"init_state.pos   = {getattr(init_state, 'pos', None)}")

  print("\nOK")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
