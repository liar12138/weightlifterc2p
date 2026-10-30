"""FastSAC RL configuration for Unitree G1 Pick-A-Box task."""

from src.rl.fastsac.config import HolosomaFastSACRunnerCfg
from src.rl.fastsac.tasks.mjlab_builtins.rl_cfg import tracking_nse_fastsac_runner_cfg


def unitree_g1_pick_box_fastsac_runner_cfg() -> HolosomaFastSACRunnerCfg:
  """FastSAC runner config derived from NSE tracking (no state estimation)."""
  cfg = tracking_nse_fastsac_runner_cfg()
  cfg.experiment_name = "g1_pick_box_fastsac"
  cfg.max_iterations = 30000
  cfg.save_interval = 5000
  # Pick-box adds a free-body box; 2048 envs OOMs with default buffer_size=4096.
  cfg.buffer_size = 1024
  return cfg
