"""Register mjlab built-in tasks with FastSAC configs."""

from src.rl.fastsac.registry_ext import extend_mjlab_task_with_fastsac
from src.rl.fastsac.runners import FastSACRunner

from .rl_cfg import (
  tracking_fastsac_runner_cfg,
  tracking_nse_fastsac_runner_cfg,
  velocity_fastsac_runner_cfg,
)

for _task_id, _cfg in [
  ("Mjlab-Velocity-Flat-Unitree-G1", velocity_fastsac_runner_cfg()),
  ("Mjlab-Velocity-Rough-Unitree-G1", velocity_fastsac_runner_cfg()),
  ("Mjlab-Tracking-Flat-Unitree-G1", tracking_fastsac_runner_cfg()),
  ("Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation", tracking_nse_fastsac_runner_cfg()),
]:
  extend_mjlab_task_with_fastsac(_task_id, _cfg, runner_cls_fastsac=FastSACRunner)
