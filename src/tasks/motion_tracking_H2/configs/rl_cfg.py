"""RL configuration for Unitree H2 motion tracking tasks."""

from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def unitree_h2_o6_motion_tracking_ppo_runner_cfg():
  """PPO runner cfg for H2+O6 motion tracking (plain MLP, no camera)."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "h2_o6_tracking"
  return cfg


def unitree_h2_31dof_motion_tracking_ppo_runner_cfg():
  """PPO runner cfg for H2 (31-DOF) motion tracking."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "h2_tracking"
  return cfg
