"""RL runner cfg placeholder (play uses --agent zero; training not required)."""

from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_revo2_touch_a_box_ppo_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_revo2_touch_a_box"
  return cfg
