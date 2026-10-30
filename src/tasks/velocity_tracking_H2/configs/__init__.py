
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from .env_cfgs import *

rl_cfg = unitree_g1_ppo_runner_cfg()
register_mjlab_task(
  task_id="Velocity-Tracking-H2",
  env_cfg=unitree_h2_velocity_env_cfg(),
  play_env_cfg=unitree_h2_velocity_env_cfg(play=True),
  rl_cfg=rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

# 完全复现宇树官方 mjlab 的 H2 速度追踪配置（不带任何自定义改动）。
rl_cfg_official = unitree_g1_ppo_runner_cfg()
register_mjlab_task(
  task_id="Velocity-Tracking-H2-Official",
  env_cfg=unitree_h2_velocity_env_cfg_official(),
  play_env_cfg=unitree_h2_velocity_env_cfg_official(play=True),
  rl_cfg=rl_cfg_official,
  runner_cls=VelocityOnPolicyRunner,
)
