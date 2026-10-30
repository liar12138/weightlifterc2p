from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_g1_revo2_touch_a_box_env_cfg
from .rl_cfg import unitree_g1_revo2_touch_a_box_ppo_runner_cfg

TASK_ID = "Unitree-G1-Revo2-Touch-A-Box"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_revo2_touch_a_box_env_cfg(play=False),
  play_env_cfg=unitree_g1_revo2_touch_a_box_env_cfg(play=True),
  rl_cfg=unitree_g1_revo2_touch_a_box_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
