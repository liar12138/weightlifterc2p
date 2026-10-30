from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from .env_cfgs import unitree_h2_pull_up_barbell_env_cfg,unitree_h2_pull_up_barbell_mt_env_cfg

_TASK_ID = "Unitree-H2-Pullup-A-Barbell-Blind"

rl_cfg_ = unitree_g1_tracking_ppo_runner_cfg()

register_mjlab_task(
  task_id=_TASK_ID,
  env_cfg=unitree_h2_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-Motion-tracking-Blind",
  env_cfg=unitree_h2_pull_up_barbell_mt_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_mt_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingOnPolicyRunner,
)
