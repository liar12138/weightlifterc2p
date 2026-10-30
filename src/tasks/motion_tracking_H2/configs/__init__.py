from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_h2_o6_motion_tracking_env_cfg,unitree_h2_o6_motion_tracking_env_cfg_hand_mass
from .env_cfgs_31dof import (
  unitree_h2_31dof_motion_tracking_env_cfg,
  unitree_h2_31dof_handmass_motion_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_h2_o6_motion_tracking_ppo_runner_cfg,
  unitree_h2_31dof_motion_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-H2-O6-Motion-Tracking",
  env_cfg=unitree_h2_o6_motion_tracking_env_cfg(),
  play_env_cfg=unitree_h2_o6_motion_tracking_env_cfg(play=True),
  rl_cfg=unitree_h2_o6_motion_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
rl_cfg_ = unitree_h2_31dof_motion_tracking_ppo_runner_cfg()
register_mjlab_task(
  task_id="Unitree-H2-Motion-Tracking",
  env_cfg=unitree_h2_31dof_motion_tracking_env_cfg(),
  play_env_cfg=unitree_h2_31dof_motion_tracking_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingOnPolicyRunner,
)

rl_cfg_handmass = unitree_h2_31dof_motion_tracking_ppo_runner_cfg()
rl_cfg_handmass.experiment_name = "h2_o6_mt"
register_mjlab_task(
  task_id="Unitree-H2-Motion-Tracking-HandMass",
  env_cfg=unitree_h2_o6_motion_tracking_env_cfg_hand_mass(),
  play_env_cfg=unitree_h2_o6_motion_tracking_env_cfg_hand_mass(play=True),
  rl_cfg=rl_cfg_handmass,
  runner_cls=MotionTrackingOnPolicyRunner,
)
