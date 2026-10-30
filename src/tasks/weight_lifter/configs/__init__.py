
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from src.rl.fastsac.registry_ext import extend_mjlab_task_with_fastsac
from src.rl.fastsac.runners import FastSACRunner

from .env_cfgs import unitree_g1_pull_up_barbell_env_cfg,unitree_g1_dex3_flat_tracking
from .rl_cfg import unitree_g1_pick_box_fastsac_runner_cfg,unitree_g1_cnn_ppo_runner_cfg,unitree_g1_cnn_gru_ppo_runner_cfg

_TASK_ID = "Unitree-G1-Dex3-Flat-Tracking"

register_mjlab_task(
  task_id=_TASK_ID,
  env_cfg=unitree_g1_dex3_flat_tracking(),
  play_env_cfg=unitree_g1_dex3_flat_tracking(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

extend_mjlab_task_with_fastsac(
  _TASK_ID,
  unitree_g1_pick_box_fastsac_runner_cfg(),
  runner_cls_fastsac=FastSACRunner,
)

from src.tasks.weight_lifter.runner import ResMLPTrackingRunner

_TASK_ID = "Unitree-G1-Pullup-A-Barbell"

rl_cfg_ = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_.actor.class_name = "src.tasks.weight_lifter.networks.resmlp:ResMLP"
rl_cfg_.experiment_name = "g1_dex3_pb"
rl_cfg_.save_interval = 50

register_mjlab_task(
  task_id=_TASK_ID,
  env_cfg=unitree_g1_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_g1_pull_up_barbell_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=ResMLPTrackingRunner,
)

extend_mjlab_task_with_fastsac(
  _TASK_ID,
  unitree_g1_pick_box_fastsac_runner_cfg(),
  runner_cls_fastsac=FastSACRunner,
)

_TASK_ID = "Unitree-G1-Pullup-A-Barbell-GrabNet"

rl_cfg_ = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_.actor.class_name = "src.tasks.weight_lifter.networks.grabnet:GrabNet"
rl_cfg_.experiment_name = "g1_dex3_pb_grabNet"

register_mjlab_task(
  task_id=_TASK_ID,
  env_cfg=unitree_g1_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_g1_pull_up_barbell_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingOnPolicyRunner,
)
