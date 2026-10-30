
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from src.rl.fastsac.registry_ext import extend_mjlab_task_with_fastsac
from src.rl.fastsac.runners import FastSACRunner

from .env_cfgs import unitree_g1_pull_up_barbell_env_cfg,unitree_g1_pull_up_barbell_mt_env_cfg,unitree_g1_pull_up_barbell_privil_env_cfg,unitree_g1_pull_up_distill
from ..runner import MotionTrackingCameraOnPolicyRunner
_TASK_ID = "Unitree-G1-O6-Pullup-A-Barbell"

rl_cfg_ = unitree_g1_tracking_ppo_runner_cfg()

register_mjlab_task(
  task_id= "Unitree-G1-O6-Pullup-A-Barbell-Privil",
  env_cfg=unitree_g1_pull_up_barbell_privil_env_cfg(),
  play_env_cfg=unitree_g1_pull_up_barbell_privil_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id= "Unitree-G1-O6-Pullup-A-Barbell-Privil-Distill",
  env_cfg=unitree_g1_pull_up_distill(),
  play_env_cfg=unitree_g1_pull_up_distill(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id= "Unitree-G1-O6-Pullup-A-Barbell-MT",
  env_cfg=unitree_g1_pull_up_barbell_mt_env_cfg(),
  play_env_cfg=unitree_g1_pull_up_barbell_mt_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id= "Unitree-G1-O6-Pullup-A-Barbell-Pure-MT-Hand-Event",
  env_cfg=unitree_g1_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_g1_pull_up_barbell_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

rl_cfg_ = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_.actor.class_name = "src.networks.simple_cnn:DepthCNNModel"
rl_cfg_.actor.hidden_dims = (768, 512, 256)
# rl_cfg_.critic.class_name = "src.networks.simple_cnn:DepthCNNModel"
rl_cfg_.critic.hidden_dims = (768, 512, 256)
register_mjlab_task(
  task_id= "Unitree-G1-O6-Pullup-A-Barbell-Pure-MT-Camera",
  env_cfg=unitree_g1_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_g1_pull_up_barbell_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingCameraOnPolicyRunner,
)