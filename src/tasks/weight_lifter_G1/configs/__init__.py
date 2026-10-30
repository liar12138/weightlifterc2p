
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from src.rl.fastsac.registry_ext import extend_mjlab_task_with_fastsac
from src.rl.fastsac.runners import FastSACRunner

from .env_cfgs import unitree_g1_pull_up_barbell_env_cfg
from .rl_cfg import unitree_g1_pick_box_fastsac_runner_cfg,unitree_g1_cnn_ppo_runner_cfg,unitree_g1_cnn_gru_ppo_runner_cfg
from ..runner import MotionTrackingCameraOnPolicyRunner
_TASK_ID = "Unitree-G1-O6-Pullup-A-Barbell"

rl_cfg_ = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_.actor.class_name = "src.networks.simple_cnn:DepthCNNModel"
rl_cfg_.actor.hidden_dims = (768, 512, 256)
# rl_cfg_.critic.class_name = "src.networks.simple_cnn:DepthCNNModel"
rl_cfg_.critic.hidden_dims = (768, 512, 256)

register_mjlab_task(
  task_id=_TASK_ID,
  env_cfg=unitree_g1_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_g1_pull_up_barbell_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingCameraOnPolicyRunner,
)

extend_mjlab_task_with_fastsac(
  _TASK_ID,
  unitree_g1_pick_box_fastsac_runner_cfg(),
  runner_cls_fastsac=FastSACRunner,
)


_TASK_ID = "Unitree-G1-O6-Pullup-A-Barbell-RNN"

rl_cfg_ = unitree_g1_cnn_gru_ppo_runner_cfg()
# rl_cfg_.actor.class_name = "src.networks.cnn_gru_model:CNNGRUModel"
# rl_cfg_.actor.hidden_dims = (768,512, 256, 128)
# rl_cfg_.critic.class_name = "src.networks.cnn_gru_model:CNNGRUModel"
# rl_cfg_.critic.hidden_dims = (768,512, 256, 128)

register_mjlab_task(
  task_id=_TASK_ID,
  env_cfg=unitree_g1_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_g1_pull_up_barbell_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingOnPolicyRunner,
)

extend_mjlab_task_with_fastsac(
  _TASK_ID,
  unitree_g1_pick_box_fastsac_runner_cfg(),
  runner_cls_fastsac=FastSACRunner,
)
