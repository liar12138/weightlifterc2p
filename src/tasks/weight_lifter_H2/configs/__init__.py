
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from src.rl.fastsac.registry_ext import extend_mjlab_task_with_fastsac
from src.rl.fastsac.runners import FastSACRunner

from .env_cfgs import *
from .c2p_cfg import (
    unitree_h2_pull_up_barbell_c2p_env_cfg,
    unitree_h2_c2p_runner_cfg,
)
from ..c2p_runner import OnlineC2PRunner
from .rl_cfg import unitree_g1_pick_box_fastsac_runner_cfg,unitree_g1_cnn_ppo_runner_cfg,unitree_g1_cnn_gru_ppo_runner_cfg
from ..runner import *
_TASK_ID = "Unitree-H2-Pullup-A-Barbell"

register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-C2P",
  env_cfg=unitree_h2_pull_up_barbell_c2p_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_c2p_env_cfg(play=True),
  rl_cfg=unitree_h2_c2p_runner_cfg(),
  runner_cls=OnlineC2PRunner,
)

rl_cfg_ = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_.actor.class_name = "src.networks.simple_cnn:DepthCNNModel"
rl_cfg_.actor.hidden_dims = (768, 512, 256)
# rl_cfg_.critic.class_name = "src.networks.simple_cnn:DepthCNNModel"
rl_cfg_.critic.hidden_dims = (768, 512, 256)

register_mjlab_task(
  task_id=_TASK_ID,
  env_cfg=unitree_h2_pull_up_barbell_camera_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_camera_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingCameraOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-No-Camera",
  env_cfg=unitree_h2_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-Motion-tracking",
  env_cfg=unitree_h2_pull_up_barbell_mt_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_mt_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingCameraOnPolicyRunner,
)

extend_mjlab_task_with_fastsac(
  _TASK_ID,
  unitree_g1_pick_box_fastsac_runner_cfg(),
  runner_cls_fastsac=FastSACRunner,
)


_TASK_ID = "Unitree-H2-Pullup-A-Barbell-RNN"

rl_cfg_gru = unitree_g1_cnn_gru_ppo_runner_cfg()
# rl_cfg_.actor.class_name = "src.networks.cnn_gru_model:CNNGRUModel"
# rl_cfg_.actor.hidden_dims = (768,512, 256, 128)
# rl_cfg_.critic.class_name = "src.networks.cnn_gru_model:CNNGRUModel"
# rl_cfg_.critic.hidden_dims = (768,512, 256, 128)

register_mjlab_task(
  task_id=_TASK_ID,
  env_cfg=unitree_h2_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_env_cfg(play=True),
  rl_cfg=rl_cfg_gru,
  runner_cls=MotionTrackingOnPolicyRunner,
)

extend_mjlab_task_with_fastsac(
  _TASK_ID,
  unitree_g1_pick_box_fastsac_runner_cfg(),
  runner_cls_fastsac=FastSACRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-Hand",
  env_cfg=unitree_h2_pull_up_barbell_hand_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_hand_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingCameraOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-Knee",
  env_cfg=unitree_h2_pull_up_barbell_knee_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_knee_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingCameraOnPolicyRunner,
)

#unitree_h2_pull_up_barbell_hand_env_cfg

register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-Hand-Event",
  env_cfg=unitree_h2_pull_up_barbell_hand_event_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_hand_event_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingCameraOnPolicyRunner,
)



register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-MT-No-Camera",
  env_cfg=unitree_h2_pull_up_barbell_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-MT-No-Camera-Hand-Event",
  env_cfg=unitree_h2_pull_up_barbell_hand_event_nocamera_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_hand_event_nocamera_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-MT-No-Camera-Hand",
  env_cfg=unitree_h2_pull_up_barbell_hand_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_hand_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-MT-Camera",
  env_cfg=unitree_h2_pull_up_barbell_camera_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_camera_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingOnPolicyRunner,
)
#unitree_h2_pull_up_barbell_hand_camera_env_cfg
rl_cfg_.experiment_name = "h2_tracking_camera"
register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-MT-Camera-Hand",
  env_cfg=unitree_h2_pull_up_barbell_hand_camera_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_hand_camera_env_cfg(play=True),
  rl_cfg=rl_cfg_,
  runner_cls=MotionTrackingCameraOnPolicyRunner,
)
#unitree_h2_pull_up_barbell_hand_res_env_cfg
rl_cfg_res = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_res.experiment_name = "h2_tracking_camera_res"
rl_cfg_res.actor.class_name = "src.tasks.weight_lifter_H2.networks.resmlp:ResMLP"
rl_cfg_res.save_interval = 100
register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-Res",
  env_cfg=unitree_h2_pull_up_barbell_hand_res_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_hand_res_env_cfg(play=True),
  rl_cfg=rl_cfg_res,
  runner_cls=ResMLPTrackingRunner,
)

rl_cfg_res_2x = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_res_2x.experiment_name = "h2_tracking_camera_res"
rl_cfg_res_2x.actor.class_name = "src.tasks.weight_lifter_H2.networks.resmlp:ResMLP2X"
rl_cfg_res_2x.save_interval = 100
register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-Res2X",
  env_cfg=unitree_h2_pull_up_barbell_hand_res_2x_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_hand_res_2x_env_cfg(play=True),
  rl_cfg=rl_cfg_res_2x,
  runner_cls=ResMLPTrackingRunner,
)

rl_cfg_res = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_res.experiment_name = "h2_tracking_camera_res"
rl_cfg_res.actor.class_name = "src.tasks.weight_lifter_H2.networks.resmlp:ResMLP"
rl_cfg_res.save_interval = 100
register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-Record",
  env_cfg=unitree_h2_pull_up_barbell_hand_res_env_cfg_record(),
  play_env_cfg=unitree_h2_pull_up_barbell_hand_res_env_cfg_record(play=True),
  rl_cfg=rl_cfg_res,
  runner_cls=ResMLPTrackingRunner,
)

rl_cfg_no_camera = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_no_camera.experiment_name = "h2_tracking_no_camera"
register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-No-Camera-Experts",
  env_cfg=unitree_h2_pull_up_barbell_hand_expert_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_hand_expert_env_cfg(play=True),
  rl_cfg=rl_cfg_no_camera,
  runner_cls=MotionTrackingOnPolicyRunner,
)

#unitree_h2_pull_up_barbell_privil_env_cfg

rl_cfg_privil = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_privil.experiment_name = "h2_tracking_privil"
rl_cfg_privil.actor.distribution_cfg["init_std"] = 0.6
rl_cfg_privil.actor.distribution_cfg["std_range"] = (1e-6,0.6)
rl_cfg_privil.actor.class_name = "src.tasks.weight_lifter_H2.networks.priv_res:PrivResMLP"
rl_cfg_privil.save_interval = 100
register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-No-Camera-Privi",
  env_cfg=unitree_h2_pull_up_barbell_privil_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_privil_env_cfg(play=True),
  rl_cfg=rl_cfg_privil,
  runner_cls=MotionTrackingOnPolicyRunner,
)

rl_cfg_privil_mlp = unitree_g1_tracking_ppo_runner_cfg()
rl_cfg_privil_mlp.experiment_name = "h2_tracking_privil"
rl_cfg_privil_mlp.actor.distribution_cfg["init_std"] = 0.6
rl_cfg_privil_mlp.actor.distribution_cfg["std_range"] = (1e-6,0.6)
rl_cfg_privil_mlp.save_interval = 100
register_mjlab_task(
  task_id="Unitree-H2-Pullup-A-Barbell-No-Camera-Privi-MLP",
  env_cfg=unitree_h2_pull_up_barbell_privil_env_cfg(),
  play_env_cfg=unitree_h2_pull_up_barbell_privil_env_cfg(play=True),
  rl_cfg=rl_cfg_privil_mlp,
  runner_cls=MotionTrackingOnPolicyRunner,
)
