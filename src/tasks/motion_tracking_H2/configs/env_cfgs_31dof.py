"""Unitree H2 (31-DOF, no hands) motion tracking environment configs."""

from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg

from src.assets.robots.unitree_h2_o6.h2_constants import (
  FOOT_COLLISION_PATTERN,
  H2_ACTION_SCALE,
  get_h2_robot_cfg,
)

# from ..events import randomize_hand_mass
# from ..curriculums import hand_mass_curriculum
#from src.sensors.depth_camera import deploy_depth_camera_to
from src.sensors.advCamera import deploy_h2_depth_camera_to
#from src.sensors.h2.d435c import deploy_depth_camera_to
from src.tasks.velocity_tracking_H2.rewards import penalty_electrical_power_cost

from src.mdr.utils import dr_to

_MOTION_DOWN = "src/assets/motion/h2/1_h2o6_promax_cut_aligned_down_hold50_31dof.npz"
_MOTION_UP = "src/assets/motion/h2/1_h2o6_promax_cut_aligned_up_31dof.npz"
_MOTION = "src/assets/motion/h2/1_h2o6_new_s3_31.npz"
_HAND_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _h2_31dof_tracking_base(play: bool):
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  cfg.scene.entities["robot"] = get_h2_robot_cfg()
  cfg.sim.nconmax = 42
  cfg.sim.njmax = 1000

  cfg.rewards["electrical_power_cost"] = RewardTermCfg(
      func=penalty_electrical_power_cost,
      weight=-0.2,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
              "deadband_w": 150.0, "normalization_w": 500.0},
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = tuple(Entity(get_h2_robot_cfg()).body_names)
  # joint-pos domain randomization: perturb the reference joint pose each reset.
  motion_cmd.joint_position_range = (-0.3, 0.3)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  return cfg


def unitree_h2_31dof_motion_tracking_env_cfg(play: bool = False):
  cfg = _h2_31dof_tracking_base(play=play)
  cfg.commands["motion"].motion_file = _MOTION_DOWN
  return cfg


def unitree_h2_31dof_handmass_motion_tracking_env_cfg(play: bool = False):
  cfg = _h2_31dof_tracking_base(play=play)
  cfg.commands["motion"].motion_file = _MOTION
  import mjlab.envs.mdp.dr as dr

  cfg.events["randomize_hand_payload"] = EventTermCfg(
    func=dr.body_mass,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=_HAND_BODY_NAMES,
        ),
        "operation": dr.add,
        "distribution": dr.uniform,
        "ranges": (0.0, 16.0),
    },
  )
#deploy_depth_camera_to
  return deploy_h2_depth_camera_to(cfg)
