"""Unitree G1 velocity environment configurations."""

import re

from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from dataclasses import asdict
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from mjlab.entity import Entity, EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from mjlab.sensor.contact_sensor import *
from ..rewards import *
from ..terminations import *
from ..observation import *
from ..commands import *

from src.assets.robots.unitree_h2_o6.h2_o6_constants import (
  FOOT_COLLISION_PATTERN,
  H2_O6_ACTION_SCALE,
  get_h2_o6_robot_cfg,
)
from src.assets.objects import get_barbell_race_cfg

def unitree_h2_pull_up_barbell_env_cfg(play : bool = False):
    
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")  
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 256
  cfg.scene.entities["robot"]   = get_h2_o6_robot_cfg()
  cfg.scene.entities["barbell"] = get_barbell_race_cfg()

  # 需要修改为h2_o6的运动文件
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_O6_ACTION_SCALE

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/h2o6/13_h2o6_pro.npz"


  cfg.commands["motion"] = motioncfg
  cfg.commands["motion"].sampling_mode = "uniform"
  motioncfg.anchor_body_name = "torso_link"
  _hand_body = re.compile(r"^(left|right)_(hand_base_link|thumb_|index_|middle_|ring_|pinky_)")
  motioncfg.body_names = tuple(
    b for b in Entity(get_h2_o6_robot_cfg()).body_names if not _hand_body.match(b)
  )
  if "motion_anchor_pos_b" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("motion_anchor_pos_b")  
  knee_barbell_contact_cfg = ContactSensorCfg(
      name="knee_barbell_contact",

      primary=ContactMatch(
          mode="body",
          pattern=r"^(left|right)_knee_link$",
          entity="robot",
      ),

      secondary=ContactMatch(
          mode="geom",
          pattern=r"^shaft_collision$",
          entity="barbell",
      ),

      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=False,
      debug=False,
  )
  left_finger_barbell_contact_cfg = ContactSensorCfg(
      name="left_finger_barbell_contact",

      primary=ContactMatch(
          mode="geom",
          pattern=r"^left_(index|middle|ring|pinky)_(proximal|distal)_collision_0$",
          entity="robot",
      ),

      secondary=ContactMatch(
          mode="geom",
          pattern=r"^shaft_collision$",
          entity="barbell",
      ),

      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=False,
      debug=False,
  )

  right_finger_barbell_contact_cfg = ContactSensorCfg(
      name="right_finger_barbell_contact",

      primary=ContactMatch(
          mode="geom",
          pattern=r"^right_(index|middle|ring|pinky)_(proximal|distal)_collision_0$",
          entity="robot",
      ),

      secondary=ContactMatch(
          mode="geom",
          pattern=r"^shaft_collision$",
          entity="barbell",
      ),

      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=False,
      debug=False,
  )
  non_foot_ground_contact_cfg = ContactSensorCfg(
      name="non_foot_ground_contact",

      primary=ContactMatch(
          mode="subtree",
          pattern="pelvis",
          entity="robot",
          exclude=(
              "left_ankle_pitch_link",
              "right_ankle_pitch_link",
              r"^(left|right)_hand_base_link$",
              r"^(left|right)_(thumb|index|middle|ring|pinky)_",
          ),
      ),

      secondary=ContactMatch(
          mode="body",
          pattern="terrain",
      ),

      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=False,
      debug=False,
  )
  bar_body_contact_cfg = ContactSensorCfg(
      name="bar_body_contact",

      primary=ContactMatch(
          mode="subtree",
          pattern="pelvis",
          entity="robot",
          exclude=(
              r"^(left|right)_hand_base_link$",
              r"^(left|right)_(thumb|index|middle|ring|pinky)_",
          ),
      ),

      secondary=ContactMatch(
          mode="geom",
          pattern=r".*",
          entity="barbell",
      ),

      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=False,
      debug=False,
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    knee_barbell_contact_cfg,
    left_finger_barbell_contact_cfg,
    right_finger_barbell_contact_cfg,
    non_foot_ground_contact_cfg,
    bar_body_contact_cfg,
  )    
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events.pop("push_robot", None)  # 移除随机推挤，利于早期稳定学习

  ## rewards

  cfg.rewards["reach_bar"] = RewardTermCfg(
    func=reach_bar_reward,
    weight=5.0,
    params={"command_name": "motion", "start_frame": 153, "end_frame": 295, "std": 0.08},
  )
  cfg.rewards["both_hands_grip_bar"] = RewardTermCfg(
    func=both_hands_grip_bar_reward,
    weight=2.0,
    params={"grip_offset": 0.25, "std": 0.10},
  )
  cfg.rewards["fingers_grip_bar"] = RewardTermCfg(
    func=fingers_grip_bar_reward,
    weight=5.0,
    params={"command_name": "motion", "start_frame": 290, "end_frame": 785, "std": 0.08},
  )
  cfg.rewards["fingers_barbell_contact"] = RewardTermCfg(
    func=fingers_barbell_contact_reward,
    weight=5.0,
    params={
      "command_name": "motion",
      "left_sensor_name": "left_finger_barbell_contact",
      "right_sensor_name": "right_finger_barbell_contact",
      "start_frame": 295,
      "end_frame": 785,
    },
  )
  cfg.rewards["fingers_barbell_contact_penalty"] = RewardTermCfg(
    func=fingers_barbell_contact_penalty,
    weight=-2.0,
    params={
      "command_name": "motion",
      "left_sensor_name": "left_finger_barbell_contact",
      "right_sensor_name": "right_finger_barbell_contact",
      "start_frame": 295,
      "end_frame": 785,
      "target_contact_ratio": 0.25,
    },
  )
  cfg.rewards["finger_dof_tracking"] = RewardTermCfg(
    func=finger_dof_tracking_error,
    weight=-1.0,
    params={"command_name": "motion", "start_frame": 295, "end_frame": 785},
  )
  cfg.rewards["hands_barbell_height_error"] = RewardTermCfg(
    func=hands_barbell_height_error,
    weight=-5.0,
    params={"command_name": "motion", "start_frame": 290},
  )
  cfg.rewards["barbell_height"] = RewardTermCfg(
    func=barbell_height_reward,
    weight=5.0,
    params={"command_name": "motion", "start_frame": 335, "end_frame": 460},
  )
  cfg.rewards["root_z_tracking"] = RewardTermCfg(
    func=root_z_tracking_error,
    weight=-10.0,
    params={"command_name": "motion", "start_frame": 335},
  )
  cfg.rewards["lower_body_tracking"] = RewardTermCfg(
    func=lower_body_dof_tracking_error_no_ankle,
    weight=-1.0,
    params={"command_name": "motion", "start_frame": 240},
  )
  cfg.rewards["lower_body_tracking_after_lift"] = RewardTermCfg(
    func=lower_body_dof_tracking_error,
    weight=-3.0,
    params={"command_name": "motion", "start_frame": 335},
  )
  cfg.rewards["knee_barbell_contact_penalty"] = RewardTermCfg(
    func=knee_barbell_contact_penalty,
    weight=-1.0,
    params={"sensor_name": "knee_barbell_contact"},
  )
  cfg.rewards["knee_dof_tracking"] = RewardTermCfg(
    func=knee_dof_tracking_error,
    weight=-1.0,
    params={"command_name": "motion"},
  )
  cfg.rewards["foot_slip_penalty"] = RewardTermCfg(
    func=foot_slip_penalty,
    weight=-2.0,
  )
  cfg.rewards["non_foot_ground_contact_penalty"] = RewardTermCfg(
    func=non_foot_ground_contact_penalty,
    weight=-2.0,
    params={"sensor_name": "non_foot_ground_contact", "force_threshold": 5.0},
  )
  cfg.rewards["bar_body_contact_penalty"] = RewardTermCfg(
    func=bar_body_contact_penalty,
    weight=-2.0,
    params={"sensor_name": "bar_body_contact"},
  )

  ## terminations

  cfg.terminations["barbell_droped_from_hands"] = TerminationTermCfg(
    func=barbell_droped_from_hands,
    params={"command_name": "motion", "limit_distant": 0.25},
  )
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
    func=far_away_root_pos,
    params={"limit_distant": 0.6},
  )
  cfg.terminations["non_foot_ground_contact"] = TerminationTermCfg(
    func=non_foot_ground_contact_termination,
    params={"sensor_name": "non_foot_ground_contact", "force_threshold": 10.0},
  )
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.35
  cfg.terminations["bar_body_contact"] = TerminationTermCfg(
    func=bar_body_contact_termination,
    params={"sensor_name": "bar_body_contact", "force_threshold": 10.0},
  )

  ## obs settings

  cfg.observations["critic"].terms["barbell_qpos"] = ObservationTermCfg(
    func=barbell_qpos
  )

  cfg.observations["critic"].terms["bellbar_vector"] = ObservationTermCfg(
    func=bellbar_vector,
    params={"command_name": "motion"}
  )
  # cfg.terminations["knee_ground_contact"] = TerminationTermCfg(
  #     func=knee_ground_contact,
  #     params={
  #         "sensor_name": "knee_ground_contact",
  #     },
  # )
  cfg.observations["critic"].terms["left_hand_plate_vector"] = ObservationTermCfg(
    func=left_hand_plate_vector,
    params={"command_name": "motion"}
  )

  cfg.observations["critic"].terms["right_hand_plate_vector"] = ObservationTermCfg(
    func=right_hand_plate_vector,
    params={"command_name": "motion"}
  )

  cfg.observations["critic"].terms["left_finger_contact"] = ObservationTermCfg(
    func=contact_found,
    params={"sensor_name": "left_finger_barbell_contact"},
  )
  cfg.observations["critic"].terms["right_finger_contact"] = ObservationTermCfg(
    func=contact_found,
    params={"sensor_name": "right_finger_barbell_contact"},
  )
  cfg.observations["critic"].terms["bar_body_contact"] = ObservationTermCfg(
    func=contact_found,
    params={"sensor_name": "bar_body_contact"},
  )
  cfg.observations["critic"].terms["non_foot_ground_contact"] = ObservationTermCfg(
    func=contact_found,
    params={"sensor_name": "non_foot_ground_contact"},
  )

  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }

  return cfg

def unitree_h2_pull_up_barbell_mt_env_cfg(play : bool = False):
    
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")  
  cfg.sim.nconmax = 42
  cfg.scene.entities["robot"]   = get_h2_o6_robot_cfg()
  cfg.scene.entities["barbell"] = get_barbell_race_cfg()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_O6_ACTION_SCALE

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/h2/1_h2o6_prosquat.npz"

  cfg.commands["motion"] = motioncfg
  cfg.commands["motion"].sampling_mode = "uniform"
  motioncfg.anchor_body_name = "torso_link"
  motioncfg.body_names = tuple(Entity(get_h2_o6_robot_cfg()).body_names)
  knee_ground_contact_cfg = ContactSensorCfg(
    name="knee_ground_contact",
    primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left|right)_knee_link$",
        entity="robot",
    ),
    secondary=ContactMatch(
        mode="body",
        pattern="terrain",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=False,
  )
 
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    knee_ground_contact_cfg
  )    
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

 

  ## obs settings

  cfg.observations["critic"].terms["barbell_qpos"] = ObservationTermCfg(
    func=barbell_qpos
  )

  cfg.observations["critic"].terms["bellbar_vector"] = ObservationTermCfg(
    func=bellbar_vector,
    params={"command_name": "motion"}
  )

  cfg.observations["critic"].terms["left_hand_plate_vector"] = ObservationTermCfg(
    func=left_hand_plate_vector,
    params={"command_name": "motion"}
  )

  cfg.observations["critic"].terms["right_hand_plate_vector"] = ObservationTermCfg(
    func=right_hand_plate_vector,
    params={"command_name": "motion"}
  )

  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }

  return cfg
