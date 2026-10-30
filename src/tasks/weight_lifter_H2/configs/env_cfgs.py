"""Unitree G1 velocity environment configurations."""

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
from mjlab.managers.event_manager import EventTermCfg
import mjlab.envs.mdp.dr as dr
from mjlab.tasks.tracking import mdp
from src.assets.robots.unitree_h2_o6.h2_o6_constants import (
  FOOT_COLLISION_PATTERN,
  H2_O6_ACTION_SCALE,
  get_h2_o6_robot_cfg,
  get_spec as get_h2_o6_spec,
)
from src.assets.objects import get_barbell_cfg
# from src.sensors.depth_camera import deploy_depth_camera_to
from src.sensors.advCamera import deploy_h2_depth_camera_to
from src.sensors.h2.d435c import deploy_depth_camera_to
import re
HAND_COLLISION_PATTERN = re.compile(r"^(left|right)_(hand_base_link|thumb_.*|index_.*|middle_.*|ring_.*|pinky_.*)_collision_0$")

def get_h2_o6_grip_spec():
    spec = get_h2_o6_spec()

    for geom in spec.geoms:
        if HAND_COLLISION_PATTERN.fullmatch(geom.name):
            # 先从 2 mm 开始，建议测试范围 0.002～0.004
            geom.margin = 0.003

    return spec
def unitree_h2_pull_up_barbell_env_cfg(play : bool = False):
    
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")  
  cfg.sim.nconmax = 200
  cfg.scene.entities["robot"]   = get_h2_o6_robot_cfg()

  cfg.scene.entities["barbell"] = get_barbell_cfg()
  # 需要修改为h2_o6的运动文件
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_O6_ACTION_SCALE

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/h2o6/1_h2o6_promax.npz"


  cfg.commands["motion"] = motioncfg
  cfg.commands["motion"].sampling_mode = "uniform"
  motioncfg.anchor_body_name = "torso_link"
  motioncfg.body_names = tuple(Entity(get_h2_o6_robot_cfg()).body_names)
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
  knee_ground_contact_cfg = ContactSensorCfg(
      name="knee_ground_contact",

      # 只检测左右 knee_link
      primary=ContactMatch(
          mode="body",
          pattern=r"^(left|right)_hip_yaw_link$",
          entity="robot",
      ),

      # 检测膝盖和地面的接触
      secondary=ContactMatch(
          mode="geom",
          pattern="terrain",
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
  feet_contact_ground_cfg = ContactSensorCfg(
      name="feet_contact_ground",

      primary=ContactMatch(
          mode="body",
          pattern=r"^(left|right)_ankle_roll_link$",
          entity="robot",
      ),

      secondary=ContactMatch(
          mode="geom",
          pattern="terrain",
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
    knee_ground_contact_cfg,
    feet_contact_ground_cfg,
  )    
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
      func=joint_vel_rel_zero_hand,
  )
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

  # cfg.terminations["bad_hands_dof"] = TerminationTermCfg(
  #   func=bad_hands_dof,
  #   params={"command_name": "motion", "threshold": torch.pi / 6},
  # )
  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }

  return cfg

def unitree_h2_pull_up_barbell_camera_env_cfg(play :bool =False):
  cfg = unitree_h2_pull_up_barbell_env_cfg(play=play)
  return deploy_depth_camera_to(cfg)

def unitree_h2_pull_up_barbell_mt_env_cfg(play : bool = False):
    
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")  
  cfg.sim.nconmax = 42
  cfg.scene.entities["robot"]   = get_h2_o6_robot_cfg()
  cfg.scene.entities["barbell"] = get_barbell_cfg()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_O6_ACTION_SCALE

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/h2o6/1_h2o6_promax.npz"

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

  # cfg.observations["critic"].terms["barbell_qpos"] = ObservationTermCfg(
  #   func=barbell_qpos
  # )

  # cfg.observations["critic"].terms["bellbar_vector"] = ObservationTermCfg(
  #   func=bellbar_vector,
  #   params={"command_name": "motion"}
  # )

  # cfg.observations["critic"].terms["left_hand_plate_vector"] = ObservationTermCfg(
  #   func=left_hand_plate_vector,
  #   params={"command_name": "motion"}
  # )

  # cfg.observations["critic"].terms["right_hand_plate_vector"] = ObservationTermCfg(
  #   func=right_hand_plate_vector,
  #   params={"command_name": "motion"}
  # )

  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }

  return cfg

def unitree_h2_pull_up_barbell_hand_env_cfg(play : bool = False):
  cfg=unitree_h2_pull_up_barbell_env_cfg(play=play)
  cfg.rewards["reward_barbell_z"] = RewardTermCfg(
    func=barbell_reward_hands_height,
    weight=1.0,
  )
  cfg.rewards["pently_hands_parrel"] = RewardTermCfg(
   func=pently_hands_parrel,
     params={"command_name":"motion"},
    weight=-1.0,
  )
  cfg.rewards["right_hand_grap_bar"] = RewardTermCfg(
    func=right_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.rewards["left_hand_grap_bar"] = RewardTermCfg(
    func=left_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.terminations["barbell_droped_from_hands"] = TerminationTermCfg(
    func=barbell_droped_from_hands,
    #params={"command_name": "motion", "limit_distance": 0.1},
    params={"command_name": "motion","limit_distant":0.1},
  )
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
    func=far_away_root_pos,
    params={"limit_distant": 0.42},#0.45
  )
  # cfg.terminations["knee_barbell_contact"] = TerminationTermCfg(
  #     func=knee_barbell_contact_termination,
  #     params={
  #         "sensor_name": "knee_barbell_contact",
  #     },
  # )
  # cfg.rewards["knee_dof_tracking"] = RewardTermCfg(
  #     func=knee_dof_tracking_error,
  #     weight=-1.0,
  #     params={
  #         "command_name": "motion",
  #     },
  # )
  # cfg.terminations["bad_lower_body_dof"] = TerminationTermCfg(
  #     func=bad_lower_body_dof,
  #     params={
  #         "command_name": "motion",
  #         "threshold": 1.0,
  #     },
  # )
  # if play:
  #   cfg.terminations = {
  #     "time_out" : cfg.terminations["time_out"]
  #   }
  return cfg

def unitree_h2_pull_up_barbell_hand_camera_env_cfg(play : bool = False):
  cfg=unitree_h2_pull_up_barbell_env_cfg(play=play)
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 512  
  # cfg.rewards["reward_barbell_z"] = RewardTermCfg(
  #   func=barbell_reward_hands_height,
  #   weight=1.0,
  # )
  cfg.rewards["pently_hands_parrel"] = RewardTermCfg(
   func=pently_hands_parrel,
     params={"command_name":"motion"},
    weight=-0.5,
  )
  cfg.rewards["right_hand_grap_bar"] = RewardTermCfg(
    func=right_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.rewards["left_hand_grap_bar"] = RewardTermCfg(
    func=left_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.rewards["both_hands_grip_bar"] = RewardTermCfg(
    func=both_hands_grip_bar_reward,
    params={
      "grip_offset": 0.25,
      "std": 0.10,
    },
    weight=2.0,
  )
  cfg.rewards["fingers_barbell_contact"] = RewardTermCfg(
      func=fingers_barbell_contact_reward,
      weight=3.0,
      params={
          "command_name": "motion",
          "left_sensor_name": "left_finger_barbell_contact",
          "right_sensor_name": "right_finger_barbell_contact",
          "start_frame": 70,
          "end_frame": 483,
      },
  )
  cfg.rewards["fingers_barbell_contact_penalty"] = RewardTermCfg(
      func=fingers_barbell_contact_penalty,
      weight=-5.0,
      params={
          "command_name": "motion",
          "left_sensor_name": "left_finger_barbell_contact",
          "right_sensor_name": "right_finger_barbell_contact",
          "start_frame": 70,
          "end_frame": 483,
          "target_contact_ratio": 0.25,
      },
  )
  # cfg.rewards["finger_dof_tracking_penalty "]=RewardTermCfg(
  #   func=finger_dof_tracking_error,
  #   weight=-0.5,
  #   params={
  #       "command_name": "motion",
  #   },
  # )
  cfg.rewards["fingers_grip_bar"] = RewardTermCfg(
      func=fingers_grip_bar_reward,
      weight=1.0,
      params={
          "command_name": "motion",
          "start_frame": 70,
          "end_frame": 483,
          "target_radius": 0.025,
          "std": 0.015,
      },
  )
  cfg.rewards["head_dof_tracking"] = RewardTermCfg(
      func=head_dof_tracking_error,
      weight=-3.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["knee_ground_contact_penalty"] = RewardTermCfg(
      func=knee_barbell_contact_penalty,
      weight=-5.0,
      params={
          "sensor_name": "knee_ground_contact",
      },
  )
  # cfg.rewards["lower_body_tracking"] = RewardTermCfg(
  #     func=lower_body_dof_tracking_error_no_ankle,
  #     weight=-1.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 150,
  #     },
  # )
  # cfg.rewards["lower_body_tracking_after_lift"] = RewardTermCfg(
  #     func=lower_body_dof_tracking_error,
  #     weight=-3.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 350,
  #     },
  # )
  # cfg.rewards["root_z_tracking"] = RewardTermCfg(
  #   func=root_z_tracking_error,
  #   weight=-10.0,
  #   params={
  #       "command_name": "motion",
  #       "start_frame": 220,
  #   },
  # )
  # cfg.rewards["barbell_height"] = RewardTermCfg(
  #     func=barbell_height_reward,
  #     weight=5.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 220,
  #     },
  # )
  cfg.rewards["hands_barbell_height_error"]=RewardTermCfg(
    func=hands_barbell_height_error,
    weight=-3.0,
    params={
        "command_name": "motion",
        "start_frame": 70,
    },
  )
  # cfg.rewards["knee_barbell_contact_penalty"] = RewardTermCfg(
  #     func=knee_barbell_contact_penalty,
  #     weight=-1.0,
  #     params={
  #         "sensor_name": "knee_barbell_contact",
  #     },
  # )
  # cfg.terminations["bad_finger_dof"]=TerminationTermCfg(
  #   func=bad_finger_dof,
  #   params={
  #       "command_name": "motion",
  #       "threshold":0.5,
  #   },
  # )
  # cfg.terminations["anchor_pos"].params["threshold"] = 0.25
  cfg.terminations["barbell_droped_from_hands"] = TerminationTermCfg(
    func=barbell_droped_from_hands,
    #params={"command_name": "motion", "limit_distance": 0.1},
    params={"command_name": "motion","limit_distant":0.05},
  )
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
    func=far_away_root_pos,
    params={"limit_distant": 0.6},#0.45
  )
  # cfg.terminations["knee_barbell_contact"] = TerminationTermCfg(
  #     func=knee_barbell_contact_termination,
  #     params={
  #         "sensor_name": "knee_barbell_contact",
  #     },
  # )
  # cfg.rewards["knee_dof_tracking"] = RewardTermCfg(
  #     func=knee_dof_tracking_error,
  #     weight=-1.0,
  #     params={
  #         "command_name": "motion",
  #     },
  # )
  # cfg.terminations["bad_lower_body_dof"] = TerminationTermCfg(
  #     func=bad_lower_body_dof,
  #     params={
  #         "command_name": "motion",
  #         "threshold": 1.0,
  #     },
  # )
  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }
  return deploy_depth_camera_to(cfg)

def unitree_h2_pull_up_barbell_knee_env_cfg(play : bool = False):
  cfg=unitree_h2_pull_up_barbell_env_cfg(play=play)
  # cfg.rewards["reward_barbell_z"] = RewardTermCfg(
  #   func=barbell_reward_hands_height,
  #   weight=1.0,
  # )
  cfg.rewards["pently_hands_parrel"] = RewardTermCfg(
   func=pently_hands_parrel,
     params={"command_name":"motion"},
    weight=-0.5,
  )
  cfg.rewards["right_hand_grap_bar"] = RewardTermCfg(
    func=right_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.rewards["left_hand_grap_bar"] = RewardTermCfg(
    func=left_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.rewards["both_hands_grip_bar"] = RewardTermCfg(
    func=both_hands_grip_bar_reward,
    params={
      "grip_offset": 0.25,
      "std": 0.10,
    },
    weight=2.0,
  )
  cfg.rewards["fingers_barbell_contact"] = RewardTermCfg(
      func=fingers_barbell_contact_reward,
      weight=5.0,
      params={
          "command_name": "motion",
          "left_sensor_name": "left_finger_barbell_contact",
          "right_sensor_name": "right_finger_barbell_contact",
          "start_frame": 70,
          "end_frame": 483,
      },
  )
  cfg.rewards["fingers_barbell_contact_penalty"] = RewardTermCfg(
      func=fingers_barbell_contact_penalty,
      weight=-2.0,
      params={
          "command_name": "motion",
          "left_sensor_name": "left_finger_barbell_contact",
          "right_sensor_name": "right_finger_barbell_contact",
          "start_frame": 70,
          "end_frame": 483,
          "target_contact_ratio": 0.25,
      },
  )
  cfg.rewards["finger_dof_tracking_penalty "]=RewardTermCfg(
    func=finger_dof_tracking_error,
    weight=-0.5,
    params={
        "command_name": "motion",
    },
  )
  cfg.rewards["fingers_grip_bar"] = RewardTermCfg(
      func=fingers_grip_bar_reward,
      weight=5.0,
      params={
          "command_name": "motion",
          "start_frame": 70,
          "std": 0.05,
      },
  )
  cfg.rewards["head_dof_tracking"] = RewardTermCfg(
      func=head_dof_tracking_error,
      weight=-3.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["lower_body_tracking"] = RewardTermCfg(
      func=lower_body_dof_tracking_error,
      weight=-2.0,
      params={
          "command_name": "motion",
          "start_frame": 270,
      },
  )
  cfg.rewards["lower_body_tracking_reward"] = RewardTermCfg(
      func=lower_body_dof_tracking_error_reward,
      weight=2.0,
      params={
          "command_name": "motion",
          "start_frame": 270,
      },
  )
  # cfg.rewards["anker_tracking"] = RewardTermCfg(
  #     func=ankle_dof_tracking_error,
  #     weight=-5.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 170,
  #     },
  # )
  cfg.rewards["knee_height_tracking"] = RewardTermCfg(
      func=knee_height_tracking_error,
      weight=5.0,
      params={
          "command_name": "motion",
          "start_frame": 90,
      },
  )
  # cfg.rewards["knee_height_tracking_pen"] = RewardTermCfg(
  #     func=knee_height_tracking_error_pen,
  #     weight=-5.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 90,
  #     },
  # )
  # cfg.rewards["lower_body_tracking_after_lift"] = RewardTermCfg(
  #     func=lower_body_dof_tracking_error,
  #     weight=-10.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 350,
  #     },
  # )
  cfg.rewards["root_z_tracking"] = RewardTermCfg(
    func=root_z_tracking_error,
    weight=-3.0,
    params={
        "command_name": "motion",
        "start_frame": 190,
    },
  )
  cfg.rewards["root_z_tracking_reward"] = RewardTermCfg(
      func=root_z_tracking_reward,
      weight=5.0,
      params={
          "command_name": "motion",
          "start_frame": 190,
          "std": 0.05,
      },
  )
  cfg.rewards["head_z_tracking"] = RewardTermCfg(
      func=head_z_tracking_reward,
      weight=2.0,
      params={
          "command_name": "motion",
          "start_frame": 190,
          "std": 0.05,
          "head_body_name": "head_yaw_link",
      },
  )
  # cfg.rewards["feet_air_penalty"] = RewardTermCfg(
  #     func=feet_air_penalty,
  #     weight=-5.0,
  #     params={
  #         "sensor_name": "feet_contact_ground",
  #         "command_name": "motion",
  #         "start_frame": 300,
  #     },
  # )
  # cfg.rewards["barbell_height"] = RewardTermCfg(
  #     func=barbell_height_reward,
  #     weight=5.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 220,
  #     },
  # )
  cfg.rewards["hands_barbell_height_error"]=RewardTermCfg(
    func=hands_barbell_height_error,
    weight=-5.0,
    params={
        "command_name": "motion",
        "start_frame": 70,
    },
  )
  cfg.rewards["hands_height_tracking"] = RewardTermCfg(
      func=hands_height_tracking_error,
      weight=2.0,
      params={
          "command_name": "motion",
          "start_frame": 270,
      },
  )
  cfg.rewards["knee_ground_contact_penalty"] = RewardTermCfg(
      func=knee_barbell_contact_penalty,
      weight=-5.0,
      params={
          "sensor_name": "knee_ground_contact",
      },
  )
  # cfg.terminations["bad_finger_dof"]=TerminationTermCfg(
  #   func=bad_finger_dof,
  #   params={
  #       "command_name": "motion",
  #       "threshold":0.5,
  #   },
  # )
  # cfg.terminations["anchor_pos"]= TerminationTermCfg(
  #     func=mdp.bad_anchor_pos_z_only,
  #     params={"command_name": "motion", "threshold": 0.25},
  # )
  cfg.terminations["barbell_droped_from_hands"] = TerminationTermCfg(
    func=barbell_droped_from_hands,
    #params={"command_name": "motion", "limit_distance": 0.1},
    params={"command_name": "motion","limit_distant":0.1},
  )
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
    func=far_away_root_pos,
    params={"limit_distant": 0.6},#0.45
  )
  # cfg.terminations["knee_barbell_contact"] = TerminationTermCfg(
  #     func=knee_barbell_contact_termination,
  #     params={
  #         "sensor_name": "knee_barbell_contact",
  #     },
  # )
  # cfg.rewards["knee_dof_tracking"] = RewardTermCfg(
  #     func=knee_dof_tracking_error,
  #     weight=-1.0,
  #     params={
  #         "command_name": "motion",
  #     },
  # )
  # cfg.terminations["bad_lower_body_dof"] = TerminationTermCfg(
  #     func=bad_lower_body_dof,
  #     params={
  #         "command_name": "motion",
  #         "threshold": 1.0,
  #     },
  # )
  cfg.events["reset_robot_joints"]= EventTermCfg(
    func=mdp.reset_joints_by_offset,
    mode="startup",
    params={
      "position_range": (1.0, 1.0),
      "velocity_range": (0.0, 0.1),
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    },
  )
  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }
  return deploy_depth_camera_to(cfg)

def unitree_h2_pull_up_barbell_hand_event_env_cfg(play : bool =False):
  cfg=unitree_h2_pull_up_barbell_camera_env_cfg(play=play)
  cfg.events["randomize_hand_payload"] = EventTermCfg(
      func=dr.body_mass,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg(
              "robot",
              body_names=[
                  "left_hand_base_link",
                  "right_hand_base_link",
              ],
          ),
          "operation": dr.add,
          "distribution": dr.uniform,
          "ranges": (5.0, 10.0),
      },
  )  
  return cfg

def unitree_h2_pull_up_barbell_hand_event_nocamera_env_cfg(play : bool =False):
  cfg=unitree_h2_pull_up_barbell_env_cfg(play=play)
  cfg.events["randomize_hand_payload"] = EventTermCfg(
      func=dr.body_mass,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg(
              "robot",
              body_names=[
                  "left_hand_base_link",
                  "right_hand_base_link",
              ],
          ),
          "operation": dr.add,
          "distribution": dr.uniform,
          "ranges": (0.0, 15.0),
      },
  )  
  return cfg

def unitree_h2_pull_up_barbell_hand_res_env_cfg(play : bool =False):
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")  
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 512
  cfg.scene.entities["robot"]   = get_h2_o6_robot_cfg()

  cfg.scene.entities["barbell"] = get_barbell_cfg(
    plate_mass=0.5,
    bar_mass=0.5,
  )
  # 需要修改为h2_o6的运动文件
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_O6_ACTION_SCALE

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/h2o6/1_h2o6_promax.npz"
  cfg.commands["motion"] = motioncfg
  cfg.commands["motion"].sampling_mode = "uniform"
  motioncfg.anchor_body_name = "torso_link"
  motioncfg.body_names = tuple(Entity(get_h2_o6_robot_cfg()).body_names)
  cfg = deploy_h2_depth_camera_to(cfg,camera_name="depth_camera")
  if "motion_anchor_pos_b" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("motion_anchor_pos_b")  
  
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
      func=joint_vel_rel_zero_hand,
  )
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

  cfg.observations["graping"] = ObservationGroupCfg(
      terms={
          "graping": ObservationTermCfg(
              func=graping,
              params={"command_name":"motion"},
          )
      },
      concatenate_terms=True,
      enable_corruption=False,
  )
  cfg.rewards["head_dof_tracking_pen"] = RewardTermCfg(
      func=head_dof_tracking_error_pen,
      weight=-1.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["head_dof_tracking"] = RewardTermCfg(
      func=head_dof_tracking_error,
      weight=1.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["reward_barbell_z"] = RewardTermCfg(
    func=barbell_reward_hands_height,
    weight=1.0,
  )
  # cfg.rewards["pently_hands_parrel"] = RewardTermCfg(
  #     func=pently_hands_parrel,
  #     params={"command_name":"motion"},
  #     weight=1.0,
  # )
  cfg.rewards["right_hand_grap_bar"] = RewardTermCfg(
    func=right_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.rewards["camera_hands_alignment_reward"] = RewardTermCfg(
    func=camera_hands_alignment_reward,
    params={"command_name":"motion","sensor_name":"depth_camera","sigma":0.5},
    weight=1.0,
  )
  cfg.rewards["left_hand_grap_bar"] = RewardTermCfg(
    func=left_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.terminations["anchor_pos"].params["threshold"] = 0.25
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
      func=far_away_root_pos,
      params={"limit_distant": 0.42},
    )
  cfg.terminations["hands_not_on_the_bar"] = TerminationTermCfg(
    func=not_on_the_bar,
    params={"command_name": "motion", "threshold": math.cos(5 / 180 * torch.pi)},
  )
  cfg.events["randomize_barbell_mass"] = EventTermCfg(
      func=dr.body_mass,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg(
              "barbell",
              body_names=["barbell"],
          ),
          "operation": dr.add,
          "distribution": dr.uniform,
          "ranges": (0.0, 16.0),
      },
  )
  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }
  return cfg

def unitree_h2_pull_up_barbell_hand_expert_env_cfg(play : bool =False):
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")  
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 512
  cfg.scene.entities["robot"]   = get_h2_o6_robot_cfg()

  cfg.scene.entities["barbell"] = get_barbell_cfg(
    plate_mass=0.5,
    bar_mass=0.5,
  )
  # 需要修改为h2_o6的运动文件
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_O6_ACTION_SCALE

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/h2o6/1_h2o6_promax.npz"
  cfg.commands["motion"] = motioncfg
  cfg.commands["motion"].sampling_mode = "uniform"
  motioncfg.anchor_body_name = "torso_link"
  motioncfg.body_names = tuple(Entity(get_h2_o6_robot_cfg()).body_names)

  if "motion_anchor_pos_b" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("motion_anchor_pos_b")  
  
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
      func=joint_vel_rel_zero_hand,
  )
  # cfg.observations["critic"].terms["barbell_qpos"] = ObservationTermCfg(
  #   func=barbell_qpos
  # )

  # cfg.observations["critic"].terms["bellbar_vector"] = ObservationTermCfg(
  #   func=bellbar_vector,
  #   params={"command_name": "motion"}
  # )

  # cfg.observations["critic"].terms["left_hand_plate_vector"] = ObservationTermCfg(
  #   func=left_hand_plate_vector,
  #   params={"command_name": "motion"}
  # )

  # cfg.observations["critic"].terms["right_hand_plate_vector"] = ObservationTermCfg(
  #   func=right_hand_plate_vector,
  #   params={"command_name": "motion"}
  # )

  cfg.observations["graping"] = ObservationGroupCfg(
      terms={
          "graping": ObservationTermCfg(
              func=graping,
              params={"command_name":"motion"},
          )
      },
      concatenate_terms=True,
      enable_corruption=False,
  )
  cfg.rewards["head_dof_tracking_pen"] = RewardTermCfg(
      func=head_dof_tracking_error_pen,
      weight=-1.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["head_dof_tracking"] = RewardTermCfg(
      func=head_dof_tracking_error,
      weight=1.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["reward_barbell_z"] = RewardTermCfg(
    func=barbell_reward_hands_height,
    weight=1.0,
  )
  cfg.rewards["right_hand_grap_bar"] = RewardTermCfg(
    func=right_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.rewards["left_hand_grap_bar"] = RewardTermCfg(
    func=left_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.terminations["anchor_pos"].params["threshold"] = 0.25
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
      func=far_away_root_pos,
      params={"limit_distant": 0.42},
    )
  cfg.terminations["hands_not_on_the_bar"] = TerminationTermCfg(
    func=not_on_the_bar,
    params={"command_name": "motion", "threshold": math.cos(5 / 180 * torch.pi)},
  )
  cfg.events["randomize_barbell_mass"] = EventTermCfg(
      func=dr.body_mass,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg(
              "barbell",
              body_names=["barbell"],
          ),
          "operation": dr.add,
          "distribution": dr.uniform,
          "ranges": (0.0, 16.0),
      },
  )
  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }
  return cfg

def unitree_h2_pull_up_barbell_hand_res_2x_env_cfg(play : bool =False):
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")  
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 512
  cfg.scene.entities["robot"]   = get_h2_o6_robot_cfg()

  cfg.scene.entities["barbell"] = get_barbell_cfg(
    plate_mass=0.5,
    bar_mass=0.5,
  )
  # 需要修改为h2_o6的运动文件
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_O6_ACTION_SCALE

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/h2o6/1_h2o6_promax.npz"
  cfg.commands["motion"] = motioncfg
  cfg.commands["motion"].sampling_mode = "uniform"
  motioncfg.anchor_body_name = "torso_link"
  motioncfg.body_names = tuple(Entity(get_h2_o6_robot_cfg()).body_names)

  if "motion_anchor_pos_b" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("motion_anchor_pos_b")  
  
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
      func=joint_vel_rel_zero_hand,
  )
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

  cfg.observations["graping"] = ObservationGroupCfg(
      terms={
          "graping": ObservationTermCfg(
              func=graping,
              params={"command_name":"motion"},
          )
      },
      concatenate_terms=True,
      enable_corruption=False,
  )
  cfg.rewards["head_dof_tracking_pen"] = RewardTermCfg(
      func=head_dof_tracking_error_pen,
      weight=-1.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["head_dof_tracking"] = RewardTermCfg(
      func=head_dof_tracking_error,
      weight=1.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["reward_barbell_z"] = RewardTermCfg(
    func=barbell_reward_hands_height,
    weight=1.0,
  )
  # cfg.rewards["pently_hands_parrel"] = RewardTermCfg(
  #     func=pently_hands_parrel,
  #     params={"command_name":"motion"},
  #     weight=1.0,
  # )
  cfg.rewards["right_hand_grap_bar"] = RewardTermCfg(
    func=right_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.rewards["left_hand_grap_bar"] = RewardTermCfg(
    func=left_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.terminations["anchor_pos"].params["threshold"] = 0.25
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
      func=far_away_root_pos,
      params={"limit_distant": 0.42},
    )
  cfg.terminations["hands_not_on_the_bar"] = TerminationTermCfg(
    func=not_on_the_bar,
    params={"command_name": "motion", "threshold": math.cos(5 / 180 * torch.pi)},
  )
  cfg.events["randomize_barbell_mass"] = EventTermCfg(
      func=dr.body_mass,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg(
              "barbell",
              body_names=["barbell"],
          ),
          "operation": dr.add,
          "distribution": dr.uniform,
          "ranges": (0.0, 16.0),
      },
  )
  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }
  return deploy_h2_depth_camera_to(cfg,camera_name="depth_camera",camera_W=96,camera_H=54)


from src.sensors.advCamera import advanced_depth_camera_record

def unitree_h2_pull_up_barbell_hand_res_env_cfg_record(play = False):
  cfg = unitree_h2_pull_up_barbell_hand_res_env_cfg(play)
  cfg.observations.pop("depth_camera")
  cfg.observations["depth_camera"] = ObservationGroupCfg(
      terms={
          "depth_camera": ObservationTermCfg(
              func=advanced_depth_camera_record, #advanced_depth_camera_save
              params={
                  "command_name": "motion",
              },
              history_length=1,
              flatten_history_dim=False,
          )
      },
      concatenate_terms=True,
      enable_corruption=False,
  )
  return cfg


def unitree_h2_pull_up_barbell_privil_env_cfg(play : bool =False):
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")  
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 512
  cfg.scene.entities["robot"]   = get_h2_o6_robot_cfg()

  cfg.scene.entities["barbell"] = get_barbell_cfg(
    plate_mass=0.5,
    bar_mass=0.5,
  )
  # 需要修改为h2_o6的运动文件
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_O6_ACTION_SCALE

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/h2o6/1_h2o6_promax.npz"
  cfg.commands["motion"] = motioncfg
  cfg.commands["motion"].sampling_mode = "uniform"
  motioncfg.anchor_body_name = "torso_link"
  motioncfg.body_names = tuple(Entity(get_h2_o6_robot_cfg()).body_names)

  if "motion_anchor_pos_b" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("motion_anchor_pos_b")  
  cfg=deploy_h2_depth_camera_to(cfg,camera_name="depth_camera")
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
      func=joint_vel_rel_zero_hand,
  )
  barbell_term = {
    "barbell_qpos": ObservationTermCfg(
      func=barbell_qpos,
    ),
    "bellbar_vector": ObservationTermCfg(
      func=bellbar_vector,
      params={"command_name": "motion"},
    ),
    "left_hand_plate_vector": ObservationTermCfg(
      func=left_hand_plate_vector,
      params={"command_name": "motion"},
    ),
    "right_hand_plate_vector": ObservationTermCfg(
      func=right_hand_plate_vector,
      params={"command_name": "motion"},
    ),
  }
  cfg.observations["barbell"]= ObservationGroupCfg(
      terms=barbell_term,
      concatenate_terms=True,
      enable_corruption=True,
  )
  cfg.observations["critic"].terms["barbell_qpos"] = ObservationTermCfg(
    func=barbell_qpos,
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

  cfg.observations["graping"] = ObservationGroupCfg(
      terms={
          "graping": ObservationTermCfg(
              func=graping,
              params={"command_name":"motion"},
          )
      },
      concatenate_terms=True,
      enable_corruption=False,
  )
  # cfg.rewards["reward_center_of_mass"] = RewardTermCfg(
  #   func=reward_center_of_mass,
  #   params={"command_name": "motion","sigma_com": 0.1},
  #   weight=1.0,
  # )
  cfg.rewards["head_dof_tracking_pen"] = RewardTermCfg(
      func=head_dof_tracking_error_pen,
      weight=-1.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["head_dof_tracking"] = RewardTermCfg(
      func=head_dof_tracking_error,
      weight=1.0,
      params={
          "command_name": "motion",
      },
  )
  cfg.rewards["reward_barbell_z"] = RewardTermCfg(
    func=barbell_reward_hands_height,
    weight=1.0,
  )
  # cfg.rewards["knee_height_tracking"] = RewardTermCfg(
  #     func=knee_height_tracking_error,
  #     weight=5.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 180,
  #     },
  # )
  cfg.rewards["anker_tracking"] = RewardTermCfg(
      func=ankle_dof_tracking_error,
      weight=-5.0,
      params={
          "command_name": "motion",
      },
  )
  # cfg.rewards["pently_hands_parrel"] = RewardTermCfg(
  #     func=pently_hands_parrel,
  #     params={"command_name":"motion"},
  #     weight=1.0,
  # )
  cfg.rewards["lower_body_tracking_after_lift"] = RewardTermCfg(
      func=lower_body_dof_tracking_error,
      weight=-3.0,
      params={
          "command_name": "motion",
          #"start_frame": 350,
      },
  )
  cfg.rewards["right_hand_grap_bar"] = RewardTermCfg(
    func=right_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.rewards["left_hand_grap_bar"] = RewardTermCfg(
    func=left_hand_grap_bar,
    params={"command_name":"motion"},
    weight=1.0,
  )
  cfg.terminations["anchor_pos"].params["threshold"] = 0.25
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
      func=far_away_root_pos,
      params={"limit_distant": 0.42},
    )
  cfg.terminations["hands_not_on_the_bar"] = TerminationTermCfg(
    func=not_on_the_bar,
    params={"command_name": "motion", "threshold": math.cos(5 / 180 * torch.pi)},
  )
  cfg.events["randomize_barbell_mass"] = EventTermCfg(
      func=dr.body_mass,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg(
              "barbell",
              body_names=["barbell"],
          ),
          "operation": dr.add,
          "distribution": dr.uniform,
          "ranges": (0.0, 16.0),
      },
  )
  if play:
    cfg.terminations = {
      "time_out" : cfg.terminations["time_out"]
    }
  return cfg 