"""Unitree G1 + O6 pull-up barbell environment configurations."""

import math
import re
from dataclasses import asdict

import torch
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.entity import Entity

from ..rewards import *
from ..terminations import *
from ..observation import *
from ..commands import *
from mjlab.managers.event_manager import EventTermCfg
from src.assets.robots.unitree_g1_o6.g1_constants import (
  FOOT_COLLISION_PATTERN,
  G1_O6_ACTION_SCALE,
  get_g1_o6_robot_cfg,
)
from src.assets.objects import get_barbell_cfg
from src.sensors.depth_camera import deploy_depth_camera_to
import mjlab.envs.mdp.dr as dr
from src.sensors.advCamera import deploy_g1_depth_camera_to
def unitree_g1_pull_up_barbell_env_cfg(play: bool = False):
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")
  # cfg.sim.nconmax = 512
  # cfg.sim.njmax = 512
  cfg.scene.entities["robot"] = get_g1_o6_robot_cfg()
  cfg.scene.entities["barbell"] = get_barbell_cfg()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  # Keep current hand travel scales; body scales from G1_O6_ACTION_SCALE.
  action_scale = dict(G1_O6_ACTION_SCALE)
  _HAND_SCALE_OVERRIDE: dict[str, float] = {
    ".*_thumb_cmc_pitch": 0.58,
    ".*_thumb_ip": 1.08,
    ".*_thumb_cmc_yaw": 1.30,
    ".*_index_dip": 1.43,
    ".*_middle_dip": 1.43,
    ".*_ring_dip": 1.43,
    ".*_pinky_dip": 1.43,
    ".*_index_mcp_pitch": 1.60,
    ".*_middle_mcp_pitch": 1.60,
    ".*_ring_mcp_pitch": 1.60,
    ".*_pinky_mcp_pitch": 1.60,
  }
  for pattern, val in _HAND_SCALE_OVERRIDE.items():
    for n in G1_O6_ACTION_SCALE:
      if re.match(pattern, n):
        G1_O6_ACTION_SCALE[n] = val
  joint_pos_action.scale = action_scale

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/g1/1_g1o6_pro.npz"
  motioncfg.anchor_body_name = "torso_link"
  motioncfg.body_names = tuple(Entity(get_g1_o6_robot_cfg()).body_names)
  motioncfg.sampling_mode = "uniform"
  cfg.commands["motion"] = motioncfg
  if "motion_anchor_pos_b" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("motion_anchor_pos_b")  
  # Finger–shaft contact sensors (O6 five-finger advantage vs Dex3).
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
  shin_barbell_contact_cfg = ContactSensorCfg(
    name="shin_barbell_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(left|right)_(knee|ankle_pitch)_link$",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="geom",
      pattern=r"^shaft_collision$",
      entity="barbell",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=4,
    track_air_time=False,
    debug=False,
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    left_finger_barbell_contact_cfg,
    right_finger_barbell_contact_cfg,
    shin_barbell_contact_cfg,
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  
  # --- Rewards v0.2.12: prioritize motion tracking; keep light true-contact grasp ---
  # Boost base tracking so torso/body signal is not drowned by grasp terms.
  cfg.rewards["motion_global_root_pos"].weight = 1.0
  cfg.rewards["motion_global_root_ori"].weight = 1.0
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
          "ranges": (3.5, 3.51),
      },
  )
  # Height-to-bar soft cue only (was 1.0; drove torso dip).
  # cfg.rewards["reward_barbell_z"] = RewardTermCfg(
  #   func=barbell_reward_hands_height,
  #   weight=0.3,
  # )
  # cfg.rewards["pently_hands_parrel"] = RewardTermCfg(
  #   func=pently_hands_parrel,
  #   params={"command_name": "motion"},
  #   weight=-0.5,
  # )
  # # Light palm-plane alignment (was 1.0 each).
  # cfg.rewards["right_hand_grap_bar"] = RewardTermCfg(
  #   func=right_hand_grap_bar,
  #   params={"command_name": "motion"},
  #   weight=0.5,
  # )
  # cfg.rewards["left_hand_grap_bar"] = RewardTermCfg(
  #   func=left_hand_grap_bar,
  #   params={"command_name": "motion"},
  #   weight=0.5,
  # )
  # # Palm-to-grip-point: keep weak (was 3–4.5); no palm orientation constraint.
  # # v0.2.13: 1.0 -> 1.5
  # cfg.rewards["both_hands_grip_bar"] = RewardTermCfg(
  #   func=both_hands_grip_bar_reward,
  #   params={"grip_offset": 0.25, "std": 0.15},
  #   weight=1.5,
  # )
  # # True contact only: delayed to graping window (160+) and weight 5/8 -> 2.
  # # v0.2.13: 2.0 -> 3.0
  # cfg.rewards["fingers_barbell_contact"] = RewardTermCfg(
  #   func=fingers_barbell_contact_reward,
  #   weight=3.0,
  #   params={
  #     "command_name": "motion",
  #     "left_sensor_name": "left_finger_barbell_contact",
  #     "right_sensor_name": "right_finger_barbell_contact",
  #     "start_frame": 150,
  #     "end_frame": 528,
  #   },
  # )
  # # Soft under-contact penalty (was -2/-4); avoid force-reach shortcuts.
  # # v0.2.13: -0.5 -> -0.8
  # cfg.rewards["fingers_barbell_contact_penalty"] = RewardTermCfg(
  #   func=fingers_barbell_contact_penalty,
  #   weight=-2.0,
  #   params={
  #     "command_name": "motion",
  #     "left_sensor_name": "left_finger_barbell_contact",
  #     "right_sensor_name": "right_finger_barbell_contact",
  #     "start_frame": 150,
  #     "end_frame": 528,
  #     "target_contact_ratio": 0.35, # v0.2.13: 0.25 -> 0.35
  #   },
  # )
  # # Prefer demo finger DOF over geometric hover grasp.
  # # v0.2.13: -1.0 -> -1.5
  # cfg.rewards["finger_dof_tracking_penalty"] = RewardTermCfg(
  #   func=finger_dof_tracking_error,
  #   weight=-1.5,
  #   params={"command_name": "motion"},
  # )
  # # OFF: fingers_grip_bar — axis proximity hover-snipe drove bend-to-reach.
  # # cfg.rewards["fingers_grip_bar"] = RewardTermCfg(
  # #   func=fingers_grip_bar_reward,
  # #   weight=5.0,
  # #   params={"command_name": "motion", "start_frame": 130, "std": 0.05},
  # # )
  # cfg.rewards["lower_body_tracking"] = RewardTermCfg(
  #   func=lower_body_dof_tracking_error,
  #   weight=-4.0,
  #   params={"command_name": "motion", "start_frame": 0},
  # )
  # # OFF: hands_barbell_height_error — worst-hand height penalty caused torso dive.
  # # cfg.rewards["hands_barbell_height_error"] = RewardTermCfg(
  # #   func=hands_barbell_height_error,
  # #   weight=-5.0,
  # #   params={"command_name": "motion", "start_frame": 130},
  # # )
  # # Re-enable root Z tracking from frame 0 to stop anchor height loss at grasp.
  # cfg.rewards["root_z_tracking"] = RewardTermCfg(
  #   func=root_z_tracking_error,
  #   weight=-5.0,
  #   params={"command_name": "motion", "start_frame": 0},
  # )
  # # cfg.rewards["barbell_height"] = RewardTermCfg(
  # #   func=barbell_height_reward,
  # #   weight=5.0,
  # #   params={"command_name": "motion", "start_frame": 220},
  # # )
  # # cfg.rewards["fingers_grap"] = RewardTermCfg(
  # #   func=fingers_grap,
  # #   params={"command_name": "motion"},
  # #   weight=-1.0,
  # # )

  # # --- Terminations ---
  # # v0.2.11: limit 0.1->0.12 — slightly more slack once hands reach the bar
  # cfg.terminations["barbell_droped_from_hands"] = TerminationTermCfg(
  #   func=barbell_droped_from_hands,
  #   params={"command_name": "motion", "limit_distant": 0.12},
  # )
  # cfg.terminations["far_away_from_root"] = TerminationTermCfg(
  #   func=far_away_root_pos,
  #   params={"limit_distant": 0.6},
  # )
  # cfg.terminations["shin_barbell_contact"] = TerminationTermCfg(
  #   func=shin_barbell_contact_termination,
  #   params={"sensor_name": "shin_barbell_contact"},
  # )
  # cfg.terminations["hands_not_on_the_bar"] = TerminationTermCfg(
  #   func=not_on_the_bar,
  #   params={
  #     "command_name": "motion",
  #     "threshold": 0.9961946980917455,
  #   },
  # )
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
          "ranges": (3.0, 5.0),
      },
  )
  cfg.observations["critic"].terms["barbell_qpos"] = ObservationTermCfg(
    func=barbell_qpos
  )
  cfg.observations["critic"].terms["bellbar_vector"] = ObservationTermCfg(
    func=bellbar_vector,
    params={"command_name": "motion"},
  )
  cfg.observations["critic"].terms["left_hand_plate_vector"] = ObservationTermCfg(
    func=left_hand_plate_vector,
    params={"command_name": "motion"},
  )
  cfg.observations["critic"].terms["right_hand_plate_vector"] = ObservationTermCfg(
    func=right_hand_plate_vector,
    params={"command_name": "motion"},
  )

  if play:
    cfg.terminations = {
      "time_out": cfg.terminations["time_out"],
    }

  return deploy_g1_depth_camera_to(cfg,camera_name="depth_camera")
  # return cfg


def unitree_g1_pull_up_barbell_mt_env_cfg(play: bool = False):
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
  if "base_lin_vel" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("base_lin_vel")
  cfg.sim.nconmax = 512
  cfg.sim.njmax = 512
  cfg.scene.entities["robot"] = get_g1_o6_robot_cfg()
  cfg.scene.entities["barbell"] = get_barbell_cfg()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  # Keep current hand travel scales; body scales from G1_O6_ACTION_SCALE.
  action_scale = dict(G1_O6_ACTION_SCALE)
  _HAND_SCALE_OVERRIDE: dict[str, float] = {
    ".*_thumb_cmc_pitch": 0.58,
    ".*_thumb_ip": 1.08,
    ".*_thumb_cmc_yaw": 1.30,
    ".*_index_dip": 1.43,
    ".*_middle_dip": 1.43,
    ".*_ring_dip": 1.43,
    ".*_pinky_dip": 1.43,
    ".*_index_mcp_pitch": 1.60,
    ".*_middle_mcp_pitch": 1.60,
    ".*_ring_mcp_pitch": 1.60,
    ".*_pinky_mcp_pitch": 1.60,
  }
  for pattern, val in _HAND_SCALE_OVERRIDE.items():
    for n in G1_O6_ACTION_SCALE:
      if re.match(pattern, n):
        G1_O6_ACTION_SCALE[n] = val
  joint_pos_action.scale = action_scale

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/g1/1_g1o6_pro.npz"
  motioncfg.anchor_body_name = "torso_link"
  motioncfg.body_names = tuple(Entity(get_g1_o6_robot_cfg()).body_names)
  motioncfg.sampling_mode = "uniform"
  cfg.commands["motion"] = motioncfg

  # Finger–shaft contact sensors (O6 five-finger advantage vs Dex3).
  left_finger_barbell_contact_cfg = ContactSensorCfg(
      name="left_finger_barbell_contact",

      primary=ContactMatch(
          mode="geom",
          pattern=(r"^left_(hand_base_link|thumb_.*|index_.*|middle_.*|ring_.*|pinky_.*)_collision_0$"),
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
          pattern=(r"^right_(hand_base_link|thumb_.*|index_.*|middle_.*|ring_.*|pinky_.*)_collision_0$"),
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
  shin_barbell_contact_cfg = ContactSensorCfg(
    name="shin_barbell_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(left|right)_(knee|ankle_pitch)_link$",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="geom",
      pattern=r"^shaft_collision$",
      entity="barbell",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=4,
    track_air_time=False,
    debug=False,
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    left_finger_barbell_contact_cfg,
    right_finger_barbell_contact_cfg,
    shin_barbell_contact_cfg,
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # cfg.observations["actor"].terms["barbell_pos_b"] = ObservationTermCfg(
  #     func=barbell_pos_b,
  # )

  # cfg.observations["actor"].terms["bellbar_vector_b"] = ObservationTermCfg(
  #     func=bellbar_vector_b,
  #     params={
  #         "command_name": "motion",
  #     },
  # )

  # cfg.observations["actor"].terms["left_hand_to_barbell_b"] = ObservationTermCfg(
  #     func=left_hand_to_barbell_b,
  #     params={
  #         "command_name": "motion",
  #     },
  # )

  # cfg.observations["actor"].terms["right_hand_to_barbell_b"] = ObservationTermCfg(
  #     func=right_hand_to_barbell_b,
  #     params={
  #         "command_name": "motion",
  #     },
  # )

  # cfg.observations["actor"].terms["barbell_qpos"] = ObservationTermCfg(
  #   func=barbell_qpos
  # )

  # cfg.observations["actor"].terms["bellbar_vector"] = ObservationTermCfg(
  #   func=bellbar_vector,
  #   params={"command_name": "motion"}
  # )
  cfg.observations["critic"].terms["barbell_qpos"] = ObservationTermCfg(
    func=barbell_qpos
  )
  cfg.observations["critic"].terms["bellbar_vector"] = ObservationTermCfg(
    func=bellbar_vector,
    params={"command_name": "motion"},
  )
  cfg.observations["critic"].terms["left_hand_plate_vector"] = ObservationTermCfg(
    func=left_hand_plate_vector,
    params={"command_name": "motion"},
  )
  cfg.observations["critic"].terms["right_hand_plate_vector"] = ObservationTermCfg(
    func=right_hand_plate_vector,
    params={"command_name": "motion"},
  )

  if play:
    cfg.terminations = {
      "time_out": cfg.terminations["time_out"],
    }

  return cfg


def unitree_g1_pull_up_barbell_privil_env_cfg(play : bool = False):
  cfg=unitree_g1_pull_up_barbell_mt_env_cfg(play=play)
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
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    knee_barbell_contact_cfg
  )  
  # --- Rewards v0.2.12: prioritize motion tracking; keep light true-contact grasp ---
  # Boost base tracking so torso/body signal is not drowned by grasp terms.
  # cfg.rewards["motion_global_root_pos"].weight = 1.0
  # cfg.rewards["motion_global_root_ori"].weight = 1.0

  # Height-to-bar soft cue only (was 1.0; drove torso dip).
  # cfg.rewards["reward_barbell_z"] = RewardTermCfg(
  #   func=barbell_reward_hands_height,
  #   weight=0.3,
  # )
  cfg.rewards["pently_hands_parrel"] = RewardTermCfg(
    func=pently_hands_parrel,
    params={"command_name": "motion"},
    weight=-0.5,
  )
  # # Light palm-plane alignment (was 1.0 each).
  cfg.rewards["right_hand_grap_bar"] = RewardTermCfg(
    func=right_hand_grap_bar,
    params={"command_name": "motion"},
    weight=0.5,
  )
  cfg.rewards["left_hand_grap_bar"] = RewardTermCfg(
    func=left_hand_grap_bar,
    params={"command_name": "motion"},
    weight=0.5,
  )
  # # Palm-to-grip-point: keep weak (was 3–4.5); no palm orientation constraint.
  # # v0.2.13: 1.0 -> 1.5
  cfg.rewards["both_hands_grip_bar"] = RewardTermCfg(
    func=both_hands_grip_bar_reward,
    params={"command_name": "motion","grip_offset": 0.25, "std": 0.15},
    weight=1.5,
  )
  # True contact only: delayed to graping window (160+) and weight 5/8 -> 2.
  # v0.2.13: 2.0 -> 3.0
  cfg.rewards["fingers_barbell_contact"] = RewardTermCfg(
    func=fingers_barbell_contact_reward,
    weight=3.0,
    params={
      "command_name": "motion",
      "left_sensor_name": "left_finger_barbell_contact",
      "right_sensor_name": "right_finger_barbell_contact",
      "start_frame": 160,
      "end_frame": 528,
    },
  )
  # Soft under-contact penalty (was -2/-4); avoid force-reach shortcuts.
  # v0.2.13: -0.5 -> -0.8
  cfg.rewards["fingers_barbell_contact_penalty"] = RewardTermCfg(
    func=fingers_barbell_contact_penalty,
    weight=-0.0,
    params={
      "command_name": "motion",
      "left_sensor_name": "left_finger_barbell_contact",
      "right_sensor_name": "right_finger_barbell_contact",
      "start_frame": 160,
      "end_frame": 528,
      "target_contact_ratio": 0.35, # v0.2.13: 0.25 -> 0.35
    },
  )
  # Prefer demo finger DOF over geometric hover grasp.
  # v0.2.13: -1.0 -> -1.5
  cfg.rewards["finger_dof_tracking_penalty"] = RewardTermCfg(
    func=finger_dof_tracking_error,
    weight=-1.5,
    params={"command_name": "motion"},
  )
  # OFF: fingers_grip_bar — axis proximity hover-snipe drove bend-to-reach.
  cfg.rewards["fingers_grip_bar"] = RewardTermCfg(
    func=fingers_grip_bar_reward,
    weight=2.0,
    params={"command_name": "motion", "start_frame": 130, "std": 0.05},
  )
  cfg.rewards["lower_body_tracking"] = RewardTermCfg(
      func=lower_body_dof_tracking_error_no_ankle,
      weight=-1.0,
      params={
          "command_name": "motion",
          "start_frame": 150,
      },
  )
  cfg.rewards["lower_body_tracking_after_lift"] = RewardTermCfg(
      func=lower_body_dof_tracking_error,
      weight=-5.0,
      params={
          "command_name": "motion",
          "start_frame": 250,
      },
  )
  # OFF: hands_barbell_height_error — worst-hand height penalty caused torso dive.
  cfg.rewards["hands_barbell_height_error"] = RewardTermCfg(
    func=hands_barbell_height_error,
    weight=-5.0,
    params={"command_name": "motion", "start_frame": 130},
  )
  # Re-enable root Z tracking from frame 0 to stop anchor height loss at grasp.
  cfg.rewards["root_z_tracking"] = RewardTermCfg(
    func=root_z_tracking_error,
    weight=-5.0,
    params={"command_name": "motion", "start_frame": 270},
  )
  cfg.rewards["fingers_grap"] = RewardTermCfg(
    func=fingers_grap,
    params={"command_name": "motion"},
    weight=-2.0,
  )
  cfg.rewards["knee_dof_tracking"] = RewardTermCfg(
      func=knee_dof_tracking_error,
      weight=-1.0,
      params={
          "command_name": "motion",
      },
  )
  # cfg.rewards["knee_barbell_contact_penalty"] = RewardTermCfg(
  #     func=knee_barbell_contact_penalty,
  #     weight=-1.0,
  #     params={
  #         "sensor_name": "knee_barbell_contact",
  #     },
  # )  
  # cfg.rewards["fingers_opposition_grasp_penalty"] = RewardTermCfg(
  #     func=fingers_opposition_grasp_penalty,
  #     weight=-3.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 150,
  #         "end_frame": 528,
  #         "target_cos": -0.5,
  #     },
  # )
  # cfg.rewards["fingers_opposition_grasp"] = RewardTermCfg(
  #     func=fingers_opposition_grasp_reward,
  #     weight=3.0,
  #     params={
  #         "command_name": "motion",
  #         "start_frame": 150,
  #         "end_frame": 528,
  #     },
  # )
  # cfg.terminations["knee_barbell_contact"] = TerminationTermCfg(
  #     func=knee_barbell_contact_termination,
  #     params={
  #         "sensor_name": "knee_barbell_contact",
  #     },
  # )
  # --- Terminations ---
  # v0.2.11: limit 0.1->0.12 — slightly more slack once hands reach the bar
  cfg.terminations["barbell_droped_from_hands"] = TerminationTermCfg(
    func=barbell_droped_from_hands,
    params={"command_name": "motion", "limit_distant": 0.12},
  )
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
    func=far_away_root_pos,
    params={"limit_distant": 0.6},
  )
  # cfg.terminations["shin_barbell_contact"] = TerminationTermCfg(
  #   func=shin_barbell_contact_termination,
  #   params={"sensor_name": "shin_barbell_contact"},
  # )
  # cfg.terminations["hands_not_on_the_bar"] = TerminationTermCfg(
  #   func=not_on_the_bar,
  #   params={
  #     "command_name": "motion",
  #     "threshold": 0.9961946980917455,
  #   },
  # )
  return cfg


from src.sensors.advCamera import deploy_g1_depth_camera_to
def unitree_g1_pull_up_distill(play = False):
  cfg = unitree_g1_pull_up_barbell_privil_env_cfg(play)
  deploy_g1_depth_camera_to(cfg)
  cfg.observations['actor'] = unitree_g1_flat_tracking_env_cfg().observations['actor']
  return cfg