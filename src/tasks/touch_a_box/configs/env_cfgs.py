"""Minimal G1+Revo2 env: hands pressed on a fixed box + hand ContactSensor."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ..events import print_hand_tactile
from ..sensors import get_hand_tactile_sensor_cfg

# Import local Revo2 asset without editing existing packages.
_ROBOTS_ROOT = Path(__file__).resolve().parents[3] / "assets" / "robots"
if str(_ROBOTS_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROBOTS_ROOT))

from unitree_g1_revo2.g1_constants import (  # noqa: E402
  G1_REVO2_ACTION_SCALE,
  get_g1_revo2_robot_cfg,
)

# Pose: standing, arms reaching forward (+x, robot face direction) onto the box.
# Note: too large shoulder_pitch folds the arms back behind the torso.
_TOUCH_POSE_JOINT_POS = {
  ".*_hip_pitch_joint": -0.312,
  ".*_knee_joint": 0.669,
  ".*_ankle_pitch_joint": -0.363,
  ".*_shoulder_pitch_joint": 0.2,
  "left_shoulder_roll_joint": 0.15,
  "right_shoulder_roll_joint": -0.15,
  ".*_shoulder_yaw_joint": 0.0,
  ".*_elbow_joint": -2.0,
  ".*_wrist_roll_joint": 0.0,
  ".*_wrist_pitch_joint": 0.0,
  ".*_wrist_yaw_joint": 0.0,
  ".*_thumb_metacarpal_joint": 0.6,
  ".*_thumb_proximal_joint": 0.55,
  ".*_thumb_distal_joint": 0.55,
  ".*_(index|middle|ring|pinky)_proximal_joint": 0.9,
  ".*_(index|middle|ring|pinky)_distal_joint": 0.85,
}

# Box in front of the robot (default facing = +x).
_BOX_POS = (0.6, 0.0, 0.45)
_BOX_SIZE = (0.18, 0.28, 0.225)  # half-sizes; top ≈ z=0.94 under forward hands


def get_fixed_box_spec():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box", pos=[0.0, 0.0, 0.0])
  body.add_geom(
    name="box_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(_BOX_SIZE),
    density=1000.0,
    rgba=[0.85, 0.35, 0.2, 1.0],
    friction=[1.0, 0.005, 0.0001],
  )
  return spec


def get_fixed_box_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=_BOX_POS),
    spec_fn=get_fixed_box_spec,
  )


def get_touch_robot_cfg() -> EntityCfg:
  """Fresh Revo2 cfg with a touch-oriented default pose (does not mutate shared CFG)."""
  cfg = get_g1_revo2_robot_cfg()
  cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.76),
    joint_pos=dict(_TOUCH_POSE_JOINT_POS),
    joint_vel={".*": 0.0},
  )
  return cfg


def unitree_g1_revo2_touch_a_box_env_cfg(play: bool = True) -> ManagerBasedRlEnvCfg:
  """Play-oriented env: stand still, hands on box, print hand_tactile periodically."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Revo2 has more DOFs / contacts than stock G1.
  cfg.sim.njmax = 600
  cfg.sim.nconmax = 200
  cfg.sim.contact_sensor_maxmatch = 256

  cfg.scene.entities["robot"] = get_touch_robot_cfg()
  cfg.scene.entities["box"] = get_fixed_box_cfg()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_REVO2_ACTION_SCALE

  # Stand still so contact is stable while inspecting in viser.
  # Keep non-zero ranges so Viser velocity GUI sliders (min=0.1) can initialize.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_standing_envs = 1.0

  # Drop G1-foot sensors/rewards that reference missing sites (Revo2 has no left_foot site).
  _drop_sensors = {"foot_height_scan", "feet_ground_contact", "terrain_scan"}
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in _drop_sensors
  )
  # Verification task: no need for locomotion rewards (many assume stock-G1 sensors/DoFs).
  cfg.rewards.clear()
  cfg.events.pop("foot_friction", None)
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "self_collision"
  )

  # Revo2 XML IMUs are named differently from stock G1; use entity root velocities.
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    terms["base_lin_vel"] = ObservationTermCfg(func=envs_mdp.base_lin_vel)
    terms["base_ang_vel"] = ObservationTermCfg(func=envs_mdp.base_ang_vel)
  for key in ("foot_height", "foot_air_time", "foot_contact", "foot_contact_forces"):
    cfg.observations["critic"].terms.pop(key, None)

  hand_tactile: ContactSensorCfg = get_hand_tactile_sensor_cfg(debug=False)
  cfg.scene.sensors = (hand_tactile,)

  # Console readout every ~0.5s (viser has no desktop; print is the primary signal).
  cfg.events["print_hand_tactile"] = EventTermCfg(
    func=print_hand_tactile,
    mode="interval",
    interval_range_s=(0.5, 0.5),
    params={"sensor_name": "hand_tactile", "force_eps": 1e-3},
  )

  # Keep pose stable for sensor check (no base randomize / DR).
  for key in (
    "reset_base",
    "randomize_terrain",
    "encoder_bias",
    "base_com",
    "push_robot",
  ):
    cfg.events.pop(key, None)

  # Keep episode alive for interactive viser tweaking.
  cfg.episode_length_s = int(1e9)
  cfg.terminations = {}

  cfg.viewer.body_name = "torso_link"
  return cfg
