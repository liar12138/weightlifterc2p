"""Unitree G1 + LinkerHand O6 (51-DOF) constants.

29 body DOFs + 22 independently actuated hand DOFs (11 per hand, no mimic).
Hand PD matches prior O6/Revo2 placeholders; effort placeholder 2.0 Nm.

Adapted for capylab's installed mjlab (BuiltinPositionActuatorCfg, no update_assets).
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

_ROBOT_ROOT = Path(__file__).resolve().parent
G1_O6_XML = _ROBOT_ROOT / "xmls" / "g1_o6.xml"
assert G1_O6_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(G1_O6_XML))


def get_checkhand_viewer_spec() -> mujoco.MjSpec:
  """Pinned floating base for upright O6 hand GUI checks."""
  spec = get_spec()
  for jnt in list(spec.joints):
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      spec.delete(jnt)
      break
  return spec


##
# Actuator config.
##

ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
)

G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
)

# O6 hand: same PD as Revo2/Dex3; effort 2.0 Nm placeholder.
G1_ACTUATOR_O6 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_thumb_cmc_yaw",
    ".*_thumb_cmc_pitch",
    ".*_thumb_ip",
    ".*_index_mcp_pitch",
    ".*_index_dip",
    ".*_middle_mcp_pitch",
    ".*_middle_dip",
    ".*_ring_mcp_pitch",
    ".*_ring_dip",
    ".*_pinky_mcp_pitch",
    ".*_pinky_dip",
  ),
  effort_limit=2.0,
  armature=1e-2,
  stiffness=30.0,
  damping=1.0,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_(thumb|index|middle|ring|pinky)_.*": 0.0,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    ".*_(thumb|index|middle|ring|pinky)_.*": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_COLLISION_PATTERN = r"^(left|right)_foot[1-4]_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={FOOT_COLLISION_PATTERN: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_PATTERN: 1},
  friction={FOOT_COLLISION_PATTERN: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={FOOT_COLLISION_PATTERN: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_PATTERN: 1},
  friction={FOOT_COLLISION_PATTERN: (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(FOOT_COLLISION_PATTERN,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_O6_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_O6,
  ),
  soft_joint_pos_limit_factor=0.9,
)

G1_O6_ROBOT_CFG = EntityCfg(
  init_state=KNEES_BENT_KEYFRAME,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=G1_O6_ARTICULATION,
)

G1_O6_VIEWER_CFG = EntityCfg(
  init_state=KNEES_BENT_KEYFRAME,
  collisions=(),
  spec_fn=get_checkhand_viewer_spec,
  articulation=G1_O6_ARTICULATION,
)


def get_g1_o6_robot_cfg() -> EntityCfg:
  """Get a fresh G1+O6 (51-DOF) robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_O6_ARTICULATION,
  )


G1_O6_ACTION_SCALE: dict[str, float] = {}
for a in G1_O6_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    G1_O6_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(G1_O6_VIEWER_CFG)
  mj_model = robot.spec.compile()
  print(f"njnt={mj_model.njnt}, nu={mj_model.nu}, nbody={mj_model.nbody}")
  print(f"nq={mj_model.nq}, nv={mj_model.nv}")
  hand_joints = [
    j
    for j in range(mj_model.njnt)
    if any(
      k in mj_model.joint(j).name
      for k in ("thumb", "index", "middle", "ring", "pinky")
    )
  ]
  print(f"hand_joint_count={len(hand_joints)} (expect 22)")
  print(f"neq={mj_model.neq} (expect 0)")

  viewer.launch(mj_model)
