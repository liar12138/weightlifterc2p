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

# ── O6 手部执行器：统一 kp/kd，1 秒内从张开到位 ──
# 设计: t_95% = 11.46/ω_n = 1.0s → ω_n=11.46 rad/s
#       kp = ω_n² × armature = 131.3 × 0.01 = 1.313
#       取 kp=1.25 (finger_mcp 的 τ_max=kp×1.60=2.0Nm, 恰好等于 effort_limit)
#       kd = 4×√(0.01×kp) = 0.447 (ζ=2.0)
#       ω_n = √(1.25/0.01) = 11.18 rad/s = 1.78 Hz
#       t_95% = 11.46/11.18 = 1.025s
G1_ACTUATOR_O6 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_thumb_cmc_yaw", ".*_thumb_cmc_pitch", ".*_thumb_ip",
    ".*_index_mcp_pitch", ".*_index_dip",
    ".*_middle_mcp_pitch", ".*_middle_dip",
    ".*_ring_mcp_pitch", ".*_ring_dip",
    ".*_pinky_mcp_pitch", ".*_pinky_dip",
  ),
  effort_limit=2.0,
  armature=1e-2,
  stiffness=1.25,    # 统一 kp
  damping=0.4472,    # ζ=2.0
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

# KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
#   pos=(0, 0, 0.76),
#   joint_pos={
#     ".*_hip_pitch_joint": -0.312,
#     ".*_knee_joint": 0.669,
#     ".*_ankle_pitch_joint": -0.363,
#     ".*_elbow_joint": 0.6,
#     "left_shoulder_roll_joint": 0.2,
#     "left_shoulder_pitch_joint": 0.2,
#     "right_shoulder_roll_joint": -0.2,
#     "right_shoulder_pitch_joint": 0.2,
#     ".*_(thumb|index|middle|ring|pinky)_.*": 0.0,
#   },
#   joint_vel={".*": 0.0},
# )

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
  init_state=HOME_KEYFRAME,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=G1_O6_ARTICULATION,
)

G1_O6_VIEWER_CFG = EntityCfg(
  init_state=HOME_KEYFRAME,
  collisions=(),
  spec_fn=get_checkhand_viewer_spec,
  articulation=G1_O6_ARTICULATION,
)


def get_g1_o6_robot_cfg() -> EntityCfg:
  """Get a fresh G1+O6 (51-DOF) robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
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

# ── 手部 scale 覆写：每关节 scale = XML max_range ──
# action=+1 → q_target = max_range (手指完全弯曲)
# 关节范围: thumb_cmc_pitch=0.58, thumb_ip=1.08, thumb_cmc_yaw=1.30,
#            finger_dip=1.43, finger_mcp=1.60
_HAND_SCALE_OVERRIDE: dict[str, float] = {
    ".*_thumb_cmc_pitch": 0.58,
    ".*_thumb_ip":        1.08,
    ".*_thumb_cmc_yaw":   1.30,
    ".*_index_dip":       1.43, ".*_middle_dip": 1.43,
    ".*_ring_dip":        1.43, ".*_pinky_dip":  1.43,
    ".*_index_mcp_pitch": 1.60, ".*_middle_mcp_pitch": 1.60,
    ".*_ring_mcp_pitch":  1.60, ".*_pinky_mcp_pitch":  1.60,
}
for pattern, val in _HAND_SCALE_OVERRIDE.items():
    for n in G1_O6_ACTION_SCALE:
        import re
        if re.match(pattern, n):
            G1_O6_ACTION_SCALE[n] = val

if __name__ == "__main__":
  import mujoco
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(G1_O6_VIEWER_CFG)
  mj_model = robot.spec.compile()
  print(f"njnt={mj_model.njnt}, nu={mj_model.nu}, nbody={mj_model.nbody}")
  print(f"nq={mj_model.nq}, nv={mj_model.nv}")

  # 重力置零（无重力仿真）。
  mj_model.opt.gravity[:] = 0.0

  # 以 init_state keyframe（HOME_KEYFRAME）作为初始状态，
  # 使直接运行时的 viewer 呈现与 play 模式一致。
  mj_data = mujoco.MjData(mj_model)
  key = mj_model.key("init_state")
  mj_data.qpos[:] = key.qpos
  mj_data.qvel[:] = key.qvel
  if key.ctrl.size:
    mj_data.ctrl[:] = key.ctrl
  mujoco.mj_forward(mj_model, mj_data)

  viewer.launch(mj_model, mj_data)
