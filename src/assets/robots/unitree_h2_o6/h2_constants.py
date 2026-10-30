"""Unitree H2 (31-DOF, 无手) constants.

31 body DOFs: L-leg(6), R-leg(6), waist(3), L-arm(7), R-arm(7), head(2).
Action 顺序 = XML 关节定义顺序，head 位于腰部之后、手臂之前。

Adapted for capylab's installed mjlab (BuiltinPositionActuatorCfg, no update_assets).
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

_ROBOT_ROOT = Path(__file__).resolve().parent
H2_XML = _ROBOT_ROOT / "xmls" / "h2.xml"
assert H2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(H2_XML))


##
# Actuator config.
##

from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)

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

# ── 腿部 ──
# hip_yaw 用 7520_14；hip_roll/hip_pitch/knee 用 7520_22
H2_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw.*",),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  # 与 XML actuatorfrcrange ±360 一致
  effort_limit=360.0,
  armature=ARMATURE_7520_14,
)
H2_ACTUATOR_HIP_ROLL_PITCH_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll.*",
    ".*_hip_pitch.*",
    ".*_knee.*",
  ),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  # 与 XML actuatorfrcrange ±360 一致
  effort_limit=360.0,
  armature=ARMATURE_7520_22,
)

# ── 脚踝: 双电机并联 (2×5020) ──
H2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll.*",),
  stiffness=2.0 * STIFFNESS_5020,
  damping=2.0 * DAMPING_5020,
  # 与 XML actuatorfrcrange ±19 一致
  effort_limit=19.0,
  armature=2.0 * ARMATURE_5020,
)
H2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch.*",),
  stiffness=2.0 * STIFFNESS_5020,
  damping=2.0 * DAMPING_5020,
  # 与 XML actuatorfrcrange ±66.88 一致
  effort_limit=66.88,
  armature=2.0 * ARMATURE_5020,
)

# ── 腰部: yaw 用 7520_14；pitch/roll 双电机 (2×5020) ──
H2_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw.*",),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  # 与 XML actuatorfrcrange ±120 一致
  effort_limit=120.0,
  armature=ARMATURE_7520_14,
)
H2_ACTUATOR_WAIST_RP = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_roll.*", "waist_pitch.*"),
  stiffness=2.0 * STIFFNESS_5020,
  damping=2.0 * DAMPING_5020,
  # 与 XML actuatorfrcrange ±180 一致
  effort_limit=180.0,
  armature=2.0 * ARMATURE_5020,
)

# ── 头部: 双电机 (2×5020) ──
H2_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("head_pitch.*", "head_yaw.*"),
  stiffness=2.0 * STIFFNESS_5020,
  damping=2.0 * DAMPING_5020,
  # 与 XML actuatorfrcrange ±50 一致
  effort_limit=50.0,
  armature=2.0 * ARMATURE_5020,
)

# ── 肩 pitch: 5020（actuatorfrcrange ±120）──
H2_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch.*",),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=120.0,
  armature=ARMATURE_5020,
)

# ── 手臂 (肩roll/yaw/肘/腕roll): 5020（actuatorfrcrange ±54）──
H2_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_roll.*",
    ".*_shoulder_yaw.*",
    ".*_elbow.*",
    ".*_wrist_roll.*",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=54.0,
  armature=ARMATURE_5020,
)

# ── 手腕 pitch/yaw: 4010 ──
H2_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch.*", ".*_wrist_yaw.*"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  # 与 XML actuatorfrcrange ±25 一致
  effort_limit=25.0,
  armature=ARMATURE_4010,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.00),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

# KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
#   pos=(0, 0, 1.04),
#   joint_pos={
#     ".*_hip_pitch_joint": -0.312,
#     ".*_knee_joint": 0.669,
#     ".*_ankle_pitch_joint": -0.363,
#     ".*_elbow_joint": 0.6,
#     "left_shoulder_roll_joint": 0.2,
#     "left_shoulder_pitch_joint": 0.2,
#     "right_shoulder_roll_joint": -0.2,
#     "right_shoulder_pitch_joint": 0.2,
#   },
#   joint_vel={".*": 0.0},
# )

##
# Collision config.
##

# Legacy single-point pattern retained for reference.
# FOOT_COLLISION_PATTERN = r"^(left|right)_foot_collision$"
FOOT_COLLISION_PATTERN = r"^(left|right)_foot[1-7]_collision$"

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

H2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H2_ACTUATOR_HIP_YAW,
    H2_ACTUATOR_HIP_ROLL_PITCH_KNEE,
    H2_ACTUATOR_ANKLE_ROLL,
    H2_ACTUATOR_ANKLE_PITCH,
    H2_ACTUATOR_WAIST_YAW,
    H2_ACTUATOR_WAIST_RP,
    H2_ACTUATOR_HEAD,
    H2_ACTUATOR_SHOULDER_PITCH,
    H2_ACTUATOR_ARM,
    H2_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)

H2_ROBOT_CFG = EntityCfg(
  init_state=HOME_KEYFRAME,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=H2_ARTICULATION,
)

H2_VIEWER_CFG = EntityCfg(
  init_state=HOME_KEYFRAME,
  collisions=(),
  spec_fn=get_spec,
  articulation=H2_ARTICULATION,
)


def get_h2_robot_cfg() -> EntityCfg:
  """Get a fresh H2 (31-DOF, 无手) robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H2_ARTICULATION,
  )


H2_ACTION_SCALE: dict[str, float] = {}
for a in H2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    H2_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(H2_VIEWER_CFG)
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
