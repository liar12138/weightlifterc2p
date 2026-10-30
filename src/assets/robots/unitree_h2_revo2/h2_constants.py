"""Unitree H2 + BrainCo Revo2 (53-DOF) constants.

31 body DOFs + 22 independently actuated hand DOFs (11 per hand, no mimic).
Body armature values are effort-tier placeholders (no official rotor data yet).
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
H2_REVO2_XML = _ROBOT_ROOT / "xmls" / "h2_revo2.xml"
assert H2_REVO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(H2_REVO2_XML))


def get_checkhand_viewer_spec() -> mujoco.MjSpec:
  """Pinned floating base for upright Revo2 hand GUI checks."""
  spec = get_spec()
  for jnt in list(spec.joints):
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      spec.delete(jnt)
      break
  return spec


##
# Actuator config.
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0


def _pd(armature: float) -> tuple[float, float]:
  stiffness = armature * NATURAL_FREQ**2
  damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
  return stiffness, damping


def _body_actuator(
  target_names_expr: tuple[str, ...],
  *,
  effort_limit: float,
  armature: float,
) -> BuiltinPositionActuatorCfg:
  stiffness, damping = _pd(armature)
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    effort_limit=effort_limit,
    armature=armature,
    stiffness=stiffness,
    damping=damping,
  )


# Placeholder armature by H2.urdf effort tier.
H2_ACTUATOR_HIP_KNEE = _body_actuator(
  (
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
  ),
  effort_limit=360.0,
  armature=0.05,
)
H2_ACTUATOR_ANKLE_PITCH = _body_actuator(
  (".*_ankle_pitch_joint",),
  effort_limit=66.88,
  armature=0.02,
)
H2_ACTUATOR_ANKLE_ROLL = _body_actuator(
  (".*_ankle_roll_joint",),
  effort_limit=19.0,
  armature=0.01,
)
H2_ACTUATOR_WAIST_YAW = _body_actuator(
  ("waist_yaw_joint",),
  effort_limit=120.0,
  armature=0.03,
)
H2_ACTUATOR_WAIST_RP = _body_actuator(
  ("waist_pitch_joint", "waist_roll_joint"),
  effort_limit=180.0,
  armature=0.04,
)
H2_ACTUATOR_HEAD = _body_actuator(
  ("head_pitch_joint", "head_yaw_joint"),
  effort_limit=50.0,
  armature=0.015,
)
H2_ACTUATOR_SHOULDER_PITCH = _body_actuator(
  (".*_shoulder_pitch_joint",),
  effort_limit=130.0,
  armature=0.03,
)
H2_ACTUATOR_ARM = _body_actuator(
  (
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
  ),
  effort_limit=60.0,
  armature=0.02,
)
H2_ACTUATOR_WRIST = _body_actuator(
  (".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  effort_limit=10.0,
  armature=0.005,
)

# Revo2 hand: same PD / effort as G1+Revo2.
H2_ACTUATOR_REVO2_THUMB_MCP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_thumb_metacarpal_joint",),
  effort_limit=0.5,
  armature=1e-2,
  stiffness=30.0,
  damping=1.0,
)
H2_ACTUATOR_REVO2_THUMB = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_thumb_proximal_joint", ".*_thumb_distal_joint"),
  effort_limit=1.1,
  armature=1e-2,
  stiffness=30.0,
  damping=1.0,
)
H2_ACTUATOR_REVO2_FINGERS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_index_proximal_joint",
    ".*_index_distal_joint",
    ".*_middle_proximal_joint",
    ".*_middle_distal_joint",
    ".*_ring_proximal_joint",
    ".*_ring_distal_joint",
    ".*_pinky_proximal_joint",
    ".*_pinky_distal_joint",
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
  pos=(0, 0, 1.2),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 0.8,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_(thumb|index|middle|ring|pinky)_.*_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.17),
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    ".*_knee_joint": 0.5,
    ".*_ankle_pitch_joint": -0.25,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    ".*_(thumb|index|middle|ring|pinky)_.*_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_COLLISION_PATTERN = r"^(left|right)_ankle_pitch_link_collision$"

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

H2_REVO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H2_ACTUATOR_HIP_KNEE,
    H2_ACTUATOR_ANKLE_PITCH,
    H2_ACTUATOR_ANKLE_ROLL,
    H2_ACTUATOR_WAIST_YAW,
    H2_ACTUATOR_WAIST_RP,
    H2_ACTUATOR_HEAD,
    H2_ACTUATOR_SHOULDER_PITCH,
    H2_ACTUATOR_ARM,
    H2_ACTUATOR_WRIST,
    H2_ACTUATOR_REVO2_THUMB_MCP,
    H2_ACTUATOR_REVO2_THUMB,
    H2_ACTUATOR_REVO2_FINGERS,
  ),
  soft_joint_pos_limit_factor=0.9,
)

H2_REVO2_ROBOT_CFG = EntityCfg(
  init_state=KNEES_BENT_KEYFRAME,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=H2_REVO2_ARTICULATION,
)

H2_REVO2_VIEWER_CFG = EntityCfg(
  init_state=KNEES_BENT_KEYFRAME,
  collisions=(),
  spec_fn=get_checkhand_viewer_spec,
  articulation=H2_REVO2_ARTICULATION,
)


def get_h2_revo2_robot_cfg() -> EntityCfg:
  """Get a fresh H2+Revo2 (53-DOF) robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H2_REVO2_ARTICULATION,
  )


H2_REVO2_ACTION_SCALE: dict[str, float] = {}
for a in H2_REVO2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    H2_REVO2_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(H2_REVO2_VIEWER_CFG)
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
  print(f"nefc_equality={mj_model.neq} (expect 0)")

  viewer.launch(mj_model)
