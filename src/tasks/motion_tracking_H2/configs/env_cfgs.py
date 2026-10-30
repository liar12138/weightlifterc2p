"""Unitree H2 + O6 (53-DOF) motion tracking environment configuration.

Pure motion tracking task (no barbell object). Inherits the G1 flat-terrain
tracking base with state estimation disabled, then swaps in the H2+O6 robot.
"""

from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.entity import Entity

from src.assets.robots.unitree_h2_o6.h2_o6_constants import (
  FOOT_COLLISION_PATTERN,
  H2_O6_ACTION_SCALE,
  get_h2_o6_robot_cfg,
)


def unitree_h2_o6_motion_tracking_env_cfg(play: bool = False):
  """Create the H2+O6 motion tracking env cfg (no state estimation).

  Args:
    play: If True, apply play-mode overrides (infinite episode, "start"
      sampling, no corruption), which the G1 base layer already handles.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  cfg.scene.entities["robot"] = get_h2_o6_robot_cfg()

  # H2+O6 has more bodies/joints than G1; give MuJoCo constraint buffers
  # enough headroom (njmax is a hard limit — overflow drops constraints).
  # cfg.sim.nconmax = 42
  # cfg.sim.njmax = 1000

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H2_O6_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = "src/assets/motion/h2o6/13_h2o6_pro.npz"
  motion_cmd.anchor_body_name = "torso_link"
  # Track every H2 body (including hands); the motion file was recorded with
  # the same H2+O6 spec, so body names line up with Entity.body_names.
  motion_cmd.body_names = tuple(Entity(get_h2_o6_robot_cfg()).body_names)
  # sampling_mode stays "adaptive" (base default) for training; play mode
  # is forced to "start" by the base cfg.

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  return cfg

def unitree_h2_o6_motion_tracking_env_cfg_hand_mass(play = False):

  cfg = unitree_h2_o6_motion_tracking_env_cfg(play=play)

  import mjlab.envs.mdp.dr as dr
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  cfg.events["randomize_hand_payload"] = EventTermCfg(
    func=dr.body_mass,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=["left_index_proximal", "right_index_proximal"],
        ),
        "operation": dr.add,
        "distribution": dr.uniform,
        "ranges": (0.0, 3.0),
    },
  )

  return cfg

