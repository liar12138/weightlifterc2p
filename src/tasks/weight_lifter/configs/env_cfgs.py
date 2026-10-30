"""Unitree G1 velocity environment configurations."""

from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from dataclasses import asdict
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from mjlab.sensor.contact_sensor import *
from mjlab.managers.event_manager import EventTermCfg

from ..rewards import *
from ..terminations import *
from ..observation import *
from ..commands import *

from src.assets.robots.unitree_g1_dex3.g1_constants import get_g1_dex3_robot_cfg
from src.assets.objects import get_barbell_cfg, MEN_COMPETITION_BAR_MASS

from src.mdr.utils import dr_to
from src.sensors.g1.camera import deploy_depth_camera_to

def unitree_g1_dex3_flat_tracking(play: bool = False):
  import mjlab.envs.mdp.dr as dr
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)
  cfg.scene.entities["robot"] = get_g1_dex3_robot_cfg()

  cfg.events["randomize_hand_payload"] = EventTermCfg(
    func=dr.body_mass,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=["right_hand_index_1_link", "left_hand_index_1_link"],
        ),
        "operation": dr.add,
        "distribution": dr.uniform,
        "ranges": (0.0, 5.0),
    },
  )

  if play:
    cfg.scene.entities["barbell"] = get_barbell_cfg()
    deploy_depth_camera_to(cfg)

  return dr_to(cfg)

def unitree_g1_pull_up_barbell_env_cfg(play : bool = False):
    
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)
  # cfg = dr_to(cfg)
  cfg = deploy_depth_camera_to(cfg)

  cfg.seed = 46
  cfg.sim.nconmax = 60
  cfg.scene.entities["robot"]   = get_g1_dex3_robot_cfg()
  # 国际标准比赛杆：整杆 20 kg，单边片重从标准片里选（此处 2.5 kg → 总重 25 kg）。
  # 常用配片：get_barbell_cfg(plate_mass=10.0, bar_mass=MEN_COMPETITION_BAR_MASS)
  cfg.scene.entities["barbell"] = get_barbell_cfg(
      plate_mass=0.5,
      bar_mass=0.5,
  )

  motioncfg = MotionBarbellCommandCfg(**asdict(cfg.commands["motion"]))
  motioncfg.viz = cfg.commands["motion"].viz
  motioncfg.motion_file = "src/assets/motion/g1/1_g1dex3_pro.npz"
  cfg.commands["motion"] = motioncfg
  cfg.commands["motion"].sampling_mode = "uniform"

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

  cfg.rewards["reward_barbell_z"] = RewardTermCfg(
    func=barbell_reward_hands_height,
    weight=1.0,
  )
  
  # cfg.rewards["fingers_grap"] = RewardTermCfg(
  #   func=fingers_grap,
  #   params={"command_name":"motion"},
  #   weight=-1.0,
  # )
  cfg.rewards["pently_hands_parrel"] = RewardTermCfg(
      func=pently_hands_parrel,
      params={"command_name":"motion"},
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

  # cfg.terminations["barbell_droped_from_hands"] = TerminationTermCfg(
  #   func=barbell_droped_from_hands,
  #   params={"command_name": "motion", "limit_distant": 0.1},
  # )
  cfg.terminations["far_away_from_root"] = TerminationTermCfg(
    func=far_away_root_pos,
    params={"limit_distant": 0.42},
  )
  cfg.terminations["hands_not_on_the_bar"] = TerminationTermCfg(
    func=not_on_the_bar,
    params={"command_name": "motion", "threshold": math.cos(5 / 180 * torch.pi)},
  )

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
      "time_out" : cfg.terminations["time_out"],
      # "hands_not_on_the_bar" : cfg.terminations["hands_not_on_the_bar"],
    }


  return cfg