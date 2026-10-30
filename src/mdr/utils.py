import math

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.managers.scene_entity_config import SceneEntityCfg

def dr_to(cfg):
    cfg.observations['actor'].terms['base_ang_vel'].noise = Gnoise(std = 0.23)
    if 'projected_gravity' in cfg.observations['actor'].terms:
        cfg.observations['actor'].terms['projected_gravity'].noise = Gnoise(std = 0.06)

    cfg.events[ "randomize_robot_inertia"] = EventTermCfg(
        func=dr.pseudo_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=(".*",),
            ),
            "alpha_range": (
                0.5 * math.log(0.8),
                0.5 * math.log(1.2),
            ),
            "distribution": "uniform",
        },
    )

    cfg.events["randomize_pd_gains"] = EventTermCfg(
        func=dr.pd_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),          # 全部执行器
            "kp_range": (0.9, 1.1), "kd_range": (0.8, 1.2),
            "operation": "scale", "distribution": "uniform",
        },
    )

    cfg.events["randomize_armature"] = EventTermCfg(
        func=dr.dof_armature, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "ranges": (0.8, 1.2),
                "operation": "scale", "distribution": "uniform"},
    )
    cfg.events["randomize_joint_damping"] = EventTermCfg(
        func=dr.dof_damping, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "ranges": (0.7, 1.3),
                "operation": "scale", "distribution": "uniform"},
    )

    cfg.events["foot_friction"].params["ranges"] = (0.3, 1.5)   # 或 (0.25, 1.8)
    cfg.events["foot_friction"].params["shared_random"] = False # 左右脚独立摩擦


    # cfg.events["reset_robot_joints"].params["position_range"] = (-0.15, 0.15)
    # cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.2, 0.2)
    # cfg.events["reset_base"].params["velocity_range"] = {"x":(-0.3,0.3), "y":(-0.2,0.2),
    #                                                     "z":(-0.1,0.1), "roll":(-0.2,0.2),
    #                                                     "pitch":(-0.2,0.2), "yaw":(-0.5,0.5)}

    return cfg