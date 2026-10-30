import math

import torch

from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg, unitree_g1_rough_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise

from mjlab.terrains.config import *
from dataclasses import replace

from src.assets.robots.unitree_h2_o6.h2_constants import (
    FOOT_COLLISION_PATTERN,
    H2_ACTION_SCALE,
    get_h2_robot_cfg,
)
from src.mdr.utils import dr_to
from src.tasks.velocity_tracking_H2.events import push_when_standing

_HAND_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")

H2_BODY_JOINT_PATTERNS = (
    r".*_hip_pitch_joint",
    r".*_hip_roll_joint",
    r".*_hip_yaw_joint",
    r".*_knee_joint",
    r".*_ankle_pitch_joint",
    r".*_ankle_roll_joint",

    r"waist_yaw_joint",
    r"waist_roll_joint",
    r"waist_pitch_joint",

    r"head_pitch_joint",
    r"head_yaw_joint",

    r".*_shoulder_pitch_joint",
    r".*_shoulder_roll_joint",
    r".*_shoulder_yaw_joint",
    r".*_elbow_joint",
    r".*_wrist_.*joint",
)

from ..rewards import *
from ..terminations import pack_timer_safe

def unitree_h2_env_cfg(cfg):
    cfg.scene.entities["robot"] = get_h2_robot_cfg()
    cfg.actions["joint_pos"].scale = H2_ACTION_SCALE
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_PATTERN
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    return cfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "random_rough": random_rough(proportion=0.1, horizontal_scale=0.2),
    "wave_terrain": wave_terrain(proportion=0.1, horizontal_scale=0.2),
  },
  add_lights=True,
)

def unitree_h2_velocity_env_cfg(play: bool = False,apply_dr = False):
    cfg = unitree_g1_flat_env_cfg(play=play)
    cfg = unitree_h2_env_cfg(cfg)
    cfg.rewards["pose"].params["asset_cfg"].joint_names = H2_BODY_JOINT_PATTERNS
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,

        # Waist
        r".*waist_yaw.*": 0.2,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.1,

        # Head
        r".*head_pitch.*": 0.08,
        r".*head_yaw.*": 0.08,

        # Arms
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.15,
        r".*wrist.*": 0.3,
    }

    cfg.rewards["pose"].params["std_running"] = {
        # Lower body
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.35,
        r".*ankle_roll.*": 0.15,

        # Waist
        r".*waist_yaw.*": 0.3,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.2,

        # Head
        r".*head_pitch.*": 0.10,
        r".*head_yaw.*": 0.10,

        # Arms
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
        r".*shoulder_yaw.*": 0.15,
        r".*elbow.*": 0.35,
        r".*wrist.*": 0.3,
    }
    cfg.rewards["pose"].func = PakPose
    cfg.rewards["pose"].weight = 1
    cfg.rewards["soft_landing"].weight = -1e-2
    cfg.rewards["action_rate_l2"].weight = 2 * cfg.rewards["action_rate_l2"].weight
    # cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    #   func= track_angular_velocity,
    #   weight=1.0,
    #   params={"command_name": "twist", "std": math.sqrt(0.25)},
    # )
    cfg.rewards["track_angular_velocity"].weight = 3

    # cfg.terminations["fell_over"].func = pack_timer_safe(cfg.terminations["fell_over"].func)
    # cfg.commands["twist"].rel_standing_envs = 0.2
    # cfg.commands["twist"].resampling_time_range = (20,40)

    cfg.observations['actor'].terms['base_ang_vel'].noise = Gnoise(std = 0.23)
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

    cfg.rewards["electrical_power_cost"] = RewardTermCfg(
        func=penalty_electrical_power_cost,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "deadband_w": 150.0, "normalization_w": 500.0},
    )

    # _fg = unitree_g1_rough_env_cfg()
    # cfg.scene.terrain = _fg.scene.terrain
    # cfg.scene.terrain.terrain_generator = replace(ROUGH_TERRAINS_CFG)
    # cfg.sim = _fg.sim
    # cfg.sim.mujoco.ccd_iterations = 30  
    # cfg.episode_length_s = 40

    # cfg.observations["actor"].enable_corruption = True

    if not play and apply_dr:
        cfg = dr_to(cfg)
        cfg.events["push_when_standing"] = EventTermCfg(
            func=push_when_standing, mode="interval", interval_range_s=(2.0, 4.0),
            params={"command_name": "twist", "command_threshold": 0.1,
                    "velocity_range": {"x": (-0.8,0.8), "y": (-0.6,0.6), "z": (-0.2,0.2),
                                    "roll": (-0.4,0.4), "pitch": (-0.4,0.4), "yaw": (-0.8,0.8)}},
        )

        # cfg.events["randomize_hand_payload"] = EventTermCfg(
        #     func=dr.body_mass,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot",
        #             body_names=_HAND_BODY_NAMES,
        #         ),
        #         "operation": dr.add,
        #         "distribution": dr.uniform,
        #         "ranges": (0.0, 16.0),
        #     },
        # )
    return cfg


# ============================================================================
# 官方复现版：完全复现宇树官方 mjlab 的 H2 速度追踪配置
# ----------------------------------------------------------------------------
# 基准 = mjlab 官方 release 的 unitree_g1_flat_env_cfg（即宇树官方设计，含
# 官方默认的奖励权重 / 事件 / 观测噪声 / 域随机化 / 速度课程 / PPO）。
#
# 只做官方口径的 H2 适配：换 H2 本体、动作缩放、脚摩擦几何、躯干质心随机化
# （与 unitree_rl_mjlab 仓库 src/tasks/velocity/config/h2/env_cfgs.py 一致）。
#
# 姿态奖励用官方 variable_posture（全关节均值 + 三档 per-joint std），
# **不使用** PakPose。其余奖励 / 事件 / 观测一律保持官方默认，不做任何覆盖。
#
# 相比当前 velocity_tracking_H2，本配置**移除**了以下非官方自定义：
#   - PakPose（min 姿态奖励，官方用 variable_posture）
#   - track_angular_velocity 权重 3（官方 1.0）
#   - action_rate_l2 权重加倍（官方默认）
#   - soft_landing 权重 -1e-2（官方默认 -1e-3）
#   - electrical_power_cost（官方无）
#   - randomize_robot_inertia（官方无）
#   - 自定义 base_ang_vel / projected_gravity 观测噪声（官方用均匀噪声默认）
#   - push_when_standing（官方用 push_robot，对全体 env 每 5-6s 推扰）
# ============================================================================


def stand_still(
    env,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """官方 stand_still：零速指令时把每个关节拉回 default 位姿（防姿态漂移）。

    复刻自 unitree_rl_mjlab src/tasks/velocity/mdp/rewards.py 的 stand_still。
    mjlab 1.5.0 的 mdp 里没有此函数，故在本地按官方语义实现。
    """
    asset = env.scene[asset_cfg.name]
    diff_angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward = reward * scale
    return reward


def feet_gait(
    env,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """官方 feet_gait：按 0.6s 周期约束左右脚触地相位，让步态更规整。

    复刻自 unitree_rl_mjlab src/tasks/velocity/mdp/rewards.py 的 feet_gait。
    mjlab 1.5.0 的 mdp 里没有此函数，故在本地按官方语义实现。
    """
    sensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(
        offset, device=env.device, dtype=global_phase.dtype
    ).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = leg_phase < threshold
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            # 只有线性指令时才施加固定周期步态约束；
            # 纯转向（lin≈0、ω≠0）时放开，避免固定 0.6s 步态压制原地转。
            scale = (linear_norm > command_threshold).float()
            reward = reward * scale
    return reward


def angular_momentum_xy_penalty(env, sensor_name: str) -> torch.Tensor:
    """只惩罚水平（xy）角动量，避免惩罚绕竖直轴（yaw）的旋转。

    官方 mjlab 的 angular_momentum_penalty 用 sum(square(angmom)) 把 z 轴角动量也罚了；
    原地转圈必然产生大竖直角动量，等于奖励在压制"转"。这里对齐社区惯例
    （legged_gym / IsaacLab 的角动量/角速度惩罚只作用 xy）只惩罚 xy 分量。
    """
    angmom = env.scene[sensor_name].data  # [B, 3]
    return torch.sum(torch.square(angmom[:, :2]), dim=1)


def track_angular_velocity_sharp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """更锐利的 yaw 跟踪奖励（对齐 GitHub main 官方口径）。

    相比 mjlab 1.5.0（ang_vel_error = z_error + xy_error，std=√0.5）：
    - xy_error 乘 0.05 衰减：转身时轻微横滚/俯仰不再被重罚；
    - 配合小 std（√0.25）+ 高权重，让"转"的正向收益压过 angular_momentum /
      pose / gait 等反向项，策略才愿意转。
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + 0.05 * xy_error
    return torch.exp(-ang_vel_error / std**2)


def unitree_h2_velocity_env_cfg_official(play: bool = False):
    """完全复现宇树官方 mjlab 的 H2 速度追踪配置（不带任何自定义改动）。"""
    cfg = unitree_g1_flat_env_cfg(play=play)  # 官方 G1 基准（含官方默认一切）
    cfg = unitree_h2_env_cfg(cfg)             # 官方口径的 H2 适配

    # ---- 姿态奖励：官方 variable_posture 的 H2 per-joint std（官方数值）----
    cfg.rewards["pose"].params["asset_cfg"].joint_names = H2_BODY_JOINT_PATTERNS
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.3,  # 放宽：允许原地转需要的髋部偏转
        r".*knee.*": 0.5,
        r".*ankle_roll.*": 0.1,
        r".*ankle_pitch.*": 0.15,
        # Waist
        r".*waist_yaw.*": 0.15,
        r".*waist_roll.*": 0.1,
        r".*waist_pitch.*": 0.1,
        # Head
        r".*head_pitch.*": 0.08,
        r".*head_yaw.*": 0.08,
        # Arms
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.1,
        r".*wrist.*": 0.1,
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Lower body
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.25,
        r".*hip_yaw.*": 0.4,  # 放宽：允许原地转需要的髋部偏转
        r".*knee.*": 0.5,
        r".*ankle_roll.*": 0.1,
        r".*ankle_pitch.*": 0.25,
        # Waist
        r".*waist_yaw.*": 0.25,
        r".*waist_roll.*": 0.1,
        r".*waist_pitch.*": 0.1,
        # Head
        r".*head_pitch.*": 0.10,
        r".*head_yaw.*": 0.10,
        # Arms
        r".*shoulder_pitch.*": 0.25,
        r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.1,
        r".*wrist.*": 0.1,
    }

    # ---- 原地转圈修复（对齐社区标准 legged_gym / IsaacLab） ----
    # 根因：mjlab 的 angular_momentum_penalty 用 sum(square(angmom)) 把绕竖直轴（z）
    # 的角动量也罚了，原地转必然产生大竖直角动量 → 奖励在压制"转"。
    #   1) 直接去掉 angular_momentum（社区人形速度追踪均不用整身角动量惩罚）。
    #   2) track_angular_velocity 加锐加重：std √0.5→√0.25、权重→3、xy_error×0.05。
    #   3) 放宽 pose 的 hip_yaw std，允许原地转需要的髋部偏转。
    cfg.rewards.pop("angular_momentum", None)
    cfg.rewards["track_angular_velocity"].func = track_angular_velocity_sharp
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params = {
        "command_name": "twist",
        "std": math.sqrt(0.25),
    }

    # ---- 对齐社区标准的一批权重修正（legged_gym / IsaacLab / GitHub main） ----
    # action_rate_l2：1.5.0 base 的 -0.1 是官方 main(-0.05) 的 2 倍、legged_gym(-0.01)
    #   的 10 倍 → 策略对指令响应迟缓。降到 -0.05。
    # track_linear_velocity：2.0 → 1.0（社区统一 1.0）。
    # foot_clearance：-2.0 → -1.0（官方 main 为 -1.0，legged_gym -0.5）。
    # air_time：启用 0.25（base 默认关闭=0.0），鼓励正常迈步（若出现跳步可调回 0）。
    # 命令重采样周期拉长到 (5,10)s，指令更稳定 → 步态更稳。
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["track_linear_velocity"].weight = 1.0
    cfg.rewards["foot_clearance"].weight = -1.0
    cfg.rewards["air_time"].weight = 0.25
    cfg.commands["twist"].resampling_time_range = (5.0, 10.0)

    # 其余（body_orientation / body_ang_vel / dof_pos_limits / foot_slip /
    # soft_landing / self_collisions / 事件 / 观测噪声 / 域随机化 / 速度课程 / PPO）
    # 保持 mjlab 官方默认。

    # ---- 最新官方（GitHub main 分支）新增项，现已补齐 ----
    # 1) is_terminated：摔倒巨额惩罚（-200），让"不摔倒"成为最高优先级。
    cfg.rewards["is_terminated"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-200.0,
    )
    # 2) stand_still：零速指令时把每个关节拉回 default 位姿（防姿态漂移）。
    #    weight -1.0 → -0.5（对齐 IsaacLab Digit 的 -0.4 量级）。
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=stand_still,
        weight=-0.5,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    # 3) foot_gait：按 0.6s 周期约束左右脚触地相位，让步态更规整。
    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=feet_gait,
        weight=0.5,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        },
    )
    return cfg


# 说明：is_terminated / stand_still / foot_gait 三个较新官方项已在上面函数中补齐。
# 其中 is_terminated 用 mjlab 自带的 mdp.is_terminated；stand_still 与 feet_gait 因
# mjlab 1.5.0 的 mdp 没有，已在本地按官方 unitree_rl_mjlab 语义等价实现。
