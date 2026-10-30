from __future__ import annotations
from typing import cast

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    subtract_frame_transforms,
)
from mjlab.sensor import CameraSensor
import torch.nn.functional as F

from mjlab.entity import Entity
from .commands import MotionBarbellCommand
from mjlab.managers.scene_entity_config import SceneEntityCfg
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
@torch.no_grad()
def barbell_qpos(env: ManagerBasedRlEnv):
    barbell_asset: Entity = env.scene["barbell"]

    return barbell_asset.data.root_com_pose_w

@torch.no_grad()
def bellbar_vector(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  return command.bellbar_vector

@torch.no_grad()
def right_hand_plate_vector(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  return command.right_hand_plate_vector

@torch.no_grad()
def left_hand_plate_vector(
    env: ManagerBasedRlEnv,
    command_name 
):
  command:MotionBarbellCommand = env.command_manager.get_term(command_name)

  return command.left_hand_plate_vector

def motion_anchor_pos_b_ref(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    ref_anchor_pos_w = command.body_pos_relative_w[
        :, command.motion_anchor_body_index
    ]

    ref_anchor_quat_w = command.body_quat_relative_w[
        :, command.motion_anchor_body_index
    ]

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        ref_anchor_pos_w,
        ref_anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b_ref(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    ref_anchor_pos_w = command.body_pos_relative_w[
        :, command.motion_anchor_body_index
    ]

    ref_anchor_quat_w = command.body_quat_relative_w[
        :, command.motion_anchor_body_index
    ]

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        ref_anchor_pos_w,
        ref_anchor_quat_w,
    )

    mat = matrix_from_quat(ori)

    return mat[..., :2].reshape(mat.shape[0], -1)


def depth_camera(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Extract depth image from a camera sensor.

  Args:
    env: The manager-based RL environment.
    sensor_name: Name of the camera sensor.

  Returns:
    Depth image tensor of shape ``[num_envs, height, width, 1]`` (float32).
  """
  sensor: CameraSensor = env.scene[sensor_name]
  return sensor.data.depth

def batch_depth_resize_nhwc(depth_batch, target_size=(180, 320), 
                            min_depth=None, max_depth=None, 
                            keep_physics_scale=True):
    """
    针对 [num_envs, height, width, 1] 格式的深度图通用变换

    Args:
        depth_batch (torch.Tensor): 形状 [B, H, W, 1], dtype=float32
        target_size (tuple): 目标尺寸 (H_out, W_out)，默认 (32, 32)
        min_depth (float, optional): 有效深度最小值（用于裁剪无效0值）
        max_depth (float, optional): 有效深度最大值（用于裁剪远处噪点）
        keep_physics_scale (bool): True=保留物理数值（如米/毫米），False=归一化到[0,1]

    Returns:
        torch.Tensor: 形状 [B, 32, 32, 1]，保持 NHWC 格式
    """
    # 1. 输入校验：确保是 4 维且通道数为 1
    assert depth_batch.dim() == 4 and depth_batch.shape[-1] == 1, \
        f"输入必须为 [B, H, W, 1]，当前形状为 {depth_batch.shape}"

    # 2. [NHWC] -> [NCHW] (因为 interpolate 要求通道在第二维)
    depth_nchw = depth_batch.permute(0, 3, 1, 2)  # (B, 1, H, W)

    # 3. 核心：Resize 到 (32, 32)，使用 nearest 防止边缘伪造深度
    #    若你的深度图是平滑曲面（如人脸），可改为 mode='bilinear', antialias=True
    depth_resized = F.interpolate(
        depth_nchw, 
        size=target_size, 
        mode='nearest'  # 通用保险选项
    )  # (B, 1, 32, 32)

    # 4. 可选：物理尺度裁剪（防止传感器噪声）
    if min_depth is not None:
        depth_resized = torch.clamp(depth_resized, min=min_depth)
    if max_depth is not None:
        depth_resized = torch.clamp(depth_resized, max=max_depth)

    # 5. 可选：归一化到 [0, 1]（如果你的网络需要标准化输入）
    if not keep_physics_scale:
        if min_depth is None or max_depth is None:
            # 如果没有提供 min/max，则从当前 batch 动态计算（注意：这会使尺度随batch变化）
            d_min = depth_resized.amin(dim=(1,2,3), keepdim=True)
            d_max = depth_resized.amax(dim=(1,2,3), keepdim=True)
            depth_resized = (depth_resized - d_min) / (d_max - d_min + 1e-8)
        else:
            depth_resized = (depth_resized - min_depth) / (max_depth - min_depth + 1e-8)

    # 6. [NCHW] 转回 [NHWC]，保持与输入一致的布局
    depth_output = depth_resized.permute(0, 2, 3, 1)  # (B, 32, 32, 1)

    return depth_output

def depth_camera_preprocessed(env: ManagerBasedRlEnv, sensor_name: str , target_size = (180, 320)) -> torch.Tensor:
    """
    depth: [B, H, W, 1]
    return: [B, 1, 32, 32]
    """    
    preprocessed = batch_depth_resize_nhwc(depth_camera(env,sensor_name),target_size=target_size)
    return preprocessed.flatten(1)

@torch.no_grad()
def frame_stage_indicator(
    env: ManagerBasedRlEnv,
    command_name: str,
    start_frame: int = 150,
    end_frame: int = 320,
) -> torch.Tensor:

    command = cast(
        MotionBarbellCommand,
        env.command_manager.get_term(command_name),
    )

    active = ((command.time_steps >= start_frame)& (command.time_steps <= end_frame))

    return active.float().unsqueeze(-1)

@torch.no_grad()
def joint_vel_rel_zero_hand(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:

    asset: Entity = env.scene[asset_cfg.name]

    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None

    jnt_ids = asset_cfg.joint_ids

    joint_vel = (
        asset.data.joint_vel[:, jnt_ids]
        - default_joint_vel[:, jnt_ids]
    ).clone()

    hand_joint_names = (
        # left
        "left_thumb_cmc_yaw",
        "left_thumb_cmc_pitch",
        "left_thumb_ip",
        "left_index_mcp_pitch",
        "left_index_dip",
        "left_middle_mcp_pitch",
        "left_middle_dip",
        "left_ring_mcp_pitch",
        "left_ring_dip",
        "left_pinky_mcp_pitch",
        "left_pinky_dip",

        # right
        "right_thumb_cmc_yaw",
        "right_thumb_cmc_pitch",
        "right_thumb_ip",
        "right_index_mcp_pitch",
        "right_index_dip",
        "right_middle_mcp_pitch",
        "right_middle_dip",
        "right_ring_mcp_pitch",
        "right_ring_dip",
        "right_pinky_mcp_pitch",
        "right_pinky_dip",
    )

    hand_joint_ids, _ = asset.find_joints(
        hand_joint_names
    )

    # 如果当前 observation 是全部 joint
    if isinstance(jnt_ids, slice):
        joint_vel[:, hand_joint_ids] = 0.0

    else:
        # 如果以后 asset_cfg 只选了部分 joint，再做映射
        id_to_local = {
            int(jid): i
            for i, jid in enumerate(jnt_ids)
        }

        hand_local_ids = [
            id_to_local[int(jid)]
            for jid in hand_joint_ids
            if int(jid) in id_to_local
        ]

        if hand_local_ids:
            joint_vel[:, hand_local_ids] = 0.0

    return joint_vel

def graping(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command : MotionBarbellCommand = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
    return command.graping_obs.unsqueeze(1)

def time_step(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command : MotionBarbellCommand = cast(MotionBarbellCommand, env.command_manager.get_term(command_name))
    return torch.zeros_like(command.time_steps.unsqueeze(-1))