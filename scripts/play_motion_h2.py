#!/usr/bin/env python3

from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import tyro

# ---------------------------------------------------------------------
# 将项目根目录加入 sys.path。
#
# 当前文件：
#   weightlifter/scripts/play_motion_with_obj.py
#
# 项目根目录：
#   weightlifter/
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

from src.tasks.weight_lifter_H2.configs.env_cfgs import (
    unitree_h2_pull_up_barbell_env_cfg,
)


MOTION_FILE = (
    PROJECT_ROOT
    / "src/assets/motion/h2/1_h2o6_prosquat.npz"
)


# H2 + O6 的 53 个关节。
#
# 这个顺序必须和 NPZ 中 joint_pos、joint_vel 的顺序一致。
JOINT_NAMES = [
    # ---------------- 左腿：6 ----------------
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",

    # ---------------- 右腿：6 ----------------
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",

    # ---------------- 腰部：3 ----------------
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    # ---------------- 头部：2 ----------------
    "head_pitch_joint",
    "head_yaw_joint",

    # ---------------- 左臂：7 ----------------
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    # ---------------- 左手 O6：11 ----------------
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

    # ---------------- 右臂：7 ----------------
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",

    # ---------------- 右手 O6：11 ----------------
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
]


def load_motion_fps(motion_file: str | Path) -> float:
    """从 NPZ 中读取动作帧率。"""
    with np.load(str(motion_file)) as data:
        if "fps" not in data.files:
            return 50.0

        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    if fps <= 0:
        return 50.0

    return fps


def validate_motion_file(motion_file: str | Path) -> None:
    """检查动作文件是否包含播放所需字段。"""
    required_keys = {
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }

    with np.load(str(motion_file)) as data:
        missing_keys = required_keys.difference(data.files)

        if missing_keys:
            raise KeyError(
                f"动作文件缺少字段：{sorted(missing_keys)}；"
                f"当前字段：{data.files}"
            )

        joint_pos_shape = data["joint_pos"].shape
        joint_vel_shape = data["joint_vel"].shape

    if joint_pos_shape[-1] != len(JOINT_NAMES):
        raise ValueError(
            "动作文件关节数量不正确："
            f"joint_pos={joint_pos_shape}，"
            f"JOINT_NAMES={len(JOINT_NAMES)}"
        )

    if joint_vel_shape != joint_pos_shape:
        raise ValueError(
            "joint_pos 和 joint_vel 形状不一致："
            f"{joint_pos_shape} vs {joint_vel_shape}"
        )


def save_video_frames(
    frames: list[np.ndarray],
    video_path: str | Path,
    video_fps: float,
) -> None:
    """将 RGB 图像帧保存为 MP4 视频。"""
    if not frames:
        print("没有可保存的视频帧。")
        return

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        video_fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件：{video_path}")

    try:
        for frame in frames:
            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR,
            )
            writer.write(frame_bgr)
    finally:
        writer.release()

    print(
        f"视频已保存至：{video_path}\n"
        f"总帧数：{len(frames)}\n"
        f"帧率：{video_fps:.2f}"
    )


def main(
    motion_file: str = str(MOTION_FILE),
    repeat: bool = True,
    device: str = "cuda:0",
    save_video: bool = False,
    video_path: str = "logs/tasks/h2/prosquat.mp4",
    video_fps: float | None = None,
):
    """播放 H2+O6 NPZ 动作。

    操作：
      Q / Esc：退出
      P / Space：暂停或继续
      A / 左箭头：暂停时上一帧
      D / 右箭头：暂停时下一帧
      R：返回第 0 帧
    """

    motion_path = Path(motion_file).expanduser().resolve()

    if not motion_path.exists():
        raise FileNotFoundError(f"找不到动作文件：{motion_path}")

    validate_motion_file(motion_path)

    motion_fps = load_motion_fps(motion_path)

    if video_fps is None:
        video_fps = motion_fps

    print(f"动作文件：{motion_path}")
    print(f"动作帧率：{motion_fps:.2f}")

    # --------------------------------------------------------------
    # 创建 H2+O6 杠铃场景
    # --------------------------------------------------------------
    env_cfg = unitree_h2_pull_up_barbell_env_cfg()

    sim_cfg = SimulationCfg()

    scene = Scene(
        env_cfg.scene,
        device=device,
    )

    model = scene.compile()

    sim = Simulation(
        num_envs=1,
        cfg=sim_cfg,
        model=model,
        device=device,
    )

    scene.initialize(
        sim.mj_model,
        sim.model,
        sim.data,
    )

    robot: Entity = scene["robot"]
    barbell: Entity = scene["barbell"]

    # barbell 在当前脚本中使用环境配置里的初始位置。
    # NPZ 中没有 barbell 的位置和姿态轨迹。
    del barbell

    # --------------------------------------------------------------
    # 查找 53 个 H2+O6 关节
    # --------------------------------------------------------------
    joint_ids, matched_joint_names = robot.find_joints(
        JOINT_NAMES,
        preserve_order=True,
    )

    if len(joint_ids) != len(JOINT_NAMES):
        raise RuntimeError(
            "机器人模型中的关节数量与 JOINT_NAMES 不一致："
            f"找到 {len(joint_ids)} 个，期望 {len(JOINT_NAMES)} 个。"
        )

    # MotionLoader 的第二个参数用于选取 body 数据。
    # 这里只需要 pelvis，后面使用它设置机器人根状态。
    pelvis_body_ids, _ = robot.find_bodies(
        ["pelvis"],
        preserve_order=True,
    )

    if len(pelvis_body_ids) != 1:
        raise RuntimeError(
            f"没有唯一找到 pelvis body：{pelvis_body_ids}"
        )

    body_ids = torch.as_tensor(
        pelvis_body_ids,
        dtype=torch.long,
        device=device,
    )

    motion_loader = MotionLoader(
        str(motion_path),
        body_ids,
        device=device,
    )

    num_frames = motion_loader.time_step_total
    motion_joint_dim = motion_loader.joint_pos.shape[-1]

    print(f"机器人模型关节总数：{len(robot.joint_names)}")
    print(f"动作使用关节数：{len(joint_ids)}")
    print(f"NPZ joint_pos：{tuple(motion_loader.joint_pos.shape)}")
    print(f"NPZ body_pos_w：{tuple(motion_loader.body_pos_w.shape)}")
    print(f"总帧数：{num_frames}")

    if motion_joint_dim != len(joint_ids):
        raise RuntimeError(
            "动作关节维度和机器人关节数量不一致："
            f"motion={motion_joint_dim}，robot={len(joint_ids)}"
        )

    # --------------------------------------------------------------
    # 创建渲染器
    # --------------------------------------------------------------
    viewer_cfg = ViewerConfig(
        height=480,
        width=640,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=2.5,
        elevation=-10.0,
        azimuth=20.0,
    )

    renderer = OffscreenRenderer(
        model=sim.mj_model,
        cfg=viewer_cfg,
        scene=scene,
    )

    renderer.initialize()

    frames: list[np.ndarray] = []
    paused = False
    quit_requested = False
    frame_idx = 0
    completed_once = False

    # OpenCV waitKey 使用毫秒。
    frame_delay_ms = max(
        1,
        int(round(1000.0 / motion_fps)),
    )

    def apply_frame(t: int) -> np.ndarray:
        """将指定动作帧写入仿真并渲染。"""

        # ----------------------------------------------------------
        # 根状态：
        # position(3) + quaternion(4) +
        # linear velocity(3) + angular velocity(3)
        # ----------------------------------------------------------
        root_state = torch.cat(
            [
                motion_loader.body_pos_w[t, 0],
                motion_loader.body_quat_w[t, 0],
                motion_loader.body_lin_vel_w[t, 0],
                motion_loader.body_ang_vel_w[t, 0],
            ],
            dim=-1,
        )

        robot.write_root_state_to_sim(
            root_state.unsqueeze(0)
        )

        # [53] -> [1, 53]
        joint_pos = motion_loader.joint_pos[t].unsqueeze(0)
        joint_vel = motion_loader.joint_vel[t].unsqueeze(0)

        robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids=joint_ids,
        )

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        renderer.update(sim.data)
        return renderer.render()

    try:
        print(
            "\n操作说明：\n"
            "  P / Space：暂停或继续\n"
            "  A / 左箭头：暂停时上一帧\n"
            "  D / 右箭头：暂停时下一帧\n"
            "  R：返回第 0 帧\n"
            "  Q / Esc：退出\n"
        )

        while not quit_requested:
            img = apply_frame(frame_idx)

            if save_video and not completed_once:
                frames.append(img.copy())

            cv2.imshow(
                "H2+O6 Motion",
                cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            wait_time = 0 if paused else frame_delay_ms
            key = cv2.waitKey(wait_time) & 0xFF

            if key in (ord("q"), 27):
                quit_requested = True
                continue

            if key in (ord("p"), ord(" ")):
                paused = not paused
                print(
                    f"{'暂停' if paused else '继续'}，"
                    f"当前帧：{frame_idx}/{num_frames - 1}"
                )
                continue

            if key == ord("r"):
                frame_idx = 0
                paused = True
                print("已返回第 0 帧")
                continue

            if paused:
                # OpenCV 不同平台的方向键编码可能不同，
                # 因此同时支持 A/D。
                if key in (ord("a"), 81):
                    frame_idx = max(
                        0,
                        frame_idx - 1,
                    )
                    print(f"当前帧：{frame_idx}")
                elif key in (ord("d"), 83):
                    frame_idx = min(
                        num_frames - 1,
                        frame_idx + 1,
                    )
                    print(f"当前帧：{frame_idx}")

                continue

            frame_idx += 1

            if frame_idx >= num_frames:
                completed_once = True

                if save_video:
                    break

                if repeat:
                    frame_idx = 0
                else:
                    break

        if save_video:
            save_video_frames(
                frames=frames,
                video_path=video_path,
                video_fps=video_fps,
            )

    finally:
        cv2.destroyAllWindows()

        # 防止异常退出时出现 EGL 析构报错。
        try:
            renderer.close()
        except Exception:
            pass


if __name__ == "__main__":
    tyro.cli(main)