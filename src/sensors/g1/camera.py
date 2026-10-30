"""
Real depth camera sensor using local video file (stored as .pt tensor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch
from pathlib import Path

from mjlab.sensor.camera_sensor import CameraSensor, CameraSensorCfg, CameraSensorData

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.sensor.sensor_context import SensorContext

from src.mdr.camera import DepthNoiseMaskTransform

from typing import cast
from mjlab.tasks.tracking.mdp.commands import MotionCommand
import numpy as np
DEPTH_BUFFER = []
LAST_TIMESTEP = None
DEPTH_SAVED = False
@dataclass
class AdvancedCameraSensorCfg(CameraSensorCfg):
    """Configuration for a real depth camera sensor that reads from a local .pt file.

    The file should contain a tensor of shape (T, H, W) with depth values in meters
    (or raw values to be scaled by depth_scale).
    """

    file_path: str = "/home/tcs/8t01/capylab/logs/merge_27_48.pt"
    """Path to the .pt file containing depth video. Required."""

    depth_scale: float = 1.0
    """Scale factor to convert raw depth to meters (raw / scale). Set to 1.0 if already in meters."""

    clip_min: float = 0.0
    """Minimum depth value (meters) to clamp. None to disable."""

    clip_max: float = 6.0
    """Maximum depth value (meters) to clamp. None to disable."""

    num_envs: int = 1
    """Number of parallel environments. The depth tensor will be expanded
    to (num_envs, H, W, 1) if > 1.
    """

    flip_horizontal: bool = False
    """Whether to flip the image horizontally (mirror)."""

    transpose_xy: bool = False
    """Whether to swap the x and y axes (transpose)."""

    noise : bool = False

    # 强制只输出深度，忽略 data_types 中的其他类型
    data_types: tuple[Literal["depth"], ...] = ("depth",)

    def __post_init__(self) -> None:
        """Ensure only depth data type is used and set a default name."""
        super().__post_init__()
        if set(self.data_types) != {"depth"}:
            import warnings
            warnings.warn(
                "FakeCameraSensor only supports 'depth'. "
                f"Ignoring extra types: {self.data_types}"
            )
        if not self.file_path:
            raise ValueError("file_path must be provided for FakeCameraSensor.")

    def build(self) -> AdvancedCameraSensor:
        return AdvancedCameraSensor(self)

class AdvancedCameraSensor(CameraSensor):
    """Real depth camera sensor that reads a pre-recorded depth video from a .pt file.

    This sensor does not rely on MuJoCo rendering and overrides the
    necessary methods to bypass scene modification and context dependencies.
    """

    def __init__(self, cfg: AdvancedCameraSensorCfg) -> None:
        super().__init__(cfg)
        self.cfg: AdvancedCameraSensorCfg = cfg
        self._device: str | None = None
        self._frames: torch.Tensor | None = None   # (T, H, W)
        self._num_frames: int = 0

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        """Load the .pt file and store device; no camera index needed."""
        self._device = device
        self._data = data
        self.nworld = self._data.nworld
        self._camera_idx = mj_model.ncam + 1

        del data, device

        self._warp_model = model

        try:
            cam = mj_model.camera(self._camera_name)
            self._camera_idx = cam.id
        except KeyError as e:
            available = [mj_model.camera(i).name for i in range(mj_model.ncam)]
            raise ValueError(
                f"Camera '{self._camera_name}' not found in model. Available: {available}"
            ) from e


        # 加载视频文件
        file_path = Path(self.cfg.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Depth video file not found: {file_path}")
        try:
            frames = torch.load(file_path, map_location=self._device)
        except Exception as e:
            raise RuntimeError(f"Failed to load .pt file: {e}")

        # 确保形状为 (T, H, W)
        if frames.ndim != 3:
            raise ValueError(f"Expected 3D tensor (T, H, W), got shape {frames.shape}")
        self._frames = frames.to(torch.float32)
        self._num_frames = self._frames.shape[0]
        if self._num_frames == 0:
            raise ValueError("Loaded video has zero frames.")

        self._frame_idx = torch.randint(0,len(frames),(self.nworld,),device=self._device)
        
        self.trasf = DepthNoiseMaskTransform()

    # ---------- 核心数据获取 ----------
    @torch.no_grad()
    def _compute_data_(self,odf:torch.Tensor) -> CameraSensorData:

        self._frame_idx += 1

        #odf = self.trasf(odf.squeeze_(-1)).unsqueeze_(-1)

        # n,h,w,c = odf.shape

        #odf = odf.clip(0,self.cfg.clip_max).div_(self.cfg.clip_max)
        odf = odf.clip(0,self.cfg.clip_max)

        return CameraSensorData(depth=odf)

    @property
    def data(self) -> CameraSensorData:
        """Override to always fetch fresh depth data from the video stream."""
        ordata = self._compute_data()
        return self._compute_data_(ordata.depth)

    @property
    def frames(self):
        return self.trasf(self._frames[self._frame_idx % len(self._frames)])

from mjlab.envs import ManagerBasedRlEnv, mdp as envs_mdp

DEPTH_BUFFER = []
LAST_TIMESTEP = None
DEPTH_SAVED = False


@torch.no_grad()
def advanced_depth_camera_save(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:

    global DEPTH_BUFFER
    global LAST_TIMESTEP
    global DEPTH_SAVED

    command = cast(
        MotionCommand,
        env.command_manager.get_term("motion"),
    )

    real_sensor: AdvancedCameraSensor = env.scene[sensor_name]

    # [1, 27, 48, 1] -> [1, 27, 48]
    camera = real_sensor.data.depth.squeeze(-1)

    # 当前深度帧
    depth_frame = (
        camera[0]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    timestep = int(command.time_steps[0].item())

    print(
        "timestep:",
        timestep,
        "last:",
        LAST_TIMESTEP,
        "buffer:",
        len(DEPTH_BUFFER),
    )

    # ==============================
    # 新 episode
    # ==============================
    if (
        LAST_TIMESTEP is not None
        and timestep < LAST_TIMESTEP
    ):
        print(
            f"[Depth Recorder] new episode: "
            f"{LAST_TIMESTEP} -> {timestep}"
        )

        DEPTH_BUFFER.clear()
        DEPTH_SAVED = False

    # ==============================
    # 每个 timestep 只保存一次
    # ==============================
    if timestep != LAST_TIMESTEP:

        DEPTH_BUFFER.append(depth_frame)

        # 注意：这里保存的是 int
        LAST_TIMESTEP = timestep

        print(
            f"[Depth Recorder] "
            f"timestep={timestep}, "
            f"frames={len(DEPTH_BUFFER)}, "
            f"shape={depth_frame.shape}"
        )

    # ==============================
    # 录到 583 时保存
    # ==============================
    if timestep == 583 and not DEPTH_SAVED:

        print(
            "[Depth Recorder] preparing save, "
            f"buffer={len(DEPTH_BUFFER)}"
        )

        if len(DEPTH_BUFFER) > 0:

            depth_video = np.stack(
                DEPTH_BUFFER,
                axis=0,
            )

            save_path = "sim_depth.npy"

            np.save(
                save_path,
                depth_video,
            )

            print(
                "\n"
                "==============================\n"
                "[Depth Recorder] SAVED\n"
                f"path: {save_path}\n"
                f"shape: {depth_video.shape}\n"
                f"frames: {len(DEPTH_BUFFER)}\n"
                "=============================="
            )

            DEPTH_SAVED = True

    return camera

@torch.no_grad()
def advanced_depth_camera(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
    """Extract depth image from a camera sensor.

    Args:
    env: The manager-based RL environment.
    sensor_name: Name of the camera sensor.

    Returns:
    Depth image tensor of shape ``[num_envs, height, width, 1]`` (float32).
    """

    real_sensor : AdvancedCameraSensor = env.scene[sensor_name]

    camera = real_sensor.data.depth.squeeze_(-1)

    return camera

import numpy as np
_RECORD_PATH = "logs/records/filtered_depth.npy"
try:
    _RECORD = torch.tensor(np.load(_RECORD_PATH))
except FileNotFoundError:
    _RECORD = torch.empty(0)

@torch.no_grad()
def advanced_depth_camera_record(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
    """Extract depth image from a camera sensor.

    Args:
    env: The manager-based RL environment.
    sensor_name: Name of the camera sensor.

    Returns:
    Depth image tensor of shape ``[num_envs, height, width, 1]`` (float32).
    """

    global _RECORD

    command : MotionCommand = env.command_manager.get_term(command_name)

    if _RECORD.device != command.command.device:
        _RECORD = _RECORD.to(command.command.device)

    camera = _RECORD[command.time_steps % _RECORD.shape[0]]

    return camera.reshape(command.time_steps.shape[0],27,48).div(6.0)

@torch.no_grad()
def advanced_depth_camera_flatten(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
    """Extract depth image from a camera sensor.

    Args:
    env: The manager-based RL environment.
    sensor_name: Name of the camera sensor.

    Returns:
    Depth image tensor of shape ``[num_envs, height, width, 1]`` (float32).
    """

    real_sensor : AdvancedCameraSensor = env.scene[sensor_name]

    camera = real_sensor.data.depth.squeeze_(-1)

    return camera.flatten(1)


from mjlab.sensor import CameraSensorCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensor
from mjlab.managers.event_manager import EventTermCfg

def deploy_depth_camera_to(cfg,depth_history_length = 1):

    from scipy.spatial.transform import Rotation as R
    import numpy as np

    base = R.from_euler('y', -47.6, degrees=True)   # 对应 base 四元数
    swap = R.from_euler('z', -90.0, degrees=True)
    down = R.from_euler('x', -5.0, degrees=True)
    total_rotation = base * swap * down

    # 获取四元数 (scipy输出顺序为 [x, y, z, w])
    quat_xyzw = total_rotation.as_quat().tolist()
    quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])

    camera_cfg = AdvancedCameraSensorCfg(
        name="depth_camera",
        parent_body="robot/torso_link",
        pos=(0.0576235, 0.01753, 0.41987),
        quat= quat_wxyz,
        fovy=54,
        width=48,
        height=27,
        data_types=("depth",),
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (camera_cfg,)

    cfg.observations["depth_camera"] = ObservationGroupCfg(
        terms={
            "depth_camera": ObservationTermCfg(
                func=advanced_depth_camera,
                params={
                    "sensor_name": "depth_camera",
                },
                history_length=depth_history_length,
                flatten_history_dim=False,
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    from src.mdr.camera import randomize_depth_camera_pose
    cfg.events["randomize_depth_camera_pose"] = EventTermCfg(
        func=randomize_depth_camera_pose,
        mode="reset",
        params={
            "sensor_name": "depth_camera",
            "pos_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
            },

            "rot_range_deg": {
                "roll": (-.1, .1),
                "pitch": (-.1, .1),
                "yaw": (-.1, .1),
            },
        },
    )

    return cfg
