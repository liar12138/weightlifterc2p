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
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.sensor.sensor_context import SensorContext

import torchvision.transforms.v2 as v2
from mjlab.managers.event_manager import EventTermCfg
from typing import cast
from mjlab.tasks.tracking.mdp.commands import MotionCommand
import numpy as np
from .depth_noise import sim_to_real_depth_noise
class DepthNoiseMaskTransform:

    def __init__(self):
        self.transforms = v2.Compose([
            # 1. 先做几何增强（裁剪翻转等），避免模糊破坏边缘结构
            # v2.RandomHorizontalFlip(0.001),
            v2.RandomRotation(degrees=3),
            v2.RandomResizedCrop((25,46),scale=(0.95,1)),  # 先裁剪
            #v2.Resize((54,96)),       # 再resize回原尺寸 v2.Resize((27,48))
            v2.Resize((27,48)),
            # 2. 再添加噪声（模拟传感器扰动）
            v2.GaussianNoise(mean=0.02, sigma=0.01, clip=False),
            # v2.RandomErasing(p=0.05, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0),
            
            # 3. 【关键新增】用 3x3 高斯核轻微模糊，抹掉高频噪声，sigma 建议 0.5~1.0
            v2.GaussianBlur(kernel_size=3, sigma=0.8),
        ])

    def __call__(self,x):
        return self.transforms(x)
    
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

        self.trasf = DepthNoiseMaskTransform()



    @torch.no_grad()
    def replace_invalid_depth(
        self,
        depth: torch.Tensor,
        max_depth: float = 6.0,
        invalid_threshold: float = 1e-4,
    ) -> torch.Tensor:
        """
        将无效深度统一替换成最大深度。

        无效深度包括：
        - NaN
        - 正负 inf
        - 小于等于 invalid_threshold 的值
        - 大于等于 max_depth 的值

        Args:
            depth: 任意形状的深度张量，单位为米。
            max_depth: 最大深度，同时也是无效值填充值。
            invalid_threshold: 判断零值或极小值的阈值。

        Returns:
            处理后的深度张量，shape 不变。
        """
        invalid_mask = (
            ~torch.isfinite(depth)
            | (depth <= invalid_threshold)
            | (depth >= max_depth)
        )

        return torch.where(
            invalid_mask,
            torch.full_like(depth, max_depth),
            depth,
        )
    
    # ---------- 核心数据获取 ----------
    @torch.no_grad()
    def _compute_data_(self,ordata:torch.Tensor) -> CameraSensorData:
        depth = ordata.squeeze(-1).clone()
        depth = self.replace_invalid_depth(depth,max_depth=self.cfg.clip_max, invalid_threshold=1e-4)
        depth = self.trasf(depth).unsqueeze(-1)
        depth = depth.clip(0,self.cfg.clip_max).div_(self.cfg.clip_max)
        # depth = depth.clip(0,self.cfg.clip_max)
        return CameraSensorData(depth=depth)

    # # ---------- 核心数据获取 ----------
    # @torch.no_grad()
    # def _compute_data_(self,ordata:torch.Tensor) -> CameraSensorData:
    #     odf = self.trasf(ordata.squeeze_(-1)).unsqueeze_(-1)
    #     odf = odf.clip(0,self.cfg.clip_max).div_(self.cfg.clip_max)
    #     return CameraSensorData(depth=odf)

    @property
    def data(self) -> CameraSensorData:
        """Override to always fetch fresh depth data from the video stream."""
        ordata = self._compute_data()
        return self._compute_data_(ordata.depth)

    @property
    def frames(self):
        return self.trasf(self._frames[self._frame_idx % len(self._frames)])

from mjlab.envs import ManagerBasedRlEnv, mdp as envs_mdp

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
    # command = env.command_manager.get_command('twist')

    real_sensor : AdvancedCameraSensor = env.scene[sensor_name]

    camera = real_sensor.data.depth.squeeze_(-1)

    return camera


# DEPTH_BUFFER = []
# LAST_TIMESTEP = None
# DEPTH_SAVED = False


@torch.no_grad()
def advanced_depth_camera_save(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    save_path=None,
) -> torch.Tensor:

    # global DEPTH_BUFFER
    # global LAST_TIMESTEP
    # global DEPTH_SAVED

    command = cast(MotionCommand,env.command_manager.get_term("motion"))

    real_sensor: AdvancedCameraSensor = env.scene[sensor_name]
    if not hasattr(advanced_depth_camera_save, "_debug"):
        print("sensor_name =", sensor_name)
        print("raw sensor depth =", real_sensor.data.depth.shape)

        if hasattr(real_sensor, "cfg"):
            print("sensor cfg width =", real_sensor.cfg.width)
            print("sensor cfg height =", real_sensor.cfg.height)

        advanced_depth_camera_save._debug = True
    # [1, 27, 48, 1] -> [1, 27, 48]
    camera = real_sensor.data.depth.squeeze(-1)
    # camera = sim_to_real_depth_noise(
    #     camera,
    #     max_depth=6.0,
    #     geometric_noise=True,
    #     output_normalized=True,
    # )
    # if not save_path:
    #     return camera
    
    # depth_frame = (camera[0].detach().cpu().numpy().copy())

    # timestep = int(command.time_steps[0].item())

    # if ( LAST_TIMESTEP is not None and timestep < LAST_TIMESTEP):
    #     print( f"[Depth Recorder] new episode: "f"{LAST_TIMESTEP} -> {timestep}")

    #     DEPTH_BUFFER.clear()
    #     DEPTH_SAVED = False

    # if timestep != LAST_TIMESTEP:

    #     DEPTH_BUFFER.append(depth_frame)

    #     LAST_TIMESTEP = timestep

    # if timestep == 583 and not DEPTH_SAVED:


    #     if len(DEPTH_BUFFER) > 0:

    #         depth_video = np.stack(DEPTH_BUFFER,axis=0)

    #         np.save(save_path,depth_video)

    #         print(
    #             "\n"
    #             "==============================\n"
    #             "[Depth Recorder] SAVED\n"
    #             f"path: {save_path}\n"
    #             f"shape: {depth_video.shape}\n"
    #             f"frames: {len(DEPTH_BUFFER)}\n"
    #             "=============================="
    #         )

    #         DEPTH_SAVED = True

    return camera

import numpy as np
_RECORD_PATH = "logs/records/filtered_depth.npy"
try:
    _RECORD = torch.tensor(np.load(_RECORD_PATH))
except FileNotFoundError:
    # Recording-only code should not prevent simulation camera imports.
    _RECORD = torch.empty(0)


def draw_camera_hands_alignment(env, visualizer) -> None:
    """Optional debug hook retained for command visualizers."""
    del env, visualizer

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

    return camera.reshape(command.time_steps.shape[0],27,48)


def deploy_h2_depth_camera_to(cfg,depth_history_length = 1,camera_name = "h2_depth_camera",camera_W = 48 , camera_H = 27):

    from scipy.spatial.transform import Rotation as R
    import numpy as np

    # 俯仰：绕Y轴旋转 -47.6°（向下看）
    pitch = R.from_euler('y', -90, degrees=True)

    # 横滚修正：如果你的图像底部朝左，说明需要绕Z轴旋转 -90° 或 +90°
    # 先尝试 -90°，如果不行改成 +90°
    roll_fix = R.from_euler('z', -90, degrees=True)  

    # 组合：先修正横滚，再俯仰
    total_rotation = pitch * roll_fix

    # 获取四元数 (scipy输出顺序为 [x, y, z, w])
    quat_xyzw = total_rotation.as_quat().tolist()
    quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])

    camera_cfg = AdvancedCameraSensorCfg(
        name=camera_name,
        parent_body="robot/head_yaw_link",
        pos=(0.09, -0.01 , 0.04),
        quat= quat_wxyz,
        fovy=50,  # 使用垂直视场角 58°
        #width=38 * 2,   # 水平分辨率，可选用848或1280等
        #height=27 * 2,  # 垂直分辨率，可选用480或720等
        width=camera_W,
        height=camera_H,
        data_types=("depth",),
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (camera_cfg,)

    cfg.observations[camera_name] = ObservationGroupCfg(
        terms={
            camera_name: ObservationTermCfg(
                func=advanced_depth_camera_save, #advanced_depth_camera_save
                params={
                    "sensor_name": camera_name,
                    "save_path": None,
                },
                history_length=depth_history_length,
                flatten_history_dim=False,
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    from src.sensors.randomize import randomize_depth_camera_pose
    cfg.events["randomize_depth_camera_pose"] = EventTermCfg(
        func=randomize_depth_camera_pose,
        mode="reset",
        params={
            "sensor_name": camera_name,
            "pos_range": {
                "x": (0, 0.03),
                "y": (-0.08, 0.08),
                "z": (-0.08, 0.08),
            },

            "rot_range_deg": {
                "roll": (-.12, .12),
                "pitch": (-.12, .12),
                "yaw": (-.12, .12),
            },
        },
    ) 

    return cfg


def deploy_g1_depth_camera_to(cfg,depth_history_length = 1,camera_name = "depth_camera"):

    from scipy.spatial.transform import Rotation as R
    import numpy as np

    # 俯仰：绕Y轴旋转 -47.6°（向下看）
    pitch = R.from_euler('y', -42.0, degrees=True)

    # 横滚修正：如果你的图像底部朝左，说明需要绕Z轴旋转 -90° 或 +90°
    # 先尝试 -90°，如果不行改成 +90°
    roll_fix = R.from_euler('z', -90, degrees=True)  

    # 组合：先修正横滚，再俯仰
    total_rotation = pitch * roll_fix

    # 获取四元数 (scipy输出顺序为 [x, y, z, w])
    quat_xyzw = total_rotation.as_quat().tolist()
    quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])

    camera_cfg = AdvancedCameraSensorCfg(
        name=camera_name,
        parent_body="robot/torso_link",
        pos=(0.0576235, 0.01753, 0.41987),
        quat= quat_wxyz,
        fovy=50,  # 使用垂直视场角 58°
        width=38  ,   # 水平分辨率，可选用848或1280等
        height=27  ,  # 垂直分辨率，可选用480或720等
        data_types=("depth",),
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (camera_cfg,)

    cfg.observations[camera_name] = ObservationGroupCfg(
        terms={
            camera_name: ObservationTermCfg(
                func=advanced_depth_camera,
                params={
                    "sensor_name": camera_name,
                },
                history_length=depth_history_length,
                flatten_history_dim=False,
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    from src.sensors.randomize import randomize_depth_camera_pose
    cfg.events["randomize_depth_camera_pose"] = EventTermCfg(
        func=randomize_depth_camera_pose,
        mode="reset",
        params={
            "sensor_name": camera_name,
            "pos_range": {
                "x": (0, 0.01),
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
