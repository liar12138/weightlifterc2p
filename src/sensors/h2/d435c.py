
from ..g1.camera import AdvancedCameraSensorCfg,ObservationGroupCfg,advanced_depth_camera_save,advanced_depth_camera,ObservationTermCfg,EventTermCfg


def deploy_depth_camera_to(cfg,depth_history_length = 1):

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
        name="depth_camera",
        parent_body="robot/head_yaw_link",
        pos=(0.09, -0.01 , 0.04),
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
                func=advanced_depth_camera,#advanced_depth_camera
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