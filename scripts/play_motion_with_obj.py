import cv2
import tyro
import mjlab
import numpy as np
import torch
import time

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

import sys

sys.path.append('./')


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

import sys
sys.path.append('./')

class ObjMotionLoader:
    def __init__(self, motion_file='/home/tcs/8t01/capylab/src/assets/motion/pickbox/box.npz', device='cuda:0'):
        data = np.load(motion_file)
        self.trans = torch.tensor(data['trans'], dtype=torch.float, device=device)
        self.trans[:,2] = self.trans[:,2] + 0.35
        self.rot = torch.tensor(data['rot'], dtype=torch.float, device=device)
        self.link_pose = torch.cat([self.trans, self.rot], dim=-1)

def main(
    motion_file='./src/assets/motion/pickbox/g1.npz',
    repeat=True,
    device='cuda:0',
    save_video=False,          # 新增：是否保存视频
    video_path='logs/tasks/pick_box/data.mp4',   # 新增：输出路径
    video_fps=50,              # 新增：帧率
):
    sim_cfg = SimulationCfg()
    scene = Scene(unitree_g1_stairs_hold_box_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
    motion_loader = MotionLoader(motion_file, robot_joint_indexes, device=device)

    box: Entity = scene["box"]
    box_motion = ObjMotionLoader()

    viewer_cfg = ViewerConfig(
        height=480,
        width=640,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=2.0,
        elevation=-5.0,
        azimuth=20,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

    frames = [] if save_video else None  # 存储图像帧

    while repeat:
        for t in range(motion_loader.time_step_total):
            box.write_root_link_pose_to_sim(box_motion.link_pose[t].unsqueeze(0))
            root_state = torch.cat([
                motion_loader.body_pos_w[t, 0],
                motion_loader.body_quat_w[t, 0],
                motion_loader.body_lin_vel_w[t, 0],
                motion_loader.body_ang_vel_w[t, 0]
            ], dim=-1)
            robot.write_root_state_to_sim(root_state.unsqueeze(0))
            robot.write_joint_state_to_sim(
                motion_loader.joint_pos[t],
                motion_loader.joint_vel[t]
            )
            sim.forward()
            scene.update(sim.mj_model.opt.timestep)

            renderer.update(sim.data)
            img = renderer.render()  # RGB (H,W,3)

            # 若开启视频保存，则复制当前帧
            if save_video:
                frames.append(img.copy())

            cv2.imshow("Simulation", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return
            elif key == ord('p'):
                print(f"Paused at time step t = {t}")   # 打印当前 t
                cv2.waitKey(0)

        # 所有帧收集完毕后写入视频
        if save_video and frames:
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, video_fps, (w, h))
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"视频已保存至: {video_path} (总帧数: {len(frames)})")
            break  # 保存后退出循环，避免重复

        if not repeat:
            break
        time.sleep(1 / 30)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
