# motion_tracking_H2

Unitree H2 纯动作跟踪任务（无杠铃实体），包含两类机器人配置：

- **H2 + LinkerHand O6（53-DOF）**：`configs/env_cfgs.py`
- **H2（31-DOF，无手，`h2.xml`）**：`configs/env_cfgs_31dof.py`

两者基础配置均为 `unitree_g1_flat_tracking_env_cfg(has_state_estimation=False)`，
无相机，runner 为 `MotionTrackingOnPolicyRunner`（纯 MLP PPO）。

## 任务

| Task ID | 机器人 | 动作文件 | 额外 |
|---|---|---|---|
| `Unitree-H2-O6-Motion-Tracking` | `get_h2_o6_robot_cfg()` | `h2o6/13_h2o6_pro.npz` | 无 |
| `Unitree-H2-Motion-Tracking` | `get_h2_robot_cfg()` | `h2/1_h2o6_promax_cut_aligned_down_hold50_31dof.npz` | `joint_position_range=(-0.3, 0.3)` |
| `Unitree-H2-Motion-Tracking-HandMass` | `get_h2_robot_cfg()` | `h2/1_h2o6_promax_cut_aligned_up_31dof.npz` | 手部（橡胶假手）质量 DR + 课程 |

31-DOF 任务 2 的手部质量课程：两手腕（`left/right_wrist_yaw_link`）合计质量上限从 0 线性
递增到 15 kg（`ramp_until_step=150_000`），每 episode 两手腕等分采样，用 `dr.body_mass` +
`operation=add` 施加点质量。

## 训练

```bash
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=8
nohup python scripts/train.py Unitree-H2-Motion-Tracking --env.scene.num-envs=8192 --agent.logger=tensorboard --env.commands.motion.motion-file=src/assets/motion/h2/h2_down_hold50.npz > logs/h2_tracking/h2_tracking_down_v1.log 2>&1 &

nohup python scripts/train.py Unitree-H2-O6-Motion-Tracking-Hand-Mass --env.scene.num-envs=12880 --env.commands.motion.motion-file=src/assets/motion/h2o6/1_h2o6_new_s3_linear.npz --agent.resume True --agent.load-run 2026-09-04_07-03-52 > logs/h2_o6_mt/h2_o6_mt_hm.log 2>&1 &

nohup python scripts/train.py Unitree-H2-Motion-Tracking --env.scene.num-envs=8192 --agent.logger=tensorboard --env.commands.motion.motion-file=src/assets/motion/h2/h2_down_hold50.npz --agent.resume=True  --agent.load_run=2026-08-27_17-49-09 --agent.load_checkpoint=model_2700.pt > logs/h2_tracking/h2_tracking_down_v1.log 2>&1 &

nohup python scripts/train.py Unitree-H2-Motion-Tracking-HandMass --env.scene.num-envs=12880 --env.commands.motion.motion-file /home/tcs/ti90004t/humanoid/weightlifter/src/assets/motion/h2/1_h2o6_new_s3_31.npz --agent.resume True --agent.load-run 2026-09-01_14-26-23  > logs/h2_tracking/h2_pullup_res_handdr.log 2>&1 &

python scripts/train.py Unitree-H2-Motion-Tracking --agent.logger=tensorboard
python scripts/train.py Unitree-H2-Motion-Tracking-HandMass --agent.logger tensorboard
```

## 播放

```bash
python scripts/play.py Unitree-H2-O6-Motion-Tracking
python scripts/play.py Unitree-H2-Motion-Tracking --checkpoint_file=logs/h2_31dof_motion_tracking/rsl_rl/2026-08-23_00-23-08/model_29999.pt --motion-file=src/assets/motion/h2/1_h2o6_promax_cut_aligned_down_hold50_31dof.npz
```
