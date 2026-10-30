/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-18_15-41-24/model_1300.pt
export CUDA_VISIBLE_DEVICES=2
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-18_15-48-16/model_2600.pt
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-No-Camera --env.scene.num-envs=4096 --motion-file /data/weightlifter/src/assets/motion/h2/1_h2o6_pro.npz --agent.resume True  --agent.load_run 2026-08-21_10-41-40 --agent.load_checkpoint model_400.pt > logs/tasks/h2/no_camera/restart/0821v0.1.log 2>&1 &
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-21_13-41-58/model_31000.pt
/data/weightlifter/logs/g1_tracking/danrsl_rl/2026-08-21_10-41-40/model_400.pt
python scripts/play.py Unitree-H2-Pullup-A-Barbell-MT-Camera-Hand --motion-file src/assets/motion/h2o6/1_h2o6_promax_cut_aligned_80_240.npz --checkpoint_file=/home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_camera/rsl_rl/2026-08-31_15-01-55/model_37400.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-10_21-37-44/model_12000.pt/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-11_13-02-32/model_13800.pt
/home/tcs/8t01/humanoid/weightlifter/logs/g1_tracking/rsl_rl/2026-08-03_20-06-34/model_1000.pt
/home/tcs/8t01/humanoid/weightlifter/logs/g1_tracking/rsl_rl/2026-08-05_19-30-00/model_59500.pt
/home/tcs/8t01/humanoid/weightlifter/logs/g1_tracking/rsl_rl/2026-08-05_21-57-30/model_83000.pt
export CUDA_VISIBLE_DEVICES=2
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-10_15-53-43/model_45100.pt
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-MT-Camera-Hand --env.scene.num-envs=4096 --env.commands.motion.motion-file src/assets/motion/h2o6/1_h2o6_promax_cut_aligned_80_240.npz --agent.resume True  --agent.load_run 3090 --agent.load_checkpoint model_37300.pt  > logs/tasks/h2_o6/only_grab/v0.8.1.addgeom_margin.log 2>&1 &
Unitree-H2-Pullup-A-Barbell
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell --env.scene.num-envs=4096 --env.commands.motion.motion-file src/assets/motion/h2o6/1_h2o6_promax_cut_aligned_80_240.npz  --agent.resume True  --agent.load_run 2026-08-31_15-05-31 --agent.load_checkpoint model_300.pt  > logs/tasks/h2_o6/only_grab/pure_mt_v0.1.log 2>&1 &

python scripts/play.py Unitree-H2-Motion-Tracking-HandMass --motion-file src/assets/motion/h2/1_h2o6_new_s3_31.npz --checkpoint_file src/tasks/weight_lifter_H2/networks/experts/model_55900_2.pt --record-depth-path sim_depth_h2_3.npy

/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-21_16-31-23-H2-Camera/model_32000.pt
Unitree-H2-Pullup-A-Barbell-MT-Camera-Hand
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell --env.scene.num-envs=4096 --agent.resume True  --agent.load_run 2026-08-10_21-37-44 --agent.load_checkpoint model_12000.pt  > logs/tasks/h2/new_v1.1.log 2>&1 &

update history:
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-07_11-44-06/model_24300.pt

/home/tcs/8t01/humanoid/weightlifter/logs/g1_tracking/rsl_rl/2026-08-07_11-12-40/model_500.pt



python scripts/play.py Unitree-H2-Pullup-A-Barbell  --motion-file src/assets/motion/h2o6/1_h2o6_promax_cut_aligned.npz --checkpoint_file=/home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_camera_res/rsl_rl/2026-09-04_11-37-23/model_2300.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-22_14-09-33/model_32400.pt
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-Res --env.scene.num-envs=4096 --env.commands.motion.motion-file src/assets/motion/h2o6/11_h2o6_new_s6_fast.npz > logs/tasks/h2_o6/res/v0.log 2>&1 &

logs/g1_tracking/rsl_rl/2026-08-22_19-40-13/model_33000.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-17_16-32-37-promax-MT/model_14400.pt
python scripts/play.py Unitree-H2-Pullup-A-Barbell-Res --motion-file src/assets/motion/h2o6/11_h2o6_new_s6_fast.npz --checkpoint_file /home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_camera_res/rsl_rl/2026-09-04_11-37-23/model_2300.pt
python scripts/play.py Unitree-H2-Pullup-A-Barbell-MT-Camera-Hand --motion-file src/assets/motion/h2o6/1_h2o6_promax_cut_aligned_80_240.npz --checkpoint_file /data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-29_23-53-35-H2/model_37300.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-20_13-05-16/model_26400.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-17_17-42-50/model_6800.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-19_17-23-04/model_22200.pt

nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-Knee --env.scene.num-envs=4096 --agent.resume True  --agent.load_run 2026-08-20_17-50-12 --agent.load_checkpoint model_28300.pt --env.commands.motion.motion-file /data/weightlifter/src/assets/motion/h2o6/1_h2o6_promax.npz > logs/tasks/h2/new_camera/normal/v0.6.8_only_z.log.log 2>&1 &
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-20_14-31-34/model_26500.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-19_18-46-42/model_20000.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-09_15-23-45/model_38800.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-09_13-37-16/model_37700.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-09_00-53-55/model_36700.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-09_13-49-40/model_38200.pt
更新了drop from hand的终止条件
/home/tcs/8t01/humanoid/weightlifter/logs/g1_tracking/rsl_rl/2026-08-07_13-09-39/model_25000.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-09_22-12-56/model_38900.pt
--env.commands.motion.motion-file /home/tcs/8t01/humanoid/weightlifter/src/assets/motion/h2o6/1_h2o6_pro_legs.npz
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-Knee --env.scene.num-envs=4096 --env.commands.motion.motion-file /data/weightlifter/src/assets/motion/h2o6/1_h2o6_promax.npz  --agent.resume True  --agent.load_run 2026-08-18_14-29-17 --agent.load_checkpoint model_19500.pt  > logs/tasks/h2/new_camera/v0.5.2.log 2>&1 &

/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-18_14-29-17/model_19500.pt

python -u scripts/grid_search_depth_noise.py \
  --sim-path logs/depth/sim/sim_depth_h2_2_30_80.npy \
  --real-path logs/depth/real/filtered_depth_30_80.npy \
  --frame-stride 1 \
  --output-csv logs/depth/depth_noise_grid_search_30_80_fixed.csv \
  --output-best-npy logs/depth/sim_depth_h2_grid_best_30_80_fixed.np是

nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-No-Camera-Privi --env.scene.num-envs=4096 --env.commands.motion.motion-file src/assets/motion/h2o6/1_h2o6_new_s6.npz  > logs/tasks/h2_o6/priv_res/new_expert_v0.1.log 2>&1 &
/home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_privil/rsl_rl/2026-09-07_13-22-31/model_3300.pt
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-Res --env.scene.num-envs=4096 --env.commands.motion.motion-file /home/tcs/8t01/humanoid/weightlifter/src/assets/motion/h2o6/1_h2o6_new_s3_linear.npz --agent.resume True  --agent.load_run 2026-09-05_23-07-17 --agent.load_checkpoint model_24000.pt > logs/tasks/h2_o6/res/v0.4.1_27x48.log 2>&1 & 
 nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-No-Camera-Privi --env.scene.num-envs=4096 --env.commands.motion.motion-file src/assets/motion/h2o6/1_h2o6_new_s6.npz --agent.resume True  --agent.load_run 2026-09-08_15-16-29 --agent.load_checkpoint model_25300.pt  > logs/tasks/h2_o6/priv_res/v0.5.2.log 2>&1 &
 /home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_privil/rsl_rl/2026-09-08_15-16-29/model_25300.pt
/home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_privil/rsl_rl/2026-09-08_10-56-25/model_24100.pt
 /home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_privil/rsl_rl/2026-09-07_21-47-29/model_21600.pt
 logs/h2_tracking_privil/rsl_rl/2026-09-07_18-25-46/model_8800.pt
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-No-Camera-Experts --env.scene.num-envs=4096 --env.commands.motion.motion-file /home/tcs/8t01/humanoid/weightlifter/src/assets/motion/h2o6/1_h2o6_new_s3_linear.npz --agent.resume True  --agent.load_run base --agent.load_checkpoint model_15200.pt > logs/tasks/h2_o6/experts/v0.1.log 2>&1 & 

/home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_camera_res/rsl_rl/2026-09-05_13-50-30/model_15700.pt
nohup python scripts/watch_h2_training.py res \
> logs/tasks/h2_o6/res/watch_v0.1.log 2>&1 &
/home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_camera_res/rsl_rl/2026-09-05_23-07-17/model_24000.pt
[text](../../../logs/h2_tracking_camera_res/rsl_rl/2026-09-05_12-31-07/model_14000.pt)
python scripts/play.py Unitree-H2-Pullup-A-Barbell-No-Camera-Privi --motion-file src/assets/motion/h2o6/1_h2o6_new_s6.npz --checkpoint_file /home/tcs/8t01/humanoid/weightlifter/logs/h2_tracking_privil/rsl_rl/2026-09-08_10-56-25/model_2400.pt --record_depth_path  /home/tcs/8t01/humanoid/weightlifter/logs/tasks/h2_o6/camera.npy
export CUDA_VISIBLE_DEVICES=2

python scripts/play.py Unitree-H2-Motion-Tracking-HandMass --motion-file src/assets/motion/h2o6/1_h2o6_new_s6.npz  --checkpoint_file logs/h2_o6_mt/rsl_rl/2026-09-08_14-29-26/model_15800.pt