
```bash
nohup python scripts/train.py Unitree-G1-O6-Pullup-A-Barbell --env.scene.num-envs=4096 --env.commands.motion.motion-file /data/weightlifter/src/assets/motion/g1/1_g1o6_pro.npz > logs/tasks/g1_o6/v0.1.log 2>&1 &

nohup python scripts/train.py Unitree-G1-O6-Pullup-A-Barbell-Pure-MT-Hand-Event --env.scene.num-envs=4096 > logs/tasks/g1_o6/pure_mt/add_hand_events/v0.0_no_camera_35.log 2>&1 &
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-18_11-13-37/model_100.pt
python scripts/play.py Unitree-G1-O6-Pullup-A-Barbell-Pure-MT-Hand-Event --checkpoint_file=/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-18_14-11-09/model_1200.pt --motion_file=/data/weightlifter/src/assets/motion/g1/1_g1o6_pro.npz
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-15_15-10-57/model_9500.pt
export CUDA_VISIBLE_DEVICES=2
nohup python scripts/train.py Unitree-G1-O6-Pullup-A-Barbell-Pure-MT-Hand-Event --env.scene.num-envs=4096 --agent.resume True  --agent.load_run 2026-08-18_13-58-02 --agent.load_checkpoint model_200.pt --env.commands.motion.motion-file /data/weightlifter/src/assets/motion/g1/1_g1o6_promax.npz > logs/tasks/g1_o6/pure_mt/add_hand_events/v0.1_no_camera.log 2>&1 &
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-17_17-21-46/model_14200.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-15_14-26-59/model_9100.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-06_21-01-31/model_200.pt
Unitree-G1-O6-Pullup-A-Barbell-MT
python scripts/play.py  Unitree-G1-O6-Pullup-A-Barbell-Pure-MT-Hand-Event --checkpoint_file=/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-18_18-23-01/model_11400.pt --motion_file=/data/weightlifter/src/assets/motion/g1/1_g1o6_pro.npz
nohup python scripts/train.py Unitree-G1-O6-Pullup-A-Barbell-Pure-MT-Hand-Event --env.scene.num-envs=4096 --agent.resume True --agent.load_run 2026-08-18_15-41-24 --agent.load_checkpoint model_1300.pt > logs/tasks/g1_o6/pure_mt/add_hand_events/v0.1_no_camera.log 2>&1
```

nohup  python scripts/train.py Unitree-G1-1307-Stage-I   --motion-file src/assets/motions/g1/42taiji.npz   --env.scene.num-envs 8192   --env.commands.motion.sampling-mode adaptive --agent.load_checkpoint=/home/tcs/8t01/KungFuAthleteBot-main/unitree_rl_mjlab/models/1307/1307.pt > unitree_rl_mjlab/logs/42.sI.log 2>&1 &