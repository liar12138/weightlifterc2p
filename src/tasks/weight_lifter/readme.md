
python scripts/train.py Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation --env.commands.motion.motion-file /home/tcs/8t01/capylab/src/assets/motion/pickbox/g1.npz

python scripts/play.py Unitree-G1-Pick-A-Box --motion-file /home/tcs/8t01/capylab/src/assets/motion/pickbox/g1.npz --checkpoint_file=/home/tcs/8t01/capylab/logs/rsl_rl/g1_tracking/2026-07-05_11-06-58/model_32000.pt

nohup python scripts/train.py Unitree-G1-Pick-A-Box --env.commands.motion.motion-file /home/tcs/8t01/capylab/src/assets/motion/pickbox/g1.npz --env.scene.num-envs=4096 > logs/tasks/pick_box/v0.3.log 2>&1 &

nohup python scripts/train.py Unitree-G1-Pick-A-Box --env.commands.motion.motion-file /home/tcs/8t01/capylab/src/assets/motion/pickbox/g1.npz --env.scene.num-envs=4096 --agent.resume True  --agent.load_run 2026-07-04_14-50-12 --agent.load_checkpoint model_29999.pt > logs/tasks/pick_box/v0.3.log 2>&1 &

nohup python scripts/train.py Unitree-G1-Pullup-A-Barbell --env.scene.num-envs=8192  --agent.resume True  --agent.load_run 2026-09-02_10-01-32 > logs/v0.3.9.log 2>&1 &

nohup python scripts/train.py Unitree-G1-Pullup-A-Barbell-RNN --env.scene.num-envs=4096  > logs/v0.1.1.log 2>&1 &

nohup python scripts/train.py Unitree-G1-Dex3-Flat-Tracking --env.scene.num-envs=12880  --env.commands.motion.motion-file src/assets/motion/g1/1_g1dex3_pro.npz > logs/v0.g1.dex3.mt.ct.log 2>&1 &

python scripts/play.py Unitree-G1-Pullup-A-Barbell --motion-file src/assets/motion/g1/1_g1dex3_pro.npz --checkpoint_file=/data/weightlifter/logs/g1_tracking/rsl_rl/2026-07-29_13-27-46/model_500.pt

python scripts/play.py Unitree-G1-Dex3-Flat-Tracking --motion-file src/assets/motion/g1/1_g1dex3_pro.npz --checkpoint_file=/home/tcs/8t01/humanoid/weightlifter/logs/g1_tracking/rsl_rl/2026-08-30_11-02-42/model_21500.pt



export CUDA_VISIBLE_DEVICES=2

nohup python scripts/train.py Unitree-G1-Pullup-A-Barbell --env.scene.num-envs=12880  --env.commands.motion.motion-file /home/tcs/8t01/humanoid/weightlifter/src/assets/motion/g1/1_g1dex3_pro.npz --agent.resume True  --agent.load_run 2026-09-03_15-15-51 > logs/v0.g1.dex3.pb.log 2>&1 &

nohup python scripts/train.py Unitree-G1-Pullup-A-Barbell --env.scene.num-envs=8192  --env.commands.motion.motion-file /home/tcs/8t01/humanoid/weightlifter/src/assets/motion/g1/1_g1dex3_pro.npz > logs/v0.g1.dex3.pb.exp.log 2>&1 &

python scripts/play.py Unitree-G1-Pullup-A-Barbell --motion-file /home/tcs/8t01/humanoid/weightlifter/src/assets/motion/g1/1_g1dex3_pro.npz --checkpoint_file=/home/tcs/8t01/humanoid/weightlifter/logs/g1_dex3_pb/rsl_rl/2026-09-01_10-04-27/model_0.pt

update history:

更新了drop from hand的终止条件