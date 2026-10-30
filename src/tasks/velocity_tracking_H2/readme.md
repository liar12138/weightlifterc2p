export CUDA_VISIBLE_DEVICES=0
nohup python scripts/train.py Velocity-Tracking-H2 --env.scene.num-envs=4096 > logs/tasks/h2/velocity/v0.1.log 2>&1 &
/data/weightlifter/logs/g1_velocity/rsl_rl/2026-08-11_22-19-55/model_12950.pt
python scripts/play.py Velocity-Tracking-H2  --checkpoint_file=/data/weightlifter/logs/g1_velocity/rsl_rl/2026-08-12_16-38-34/model_8750.pt
/data/weightlifter/logs/g1_velocity/rsl_rl/2026-08-12_16-38-34/model_8750.pt
/data/weightlifter/logs/g1_velocity/rsl_rl/2026-08-12_13-17-31/model_850.pt
nohup python scripts/train.py Velocity-Tracking-H2 --env.scene.num-envs=4096 --agent.resume True  --agent.load_run 2026-08-12_13-17-31 --agent.load_checkpoint model_850.pt > logs/tasks/h2/privil/v0.0.3.1.log 2>&1 &
nohup python scripts/train.py Velocity-Tracking-Fine-H2 --env.scene.num-envs=4096 > logs/tasks/h2/velocity/defaut_pos.log 2>&1 &
python scripts/play.py Velocity-Tracking-Fine-H2  --checkpoint_file=logs/g1_velocity/rsl_rl/2026-08-14_18-17-06/model_19500.pt
python scripts/play.py Velocity-Tracking-H2  --checkpoint_file=logs/g1_velocity/rsl_rl/2026-08-15_20-26-57/model_4650.pt

python scripts/play.py Velocity-Tracking-H2  --checkpoint_file=logs/g1_velocity/rsl_rl/2026-08-29_17-15-37/model_59900.pt

nohup python scripts/train.py Velocity-Tracking-H2 --env.scene.num-envs=12880 > logs/h2.vel.ct.pose.log 2>&1 &
