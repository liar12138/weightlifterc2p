
nohup python scripts/train.py Unitree-G1-Pick-A-Box --env.commands.motion.motion-file /home/tcs/8t01/capylab/src/assets/motion/pickbox/g1.npz --env.scene.num-envs=4096 > logs/tasks/pick_box/v0.3.log 2>&1 &

nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-Privil-MT --env.scene.num-envs=4096 > logs/tasks/h2/privil/mt_pri_v0.0.log 2>&1 &

nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-Privil --env.scene.num-envs=4096 --agent.resume True  --agent.load_run 2026-08-14_10-54-07 --agent.load_checkpoint model_7200.pt > logs/tasks/h2/privil/vnew_0.4.log 2>&1 &
python scripts/play.py Unitree-H2-Pullup-A-Barbell-Privil-MT --motion-file src/assets/motion/h2/1_h2o6_pro.npz --checkpoint_file=/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-14_11-06-52-G1-MT/model_4300.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-14_11-06-52-G1-MT/model_4300.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-12_22-03-03/model_3100.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-12_18-30-57/model_2200.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-12_19-29-10/model_2500.pt
export CUDA_VISIBLE_DEVICES=2
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-12_14-35-15/model_1200.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-12_13-40-10/model_1000.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-11_21-24-39/model_300.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-11_22-29-08/model_700.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-11_22-29-08/model_700.pt
python scripts/play.py Unitree-H2-Pullup-A-Barbell-Privil --motion-file src/assets/motion/h2/1_h2o6_pro.npz --checkpoint_file=/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-14_13-08-44/model_9500.pt
python scripts/play.py Unitree-H2-Pullup-A-Barbell-Privil --motion-file src/assets/motion/h2/1_h2o6_pro.npz --checkpoint_file=/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-14_10-54-07/model_7200.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-13_13-07-57/model_12900.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-13_16-26-13/model_4200.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-13_13-07-57/model_3600.pt
Unitree-H2-Pullup-A-Barbell-Privil
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-12_16-18-54/model_1700.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-12_17-27-30/model_1900.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-12_21-14-52/model_2700.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-12_21-47-53/model_2900.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-14_10-54-07/model_7200.pt
