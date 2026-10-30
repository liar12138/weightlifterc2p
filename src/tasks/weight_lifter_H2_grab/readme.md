export CUDA_VISIBLE_DEVICES=2
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-Grab --env.scene.num-envs=4096 --env.commands.motion.motion-file src/assets/motion/h2o6/1_h2o6_promax_cut_aligned.npz --agent.resume True  --agent.load_run 2026-08-22_17-10-09 --agent.load_checkpoint model_3000.pt > logs/tasks/h2/new_camera/grab/v0.1.4.log 2>&1 &
logs/g1_tracking/rsl_rl/2026-08-22_01-39-25/model_convert.pt
/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-22_17-10-09/model_3000.pt
python scripts/play.py Unitree-H2-Pullup-A-Barbell-Grab --motion-file src/assets/motion/h2o6/1_h2o6_promax_cut_aligned.npz --checkpoint_file=/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-22_17-10-09/model_3000.pt
logs/g1_tracking/rsl_rl/2026-08-22_14-10-50/model_convert333.pt
python scripts/play.py Unitree-H2-Pullup-A-Barbell-Grab --motion-file src/assets/motion/h2o6/1_h2o6_promax_cut_aligned.npz --chec
kpoint_file=/data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-22_10-59-05/model_7000.pt 
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-Grab --env.scene.num-envs=4096 --env.commands.motion.motion-file /data/weightlifter/src/assets/motion/h2/1_h2o6_promax_cut.npz  > logs/tasks/h2_grab/v0.0.log 2>&1 &