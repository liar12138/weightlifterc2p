
```bash
cd /data/weightlifter
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/train.py Unitree-G1-O6-Pullup-A-Barbell \
  --env.scene.num-envs=2048 \
  --env.commands.motion.motion-file /data/weightlifter/src/assets/motion/g1/1_g1o6_pro.npz \
  > logs/tasks/g1_o6/v0.2.13.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
cd /data/weightlifter
nohup python scripts/train.py Unitree-G1-O6-Pullup-A-Barbell \
  --env.scene.num-envs=2048 \
  --env.commands.motion.motion-file /data/weightlifter/src/assets/motion/g1/1_g1o6_pro.npz \
  --agent.resume True \
  --agent.load_run 2026-08-13_12-01-17 \
  --agent.load_checkpoint model_8500.pt \
  > logs/tasks/g1_o6/v0.2.12_new_3.log 2>&1 &

python scripts/play.py Unitree-G1-O6-Pullup-A-Barbell \
  --viewer viser \
  --num_envs 1 \
  --checkpoint_file /data/weightlifter/logs/g1_tracking/rsl_rl/2026-08-13_12-01-17/model_8000.pt \
  --motion_file /data/weightlifter/src/assets/motion/g1/1_g1o6_pro.npz
```


