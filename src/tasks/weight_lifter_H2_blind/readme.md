export CUDA_VISIBLE_DEVICES=2
nohup python scripts/train.py Unitree-H2-Pullup-A-Barbell-Blind --env.scene.num-envs=8192 --agent.logger tensorboard > logs/g1_tracking/blind_v0.0.log 2>&1 &

python scripts/train.py Unitree-H2-Pullup-A-Barbell-Blind --env.scene.num-envs=8192 \
--agent.resume True --agent.load_run <run目录名> --agent.load_checkpoint model_XXXX.pt