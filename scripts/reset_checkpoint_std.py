import torch

path = "/home/tcs/ti90004t/humanoid/weightlifter/logs/h2_tracking/rsl_rl/2026-08-31_13-50-17/model_25700.pt"

ckpt = torch.load(path, map_location="cpu", weights_only=False)

for k, v in ckpt["actor_state_dict"].items():
    if "std_param" in k:
        if "log_" in k:      # std_type="log" 时是 log_std_param
            v.fill_(0.0)     # log(1.0)=0
        else:                # 你这个任务用 "scalar"，走这里
            v.fill_(1.0)
        print("reset:", k, "-> std=1.0")

torch.save(ckpt, path + 'reset.std')