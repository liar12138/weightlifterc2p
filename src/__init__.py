# 这里用来导入所有我们自己的包，例如注册任务等，避免train.py文件中大量导入我们自己的东西。

import torch
# import TensorDict

# import rsl_rl.utils  # 先导入模块
# from rsl_rl import utils  # 也可以

# def check_nan_(obs, rewards, dones):
#     # 你的自定义实现（注意修正了原代码中的缩进和迭代问题）
#     for key, tensor in obs.items():
#         if isinstance(tensor, TensorDict):
#             for sub_key, sub_tensor in tensor.items():
#                 if torch.isnan(sub_tensor).any():
#                     raise ValueError(...)
#         elif torch.isnan(tensor).any():
#             raise ValueError(...)
#     if torch.isnan(rewards).any():
#         raise ValueError(...)
#     if torch.isnan(dones).any():
#         raise ValueError(...)

# def patch_check_nan():
#     # 替换模块属性
#     import rsl_rl.utils,sys
#     rsl_rl.utils.check_nan = check_nan_
#     if hasattr(rsl_rl, 'check_nan'):
#         rsl_rl.check_nan = check_nan_
    
#     # 强制替换所有已导入模块中的绑定
#     for module_name, module in list(sys.modules.items()):
#         if module is None:
#             continue
#         # 如果该模块有 check_nan 属性且指向旧函数（可选），则替换
#         if hasattr(module, 'check_nan'):
#             # 检查是否是我们想要替换的旧函数（避免误替换其他同名函数）
#             # 这里假设原始函数是 rsl_rl.utils.check_nan 的旧版本
#             # 可以添加更严格的检查：比如判断函数名或位置
#             setattr(module, 'check_nan', check_nan_)

# patch_check_nan()
from .tasks.weight_lifter import *
from .tasks.weight_lifter_G1 import *
from .tasks.weight_lifter_H2 import *
from .tasks.weight_lifter_H2_privil import *
from .tasks.weight_lifter_H2_blind import *
from .tasks.weight_lifter_H2_grab import *
from .tasks.velocity_tracking_H2 import *
from .tasks.motion_tracking_H2 import *
from .tasks.weight_lifter_G1_privil import *
from .rl.fastsac.tasks.mjlab_builtins import *  # noqa: F401
from .rl.fastsac.config import HolosomaFastSACRunnerCfg
from .rl.fastsac.log_paths import resolve_run_log_dir
from .rl.fastsac.registry_ext import assert_fastsac_task, load_fastsac_rl_cfg, load_fastsac_runner_cls
from .rl.fastsac.runners import FastSACRunner
from .envs.ManagerBasedRlEnvFinalObs import ManagerBasedRlEnvFinalObs
from .envs.nan_safe_env import NanSafeRlEnvFinalObs
