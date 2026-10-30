#!/usr/bin/env python3
import os
import time
import subprocess
import sys

LOG_PATH = "logs/v0.g1.dex3.pb.exp.log"
RUN_BASE = "logs/g1_dex3_pb/rsl_rl"
TRIGGER_STR = "The observation group 'actor' returned by the environment contains NaN values."
# 训练命令（不含 --agent.load-run）
CMD_TEMPLATE = [
    "python", "scripts/train.py",
    "Unitree-G1-Pullup-A-Barbell",
    "--env.scene.num-envs=8192",
    "--env.commands.motion.motion-file",
    "/home/tcs/8t01/humanoid/weightlifter/src/assets/motion/g1/1_g1dex3_pro.npz",
    "--agent.resume", "True",
]

def get_latest_run_dir():
    """返回 RUN_BASE 下最新子目录的完整路径，若无则返回 None"""
    if not os.path.isdir(RUN_BASE):
        return None
    dirs = [os.path.join(RUN_BASE, d) for d in os.listdir(RUN_BASE)
            if os.path.isdir(os.path.join(RUN_BASE, d))]
    if not dirs:
        return None
    latest = max(dirs, key=os.path.getmtime)
    return latest

def start_training(run_dir):
    """启动训练进程，返回 Popen 对象，并设置 CUDA_VISIBLE_DEVICES=2"""
    # get_checkpoint_path 用 re.match 匹配的是目录名，不能传完整路径，需取 basename
    cmd = CMD_TEMPLATE + ["--agent.load-run", os.path.basename(run_dir)]
    log_file = open(LOG_PATH, 'w')
    # 为子进程独立设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    log_file.close()
    return proc

def monitor():
    current_proc = None
    last_pos = 0  # 已读到的字节偏移；0 表示从文件头开始（能命中启动前已存在的触发行）
    print(f"Monitoring {LOG_PATH} for trigger: '{TRIGGER_STR}'")

    while True:
        try:
            with open(LOG_PATH, 'r') as f:
                # 首次从文件头扫描，之后从上次位置继续
                f.seek(min(last_pos, os.path.getsize(LOG_PATH)))
                while True:
                    line = f.readline()
                    if line:
                        if TRIGGER_STR in line:
                            print(f"[{time.ctime()}] Trigger detected, restarting...")
                            if current_proc is not None and current_proc.poll() is None:
                                current_proc.terminate()
                                try:
                                    current_proc.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    current_proc.kill()
                                current_proc = None

                            latest_run = get_latest_run_dir()
                            if latest_run is None:
                                print("No run directory found, skip restart")
                            else:
                                print(f"Using latest run: {latest_run}")
                                current_proc = start_training(latest_run)
                                print(f"New training process started (PID: {current_proc.pid})")
                            # start_training 会用 'w' 截断日志，重置偏移避免同一行反复触发
                            last_pos = 0
                            break
                    else:
                        # 读到文件末尾：记录位置，等待新内容写入
                        last_pos = f.tell()
                        if os.path.getsize(LOG_PATH) < last_pos:
                            print("Log file truncated, reopening...")
                            last_pos = 0
                            break
                        time.sleep(0.5)
        except FileNotFoundError:
            time.sleep(1)
        except KeyboardInterrupt:
            print("Monitor stopped by user.")
            if current_proc is not None and current_proc.poll() is None:
                current_proc.terminate()
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    monitor()