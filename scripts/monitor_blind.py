#!/usr/bin/env python3
"""监控 Blind 训练日志,检测到 NaN 崩溃后自动重启训练。

每次重启都会生成一个带时间戳的新日志文件(不覆盖旧日志),
并使用最新的、包含 checkpoint 的运行目录进行 resume。
"""
import os
import subprocess
import time
from pathlib import Path

# ==================== 可配置参数 ====================
ROOT = Path(__file__).resolve().parent.parent            # 项目根目录
LOG_DIR = ROOT / "logs" / "g1_tracking"                  # 训练 stdout 日志目录
RUNS_DIR = LOG_DIR / "rsl_rl"                            # 训练运行目录(resume 用)
INITIAL_LOG = LOG_DIR / "blind_v0.0.log"                 # 首次要监控的日志文件
LOG_PREFIX = "blind"                                     # 重启日志文件名前缀

TASK = "Unitree-H2-Pullup-A-Barbell-Blind"
NUM_ENVS = "8192"
LOGGER = "tensorboard"
PYTHON = "/root/miniconda3/envs/capylab/bin/python"      # capylab 环境的 python

# rsl_rl 检测到 NaN 时的报错关键字(覆盖 actor/critic/rewards/dones)
ERROR_PATTERN = "returned by the environment contain"
WATCH_INTERVAL = 5                                       # 轮询间隔(秒)
# ====================================================


def tail_bytes(path: Path, n: int = 8192) -> str:
    """读取文件末尾 n 字节,以文本返回(容忍未解码字节)。"""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def find_latest_runnable_dir() -> str | None:
    """返回最新且包含 checkpoint 的运行目录名(用于 --agent.load_run)。

    checkpoint 文件名为 model_*.pt;若没有任何目录包含 checkpoint 则返回 None。
    """
    if not RUNS_DIR.exists():
        return None
    candidates = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        if any(f.name.endswith(".pt") for f in d.iterdir()):
            candidates.append(d)
    if not candidates:
        return None
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    return latest.name


def make_new_log() -> Path:
    """生成一个带时间戳、且不与现有文件重名的新日志文件路径。

    正常情况下时间戳(精确到秒)已足够区分;若同一秒内多次重启导致重名,
    则追加 ``_2``、``_3``... 后缀,保证绝不覆盖旧日志。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{LOG_PREFIX}_{ts}.log"
    idx = 1
    while path.exists():
        idx += 1
        path = LOG_DIR / f"{LOG_PREFIX}_{ts}_{idx}.log"
    return path


def launch_training(run_dir: str | None, log_file: Path) -> None:
    """启动训练;run_dir 为 None 时从头训练,否则 resume。"""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "8"

    cmd = [
        PYTHON, "scripts/train.py", TASK,
        f"--env.scene.num-envs={NUM_ENVS}",
        f"--agent.logger={LOGGER}",
    ]
    if run_dir is not None:
        cmd += ["--agent.resume", "True", "--agent.load_run", run_dir]

    with open(log_file, "wb") as out:
        subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env)

    resume_info = f"resume 自 {run_dir}" if run_dir else "从头训练"
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 重启训练({resume_info}) -> 日志 {log_file}",
        flush=True,
    )


def main() -> None:
    current_log: Path | None = INITIAL_LOG if INITIAL_LOG.exists() else None
    if current_log is None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 等待初始日志 {INITIAL_LOG} 出现...", flush=True)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始监控,错误模式: {ERROR_PATTERN!r}", flush=True)

    while True:
        try:
            # 初始日志还没出现(比如当前训练还没启动)则继续等待。
            if current_log is None:
                if INITIAL_LOG.exists():
                    current_log = INITIAL_LOG
                    print(f"[{time.strftime('%H:%M:%S')}] 开始监控 {current_log}", flush=True)
                else:
                    time.sleep(WATCH_INTERVAL)
                    continue

            content = tail_bytes(current_log)
            if ERROR_PATTERN in content:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测到 NaN 错误!", flush=True)
                run_dir = find_latest_runnable_dir()
                new_log = make_new_log()
                launch_training(run_dir, new_log)
                # 下次监控新的日志文件。
                current_log = new_log

            time.sleep(WATCH_INTERVAL)
        except Exception as e:  # noqa: BLE001
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 异常: {e},5s 后重试", flush=True)
            time.sleep(WATCH_INTERVAL)


if __name__ == "__main__":
    main()
