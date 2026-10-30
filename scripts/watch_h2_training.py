#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import time
from pathlib import Path


# ============================================================
# 项目根目录
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PYTHON = "python"

WATCH_INTERVAL = 5

ERROR_PATTERN = "returned by the environment contain"

MOTION_FILE = (
    "src/assets/motion/h2o6/"
    "1_h2o6_promax_cut_aligned_80_240.npz"
)


# ============================================================
# 两个训练任务分别配置
# ============================================================

CONFIGS = {

    # --------------------------------------------------------
    # 模型 1
    #
    # 你启动时：
    #
    # Unitree-H2-Pullup-A-Barbell-Knee
    # num_envs=4096
    # GPU 0
    # --------------------------------------------------------

    "normal": {
        "task": "Unitree-H2-Pullup-A-Barbell-Knee",

        "num_envs": "4096",

        "cuda_visible_devices": "1",

        "initial_log": (
            ROOT
            / "logs/tasks/h2/new_camera/normal/grab_not_only_v0.0.log"
        ),

        "log_dir": (
            ROOT
            / "logs"
            / "tasks"
            / "h2"
            / "new_camera"
            / "normal"
        ),

        "log_prefix": "grab_not_only_v0.auto",
        "motion_file": (
            "src/assets/motion/h2o6/"
            "1_h2o6_promax_cut_aligned_80_240.npz"
        ),
    },


    # --------------------------------------------------------
    # 模型 2
    #
    # Unitree-H2-Pullup-A-Barbell-Grab
    # num_envs=2048
    # GPU 1
    # --------------------------------------------------------

    "grab": {
        "task": "Unitree-H2-Pullup-A-Barbell-Grab",

        "num_envs": "4096",

        # 如果这个 grab 任务跑在物理 GPU 1，就保持 1
        "cuda_visible_devices": "0",

        "initial_log": (
            ROOT
            / "logs"
            / "tasks"
            / "h2"
            / "new_camera"
            / "grab"
            / "v0.1.3.log"
        ),

        "log_dir": (
            ROOT
            / "logs"
            / "tasks"
            / "h2"
            / "new_camera"
            / "grab"
        ),

        "log_prefix": "v0.1.3.auto",
        "motion_file": (
            "src/assets/motion/h2o6/"
            "1_h2o6_promax_cut_aligned_80_240.npz"
        ),
    },
    "res": {
        "task": "Unitree-H2-Pullup-A-Barbell-Res",

        "num_envs": "4096",

        # 改成你实际训练使用的 GPU
        "cuda_visible_devices": "2",

        "initial_log": (
            ROOT
            / "logs"
            / "tasks"
            / "h2_o6"
            / "res"
            / "v0.1.log"
        ),

        "log_dir": (
            ROOT
            / "logs"
            / "tasks"
            / "h2_o6"
            / "res"
        ),

        "log_prefix": "v0.1.auto",

        "motion_file": (
            "/home/tcs/8t01/humanoid/weightlifter/"
            "src/assets/motion/h2o6/"
            "1_h2o6_new_s3_linear.npz"
        ),
    },
}


# ============================================================
# 读取日志末尾
# ============================================================

def tail_bytes(path: Path, n: int = 1024 * 1024) -> str:
    """
    读取日志末尾最多 n 字节。

    这里设成 1MB，
    因为 Logging experiment in directory 一般出现在日志开头，
    日志不太大时可以直接解析到。
    """

    try:

        with open(path, "rb") as f:

            f.seek(
                0,
                os.SEEK_END,
            )

            size = f.tell()

            f.seek(
                max(0, size - n)
            )

            return f.read().decode(
                "utf-8",
                errors="ignore",
            )

    except FileNotFoundError:

        return ""


# ============================================================
# 从当前 stdout log 中解析对应的 rsl_rl run 目录
# ============================================================

def get_run_dir_from_log(log_file: Path) -> Path | None:

    if not log_file.exists():
        return None

    try:

        # 为了保证能找到日志开头的：
        #
        # [INFO] Logging experiment in directory: xxx
        #
        # 直接读取整个文件。
        text = log_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception as e:

        print(
            f"[WARNING] 读取日志失败: {e}",
            flush=True,
        )

        return None


    pattern = (
        r"\[INFO\]\s*Logging experiment in directory:\s*(.+)"
    )

    matches = re.findall(
        pattern,
        text,
    )

    if not matches:
        return None


    # 如果日志里意外出现多条，取最后一条
    run_path = matches[-1].strip()

    return Path(run_path)


# ============================================================
# 找这个 run 里面最新的 model_xxx.pt
# ============================================================

def get_latest_checkpoint(
    run_dir: Path,
) -> Path | None:

    if not run_dir.exists():
        return None


    checkpoints = []

    for path in run_dir.glob("model_*.pt"):

        match = re.fullmatch(
            r"model_(\d+)\.pt",
            path.name,
        )

        if match is None:
            continue

        iteration = int(
            match.group(1)
        )

        checkpoints.append(
            (
                iteration,
                path,
            )
        )


    if not checkpoints:
        return None


    # 按 iteration 最大值选择
    #
    # model_32000 > model_31000
    #
    # 比单纯看修改时间更可靠
    checkpoints.sort(
        key=lambda x: x[0]
    )

    return checkpoints[-1][1]


# ============================================================
# 自动生成新的 stdout log
# ============================================================

def make_new_log(config) -> Path:

    log_dir = config["log_dir"]

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ts = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        log_dir
        / f"{config['log_prefix']}_{ts}.log"
    )


    idx = 1

    while path.exists():

        idx += 1

        path = (
            log_dir
            / f"{config['log_prefix']}_{ts}_{idx}.log"
        )

    return path


# ============================================================
# 重启训练
# ============================================================

def launch_training(
    config,
    run_dir: Path,
    checkpoint: Path,
    new_log: Path,
):

    env = dict(
        os.environ
    )

    # ----------------------------------------
    # 非常重要
    #
    # normal 使用物理 GPU 0
    # grab   使用物理 GPU 1
    # ----------------------------------------

    env["CUDA_VISIBLE_DEVICES"] = (
        config["cuda_visible_devices"]
    )

    env["OMP_NUM_THREADS"] = "8"


    cmd = [
        PYTHON,

        "scripts/train.py",

        config["task"],

        f"--env.scene.num-envs={config['num_envs']}",

        "--env.commands.motion.motion-file",
        config["motion_file"],

        "--agent.resume",
        "True",

        "--agent.load_run",
        run_dir.name,

        "--agent.load_checkpoint",
        checkpoint.name,
    ]


    with open(
        new_log,
        "wb",
    ) as out:

        subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )


    print(
        "\n"
        "========================================\n"
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n"
        "自动重启训练\n"
        f"TASK       : {config['task']}\n"
        f"GPU        : {config['cuda_visible_devices']}\n"
        f"RUN        : {run_dir.name}\n"
        f"CHECKPOINT : {checkpoint.name}\n"
        f"LOG        : {new_log}\n"
        "========================================",
        flush=True,
    )


# ============================================================
# 主循环
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "profile",
        choices=[
            "normal",
            "grab",
            "res",
        ],
    )

    args = parser.parse_args()

    config = CONFIGS[
        args.profile
    ]


    current_log = config[
        "initial_log"
    ]


    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"启动 {args.profile} 监控",
        flush=True,
    )

    print(
        f"GPU: {config['cuda_visible_devices']}",
        flush=True,
    )

    print(
        f"监控日志: {current_log}",
        flush=True,
    )

    print(
        f"错误模式: {ERROR_PATTERN!r}",
        flush=True,
    )


    while True:

        try:

            # ================================================
            # 日志还没出现
            # ================================================

            if not current_log.exists():

                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"等待日志出现: {current_log}",
                    flush=True,
                )

                time.sleep(
                    WATCH_INTERVAL
                )

                continue


            # ================================================
            # 检查日志末尾是否出现 NaN
            # ================================================

            content = tail_bytes(
                current_log,
                16384,
            )


            if ERROR_PATTERN not in content:

                time.sleep(
                    WATCH_INTERVAL
                )

                continue


            # ================================================
            # 发现 NaN
            # ================================================

            print(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[{args.profile}] 检测到 NaN！",
                flush=True,
            )


            # ================================================
            # 从这个模型自己的 stdout log 获取 run dir
            # ================================================

            run_dir = get_run_dir_from_log(
                current_log
            )


            if run_dir is None:

                print(
                    "[ERROR] 无法从日志中找到：\n"
                    "Logging experiment in directory",
                    flush=True,
                )

                time.sleep(
                    WATCH_INTERVAL
                )

                continue


            print(
                f"[INFO] 当前 run: {run_dir}",
                flush=True,
            )


            # ================================================
            # 只在这个 run 里面找 checkpoint
            # ================================================

            checkpoint = get_latest_checkpoint(
                run_dir
            )


            if checkpoint is None:

                print(
                    f"[ERROR] {run_dir} "
                    f"没有找到 model_*.pt",
                    flush=True,
                )

                time.sleep(
                    WATCH_INTERVAL
                )

                continue


            print(
                f"[INFO] 最新 checkpoint: "
                f"{checkpoint.name}",
                flush=True,
            )


            # ================================================
            # 新 stdout 日志
            # ================================================

            new_log = make_new_log(
                config
            )


            # ================================================
            # 重启
            # ================================================

            launch_training(
                config,
                run_dir,
                checkpoint,
                new_log,
            )


            # ================================================
            # 从现在开始监控新的 stdout log
            #
            # 新训练启动后，它又会输出：
            #
            # Logging experiment in directory: ...
            #
            # 下一次崩溃时就会解析这个新的 run。
            # ================================================

            current_log = new_log


            # 给 MuJoCo / CUDA / env 一点启动时间
            time.sleep(30)


        except Exception as e:

            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"监控异常: {e}",
                flush=True,
            )

            time.sleep(
                WATCH_INTERVAL
            )


if __name__ == "__main__":
    main()