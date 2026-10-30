#!/usr/bin/env python3
"""Config-driven training keepalive: resume only after actor-NaN crashes.

Usage:
  python scripts/watch_g1_o6_resume.py --config scripts/watch_configs/v0.2.14.json

Rules:
- Poll every check_interval_sec (also check once on start).
- Resume only when the monitored log ends with the actor-NaN ValueError.
- Non-NaN process exit => stop watcher (leave running trains alone).
- Need resume but no eligible GPU => stop watcher (leave running trains alone).
- Multiple free GPUs: resume all needing jobs in priority order.
- One free GPU: only the highest-priority needing job.
- Already-running jobs: monitor only, do not relaunch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

NAN_NEEDLE = (
  "The observation group 'actor' returned by the environment contains NaN values"
)
RE_LOG_DIR = re.compile(
  r"Logging experiment in directory:\s*(/data/weightlifter/logs/g1_tracking/rsl_rl/\S+)"
)
RE_LOAD_CKPT = re.compile(
  r"Loading model checkpoint from:\s*(/data/weightlifter/logs/g1_tracking/rsl_rl/\S+/model_\d+\.pt)"
)
RE_MODEL = re.compile(r"model_(\d+)\.pt$")


@dataclass
class Job:
  key: str
  log_prefix: str
  task_id: str
  motion_file: str
  num_envs: int
  log_path: str
  run_dir: str
  load_run_hint: str
  next_log_index: int
  pid: Optional[int] = None
  cmdline_match: list[str] = field(default_factory=list)
  extra_args: list[str] = field(default_factory=list)


@dataclass
class WatchConfig:
  name: str
  root: Path
  log_dir: Path
  run_root: Path
  state_path: Path
  watch_log: Path
  python: str
  check_interval_sec: int
  min_free_mib: int
  gpu_release_wait_sec: int
  gpu_release_retries: int
  post_launch_wait_sec: int
  priority: list[str]
  jobs: dict[str, Job]


# Filled after --config load; used by log().
_WATCH_LOG: Optional[Path] = None


def now() -> str:
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
  line = f"[{now()}] {msg}"
  print(line, flush=True)
  if _WATCH_LOG is None:
    return
  _WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
  with _WATCH_LOG.open("a", encoding="utf-8") as f:
    f.write(line + "\n")


def _resolve(root: Path, p: str | Path) -> Path:
  path = Path(p)
  return path if path.is_absolute() else (root / path)


def load_config(path: Path) -> WatchConfig:
  raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
  root = Path(raw.get("root", "/data/weightlifter"))
  jobs: dict[str, Job] = {}
  for j in raw["jobs"]:
    key = j["key"]
    log_path = str(_resolve(root, j["log_path"]))
    jobs[key] = Job(
      key=key,
      log_prefix=j["log_prefix"],
      task_id=j["task_id"],
      motion_file=str(_resolve(root, j["motion_file"])),
      num_envs=int(j["num_envs"]),
      log_path=log_path,
      run_dir=j["run_dir"],
      load_run_hint=j.get("load_run_hint") or j["run_dir"],
      next_log_index=int(j.get("next_log_index", 2)),
      pid=j.get("pid"),
      cmdline_match=list(j.get("cmdline_match") or []),
      extra_args=list(j.get("extra_args") or []),
    )
  priority = list(raw.get("priority") or list(jobs.keys()))
  for k in jobs:
    if k not in priority:
      priority.append(k)
  return WatchConfig(
    name=raw.get("name", path.stem),
    root=root,
    log_dir=_resolve(root, raw.get("log_dir", "logs/tasks/g1_o6")),
    run_root=_resolve(root, raw.get("run_root", "logs/g1_tracking/rsl_rl")),
    state_path=_resolve(
      root, raw.get("state_file", f"logs/tasks/g1_o6/.watch_{path.stem}_state.json")
    ),
    watch_log=_resolve(
      root, raw.get("watch_log", f"logs/tasks/g1_o6/watch_{path.stem}.log")
    ),
    python=raw.get("python", "/opt/conda/envs/capylab/bin/python"),
    check_interval_sec=int(raw.get("check_interval_sec", 900)),
    min_free_mib=int(raw.get("min_free_mib", 17000)),
    gpu_release_wait_sec=int(raw.get("gpu_release_wait_sec", 30)),
    gpu_release_retries=int(raw.get("gpu_release_retries", 4)),
    post_launch_wait_sec=int(raw.get("post_launch_wait_sec", 45)),
    priority=priority,
    jobs=jobs,
  )


def save_state(cfg: WatchConfig, jobs: dict[str, Job]) -> None:
  payload = {k: asdict(v) for k, v in jobs.items()}
  cfg.state_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_state(cfg: WatchConfig) -> Optional[dict[str, Job]]:
  if not cfg.state_path.exists():
    return None
  raw = json.loads(cfg.state_path.read_text(encoding="utf-8"))
  out: dict[str, Job] = {}
  for k, v in raw.items():
    # Merge with template from config for fields that may be missing in old state.
    base = cfg.jobs.get(k)
    if base is None:
      continue
    merged = asdict(base)
    merged.update(v)
    out[k] = Job(**merged)
  return out or None


def next_free_log_index(cfg: WatchConfig, prefix: str, start_at: int) -> int:
  idx = max(start_at, 1)
  while True:
    p = cfg.log_dir / f"{prefix}_{idx}.log"
    if not p.exists() or p.stat().st_size == 0:
      return idx
    idx += 1


def parse_log_meta(log_path: Path, max_bytes: int = 256 * 1024) -> tuple[Optional[str], Optional[str]]:
  """Return (run_dir_basename, load_run_basename); scan head then tail."""
  if not log_path.exists():
    return None, None
  size = log_path.stat().st_size
  chunks: list[str] = []
  with log_path.open("rb") as f:
    chunks.append(f.read(min(size, max_bytes)).decode("utf-8", errors="ignore"))
    if size > max_bytes * 2:
      f.seek(size - max_bytes)
      chunks.append(f.read().decode("utf-8", errors="ignore"))
  text = "\n".join(chunks)
  run_dir = None
  load_run = None
  for m in RE_LOG_DIR.finditer(text):
    run_dir = Path(m.group(1)).name
  for m in RE_LOAD_CKPT.finditer(text):
    load_run = Path(m.group(1)).parent.name
  return run_dir, load_run


def log_ends_with_nan(log_path: Path, tail_bytes: int = 16384) -> bool:
  if not log_path.exists():
    return False
  size = log_path.stat().st_size
  with log_path.open("rb") as f:
    if size > tail_bytes:
      f.seek(size - tail_bytes)
    tail = f.read().decode("utf-8", errors="ignore")
  return NAN_NEEDLE in tail


def read_cmdline(pid: int) -> Optional[str]:
  try:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
  except FileNotFoundError:
    return None
  return raw.replace(b"\0", b" ").decode("utf-8", errors="ignore")


def is_target_train_cmdline(args: str, task_id: str) -> bool:
  if "train.py" not in args:
    return False
  if task_id not in args:
    return False
  # Exact task token: reject "...-Privil" when watching the non-Privil task.
  if re.search(rf"{re.escape(task_id)}-", args):
    return False
  return True


def pid_alive(pid: Optional[int], task_id: str) -> bool:
  if not pid:
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  cmdline = read_cmdline(pid)
  return cmdline is not None and is_target_train_cmdline(cmdline, task_id)


def find_train_processes(task_id: str) -> list[tuple[int, str]]:
  out: list[tuple[int, str]] = []
  try:
    ps = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
  except subprocess.CalledProcessError:
    return out
  for line in ps.splitlines():
    line = line.strip()
    if not line:
      continue
    parts = line.split(None, 1)
    if len(parts) != 2:
      continue
    pid_s, args = parts
    if not is_target_train_cmdline(args, task_id):
      continue
    try:
      out.append((int(pid_s), args))
    except ValueError:
      continue
  return out


def match_process(job: Job) -> Optional[int]:
  if pid_alive(job.pid, job.task_id):
    return job.pid
  hints = {h for h in (job.load_run_hint, job.run_dir, *job.cmdline_match) if h}
  for pid, args in find_train_processes(job.task_id):
    if any(h in args for h in hints):
      return pid
  return None


def latest_checkpoint(cfg: WatchConfig, run_dir: str) -> Optional[str]:
  d = cfg.run_root / run_dir
  if not d.is_dir():
    return None
  best_n = -1
  best_name = None
  for p in d.glob("model_*.pt"):
    m = RE_MODEL.search(p.name)
    if not m:
      continue
    n = int(m.group(1))
    if n > best_n:
      best_n = n
      best_name = p.name
  return best_name


def free_gpus(min_free_mib: int) -> list[int]:
  try:
    out = subprocess.check_output(
      [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
      ],
      text=True,
    )
  except (subprocess.CalledProcessError, FileNotFoundError) as e:
    log(f"nvidia-smi failed: {e}")
    return []
  free: list[int] = []
  for line in out.splitlines():
    line = line.strip()
    if not line:
      continue
    idx_s, free_s = [x.strip() for x in line.split(",")]
    if int(float(free_s)) >= min_free_mib:
      free.append(int(idx_s))
  return free


def wait_for_free_gpus(cfg: WatchConfig) -> list[int]:
  gpus: list[int] = []
  for i in range(cfg.gpu_release_retries):
    gpus = free_gpus(cfg.min_free_mib)
    if gpus:
      return gpus
    log(
      f"No GPU with >={cfg.min_free_mib} MiB free "
      f"(retry {i + 1}/{cfg.gpu_release_retries}, sleep {cfg.gpu_release_wait_sec}s)"
    )
    time.sleep(cfg.gpu_release_wait_sec)
  return free_gpus(cfg.min_free_mib)


def launch_resume(cfg: WatchConfig, job: Job, gpu: int) -> int:
  ckpt = latest_checkpoint(cfg, job.run_dir)
  if not ckpt:
    raise RuntimeError(f"{job.key}: no checkpoint in run_dir={job.run_dir}")

  idx = next_free_log_index(cfg, job.log_prefix, job.next_log_index)
  log_path = cfg.log_dir / f"{job.log_prefix}_{idx}.log"
  job.next_log_index = idx + 1
  job.log_path = str(log_path)
  job.load_run_hint = job.run_dir

  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = str(gpu)
  env["PATH"] = f"/opt/conda/envs/capylab/bin:{env.get('PATH', '')}"
  env["CONDA_DEFAULT_ENV"] = "capylab"
  env["CONDA_PREFIX"] = "/opt/conda/envs/capylab"

  cmd = [
    cfg.python,
    str(cfg.root / "scripts" / "train.py"),
    job.task_id,
    f"--env.scene.num-envs={job.num_envs}",
    "--env.commands.motion.motion-file",
    job.motion_file,
    "--agent.resume",
    "True",
    "--agent.load_run",
    job.run_dir,
    "--agent.load_checkpoint",
    ckpt,
    *job.extra_args,
  ]
  log(f"{job.key}: resume on GPU {gpu} from {job.run_dir}/{ckpt} -> {log_path.name}")
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("w", encoding="utf-8") as lf:
    proc = subprocess.Popen(
      cmd,
      cwd=str(cfg.root),
      env=env,
      stdout=lf,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  job.pid = proc.pid
  log(f"{job.key}: launched pid={job.pid}")

  deadline = time.time() + cfg.post_launch_wait_sec
  while time.time() < deadline:
    run_dir, load_run = parse_log_meta(log_path)
    if run_dir:
      job.run_dir = run_dir
      if load_run:
        job.load_run_hint = load_run
      log(f"{job.key}: new run_dir={job.run_dir} (load_run_hint={job.load_run_hint})")
      break
    if not pid_alive(job.pid, job.task_id):
      break
    time.sleep(2)
  else:
    log(f"{job.key}: warning: timed out waiting for Logging experiment line")
  return job.pid


def refresh_run_dir_from_log(job: Job) -> None:
  run_dir, load_run = parse_log_meta(Path(job.log_path))
  if run_dir:
    job.run_dir = run_dir
  if load_run:
    job.load_run_hint = load_run


def classify_job(job: Job) -> str:
  refresh_run_dir_from_log(job)
  pid = match_process(job)
  job.pid = pid
  if pid:
    return "running"
  if log_ends_with_nan(Path(job.log_path)):
    return "need_resume_nan"
  return "dead_other"


def bootstrap_from_config(cfg: WatchConfig) -> dict[str, Job]:
  jobs = {k: Job(**asdict(v)) for k, v in cfg.jobs.items()}
  for job in jobs.values():
    # Prefer meta from current log if present.
    run_dir, load_run = parse_log_meta(Path(job.log_path))
    if run_dir:
      job.run_dir = run_dir
    if load_run:
      job.load_run_hint = load_run
    pid = match_process(job)
    job.pid = pid
    status = f"pid={pid}" if pid else "not running"
    log(
      f"bootstrap {job.key}: log={Path(job.log_path).name} "
      f"run_dir={job.run_dir} load_hint={job.load_run_hint} {status}"
    )
  return jobs


def main(argv: Optional[list[str]] = None) -> int:
  global _WATCH_LOG
  parser = argparse.ArgumentParser(description="Config-driven NaN resume watcher")
  parser.add_argument(
    "--config",
    required=True,
    help="Path to watch JSON config (e.g. scripts/watch_configs/v0.2.14.json)",
  )
  parser.add_argument(
    "--force-bootstrap",
    action="store_true",
    help="Ignore existing state file and bootstrap from config",
  )
  args = parser.parse_args(argv)

  cfg = load_config(Path(args.config).resolve())
  _WATCH_LOG = cfg.watch_log
  cfg.log_dir.mkdir(parents=True, exist_ok=True)

  log(f"==== watch start name={cfg.name} config={args.config} ====")

  jobs: Optional[dict[str, Job]] = None
  if not args.force_bootstrap:
    jobs = load_state(cfg)
  if jobs is None:
    jobs = bootstrap_from_config(cfg)
  else:
    log(f"loaded state file: {cfg.state_path}")
    # Keep train hyperparams from config; keep runtime fields from state.
    for k, job in list(jobs.items()):
      tmpl = cfg.jobs[k]
      job.task_id = tmpl.task_id
      job.motion_file = tmpl.motion_file
      job.num_envs = tmpl.num_envs
      job.log_prefix = tmpl.log_prefix
      job.cmdline_match = list(tmpl.cmdline_match)
      job.extra_args = list(tmpl.extra_args)
      pid = match_process(job)
      job.pid = pid
      refresh_run_dir_from_log(job)
      log(
        f"state {job.key}: log={Path(job.log_path).name} "
        f"run_dir={job.run_dir} pid={pid}"
      )
  save_state(cfg, jobs)

  while True:
    statuses = {k: classify_job(j) for k, j in jobs.items()}
    save_state(cfg, jobs)
    for k, st in statuses.items():
      log(
        f"check {k}: {st} (log={Path(jobs[k].log_path).name}, run_dir={jobs[k].run_dir})"
      )

    for k, st in statuses.items():
      if st == "dead_other":
        log(
          f"{k} exited without NaN marker in {jobs[k].log_path}; "
          "stopping watcher (training processes left untouched)"
        )
        return 1

    need = [k for k, st in statuses.items() if st == "need_resume_nan"]
    if need:
      order = [k for k in cfg.priority if k in need]
      gpus = wait_for_free_gpus(cfg)
      log(f"free GPUs (>= {cfg.min_free_mib} MiB): {gpus}")
      if not gpus:
        log("need resume but no eligible GPU; stopping watcher")
        return 2

      if len(gpus) < len(order):
        log(
          f"only {len(gpus)} eligible GPU(s); "
          f"prioritizing {order[: len(gpus)]} over {order[len(gpus) :]}"
        )
        order = order[: len(gpus)]

      for job_key, gpu in zip(order, gpus):
        try:
          launch_resume(cfg, jobs[job_key], gpu)
        except Exception as e:
          log(f"{job_key}: launch failed: {e}")
          return 3
        save_state(cfg, jobs)
        time.sleep(5)

    log(f"sleep {cfg.check_interval_sec}s until next check")
    time.sleep(cfg.check_interval_sec)


if __name__ == "__main__":
  try:
    sys.exit(main())
  except KeyboardInterrupt:
    log("interrupted by user; exiting")
    sys.exit(130)
