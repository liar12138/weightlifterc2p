"""FastSAC training runner for mjlab environments (single-GPU)."""

from __future__ import annotations

from typing import Any

from holosoma_min.agents.fast_sac.fast_sac_agent import FastSACAgent
from holosoma_min.envs.mjlab_env import MjlabFastSACEnv
from mjlab.envs import ManagerBasedRlEnv

from src.rl.fastsac.config import HolosomaFastSACRunnerCfg, to_fastsac_config
from src.rl.fastsac.training.reward_anomaly_logger import RewardAnomalyCfg


class FastSACRunner:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    train_cfg: HolosomaFastSACRunnerCfg,
    log_dir: str,
    device: str,
  ) -> None:
    self.train_cfg = train_cfg
    self.log_dir = log_dir
    self.device = device
    self.git_status_repos: list[str] = []

    mj_env = MjlabFastSACEnv(env)
    sac_cfg = to_fastsac_config(train_cfg)
    self.agent = FastSACAgent(
      mj_env,
      sac_cfg,
      device=device,
      log_dir=log_dir,
      multi_gpu_cfg=None,
    )
    self.agent.reward_anomaly_cfg = RewardAnomalyCfg.from_runner(train_cfg)
    self.agent.setup()
    self._disable_holosoma_onnx_export()

  def _disable_holosoma_onnx_export(self) -> None:
    def _noop_export(*_args, **_kwargs) -> None:
      return
    self.agent.export = _noop_export  # type: ignore[method-assign]

  def learn(self, num_learning_iterations: int | None = None) -> None:
    del num_learning_iterations
    self.agent.learn()

  def load(self, path: str) -> None:
    self.agent.load(path)

  def save(self, path: str, infos: Any = None) -> None:
    del infos
    self.agent.save(path)

  def add_git_repo_to_log(self, file_path: str) -> None:
    self.git_status_repos.append(file_path)

  def get_inference_policy(self, device: str | None = None):
    return self.agent.get_inference_policy(device=device)
