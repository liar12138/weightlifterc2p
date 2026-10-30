"""ManagerBasedRlEnvFinalObs 的 NaN 安全子类（用于 weight_lifter 举重/杠铃任务）。

设计目标（不改动 mjlab / rsl_rl 的任何类）：
  1. 继承 ``ManagerBasedRlEnvFinalObs`` 并重写 ``step()``；
  2. 沿用基类 step 的完整逻辑，只在“结束/奖励计算”处插入 NaN 保护；
  3. 一旦某个 env 的物理量 qpos/qvel 出现 NaN/Inf（物理发散），就：
       - 本步奖励置 0（避免把 NaN / 垃圾 reward 写进 PPO 缓存）；
       - 标记 done/terminated=True 且非 time_out（不会被 bootstrap，
         等价于把发散这一条 transition 从 PPO 学习中剔除）；
       - 立刻 ``_reset_idx`` 硬重置并返回干净的 reset 后 obs。

NanGuard（``cfg.sim.nan_guard``）仍负责在检测到 NaN 时 dump 取证，
本类负责“兜底不崩”，二者互不干扰。
"""

from __future__ import annotations

import torch

from .ManagerBasedRlEnvFinalObs import ManagerBasedRlEnvFinalObs


class NanSafeRlEnvFinalObs(ManagerBasedRlEnvFinalObs):
  """在 ManagerBasedRlEnvFinalObs 基础上加入“NaN env 硬重置 + 不进缓存”保护。"""

  def _nan_env_ids(self) -> torch.Tensor:
    """返回 qpos / qvel 存在 NaN 或 Inf 的 env id（可能为空）。

    说明：robot 实体持有的 mujoco_warp Data 是全模型共享的，
    ``data.qpos / data.qvel`` 已包含 robot + barbell 等所有实体。
    """
    data = self.scene["robot"].data.data  # mujoco_warp.Data
    bad = torch.logical_or(
      ~torch.isfinite(data.qpos).all(dim=-1),
      ~torch.isfinite(data.qvel).all(dim=-1),
    )
    return bad.nonzero(as_tuple=False).squeeze(-1)

  def step(self, action: torch.Tensor):
    # ------------------------------------------------------------------
    # 与基类 ManagerBasedRlEnvFinalObs.step 一致的开头。
    # ------------------------------------------------------------------
    if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
      pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
      raise RuntimeError(
        f"Environments {pending_ids.cpu().tolist()} must be reset via "
        "reset(env_ids=...) before calling step() again when auto_reset=False."
      )

    self.extras["log"] = dict()
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # ------------------------------------------------------------------
    # NaN 保护：物理量(qpos/qvel)非有限的 env。
    # ------------------------------------------------------------------
    nan_ids = self._nan_env_ids()
    if len(nan_ids) > 0:
      if not self.cfg.auto_reset:
        raise RuntimeError(
          "NaN envs detected but auto_reset=False, cannot auto-recover. "
          f"envs: {nan_ids.cpu().tolist()}"
        )
      print(
        f"[NaN-AutoReset] non-finite qpos/qvel -> hard resetting "
        f"{len(nan_ids)} env(s): {nan_ids.cpu().tolist()[:32]}"
      )
    # 暴露给上层/日志，便于观察出现频率。
    self.extras["nan_reset_envs"] = nan_ids

    # ------------------------------------------------------------------
    # 结束条件 + 奖励（在基类基础上强制处理 NaN env）。
    # ------------------------------------------------------------------
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    # 发散 env 视作“本步终止”：不进缓存、不 bootstrap。
    if len(nan_ids) > 0:
      self.reset_buf[nan_ids] = True
      self.reset_terminated[nan_ids] = True
      self.reset_time_outs[nan_ids] = False

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    # 发散 env 的奖励作废，避免把 NaN / 垃圾 reward 存进 PPO 缓存。
    if len(nan_ids) > 0:
      self.reward_buf[nan_ids] = 0.0
    self.metrics_manager.compute()

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      final_obs = self.observation_manager.compute(update_history=False)
      if isinstance(final_obs, dict):
        self.extras["final_obs"] = {
          key: value for key, value in final_obs.items()
        }
      else:
        self.extras["final_obs"] = final_obs.clone()
    else:
      self.extras.pop("final_obs", None)

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_pre_reset(reset_env_ids)
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # 单次 forward()：统一刷新全部 env 的派生量。
    self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_post_reset(reset_env_ids)
    elif len(reset_env_ids) > 0:
      self._manual_reset_pending[reset_env_ids] = True

    self.recorder_manager.record_post_step()

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )
