from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg,ManagerBasedRlEnv
import torch
from mjlab.envs import types

class ManagerBasedRlEnvFinalObs(ManagerBasedRlEnv):
    def __init__(
        self,
        cfg: ManagerBasedRlEnvCfg,
        device: str,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg,device,render_mode)
    
    def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
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

        # Check terminations and compute rewards.
        # NOTE: Derived quantities (xpos, xquat, ...) are stale by one physics
        # substep here. See the docstring above for why this is acceptable.
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
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

        # Single forward() call: recompute derived quantities from current
        # qpos/qvel for every env. For non-reset envs this resolves the
        # one-substep staleness left by mj_step; for reset envs it picks up
        # the freshly written reset state.
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