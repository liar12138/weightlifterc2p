from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

def pack_timer_safe(func):

  class safe_timer_termi:

    def __init__(self, cfg, env : ManagerBasedRlEnv):
      self.wait_count = torch.zeros(env.num_envs,device=env.device)

    def __call__(
      self,
      env: ManagerBasedRlEnv,
      **kws
    ):
      """Terminate when the asset's orientation exceeds the limit angle."""
      termi = func(env,**kws)
      self.wait_count[termi] = self.wait_count[termi] + 1
      result = self.wait_count > (0.3 / env.step_dt)
      self.wait_count[result] = 0
      return result

  return safe_timer_termi