"""Custom domain-randomization event functions for the H2 (31-DOF) tasks."""

from __future__ import annotations

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import dr


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def randomize_hand_mass(
  env,
  env_ids,
  *,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Randomize the hand (wrist) mass on every episode reset.

  The upper bound of the total added mass is produced by the ``hand_mass``
  curriculum term (stored on ``env._hand_mass_upper_total``), which ramps it
  from 0 to 15 kg over training.

  The total mass is split equally across the matched wrist bodies: we use
  ``shared_random=True`` so both wrists draw the *same* sample per environment
  (equal split), keeping the total added mass within the curriculum bound.
  ``operation="add"`` adds a point mass on top of the default wrist mass
  without accumulating across episodes.
  """
  upper_total = float(getattr(env, "_hand_mass_upper_total", 0.0))
  num_hands = len(asset_cfg.body_names) if asset_cfg.body_names else 2
  upper_each = upper_total / num_hands

  dr.body_mass(
    env,
    env_ids,
    ranges=(0.0, upper_each),
    operation="add",
    shared_random=True,
    asset_cfg=asset_cfg,
  )
