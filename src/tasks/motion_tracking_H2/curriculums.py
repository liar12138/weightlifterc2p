"""Custom curriculum terms for the H2 (31-DOF) tasks."""

from __future__ import annotations


def hand_mass_curriculum(
  env,
  env_ids,
  *,
  max_mass: float = 15.0,
  ramp_until_step: int = 150_000,
):
  """Linearly ramp the hand-mass upper bound from 0 to ``max_mass`` kg.

  Progress is based on the global environment step counter
  (``env.common_step_counter``), so difficulty increases as training proceeds.
  The current upper bound is stored on ``env._hand_mass_upper_total`` for the
  ``randomize_hand_mass`` reset event to consume, and returned as a dict for
  tensorboard logging (``Curriculum/hand_mass/upper_total_kg``).
  """
  progress = min(1.0, env.common_step_counter / ramp_until_step)
  upper = float(max_mass * progress)
  env._hand_mass_upper_total = upper
  return {"upper_total_kg": upper}
