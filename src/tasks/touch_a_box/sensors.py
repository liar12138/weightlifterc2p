"""Hand ContactSensor factory for G1 + Revo2."""

from __future__ import annotations

from mjlab.sensor import ContactMatch, ContactSensorCfg

# Palm + phalanges + fingertip touch pads (excludes empty *_tip_link).
# Expected: 2*(1 palm + 3 thumb links + 8 finger links + 5 touch) = 34.
HAND_TACTILE_BODY_PATTERN = (
  r"^(left|right)_hand_base_link$"
  r"|^(left|right)_thumb_(metacarpal|proximal|distal)_link$"
  r"|^(left|right)_(index|middle|ring|pinky)_(proximal|distal)_link$"
  r"|^(left|right)_(thumb|index|middle|ring|pinky)_touch_link$"
)


def get_hand_tactile_sensor_cfg(
  name: str = "hand_tactile",
  *,
  debug: bool = True,
) -> ContactSensorCfg:
  """One ContactSensor → one netforce column per matched hand body.

  ``secondary=None``: any contact on those bodies counts (easiest to get readings).
  """
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(
      mode="body",
      pattern=HAND_TACTILE_BODY_PATTERN,
      entity="robot",
    ),
    secondary=None,
    fields=("found", "force", "pos"),
    reduce="netforce",
    num_slots=1,
    history_length=0,
    track_air_time=False,
    debug=debug,
  )
