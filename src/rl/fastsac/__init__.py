"""FastSAC optional training backend for capylab_wt."""

import sys
from pathlib import Path

_FASTSAC_ROOT = Path(__file__).resolve().parent
if str(_FASTSAC_ROOT) not in sys.path:
  sys.path.insert(0, str(_FASTSAC_ROOT))
