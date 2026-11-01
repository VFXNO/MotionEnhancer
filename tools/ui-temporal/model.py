"""Compatibility imports for the isolated research tools; no app integration."""
from pathlib import Path
import sys

TRAINER = Path(__file__).resolve().parents[1] / 'ai-train'
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from mefi.ui_model import (FORMAT, HEAD_ARCH, MODES, ExportedTap, ExportedTemporal,
                           FeatureTap, SpatialBlock, TemporalUi)
