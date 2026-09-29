"""Minimal, paper-audited BTSP-CAM reproduction core."""

from .binary_memory import BinaryMemory, official_simplebtsp_step
from .ca1 import ca1_features
from .config import BTSPCAMConfig, ParameterProvenance
from .eligibility import ClassEligibilityTraces, GatingSignals
from .model import BTSPCAM
from .plateau import plateau_probability, sample_slot_plateaus
from .ridge import StreamingRidge
from .rng import RNGProvenance
from .sparse_encoder import RandomSparseEncoder

__all__ = [
    "BTSPCAM",
    "BTSPCAMConfig",
    "BinaryMemory",
    "ClassEligibilityTraces",
    "GatingSignals",
    "RandomSparseEncoder",
    "ParameterProvenance",
    "RNGProvenance",
    "StreamingRidge",
    "ca1_features",
    "official_simplebtsp_step",
    "plateau_probability",
    "sample_slot_plateaus",
]
