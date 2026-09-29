"""Equation (5) adaptive plateau probability and slot-wise sampling."""

from __future__ import annotations

import numpy as np

from .eligibility import GatingSignals


def plateau_probability(base_probability: float, signals: GatingSignals) -> float:
    if not np.isfinite(base_probability) or not 0.0 <= base_probability <= 1.0:
        raise ValueError("base_probability must be in [0, 1]")
    for name, value in (
        ("familiarity", signals.familiarity),
        ("novelty", signals.novelty),
        ("interference", signals.interference),
    ):
        if not np.isfinite(value) or not -1e-12 <= value <= 1.0 + 1e-12:
            raise ValueError(f"{name} must be in [0, 1]")
    raw = (
        float(base_probability)
        * 4.0
        * signals.novelty
        * (1.0 - signals.novelty)
        * signals.familiarity
        * signals.interference
    )
    # Only remove floating-point excursions at the boundaries. With valid
    # inputs, every analytic factor already lies in [0, 1].
    return float(np.clip(raw, 0.0, 1.0))


def sample_slot_plateaus(
    probability: float, slots: int, rng: np.random.Generator
) -> np.ndarray:
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    return (rng.random(int(slots)) < probability).astype(np.uint8)
