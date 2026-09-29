"""Small exactness tests for the independent parameter-screen acceleration."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_unresolved_parameter_screen import FastTraces, SplitDecayTraces
from btsp_cam.eligibility import ClassEligibilityTraces


@pytest.mark.parametrize("current,inactive", [(0.0, 0.0), (.01, .01), (0.0, .01), (.01, .05)])
def test_fast_trace_signals_match_reference(current: float, inactive: float) -> None:
    rng = np.random.default_rng(1993)
    if current == inactive:
        reference = ClassEligibilityTraces(12, 64, 1.0, current)
    else:
        reference = SplitDecayTraces(12, 64, 1.0, current, inactive)
    fast = FastTraces(12, 64, 1.0, current, inactive)
    for _ in range(500):
        code = np.zeros(64, dtype=np.uint8)
        code[rng.choice(64, 5, replace=False)] = 1
        label = int(rng.integers(12))
        reference.update(code, label)
        fast.update(code, label)
        np.testing.assert_allclose(fast.values, reference.values, rtol=0, atol=1e-13)
        expected = reference.signals(code, label, 2.0)
        actual = fast.signals(code, label, 2.0)
        for name in ("familiarity", "novelty", "interference", "max_rival_similarity"):
            assert np.isclose(getattr(actual, name), getattr(expected, name), rtol=0, atol=1e-12)
