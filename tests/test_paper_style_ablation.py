"""Small deterministic checks for the isolated Section-6 operations."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.ca1 import ca1_features
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from run_paper_style_ablation import features, writer_step_variant


def tiny_model():
    cfg = BTSPCAMConfig(feature_dim=3, code_dim=8, k=2, memory_slots=4,
                        num_classes=3, base_plateau_probability=1.0, beta=1.0,
                        flip_probability=1.0, eta=1.0, trace_decay=0.01,
                        tau=0.25, gamma=0.1, seed=1993)
    return BTSPCAM(cfg)


def test_fixed_gate_is_exact_base_rate():
    model = tiny_model()
    result = writer_step_variant(model, np.array([1, 1, 0, 0, 0, 0, 0, 0],
                                                   dtype=np.uint8), 0, "A1_FIXED_GATE")
    assert result["f_eff"] == model.config.base_plateau_probability


def test_collision_removal_sets_only_effective_interference_to_one():
    model = tiny_model()
    code = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    result = writer_step_variant(model, code, 0, "A2_NO_COLLISION")
    assert result["I_effective"] == 1.0
    assert model.traces.values[0].sum() > 0  # eligibility remains active


def test_only_ltp_never_depresses_an_existing_one():
    model = tiny_model()
    code = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    model.memory.weights[:, 0] = 1
    result = writer_step_variant(model, code, 0, "A3_ONLY_LTP")
    assert result["flip_10"] == 0
    np.testing.assert_array_equal(model.memory.weights[:, 0], np.ones(4, dtype=np.uint8))


def test_ca1_removals_and_full_operator():
    counts = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(features(counts, "A4A_NO_THRESHOLD", 4), counts/4)
    np.testing.assert_array_equal(features(counts, "A4B_NO_CA1_OPERATOR", 4), counts)
    np.testing.assert_allclose(features(counts, "FULL", 4),
                               np.maximum(counts/4-4/96, 0))
    code = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    memory = np.array([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
    overlap = code @ memory.T
    np.testing.assert_allclose(features(overlap, "FULL", 2),
                               ca1_features(code, memory, 4/96))


def test_partial_cue_seed_namespace_has_no_cross_task_reuse():
    seeds = [2026101001, 2026101002, 2026101003, 2026101004, 2026101005]
    old_keys = [seed + task for seed in seeds for task in range(1, 20)]
    assert len(set(old_keys)) == 23  # documented defect in preserved old CSV
    streams = [np.random.default_rng(np.random.SeedSequence([seed, task]))
               .integers(0, 2**63, dtype=np.int64)
               for seed in seeds for task in range(1, 20)]
    assert len(set(streams)) == 95
