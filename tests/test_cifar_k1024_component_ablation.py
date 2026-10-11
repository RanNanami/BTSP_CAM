"""Regression checks for the isolated k=1024, q=4 CIFAR ablation."""

import ast
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "audit"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
import cifar_k1024_component_ablation as audit


def tiny_model():
    return BTSPCAM(BTSPCAMConfig(feature_dim=3, code_dim=8, k=2,
                                memory_slots=4, num_classes=3,
                                base_plateau_probability=1.0, beta=1.0,
                                flip_probability=1.0, eta=1.0,
                                trace_decay=0.01, tau=0.25,
                                gamma=0.1, seed=1993))


def test_k1024_codes_are_exact_binary_and_shared_across_writer_branches():
    codes, _ = audit.cache_pair("train", 1)
    assert codes.dtype == np.uint8
    assert np.all((codes == 0) | (codes == 1))
    assert np.all(codes.sum(axis=1) == 1024)
    # No variant-specific path is accepted by the cache loader.
    assert set(inspect.signature(audit.cache_pair).parameters) == {"split", "task"}
    assert set(audit.WRITER_BRANCHES) == {audit.A1, audit.A2, audit.A3}


def test_q4_maps_to_exact_discrete_tau():
    assert audit.Q == 4 and audit.TAU == 4 / 1024
    counts = np.arange(7, dtype=np.float64)[None, :]
    z = audit.features(counts, audit.FULL)
    assert np.array_equal(z > 0, counts > 4)


def test_a1_fixed_plateau_ignores_trace_probability():
    model = tiny_model()
    log = audit.writer_step(model, np.array([1, 1, 0, 0, 0, 0, 0, 0],
                                             dtype=np.uint8), 0, audit.A1)
    assert log["f_eff"] == model.config.base_plateau_probability
    assert model.traces.values[0].sum() > 0


def test_a2_sets_interference_to_one_without_removing_eligibility():
    model = tiny_model()
    log = audit.writer_step(model, np.array([1, 1, 0, 0, 0, 0, 0, 0],
                                             dtype=np.uint8), 0, audit.A2)
    assert log["I_effective"] == 1.0
    assert model.traces.values[0].sum() > 0


def test_a3_keeps_plateau_draws_and_sampled_flip_attempts():
    reference, ltp = tiny_model(), tiny_model()
    code = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    reference.memory.weights[:, 0] = 1
    ltp.memory.weights[:, 0] = 1
    full = reference._learn_code(code, 0)
    altered = audit.writer_step(ltp, code, 0, audit.A3)
    assert altered["plateau_count"] == int(full["plateau_count"])
    assert altered["sampled_flip_attempts"] == int(full["flips"])
    assert altered["LTD_count"] == 0
    assert np.all(ltp.memory.weights[:, 0] == 1)
    assert reference.plateau_rng.bit_generator.state == ltp.plateau_rng.bit_generator.state
    assert reference.synapse_rng.bit_generator.state == ltp.synapse_rng.bit_generator.state


def test_a4a_and_a4b_are_only_readout_operators():
    counts = np.array([[0, 1, 4, 5, 6]], dtype=np.float64)
    np.testing.assert_array_equal(audit.features(counts, audit.A4A), counts / 1024)
    np.testing.assert_array_equal(audit.features(counts, audit.A4B), counts)
    assert audit.A4A not in audit.WRITER_BRANCHES
    assert audit.A4B not in audit.WRITER_BRANCHES


def test_a4_branches_reuse_full_memory_hashes():
    for task in (1, 5, 10, 20):
        full = audit.wb_for(audit.FULL, task)
        assert audit.sha(full) == audit.sha(audit.wb_for(audit.A4A, task))
        assert audit.sha(full) == audit.sha(audit.wb_for(audit.A4B, task))


def test_fit_reads_training_cache_only_from_seen_tasks():
    tree = ast.parse(inspect.getsource(audit.fit_evaluate))
    updates = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "update"]
    assert len(updates) == 1
    source = inspect.getsource(audit.fit_evaluate)
    assert "for source in range(1, task + 1):" in source
    assert "s, y = train_cache[source]" in source
    assert source.index("ridge.update(") < source.index("s, y = test_cache[source]")
    assert "ridge.update(features(j, branch), y)" in source
    assert "ridge.update(z" not in source


def test_train_and_test_caches_are_distinct_files():
    for task in (1, 5, 20):
        train = audit.code_path(audit.CACHE, 1024, "train", task)
        test = audit.code_path(audit.CACHE, 1024, "test", task)
        assert train != test and train.exists() and test.exists()


def test_abar_arithmetic_recomputed_independently():
    rows = audit.csv_rows(audit.FULL20)
    summary = json.loads(audit.SUMMARY.read_text(encoding="utf-8"))
    for branch in audit.BRANCHES:
        subset = [float(r["cumulative_accuracy"]) for r in rows
                  if r["variant"] == branch]
        assert len(subset) == 20
        assert abs(sum(subset) / 20 - summary["variants"][branch]["Abar"]) < 1e-12


def test_archived_k96_regression_is_unchanged():
    old = audit.csv_rows(audit.OLD_ABLATION)
    full = [r for r in old if r["variant"] == "FULL"]
    assert len(full) == 20
    assert abs(sum(float(r["cumulative_accuracy"]) for r in full) / 20 - .85553) < .00002
    assert abs(float(full[-1]["cumulative_accuracy"]) - .7918) < 1e-12


def test_a3_full_plateau_and_attempt_regression():
    rows = audit.csv_rows(audit.MECHANISMS)
    full = {int(r["task"]): r for r in rows if r["variant"] == audit.FULL}
    ltp = {int(r["task"]): r for r in rows if r["variant"] == audit.A3}
    assert len(full) == len(ltp) == 20
    for task in full:
        assert full[task]["plateau_count"] == ltp[task]["plateau_count"]
        assert full[task]["sampled_flip_attempts"] == ltp[task]["sampled_flip_attempts"]
        assert int(ltp[task]["LTD_count"]) == 0
