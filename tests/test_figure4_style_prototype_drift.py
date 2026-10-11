"""Regression checks for the read-only Figure-4-style drift diagnostic."""

import csv
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "audit"))
import figure4_style_prototype_drift as audit


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_locked_k_q_tau():
    assert (audit.K, audit.Q, audit.TAU) == (96, 4, 4 / 96)
    assert audit.METHOD == "FIGURE4_STYLE_LOCAL_OPERATIONALIZATION"


def test_requested_output_shapes():
    assert len(rows(audit.CLASSWISE)) == 3150
    assert len(rows(audit.SUMMARY)) == 438
    assert len(rows(audit.VS_SAMPLE)) == 420
    assert len(rows(audit.RELATION)) == 20


def test_classwise_task_domain_and_fixed_sample_hashes():
    data = rows(audit.CLASSWISE)
    keys = {}
    for row in data:
        source, checkpoint = int(row["source_task"]), int(row["checkpoint_task"])
        assert 1 <= source <= checkpoint <= 20
        key = (row["dataset"], int(row["class_id"]))
        keys.setdefault(key, (row["sample_ids_sha256"], int(row["sample_count"])))
        assert keys[key] == (row["sample_ids_sha256"], int(row["sample_count"]))
    assert len([k for k in keys if k[0] == "CIFAR100"]) == 100
    assert len([k for k in keys if k[0] == "CUB200"]) == 200


def test_introduction_checkpoint_is_identity_for_nonzero_prototypes():
    identity = [r for r in rows(audit.CLASSWISE)
                if r["source_task"] == r["checkpoint_task"]]
    assert len(identity) == 300
    for row in identity:
        assert abs(float(row["prototype_cosine_rawmean"]) - 1) < 1e-12
        assert abs(float(row["prototype_cosine_sample_norm_mean"]) - 1) < 1e-12
        assert abs(float(row["prototype_cosine_a"]) - 1) < 1e-12
        assert abs(float(row["sample_cosine_mean"]) - 1) < 2e-6


def test_task1_to_20_summary_is_independently_recomputed():
    detail = rows(audit.CLASSWISE)
    summary = rows(audit.SUMMARY)
    for dataset, classes in (("CIFAR100", 5), ("CUB200", 10)):
        selected = [r for r in detail if r["dataset"] == dataset
                    and r["source_task"] == "1" and r["checkpoint_task"] == "20"]
        target = next(r for r in summary if r["dataset"] == dataset
                      and r["summary_scope"] == "TASK1_CLASSES_TO_TASK20")
        assert len(selected) == classes
        proto = np.array([float(r["prototype_cosine_rawmean"]) for r in selected])
        sample = np.array([float(r["sample_cosine_mean"]) for r in selected])
        assert abs(proto.mean() - float(target["prototype_cosine_mean"])) < 1e-14
        assert abs(np.median(proto) - float(target["prototype_cosine_median"])) < 1e-14
        assert abs(sample.mean() - float(target["sample_cosine_mean"])) < 1e-14


def test_cub_first100_sample_drift_matches_archived_diagnostic():
    _, codes, _, _, loader = audit.dataset_inputs("CUB200")
    wb1, _ = loader(1)
    wb20, _ = loader(20)
    _, z1 = audit.extract(codes[1][:100], np.ascontiguousarray(wb1.T, dtype=np.float32))
    _, z20 = audit.extract(codes[1][:100], np.ascontiguousarray(wb20.T, dtype=np.float32))
    u1, _ = audit.row_unit(z1)
    u20, _ = audit.row_unit(z20)
    actual = float(np.einsum("ij,ij->i", u1, u20).mean())
    archived = next(r for r in rows(ROOT / "diagnostics/cub200_readout_feature_drift.csv")
                    if r["source_task"] == "1" and r["checkpoint_task"] == "20"
                    and r["split"] == "test")
    assert abs(actual - float(archived["mean_cosine"])) < 2e-7


def test_readout_relation_uses_archived_r0_r1_without_refit():
    source = rows(ROOT / "diagnostics/cub200_readout_decomposition_taskwise.csv")
    relation = {int(r["task"]): r for r in rows(audit.RELATION)}
    for branch, field in (("R0_CURRENT_WB_REPLAY_REFIT", "R0_current_WB_replay_refit_accuracy"),
                          ("R1_LITERAL_STREAMING_EQ8", "R1_literal_streaming_accuracy")):
        expected = {int(r["task"]): float(r["cumulative_accuracy"])
                    for r in source if r["branch"] == branch}
        assert len(expected) == 20
        for task in expected:
            assert float(relation[task][field]) == expected[task]


def test_descriptive_correlations_are_independently_recomputed():
    relation = rows(audit.RELATION)[1:]
    gap = np.array([float(r["R0_minus_R1_gap"]) for r in relation])
    proto = np.array([float(r["prototype_similarity_mean_old_classes"]) for r in relation])
    sample = np.array([float(r["sample_similarity_mean_old_classes"]) for r in relation])
    assert abs(np.corrcoef(proto, gap)[0, 1]
               - float(relation[0]["prototype_vs_readout_gap_pearson"])) < 1e-14
    assert abs(np.corrcoef(sample, gap)[0, 1]
               - float(relation[0]["sample_vs_readout_gap_pearson"])) < 1e-14
    assert all(r["correlation_status"] == "DESCRIPTIVE_CORRELATION_ONLY" for r in relation)


def test_summary_declares_paper_boundary_and_p2():
    result = json.loads(audit.SUMMARY_JSON.read_text(encoding="utf-8"))
    assert result["paper_claim"] == "prototype similarity > 0.9"
    assert result["exact_paper_metric_formula"] == "NOT_FULLY_SPECIFIED"
    assert result["paper_metric_exactly_reproducible"] is False
    assert result["interpretation_case"].startswith("P2_")
    assert result["does_our_prototype_look_slowly_drifting"] == "NO"
    assert result["does_prototype_stability_guarantee_readout_coordinate_consistency"] == "NOT_DETERMINED"


def test_script_contains_no_training_or_ridge_fit_path():
    source = inspect.getsource(audit)
    assert "StreamingRidge" not in source
    assert "_learn_code" not in source
    assert "fit_replay" not in source
    assert "train_task" not in source
    assert "cub200_readout_decomposition_taskwise.csv" in source
