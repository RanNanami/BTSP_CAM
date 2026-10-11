"""Artifact and guardrail tests for the isolated CUB readout decomposition."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cub200_readout_decomposition import (
    BRANCH_R0, BRANCH_R1, BRANCH_R3, DIAG, DRIFT, MISMATCH, PREFLIGHT,
    SOURCE_MATRIX, SUMMARY, TASKWISE, guard_ids, load_wb,
)


def artifact():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def rows(path: Path):
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def test_r0_reconstruction_matches_all_archived_tasks():
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    actual = rows(TASKWISE)
    archived = rows(DIAG / "cub200_full_taskwise.csv")
    assert preflight["status"] == "PASS" and preflight["task_count"] == 20
    assert preflight["max_absolute_accuracy_error"] < 1e-12
    r0 = [r for r in actual if r["branch"] == BRANCH_R0]
    assert len(r0) == len(archived) == 20
    assert all(abs(float(a["cumulative_accuracy"]) - float(b["cumulative_accuracy"])) < 1e-12
               for a, b in zip(r0, archived))


def test_same_writer_hash_r0_r1_r2_and_checkpoint():
    summary = artifact()
    all_rows = rows(TASKWISE)
    assert summary["same_WB_R0_R1"] is True
    for task in range(1, 21):
        task_rows = [r for r in all_rows if int(r["task"]) == task]
        assert len(task_rows) == 3
        same = [r["W_B_sha256"] for r in task_rows if r["branch"] in (BRANCH_R0, BRANCH_R1)]
        assert same == [summary["writer_sha256"][str(task)]] * 2
        assert load_wb(task, same[0]).shape == (4096, 8192)


def test_r2_equivalence_all_tasks():
    eq = artifact()["R2_equivalence_by_task"]
    assert len(eq) == 20 and artifact()["R2_equivalence"] == "PASS"
    assert all(item["accuracy_exact_equal"] and item["logit_max_abs_diff"] <= 1e-10
               and item["A_max_abs_diff"] <= 1e-8 and item["B_max_abs_diff"] <= 1e-10
               for item in eq)


def test_streaming_guard_rejects_historical_rebuild():
    ids = {"train": {1: ["train:1"], 2: ["train:2"]}}
    for branch in (BRANCH_R1, BRANCH_R3):
        try:
            guard_ids(1, 2, branch, ids, set(), {"train:1", "train:2"})
        except AssertionError as exc:
            assert "historical" in str(exc)
        else:
            raise AssertionError("streaming branch accepted historical training data")


def test_current_task_training_only_allowed_for_streaming():
    ids = {"train": {1: ["train:1"], 2: ["train:2"]}}
    assert guard_ids(2, 2, BRANCH_R1, ids, set(), {"train:1", "train:2"}) == ["train:2"]


def test_r0_seen_train_allowed_without_future_train():
    ids = {"train": {1: ["train:1"], 2: ["train:2"], 3: ["train:3"]}}
    assert guard_ids(1, 2, BRANCH_R0, ids, set(), set().union(*map(set, ids["train"].values()))) == ["train:1"]
    assert guard_ids(2, 2, BRANCH_R0, ids, set(), set().union(*map(set, ids["train"].values()))) == ["train:2"]


def test_no_test_ids_in_fit_guard():
    ids = {"train": {1: ["test:1"]}}
    try:
        guard_ids(1, 1, BRANCH_R0, ids, {"test:1"}, {"test:1"})
    except AssertionError:
        pass
    else:
        raise AssertionError("fit guard accepted test identity")


def test_no_future_ids_in_fit_guard():
    ids = {"train": {1: ["train:1"], 2: ["train:2"]}}
    try:
        guard_ids(2, 1, BRANCH_R0, ids, set(), {"train:1", "train:2"})
    except AssertionError:
        pass
    else:
        raise AssertionError("fit guard accepted a future task")


def test_frozen_tau_gamma_all_branches():
    p = artifact()["frozen_parameters"]
    assert p["tau"] == 4 / 96 and p["gamma"] == 0.1
    assert {r["branch"] for r in rows(TASKWISE)} == {BRANCH_R0, BRANCH_R1, BRANCH_R3}


def test_metric_arithmetic_independent():
    all_rows = rows(TASKWISE)
    for branch, key in ((BRANCH_R0, "R0"), (BRANCH_R1, "R1"), (BRANCH_R3, "R3")):
        branch_rows = [r for r in all_rows if r["branch"] == branch]
        assert len(branch_rows) == 20
        average = sum(float(r["cumulative_accuracy"]) for r in branch_rows) / 20
        assert abs(average - artifact()[key]["average_incremental_accuracy"]) < 1e-12
        assert abs(float(branch_rows[-1]["cumulative_accuracy"]) -
                   artifact()[key]["final_accuracy"]) < 1e-12


def test_branch_order_does_not_change_fixed_input_update():
    from btsp_cam.ridge import StreamingRidge
    z = np.array([[1., 0.], [0., 1.], [1., 1.]])
    y = np.array([0, 1, 0])
    def independent_fit(order):
        result = {}
        for branch in order:
            ridge = StreamingRidge(2, 2, .1)
            ridge.update(z.copy(), y.copy())
            ridge.solve()
            result[branch] = ridge.logits(z).copy()
        return result
    forward = independent_fit((BRANCH_R0, BRANCH_R1))
    reverse = independent_fit((BRANCH_R1, BRANCH_R0))
    assert all(np.array_equal(forward[name], reverse[name]) for name in forward)


def test_actual_task2_branch_order_and_literal_increment():
    from run_btsp_cam_cub200 import ca1, fit_replay, load_codes
    from btsp_cam.ridge import StreamingRidge

    hashes = artifact()["writer_sha256"]
    wb1, wb2 = load_wb(1, hashes["1"]), load_wb(2, hashes["2"])
    def cached(split, task):
        path = DIAG / "cub200_sparse_cache" / f"{split}_task{task:02d}.npz"
        with np.load(path, allow_pickle=False) as values:
            metadata = json.loads(values["metadata"].tobytes())
        return load_codes(path, metadata)
    train1, train2, test2 = cached("train", 1), cached("train", 2), cached("test", 2)
    wt1, wt2 = np.ascontiguousarray(wb1.T, dtype=np.float32), np.ascontiguousarray(wb2.T, dtype=np.float32)
    def run(order):
        output = {}
        for branch in order:
            if branch == BRANCH_R0:
                ridge, _ = fit_replay(wb2, 2, .1, lambda source: (train1, train2)[source - 1])
            else:
                ridge = StreamingRidge(4096, 10, .1)
                _, z1 = ca1(train1[0], wt1)
                ridge.update(z1, train1[1])
                ridge.solve()
                old_a, old_b = ridge.a.copy(), ridge.b.copy()
                ridge.expand_classes(20)
                _, z2 = ca1(train2[0], wt2)
                ridge.update(z2, train2[1])
                assert np.array_equal(ridge.a, old_a + z2.T @ z2)
                assert np.array_equal(ridge.b[:, :10], old_b)
                ridge.solve()
            _, ztest = ca1(test2[0], wt2)
            output[branch] = ridge.logits(ztest)
        return output
    forward = run((BRANCH_R0, BRANCH_R1))
    reverse = run((BRANCH_R1, BRANCH_R0))
    assert all(np.array_equal(forward[name], reverse[name]) for name in forward)


def test_source_matrix_and_drift_record_counts():
    source = rows(SOURCE_MATRIX)
    drift = rows(DRIFT)
    mismatch = rows(MISMATCH)
    assert len(source) == 3 * (20 * 21 // 2) == 630
    assert len(drift) == 2 * (20 * 21 // 2) == 420
    assert len(mismatch) == 20
    assert all(int(r["sample_count"]) >= 100 for r in drift)
