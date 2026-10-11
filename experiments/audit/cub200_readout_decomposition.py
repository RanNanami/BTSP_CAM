"""Readout-only CUB decomposition on one verified, frozen writer trajectory.

Run from repository root. This never edits the formal model/config or existing
CUB caches/results. Outputs are new diagnostics only.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from btsp_cam.ridge import StreamingRidge
from run_btsp_cam_cub200 import ca1, evaluate, fit_replay, load_codes, load_config, make_model

DIAG = ROOT / "diagnostics"
SPARSE = DIAG / "cub200_sparse_cache"
CHECKPOINTS = DIAG / "cub200_readout_wb_checkpoints"
TASKWISE = DIAG / "cub200_readout_decomposition_taskwise.csv"
SOURCE_MATRIX = DIAG / "cub200_readout_source_task_matrix.csv"
DRIFT = DIAG / "cub200_readout_feature_drift.csv"
MISMATCH = DIAG / "cub200_readout_coordinate_mismatch.csv"
SUMMARY = DIAG / "cub200_readout_decomposition_summary.json"
PREFLIGHT = DIAG / "cub200_readout_writer_verification.json"
BRANCH_R0 = "R0_CURRENT_WB_REPLAY_REFIT"
BRANCH_R1 = "R1_LITERAL_STREAMING_EQ8"
BRANCH_R3 = "R3_STABLE_WB1_STREAMING_CONTROL"
SLOTS, CODE_BITS, TASKS, INCREMENT = 4096, 8192, 20, 10
EPS = 1e-12


def sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError(f"empty output: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs():
    provenance = {split: defaultdict(list) for split in ("train", "test")}
    with (DIAG / "cub200_cache_row_provenance.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            provenance[row["split"]][int(row["task"])].append(row)
    cache = {split: {} for split in ("train", "test")}
    ids = {split: {} for split in ("train", "test")}
    for split in cache:
        for task in range(1, TASKS + 1):
            rows = sorted(provenance[split][task], key=lambda r: r["sparse_row_index"])
            assert [r["sparse_row_index"] for r in rows] == list(range(len(rows)))
            ids[split][task] = [r["sample_id"] for r in rows]
            path = SPARSE / f"{split}_task{task:02d}.npz"
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(data["metadata"].tobytes())
            cache[split][task] = load_codes(path, meta)
            assert np.array_equal(cache[split][task][1], [r["label"] for r in rows])
            assert len(cache[split][task][1]) == len(ids[split][task])
    all_train = set().union(*(set(x) for x in ids["train"].values()))
    all_test = set().union(*(set(x) for x in ids["test"].values()))
    assert len(all_train) == 5994 and len(all_test) == 5794 and not all_train & all_test
    archived = list(csv.DictReader((DIAG / "cub200_full_taskwise.csv").open(newline="", encoding="utf-8")))
    assert len(archived) == TASKS
    return cache, ids, all_train, all_test, archived


def checkpoint_path(task: int) -> Path:
    return CHECKPOINTS / f"WB_task{task:02d}.npz"


def save_wb(task: int, wb: np.ndarray) -> str:
    path = checkpoint_path(task)
    if path.exists():
        raise FileExistsError(path)
    assert wb.shape == (SLOTS, CODE_BITS) and wb.dtype == np.uint8
    assert np.logical_or(wb == 0, wb == 1).all()
    packed = np.packbits(wb, axis=1)
    digest = sha(wb)
    with path.open("xb") as stream:
        np.savez_compressed(stream, packed=packed,
                            sha256=np.array(digest), task=np.array(task, dtype=np.int16))
    return digest


def load_wb(task: int, expected_hash: str) -> np.ndarray:
    with np.load(checkpoint_path(task), allow_pickle=False) as data:
        assert int(data["task"]) == task
        wb = np.unpackbits(data["packed"], axis=1, count=CODE_BITS)
        assert str(data["sha256"]) == expected_hash
    assert wb.shape == (SLOTS, CODE_BITS) and sha(wb) == expected_hash
    return wb


def guard_ids(source: int, task: int, branch: str, ids: dict, all_test: set, all_train: set):
    row_ids = ids["train"][source]
    assert len(row_ids) == len(set(row_ids))
    assert not (set(row_ids) & all_test)
    seen = set().union(*(set(ids["train"][i]) for i in range(1, task + 1)))
    assert set(row_ids) <= seen and not (set(row_ids) & (all_train - seen))
    if branch in (BRANCH_R1, BRANCH_R3):
        assert source == task, "streaming branch reread a historical task"
    return row_ids


def preflight(cache, ids, all_train, all_test, archived, gamma: float):
    """One writer replay; gate all downstream branches on archived R0 equality."""
    model = make_model(load_config()[1])
    hashes = {}
    accuracies = []
    CHECKPOINTS.mkdir(exist_ok=False)
    for task in range(1, TASKS + 1):
        for code, label, sample_id in zip(cache["train"][task][0].toarray(),
                                          cache["train"][task][1], ids["train"][task]):
            assert sample_id in all_train and sample_id not in all_test
            model._learn_code(code, int(label))
        wb = model.memory.weights
        hashes[task] = save_wb(task, wb)
        def load_training(source):
            guard_ids(source, task, BRANCH_R0, ids, all_test, all_train)
            return cache["train"][source]
        ridge, wt = fit_replay(wb, task, gamma, load_training)
        result, _ = evaluate(ridge, wt, task,
                             [cache["test"][i] for i in range(1, task + 1)])
        actual = float(result["cumulative_accuracy"])
        expected = float(archived[task - 1]["cumulative_accuracy"])
        assert abs(actual - expected) < 1e-12, (task, actual, expected)
        accuracies.append(actual)
        print(json.dumps(dict(phase="preflight", task=task, accuracy=actual,
                              archived_error=abs(actual - expected), wb_sha256=hashes[task])), flush=True)
        ridge._last_cholesky = None
        del ridge, wt
        gc.collect()
    assert abs(float(np.mean(accuracies)) - 0.8786059546186593) < 1e-12
    assert abs(accuracies[-1] - 0.8060062133241284) < 1e-12
    output = dict(status="PASS", task_count=TASKS, max_absolute_accuracy_error=max(
        abs(a - float(archived[i]["cumulative_accuracy"])) for i, a in enumerate(accuracies)),
        wb_sha256={str(t): hashes[t] for t in hashes},
        average_incremental_accuracy=float(np.mean(accuracies)), final_accuracy=accuracies[-1])
    if PREFLIGHT.exists():
        raise FileExistsError(PREFLIGHT)
    PREFLIGHT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return hashes


def metrics_for_branch(ridge, wb: np.ndarray, task: int, cache: dict, branch: str,
                       source_rows: list[dict], r0_logits: dict | None = None):
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    old_ok = old_n = current_ok = current_n = restricted_ok = 0
    test_nonzero = test_zero_rows = test_norm = total_elements = 0
    max_logit_diff = 0.0
    logits_by_source = {}
    for source in range(1, task + 1):
        codes, labels = cache["test"][source]
        _, z = ca1(codes, wt)
        logits = ridge.logits(z)
        pred = logits.argmax(axis=1)
        correct = int(np.count_nonzero(pred == labels))
        source_rows.append(dict(branch=branch, checkpoint_task=task, source_task=source,
                                seen_classes=task * INCREMENT, test_count=len(labels),
                                correct=correct, accuracy=correct / len(labels)))
        if source == task:
            current_ok += correct
            current_n += len(labels)
            start = (task - 1) * INCREMENT
            restricted = logits[:, start:task * INCREMENT].argmax(axis=1) + start
            restricted_ok += int(np.count_nonzero(restricted == labels))
        else:
            old_ok += correct
            old_n += len(labels)
        test_nonzero += int(np.count_nonzero(z))
        test_zero_rows += int(np.count_nonzero(~np.any(z > 0, axis=1)))
        test_norm += float(np.linalg.norm(z, axis=1).sum())
        total_elements += z.size
        logits_by_source[source] = logits
        if r0_logits is not None:
            max_logit_diff = max(max_logit_diff,
                                 float(np.max(np.abs(logits - r0_logits[source]))))
            assert np.array_equal(pred, r0_logits[source].argmax(axis=1))
    n = old_n + current_n
    result = dict(branch=branch, task=task, seen_classes=task * INCREMENT,
                  cumulative_accuracy=(old_ok + current_ok) / n,
                  old_accuracy=old_ok / old_n if old_n else float("nan"),
                  current_accuracy=current_ok / current_n,
                  current_restricted_10way_accuracy=restricted_ok / current_n,
                  ridge_condition=ridge.condition_number,
                  CA1_density_test=test_nonzero / total_elements,
                  CA1_zero_row_fraction_test=test_zero_rows / n,
                  mean_z_norm_test=test_norm / n,
                  W_B_density=float(wb.mean()),
                  W_B_source="WB1_FIXED" if branch == BRANCH_R3 else f"WB_task{task:02d}",
                  W_B_sha256=sha(wb))
    return result, logits_by_source, max_logit_diff


def drift_record(source: int, task: int, split: str, initial: np.ndarray,
                 current: np.ndarray, sample_ids: list[str]) -> dict:
    assert initial.shape == current.shape and len(initial) == len(sample_ids) >= 100
    old_norm = np.linalg.norm(initial, axis=1)
    new_norm = np.linalg.norm(current, axis=1)
    dot = np.einsum("ij,ij->i", initial, current)
    both_zero = (old_norm == 0) & (new_norm == 0)
    cosine = np.divide(dot, old_norm * new_norm, out=np.zeros(len(old_norm)),
                       where=(old_norm * new_norm) > 0)
    cosine[both_zero] = 1.0
    relative = np.linalg.norm(current - initial, axis=1) / (old_norm + EPS)
    support_change = np.logical_xor(initial > 0, current > 0).mean(axis=1)
    digest = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    return dict(source_task=source, checkpoint_task=task, split=split,
                subset_rule="FIRST_100_CACHE_ROWS_FIXED_AT_SOURCE_TASK",
                sample_count=len(sample_ids), sample_ids_sha256=digest,
                mean_cosine=float(cosine.mean()), mean_relative_l2_drift=float(relative.mean()),
                mean_support_change_rate=float(support_change.mean()),
                norm_ratio=float(new_norm.mean() / (old_norm.mean() + EPS)),
                initial_zero_row_fraction=float(np.mean(old_norm == 0)),
                current_zero_row_fraction=float(np.mean(new_norm == 0)))


def main() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from repository root: {ROOT}")
    raw, config = load_config()
    assert (config.code_dim, config.memory_slots, config.k, config.tau, config.gamma,
            config.base_plateau_probability, config.beta, config.trace_decay, config.eta,
            config.flip_probability, config.seed) == (
            8192, 4096, 96, 4 / 96, .1, .005, 1., .01, 1., .5, 1993)
    assert raw["readout"] == "CURRENT_WB_REPLAY_REFIT"
    for path in (TASKWISE, SOURCE_MATRIX, DRIFT, MISMATCH, SUMMARY, PREFLIGHT):
        if path.exists():
            raise FileExistsError(path)
    if CHECKPOINTS.exists():
        raise FileExistsError(CHECKPOINTS)
    start_time = time.perf_counter()
    cache, ids, all_train, all_test, archived = load_inputs()
    hashes = preflight(cache, ids, all_train, all_test, archived, config.gamma)
    assert len(hashes) == 20

    r1 = StreamingRidge(SLOTS, 0, config.gamma)
    r3 = StreamingRidge(SLOTS, 0, config.gamma)
    wb1 = load_wb(1, hashes[1])
    r0_rows, r1_rows, r3_rows = [], [], []
    source_rows, drift_rows, mismatch_rows, equivalence = [], [], [], []
    baseline = {split: {} for split in ("train", "test")}
    for task in range(1, TASKS + 1):
        wb = load_wb(task, hashes[task])
        assert sha(wb) == hashes[task]  # same W_B for R0, R1 and R2
        seen = set().union(*(set(ids["train"][i]) for i in range(1, task + 1)))
        future = all_train - seen
        assert not seen & future and not seen & all_test
        def load_training(source):
            guard_ids(source, task, BRANCH_R0, ids, all_test, all_train)
            return cache["train"][source]
        r0, wt = fit_replay(wb, task, config.gamma, load_training)
        assert sha(wb) == hashes[task]

        # R2: rebuild aligned A/B independently, source-task by source-task.
        r2 = StreamingRidge(SLOTS, task * INCREMENT, config.gamma)
        for source in range(1, task + 1):
            guard_ids(source, task, "R2_ALIGNED_REBUILD_CHECK", ids, all_test, all_train)
            codes, labels = cache["train"][source]
            _, z = ca1(codes, wt)
            r2.update(z, labels)
        a_diff = float(np.max(np.abs(r0.a - r2.a)))
        b_diff = float(np.max(np.abs(r0.b - r2.b)))
        assert a_diff <= 1e-8 and b_diff <= 1e-10, (task, a_diff, b_diff)
        r2.solve()
        w_diff = float(np.max(np.abs(r0.weights - r2.weights)))
        assert w_diff <= 1e-7, (task, w_diff)

        # R1 and R3 add only the current task's training IDs to old A/B.
        guard_ids(task, task, BRANCH_R1, ids, all_test, all_train)
        current_codes, current_labels = cache["train"][task]
        _, current_z = ca1(current_codes, wt)
        r1.expand_classes(task * INCREMENT)
        old_a_hash, old_b_hash = sha(r1.a), sha(r1.b)
        r1.update(current_z, current_labels)
        assert old_a_hash != sha(r1.a) and old_b_hash != sha(r1.b)
        r1.solve()

        guard_ids(task, task, BRANCH_R3, ids, all_test, all_train)
        stable_wt = np.ascontiguousarray(wb1.T, dtype=np.float32)
        _, stable_z = ca1(current_codes, stable_wt)
        r3.expand_classes(task * INCREMENT)
        r3.update(stable_z, current_labels)
        r3.solve()

        # Exact Gram/B mismatch in the same dimensionality and class mask.
        relative_a = float(np.linalg.norm(r1.a - r0.a, "fro") /
                           (np.linalg.norm(r0.a, "fro") + EPS))
        relative_b = float(np.linalg.norm(r1.b - r0.b, "fro") /
                           (np.linalg.norm(r0.b, "fro") + EPS))

        r0_result, r0_logits, _ = metrics_for_branch(r0, wb, task, cache, BRANCH_R0, source_rows)
        r2_source_rows = []
        r2_result, _, logit_diff = metrics_for_branch(
            r2, wb, task, cache, "R2_ALIGNED_REBUILD_CHECK", r2_source_rows, r0_logits)
        assert r0_result["cumulative_accuracy"] == r2_result["cumulative_accuracy"]
        assert logit_diff <= 1e-10, (task, logit_diff)
        assert abs(r0_result["cumulative_accuracy"] -
                   float(archived[task - 1]["cumulative_accuracy"])) < 1e-12
        r1_result, _, _ = metrics_for_branch(r1, wb, task, cache, BRANCH_R1, source_rows)
        r3_result, _, _ = metrics_for_branch(r3, wb1, task, cache, BRANCH_R3, source_rows)
        r0_rows.append(r0_result)
        r1_rows.append(r1_result)
        r3_rows.append(r3_result)
        mismatch_rows.append(dict(task=task, relative_A_mismatch=relative_a,
                                  relative_B_mismatch=relative_b,
                                  R0_accuracy=r0_result["cumulative_accuracy"],
                                  R1_accuracy=r1_result["cumulative_accuracy"],
                                  accuracy_gap=r0_result["cumulative_accuracy"] -
                                  r1_result["cumulative_accuracy"]))
        equivalence.append(dict(task=task, A_max_abs_diff=a_diff, B_max_abs_diff=b_diff,
                                Wout_max_abs_diff=w_diff, logit_max_abs_diff=logit_diff,
                                accuracy_exact_equal=True))

        # Predeclared deterministic subset: first 100 cache rows per source/split.
        for source in range(1, task + 1):
            for split in ("train", "test"):
                codes, _ = cache[split][source]
                subset_ids = ids[split][source][:100]
                assert len(subset_ids) == 100
                _, current = ca1(codes[:100], wt)
                if source == task:
                    baseline[split][source] = current.copy()
                drift_rows.append(drift_record(source, task, split,
                                               baseline[split][source], current, subset_ids))

        assert r0_result["W_B_sha256"] == r1_result["W_B_sha256"] == hashes[task]
        print(json.dumps(dict(phase="readout", task=task,
                              R0=r0_result["cumulative_accuracy"],
                              R1=r1_result["cumulative_accuracy"],
                              R3=r3_result["cumulative_accuracy"],
                              R2_logit_max_abs_diff=logit_diff,
                              relative_A_mismatch=relative_a)), flush=True)
        r0._last_cholesky = r2._last_cholesky = None
        r1._last_cholesky = r3._last_cholesky = None
        del r0, r2, wt, current_z, stable_z, r0_logits
        gc.collect()

    all_task_rows = [item for task in range(1, 21) for item in
                     (r0_rows[task - 1], r1_rows[task - 1], r3_rows[task - 1])]
    assert len(all_task_rows) == 60 and len(source_rows) == 630 and len(drift_rows) == 420
    def headline(rows):
        return dict(average_incremental_accuracy=float(np.mean(
                        [r["cumulative_accuracy"] for r in rows])),
                    final_accuracy=rows[-1]["cumulative_accuracy"])
    h0, h1, h3 = headline(r0_rows), headline(r1_rows), headline(r3_rows)
    assert abs(h0["average_incremental_accuracy"] - 0.8786059546186593) < 1e-12
    assert abs(h0["final_accuracy"] - 0.8060062133241284) < 1e-12
    x = np.asarray([r["relative_A_mismatch"] for r in mismatch_rows])
    y = np.asarray([r["accuracy_gap"] for r in mismatch_rows])
    descriptive_corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    summary = dict(status="PASS", preflight="PASS", same_WB_R0_R1=True,
                   R2_equivalence="PASS", frozen_parameters=raw["parameters"],
                   writer_sha256={str(t): hashes[t] for t in hashes},
                   R0=h0, R1=h1, R3=h3,
                   replay_advantage=dict(average_incremental_accuracy=h0["average_incremental_accuracy"]-
                                         h1["average_incremental_accuracy"],
                                         final_accuracy=h0["final_accuracy"]-h1["final_accuracy"]),
                   stable_vs_literal=dict(average_incremental_accuracy=h3["average_incremental_accuracy"]-
                                          h1["average_incremental_accuracy"],
                                          final_accuracy=h3["final_accuracy"]-h1["final_accuracy"]),
                   R2_equivalence_by_task=equivalence,
                   A_mismatch_accuracy_gap_descriptive_pearson=descriptive_corr,
                   task20_coordinate_mismatch=mismatch_rows[-1],
                   runtime_seconds=time.perf_counter()-start_time,
                   historical_train_replay=dict(R0=True, R1=False, R3=False),
                   strict_exemplar_free=dict(R0=False, R1=True, R3=True))
    write_csv_new(TASKWISE, all_task_rows)
    write_csv_new(SOURCE_MATRIX, source_rows)
    write_csv_new(DRIFT, drift_rows)
    write_csv_new(MISMATCH, mismatch_rows)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(status="PASS", R0=h0, R1=h1, R3=h3,
                          replay_advantage=summary["replay_advantage"],
                          stable_vs_literal=summary["stable_vs_literal"],
                          runtime_seconds=summary["runtime_seconds"])), flush=True)


if __name__ == "__main__":
    main()
