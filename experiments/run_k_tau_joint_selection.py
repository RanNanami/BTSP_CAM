"""Train-side k/q selection, then one locked full-20 replay confirmation.

This is an experimental coordinate-consistent replay readout, not the paper's
exemplar-free implementation. No formal config or checkpoint is ever written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.ridge import StreamingRidge
from run_coordinate_aligned_full20 import CosineMoments
from run_high_k_mechanism_decomposition import (code_path as old_code_path,
                                                  load_codes as old_load_codes)
from run_high_k_stress_test import writer_step
from run_k_robustness_and_selection import development_split, make_model
from run_repair_ladder_task1_5 import load_split

SPEC = ROOT / "experiments/configs/k_tau_joint_selection.yaml"
ARCHIVE = ROOT / "diagnostics/k_train_only_selection.csv"
OLD_K96_CODES = ROOT / "diagnostics/high_k_mechanism_cache"
HIST_MAX = 30


def config():
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(
        (SPEC.parent / spec["base_config"]).resolve().read_text(encoding="utf-8")))
    assert spec["k_candidates"] == [41, 64, 80, 88, 96]
    assert spec["q_candidates"] == [0, 1, 2, 3, 4, 5, 8]
    assert spec["boundary_extension"] == [12, 16]
    assert (spec["class_order_seed"], spec["rf_seed"], spec["writer_seed"],
            spec["split_seed"], spec["validation_fraction"]) == (1993, 1993, 1993, 20260928, 0.2)
    assert (base.seed, base.code_dim, base.memory_slots, base.flip_probability,
            base.base_plateau_probability, base.beta, base.trace_decay, base.gamma) == (
                1993, 8192, 4096, 0.5, 0.005, 1.0, 0.01, 0.1)
    assert spec["fixed"] == dict(m=8192, slots=4096, f_base=0.005,
                                  beta=1.0, **{"lambda": 0.01}, p=0.5, gamma=0.1)
    paths = {key: (SPEC.parent / spec[key]).resolve() for key in (
        "cache_dir", "validation_csv", "overlap_csv", "confirmation_csv", "frozen_config")}
    return spec, base, paths


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic: {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def code_file(cache: Path, k: int, split: str, task: int) -> Path:
    return cache / f"k{k}" / f"{split}_task{task:02d}.npz"


def wb_file(cache: Path, k: int, task: int, *, development: bool) -> Path:
    part = "development" if development else "full20"
    return cache / f"k{k}" / part / f"WB_task{task:02d}.npz"


def encode_sources(base, cache: Path, k: int, task: int, *, development: bool):
    """One encoder, same RF seed; validation is isolated from writer samples."""
    model = make_model(base, k, 1993, 1993)
    if development:
        h, y = load_split("train", task)
        tr, va = development_split(y, 20260928, 0.2)
        s = model.encoder.encode(h, base.encoder_batch_size)
        if k == 96:
            old_s, old_y = old_load_codes(old_code_path(OLD_K96_CODES, k, "train", task), k)
            if not np.array_equal(s, old_s) or not np.array_equal(y, old_y):
                raise AssertionError(f"k96 sparse-cache bitwise mismatch Task{task}")
        return (s[tr].copy(), y[tr].copy()), (s[va].copy(), y[va].copy())
    h, y = load_split("train", task)
    s = model.encoder.encode(h, base.encoder_batch_size)
    return (s, y), None


def save_codes(path: Path, s: np.ndarray, y: np.ndarray, k: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    if s.dtype != np.uint8 or s.shape[1] != 8192 or not np.all(s.sum(axis=1) == k):
        raise AssertionError("invalid exact-k code")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=np.packbits(s, axis=1), labels=y.astype(np.int16))


def load_codes(path: Path, k: int):
    with np.load(path, allow_pickle=False) as data:
        s = np.unpackbits(data["packed"], axis=1, count=8192)
        y = data["labels"].astype(np.int64)
    if not np.all(s.sum(axis=1) == k):
        raise AssertionError(f"non-exact-k codes: {path}")
    return s, y


def train_writer(base, cache: Path, k: int, tasks: int, *, development: bool):
    """Fresh writer from Task1; all q share the same persisted W_B trajectory."""
    model = make_model(base, k, 1993, 1993)
    for task in range(1, tasks + 1):
        start = time.perf_counter()
        train_path = code_file(cache, k, "dev_train" if development else "full_train", task)
        val_path = code_file(cache, k, "dev_validation", task)
        if development:
            (train_s, train_y), (val_s, val_y) = encode_sources(base, cache, k, task, development=True)
            save_codes(train_path, train_s, train_y, k)
            save_codes(val_path, val_s, val_y, k)
        else:
            train_s, train_y = encode_sources(base, cache, k, task, development=False)[0]
            save_codes(train_path, train_s, train_y, k)
        log = writer_step(model, train_s, train_y)
        path = wb_file(cache, k, task, development=development)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        packed = np.packbits(model.memory.weights, axis=1)
        np.savez_compressed(path, packed=packed, k=k, task=task,
                            gross_flips=log["gross_flips"])
        if task == 1:
            with np.load(path, allow_pickle=False) as state:
                restored = np.unpackbits(state["packed"], axis=1, count=8192)
            np.testing.assert_array_equal(restored, model.memory.weights)
        print(json.dumps(dict(phase="development_writer" if development else "full_writer",
                              k=k, task=task, WB_density=float(model.memory.weights.mean()),
                              gross_flips=log["gross_flips"], seconds=time.perf_counter()-start)), flush=True)


def load_wb(path: Path, k: int, task: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if int(data["k"]) != k or int(data["task"]) != task:
            raise AssertionError("checkpoint metadata mismatch")
        w = np.unpackbits(data["packed"], axis=1, count=8192)
    if w.shape != (4096, 8192):
        raise AssertionError("checkpoint shape mismatch")
    return w


def overlaps(sources, wt: np.ndarray, k: int):
    result = []
    for s, y in sources:
        j = np.asarray(sparse.csr_matrix(s, dtype=np.float32) @ wt)
        if np.any(j != np.floor(j)) or np.any(j > k):
            raise AssertionError("overlap counts are not exact integers")
        result.append((j.astype(np.uint8), y))
    return result


def source_codes(cache: Path, k: int, task: int, *, development: bool, validation: bool):
    split = ("dev_validation" if validation else "dev_train") if development else (
        "full_test" if validation else "full_train")
    if not development and validation:
        path = code_file(cache, k, split, task)
        if not path.exists():
            model = make_model(config()[1], k, 1993, 1993)
            h, y = load_split("test", task)
            s = model.encoder.encode(h, model.config.encoder_batch_size)
            save_codes(path, s, y, k)
    return load_codes(code_file(cache, k, split, task), k)


def overlap_row(k: int, task: int, split: str, parts: list[tuple[np.ndarray, np.ndarray]]):
    hist = sum((np.bincount(j.ravel(), minlength=k+1) for j, _ in parts),
               np.zeros(k+1, dtype=np.int64))
    values = np.arange(k+1)
    n = int(hist.sum())
    def quantile(p):
        return int(np.searchsorted(np.cumsum(hist), p*n, side="left"))
    row = dict(k=k, task=task, split=split, J_mean=float(hist @ values / n),
               J_std=float(np.sqrt(hist @ (values**2) / n - (hist @ values / n)**2)),
               J_median=quantile(.5), J_p75=quantile(.75), J_p90=quantile(.9),
               J_p95=quantile(.95), J_p99=quantile(.99), J_max=int(np.flatnonzero(hist)[-1]),
               sample_count=sum(len(y) for _, y in parts), element_count=n)
    row.update({f"J_count_{i}": int(hist[i]) for i in range(HIST_MAX+1)})
    row["J_count_gt30"] = int(hist[HIST_MAX+1:].sum())
    return row


def fit_eval(k: int, task: int, q: int, gamma: float, train, evaluation, wb_density: float):
    """Exact current-WB nonlinear replay; no C/D shortcut for q > 0."""
    lookup = np.maximum(np.arange(k+1, dtype=np.float64)-q, 0) / k
    ridge = StreamingRidge(4096, task*5, gamma)
    for j, y in train:
        ridge.update(lookup[j], y)
    ridge.solve()
    ridge._last_cholesky = None
    old_ok = old_n = current_ok = current_n = five_ok = 0
    active = nonzero_samples = total_elements = total_samples = 0
    z_sum = z_sq = 0.0
    z_hist = np.zeros(k+1, dtype=np.int64)
    geom = CosineMoments(4096)
    for source, (j, y) in enumerate(evaluation, 1):
        z = lookup[j]
        logits = ridge.logits(z)
        pred = logits.argmax(axis=1)
        correct = int(np.count_nonzero(pred == y))
        if source == task:
            current_ok += correct
            current_n += len(y)
            five = logits[:, (task-1)*5:task*5].argmax(axis=1)+(task-1)*5
            five_ok += int(np.count_nonzero(five == y))
        else:
            old_ok += correct
            old_n += len(y)
        mask = j > q
        active += int(mask.sum())
        nonzero_samples += int(mask.any(axis=1).sum())
        total_elements += j.size
        total_samples += len(y)
        z_sum += float(z.sum())
        z_sq += float(np.einsum("ij,ij->", z, z))
        z_hist += np.bincount(j.ravel(), minlength=k+1)
        geom.add(z, y)
    same, diff, margin = geom.values()
    def z_percentile(p):
        idx = int(np.searchsorted(np.cumsum(z_hist), p*total_elements, side="left"))
        return float(lookup[idx])
    return dict(k=k, q=q, tau=q/k, task=task,
                cumulative_validation_accuracy=(old_ok+current_ok)/(old_n+current_n),
                old_accuracy=old_ok/old_n if old_n else float("nan"),
                current_validation_accuracy=current_ok/current_n,
                current_5way_validation_accuracy=five_ok/current_n,
                CA1_nonzero_density=active/total_elements,
                sample_nonzero_fraction=nonzero_samples/total_samples,
                zero_sample_fraction=1-nonzero_samples/total_samples,
                mean_z=z_sum/total_elements,
                std_z=float(np.sqrt(max(0, z_sq/total_elements-(z_sum/total_elements)**2))),
                p90_z=z_percentile(.9), p99_z=z_percentile(.99),
                same_class_cosine=same, different_class_cosine=diff,
                cosine_margin=margin, WB_density=wb_density,
                evaluation_samples=total_samples)


def archived_k96():
    rows = [r for r in read_csv(ARCHIVE) if r["phase"] == "development_validation"
            and int(r["k"]) == 96]
    out = {int(r["task"]): float(r["cumulative_accuracy"]) for r in rows}
    if set(out) != set(range(1, 6)):
        raise AssertionError("archived k96 validation trajectory missing")
    return out


def development(spec, base, paths):
    if paths["validation_csv"].exists() or paths["overlap_csv"].exists():
        raise FileExistsError("joint-selection outputs already exist; refuse to overwrite")
    cache = paths["cache_dir"]
    grid = []
    overlap_rows = []
    reference = archived_k96()
    for k in spec["k_candidates"]:
        train_writer(base, cache, k, 5, development=True)
        k_rows = []
        for task in range(1, 6):
            start = time.perf_counter()
            w = load_wb(wb_file(cache, k, task, development=True), k, task)
            wt = np.ascontiguousarray(w.T, dtype=np.float32)
            train = overlaps([source_codes(cache, k, t, development=True, validation=False)
                              for t in range(1, task+1)], wt, k)
            val = overlaps([source_codes(cache, k, t, development=True, validation=True)
                            for t in range(1, task+1)], wt, k)
            overlap_rows.append(overlap_row(k, task, "seen_validation", val))
            for q in spec["q_candidates"]:
                result = fit_eval(k, task, q, base.gamma, train, val, float(w.mean()))
                if k == 96 and q == 0 and abs(result["cumulative_validation_accuracy"]-reference[task]) > 1e-12:
                    raise AssertionError(f"k96 q0 archived mismatch at Task{task}: {result['cumulative_validation_accuracy']} vs {reference[task]}")
                k_rows.append(result)
                print(json.dumps(dict(phase="validation", k=k, task=task, q=q,
                                      accuracy=result["cumulative_validation_accuracy"])), flush=True)
            print(json.dumps(dict(phase="validation_task_complete", k=k, task=task,
                                  seconds=time.perf_counter()-start)), flush=True)
        scores = {q: float(np.mean([r["cumulative_validation_accuracy"] for r in k_rows
                                     if r["q"] == q])) for q in spec["q_candidates"]}
        best = max(scores, key=scores.get)
        if best == 8 and 100*(scores[8]-scores[5]) >= spec["boundary_gain_pp"]:
            for task in range(1, 6):
                w = load_wb(wb_file(cache, k, task, development=True), k, task)
                wt = np.ascontiguousarray(w.T, dtype=np.float32)
                train = overlaps([source_codes(cache, k, t, development=True, validation=False)
                                  for t in range(1, task+1)], wt, k)
                val = overlaps([source_codes(cache, k, t, development=True, validation=True)
                                for t in range(1, task+1)], wt, k)
                for q in spec["boundary_extension"]:
                    k_rows.append(fit_eval(k, task, q, base.gamma, train, val, float(w.mean())))
        grid.extend(k_rows)
    for row in grid:
        same = [r["cumulative_validation_accuracy"] for r in grid
                if r["k"] == row["k"] and r["q"] == row["q"]]
        row["mean_task1_5_validation_accuracy"] = float(np.mean(same))
        row["selection_split_seed"] = spec["split_seed"]
        row["readout_mode"] = spec["readout_mode"]
    write_csv(paths["validation_csv"], grid)
    write_csv(paths["overlap_csv"], overlap_rows)
    print(json.dumps(dict(phase="baseline_consistency", status="PASS",
                          k96_q0=[r["cumulative_validation_accuracy"] for r in grid
                                  if r["k"] == 96 and r["q"] == 0])), flush=True)


def freeze(spec, paths):
    if paths["frozen_config"].exists():
        raise FileExistsError("frozen config already exists; never reselect")
    rows = read_csv(paths["validation_csv"])
    expected = {(k, q, task) for k in spec["k_candidates"]
                for q in spec["q_candidates"] for task in range(1, 6)}
    found = {(int(r["k"]), int(r["q"]), int(r["task"])) for r in rows}
    if not expected.issubset(found):
        raise AssertionError("incomplete validation grid")
    combos = {(int(r["k"]), int(r["q"])) for r in rows}
    scores = {(k, q): float(np.mean([float(r["cumulative_validation_accuracy"])
                                    for r in rows if int(r["k"]) == k and int(r["q"]) == q]))
              for k, q in combos}
    peak = max(scores.values())
    tied = [(k, q) for (k, q), value in scores.items()
            if 100*(peak-value) <= spec["tie_tolerance_pp"] + 1e-12]
    k, q = min(tied)
    assert 0 <= q < k
    payload = dict(status="LOCAL_WORKING_ABLATION_BASELINE",
                   provenance="train_side_validation_only_not_author_configuration",
                   dataset="CIFAR100", protocol="B0-Inc5", seed=1993,
                   class_order_seed=1993, rf_seed=1993, writer_seed=1993,
                   validation_split_seed=20260928, m=8192, slots=4096,
                   k=k, q=q, tau=q/k, f_base=0.005, beta=1.0,
                   **{"lambda": 0.01}, p=0.5, gamma=0.1,
                   readout_mode="CURRENT_WB_REPLAY_REFIT",
                   replay_boundary="Historical sparse-code replay aligns nonlinear CA1 features to current W_B; not a strict exemplar-free BTSP-CAM algorithm.",
                   selection_metric="mean_Task1_5_cumulative_train_side_validation_accuracy",
                   mean_validation_accuracy=scores[k, q],
                   same_k_q0_validation_accuracy=scores[k, 0],
                   validation_csv_sha256=hashlib.sha256(paths["validation_csv"].read_bytes()).hexdigest(),
                   overlap_csv_sha256=hashlib.sha256(paths["overlap_csv"].read_bytes()).hexdigest())
    paths["frozen_config"].write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps(dict(phase="freeze", k=k, q=q, tau=q/k,
                          mean_validation=scores[k, q], same_k_q0=scores[k, 0])), flush=True)


def confirmation(spec, base, paths):
    lock = yaml.safe_load(paths["frozen_config"].read_text(encoding="utf-8"))
    if lock["status"] != "LOCAL_WORKING_ABLATION_BASELINE":
        raise AssertionError("invalid frozen status")
    if hashlib.sha256(paths["validation_csv"].read_bytes()).hexdigest() != lock["validation_csv_sha256"]:
        raise AssertionError("selection grid changed after freeze")
    if hashlib.sha256(paths["overlap_csv"].read_bytes()).hexdigest() != lock["overlap_csv_sha256"]:
        raise AssertionError("overlap diagnostic changed after freeze")
    if paths["confirmation_csv"].exists():
        raise FileExistsError("confirmation exists; never run test twice")
    k, q = int(lock["k"]), int(lock["q"])
    cache = paths["cache_dir"]
    train_writer(base, cache, k, 20, development=False)
    rows = []
    for task in range(1, 21):
        start = time.perf_counter()
        w = load_wb(wb_file(cache, k, task, development=False), k, task)
        wt = np.ascontiguousarray(w.T, dtype=np.float32)
        train = overlaps([source_codes(cache, k, t, development=False, validation=False)
                          for t in range(1, task+1)], wt, k)
        test = overlaps([source_codes(cache, k, t, development=False, validation=True)
                         for t in range(1, task+1)], wt, k)
        row = fit_eval(k, task, q, base.gamma, train, test, float(w.mean()))
        row["raw_J_summary_split"] = "seen_test"
        row.update(overlap_row(k, task, "seen_test", test))
        row["average_incremental_to_task"] = float(np.mean(
            [r["cumulative_validation_accuracy"] for r in rows] +
            [row["cumulative_validation_accuracy"]]))
        row["evaluation_split"] = "formal_test_confirmation_only_after_freeze"
        row["task_seconds"] = time.perf_counter()-start
        rows.append(row)
        print(json.dumps(dict(phase="confirmation", task=task, k=k, q=q,
                              cumulative=row["cumulative_validation_accuracy"],
                              seconds=row["task_seconds"])), flush=True)
    write_csv(paths["confirmation_csv"], rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("validation", "freeze", "confirm"))
    args = parser.parse_args()
    spec, base, paths = config()
    if args.phase == "validation":
        development(spec, base, paths)
    elif args.phase == "freeze":
        freeze(spec, paths)
    else:
        confirmation(spec, base, paths)


if __name__ == "__main__":
    main()
