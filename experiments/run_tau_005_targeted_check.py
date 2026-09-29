"""Read-only current-WB oracle at five fixed tau values for k=128/160.

This deliberately reuses the independently saved WB trajectories. It never
instantiates a BTSP writer or changes formal configuration/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.ridge import StreamingRidge
from run_coordinate_aligned_full20 import CosineMoments
from run_tau_q_oracle_k128_k160 import (
    archived,
    config,
    load_checkpoint,
    overlaps,
    require_trajectory,
    rows_csv,
    sparse_sources,
    transformed_quantile_from_hist,
    write_csv,
)

TAU_TEXT = ("0", "0.0025", "0.005", "0.00625", "0.0078125")
OUTPUT = ROOT / "diagnostics/tau_005_targeted_check.csv"
PREVIOUS_ORACLE = ROOT / "diagnostics/tau_q_oracle_k{k}.csv"
TASKS = 20


def previous_rows(k: int) -> dict[tuple[int, int], dict]:
    rows = rows_csv(Path(str(PREVIOUS_ORACLE).format(k=k)))
    result = {(int(row["task"]), int(row["q"])): row for row in rows}
    if len(result) != 240:
        raise ValueError(f"k{k} discrete-q oracle is incomplete")
    return result


def archive_q(k: int, tau_text: str) -> int | None:
    if tau_text == "0":
        return 0
    if (k, tau_text) in ((128, "0.0078125"), (160, "0.00625")):
        return 1
    return None


def support_difference(hist: np.ndarray, lookup: np.ndarray) -> int:
    return int(hist[1:][lookup[1:] == 0].sum())


def oracle_metrics(k: int, task: int, tau: float, lookup: np.ndarray,
                   train_counts: list[np.ndarray], train_y: list[np.ndarray],
                   test_counts: list[np.ndarray], test_y: list[np.ndarray],
                   test_hist: np.ndarray, gamma: float) -> dict:
    start = time.perf_counter()
    feature_dim = test_counts[0].shape[1]
    ridge = StreamingRidge(feature_dim, task * 5, gamma)
    for counts, labels in zip(train_counts, train_y[:task]):
        ridge.update(lookup[counts], labels)
    ridge.solve()
    ridge._last_cholesky = None
    geometry = CosineMoments(feature_dim)
    old_ok = old_n = current_ok = current_n = five_ok = nonzero_samples = 0
    for source, (counts, labels) in enumerate(zip(test_counts, test_y[:task]), 1):
        z = lookup[counts]
        logits = ridge.logits(z)
        pred = logits.argmax(axis=1)
        ok = int(np.count_nonzero(pred == labels))
        if source == task:
            current_ok += ok
            current_n += len(labels)
            five_ok += int(np.count_nonzero(
                logits[:, (task - 1) * 5:task * 5].argmax(axis=1)
                + (task - 1) * 5 == labels))
        else:
            old_ok += ok
            old_n += len(labels)
        nonzero_samples += int(np.count_nonzero(np.any(z > 0, axis=1)))
        geometry.add(z, labels)
    same, different, margin = geometry.values()
    elements = int(test_hist.sum())
    return {
        "cumulative_accuracy": (old_ok + current_ok) / (old_n + current_n),
        "old_accuracy": old_ok / old_n if old_n else float("nan"),
        "current_accuracy": current_ok / current_n,
        "current_5way_accuracy": five_ok / current_n,
        "CA1_element_density": float(test_hist @ (lookup > 0) / elements),
        "fraction_nonzero_samples": nonzero_samples / (old_n + current_n),
        "mean_z": float(test_hist @ lookup / elements),
        "p99_z": transformed_quantile_from_hist(test_hist, lookup, 0.99),
        "same_class_cosine": same,
        "different_class_cosine": different,
        "cosine_margin": margin,
        "fit_eval_seconds": time.perf_counter() - start,
        "source": "fresh_current_WB_replay_oracle",
    }


def reused_metrics(prior: dict, hist: np.ndarray, lookup: np.ndarray) -> dict:
    fields = ("cumulative_accuracy", "old_accuracy", "current_accuracy",
              "current_5way_accuracy", "CA1_element_density",
              "fraction_nonzero_samples", "mean_z", "p99_z",
              "same_class_cosine", "different_class_cosine", "cosine_margin")
    out = {field: prior[field] for field in fields}
    elements = int(hist.sum())
    checks = {
        "CA1_element_density": hist @ (lookup > 0) / elements,
        "mean_z": hist @ lookup / elements,
        "p99_z": transformed_quantile_from_hist(hist, lookup, 0.99),
    }
    for name, value in checks.items():
        if abs(float(out[name]) - float(value)) > 1e-12:
            raise AssertionError(f"reused {name} disagrees with current WB/counts")
    out["fit_eval_seconds"] = 0.0
    out["source"] = "validated_prior_current_WB_replay_oracle"
    return out


def run(k: int) -> None:
    cfg = config()
    require_trajectory(k)
    prior = previous_rows(k)
    archive = archived(k)
    train_s, train_y, test_s, test_y = sparse_sources(k, cfg)
    rows = rows_csv(OUTPUT)
    existing = {(int(row["k"]), int(row["task"]), row["tau_text"]) for row in rows}
    for task in range(1, TASKS + 1):
        start = time.perf_counter()
        wb = load_checkpoint(k, task, cfg.memory_slots, cfg.code_dim)
        w_t = np.ascontiguousarray(wb.T, dtype=np.float32)
        train_counts = overlaps(train_s, w_t, task, k)
        test_counts = overlaps(test_s, w_t, task, k)
        train_hist = sum((np.bincount(part.ravel(), minlength=k + 1)
                          for part in train_counts), np.zeros(k + 1, dtype=np.int64))
        test_hist = sum((np.bincount(part.ravel(), minlength=k + 1)
                         for part in test_counts), np.zeros(k + 1, dtype=np.int64))
        q0 = prior[(task, 0)]
        if abs(float(q0["cumulative_accuracy"])
               - float(archive[task]["cumulative_accuracy"])) > 1e-12:
            raise AssertionError(f"k{k} Task{task} prior q0/archive mismatch")
        # An independent direct array comparison is required for the central
        # tau=0.005 claim; the histogram counts in each output row are additive.
        tau005 = 0.005
        direct_train = sum(int(np.count_nonzero(
            (part > 0) != (np.maximum(part.astype(np.float64) / k - tau005, 0) > 0)))
            for part in train_counts)
        direct_test = sum(int(np.count_nonzero(
            (part > 0) != (np.maximum(part.astype(np.float64) / k - tau005, 0) > 0)))
            for part in test_counts)
        if direct_train or direct_test:
            raise AssertionError(f"k{k} Task{task}: tau=0.005 changed support")
        for tau_text in TAU_TEXT:
            if (k, task, tau_text) in existing:
                continue
            tau = float(tau_text)
            lookup = np.maximum(np.arange(k + 1, dtype=np.float64) / k - tau, 0.0)
            if tau_text == "0.005":
                if (support_difference(train_hist, lookup),
                        support_difference(test_hist, lookup)) != (direct_train, direct_test):
                    raise AssertionError("histogram/direct support counts disagree")
            q = int(np.flatnonzero(lookup == 0)[-1])
            source_q = archive_q(k, tau_text)
            if source_q is not None:
                result = reused_metrics(prior[(task, source_q)], test_hist, lookup)
            else:
                result = oracle_metrics(k, task, tau, lookup, train_counts,
                                        train_y, test_counts, test_y, test_hist, cfg.gamma)
            base = float(q0["cumulative_accuracy"])
            row = {
                "k": k, "task": task, "tau_text": tau_text, "tau": tau,
                "k_tau": k * tau, "max_eliminated_J": q,
                "method": "current_WB_replay_oracle_not_exemplar_free",
                **result,
                "support_difference_count": support_difference(train_hist, lookup)
                                            + support_difference(test_hist, lookup),
                "train_support_difference_count": support_difference(train_hist, lookup),
                "test_support_difference_count": support_difference(test_hist, lookup),
                "delta_cumulative_pp_vs_tau0": 100 * (float(result["cumulative_accuracy"]) - base),
                "train_samples": sum(len(y) for y in train_y[:task]),
                "test_samples": sum(len(y) for y in test_y[:task]),
                "WB_density": float(wb.mean()),
            }
            rows.append(row)
            existing.add((k, task, tau_text))
            write_csv(OUTPUT, sorted(rows, key=lambda r: (int(r["k"]), int(r["task"]), float(r["tau"]))))
            print(json.dumps({"k": k, "task": task, "tau": tau_text,
                              "accuracy": float(result["cumulative_accuracy"]),
                              "density": float(result["CA1_element_density"]),
                              "support_difference_count": row["support_difference_count"],
                              "source": result["source"]}), flush=True)
        del wb, w_t, train_counts, test_counts
        gc.collect()
        print(json.dumps({"k": k, "task": task,
                          "task_seconds": round(time.perf_counter() - start, 2)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, choices=(128, 160))
    run(parser.parse_args().k)


if __name__ == "__main__":
    main()
