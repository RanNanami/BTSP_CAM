"""Current-WB replay-oracle threshold sweep, CIFAR100 Task1-20.

All tau values, including zero, refit ridge from all *seen* train samples in
the same current-WB feature space. This is deliberately not exemplar-free.
The formal model and coordinate-aligned C/D implementation are never written.
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
from scipy import sparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder
from run_coordinate_aligned_full20 import CosineMoments, formal_wb
from run_repair_ladder_task1_5 import load_split, overlap

CONFIG_PATH = ROOT / "experiments" / "configs" / "aligned_threshold_oracle_full20.yaml"
FULL20_REFERENCE = ROOT / "diagnostics" / "coordinate_aligned_full20.csv"
TAU_GRID = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10,
            0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
COUNT_VALUES = np.arange(42, dtype=np.float64)
PERCENTILES = (0.50, 0.75, 0.90, 0.95, 0.99, 0.999)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_overlap(codes: sparse.csr_matrix, w_float_t: np.ndarray, k: int) -> np.ndarray:
    """Exact integer overlaps in float32 (at most k=41), stored as uint8."""
    counts = codes @ w_float_t
    if not np.all((counts >= 0) & (counts <= k) & (counts == np.floor(counts))):
        raise AssertionError("sparse/WB multiplication lost integer overlaps")
    return np.asarray(counts, dtype=np.uint8)


def histogram(parts: list[np.ndarray]) -> np.ndarray:
    result = np.zeros(42, dtype=np.int64)
    for part in parts:
        result += np.bincount(part.ravel(), minlength=42)
    return result


def histogram_quantile(hist: np.ndarray, probability: float) -> float:
    """NumPy's default linear quantile, applied to the discrete count histogram."""
    n = int(hist.sum())
    if not n:
        return float("nan")
    position = (n - 1) * probability
    low = int(np.floor(position))
    high = int(np.ceil(position))
    cumulative = np.cumsum(hist)
    low_value = int(np.searchsorted(cumulative, low + 1, side="left"))
    high_value = int(np.searchsorted(cumulative, high + 1, side="left"))
    return (low_value + (position - low) * (high_value - low_value)) / 41.0


def a_scale_row(task: int, split: str, parts: list[np.ndarray], wb_density: float) -> dict:
    hist = histogram(parts)
    n = int(hist.sum())
    mean = float(hist @ (COUNT_VALUES / 41.0) / n)
    second = float(hist @ ((COUNT_VALUES / 41.0) ** 2) / n)
    q = [histogram_quantile(hist, p) for p in PERCENTILES]
    return {
        "task": task, "split": split, "WB_density": wb_density,
        "sample_count": sum(len(part) for part in parts), "element_count": n,
        "mean_a": mean, "std_a": float(np.sqrt(max(0.0, second - mean * mean))),
        "p50_a": q[0], "p75_a": q[1], "p90_a": q[2],
        "p95_a": q[3], "p99_a": q[4], "p99_9_a": q[5],
        "source": "current_WB_replay_oracle",
    }


def fit_ridge(train_counts: list[np.ndarray], train_labels: list[np.ndarray],
              tau: float, slots: int, k: int, gamma: float) -> StreamingRidge:
    ridge = StreamingRidge(slots, int(train_labels[-1].max()) + 1, gamma)
    table = np.maximum(0.0, COUNT_VALUES / k - tau)
    for counts, labels in zip(train_counts, train_labels):
        z = table[counts]
        ridge.update(z, labels)
        del z
    ridge.solve()
    ridge._last_cholesky = None
    return ridge


def evaluate(ridge: StreamingRidge, test_counts: list[np.ndarray],
             test_labels: list[np.ndarray], tau: float, k: int,
             density: float, fraction_nonzero_samples: float,
             mean_z: float, p99_z: float) -> dict:
    task = len(test_counts)
    table = np.maximum(0.0, COUNT_VALUES / k - tau)
    geometry = CosineMoments(test_counts[0].shape[1])
    old_correct = old_total = current_correct = current_total = five_correct = 0
    for source_task, (counts, labels) in enumerate(zip(test_counts, test_labels), 1):
        z = table[counts]
        logits = ridge.logits(z)
        predictions = logits.argmax(axis=1)
        correct = int(np.count_nonzero(predictions == labels))
        if source_task == task:
            current_correct += correct
            current_total += len(labels)
            local = logits[:, (task - 1) * 5:task * 5]
            five_correct += int(np.count_nonzero(local.argmax(axis=1) + (task - 1) * 5 == labels))
        else:
            old_correct += correct
            old_total += len(labels)
        geometry.add(z, labels)
        del z, logits
    same, different, margin = geometry.values()
    return {
        "cumulative_accuracy": (old_correct + current_correct) / (old_total + current_total),
        "old_accuracy": old_correct / old_total if old_total else float("nan"),
        "current_accuracy": current_correct / current_total,
        "current_5way_accuracy": five_correct / current_total,
        "CA1_element_density": density,
        "fraction_nonzero_samples": fraction_nonzero_samples,
        "mean_z": mean_z, "p99_z": p99_z,
        "same_class_cosine": same, "different_class_cosine": different, "margin": margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-task", type=int, default=20,
                        help="Development smoke limit; full deliverable uses 20")
    args = parser.parse_args()
    experiment = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if experiment["tasks"] != 20 or experiment["readout"] != "current_WB_replay_oracle":
        raise ValueError("this runner is replay-oracle only")
    if tuple(float(x) for x in experiment["tau_grid"]) != TAU_GRID or not 1 <= args.max_task <= 20:
        raise ValueError("unexpected tau grid or task count")
    config_path = (CONFIG_PATH.parent / experiment["base_config"]).resolve()
    config = BTSPCAMConfig.from_mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    if (config.seed, config.feature_dim, config.code_dim, config.k, config.memory_slots,
            config.num_classes, config.gamma) != (1993, 768, 8192, 41, 4096, 100, 0.1):
        raise ValueError("formal model parameters changed")
    sweep_path = (CONFIG_PATH.parent / experiment["sweep_csv"]).resolve()
    scale_path = (CONFIG_PATH.parent / experiment["scale_csv"]).resolve()
    if sweep_path != ROOT / "diagnostics" / "aligned_tau_sweep_full20.csv" or scale_path != ROOT / "diagnostics" / "a_scale_evolution_full20.csv":
        raise ValueError("unexpected output paths")
    encoder = RandomSparseEncoder(config.feature_dim, config.code_dim, config.k,
                                  RNGProvenance.from_seed(config.seed).projection_seed)
    with np.load(ROOT / "results/btsp_cam_cifar100/checkpoint_task1.npz", allow_pickle=False) as state:
        np.testing.assert_array_equal(encoder.projection, state["projection"])
        np.testing.assert_array_equal(encoder.bias, state["bias"])
    with FULL20_REFERENCE.open(newline="", encoding="utf-8") as handle:
        reference = {int(row["task"]): row for row in csv.DictReader(handle)
                     if row["branch"] == "coordinate_aligned_ridge"}
    if len(reference) != 20:
        raise ValueError("full20 aligned reference missing")
    sweep_rows = read_csv(sweep_path)
    scale_rows = read_csv(scale_path)
    done = {(int(r["task"]), float(r["tau"])) for r in sweep_rows}
    scale_done = {(int(r["task"]), r["split"]) for r in scale_rows}
    train_codes: list[sparse.csr_matrix] = []
    test_codes: list[sparse.csr_matrix] = []
    train_labels: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    started = time.perf_counter()
    for task in range(1, args.max_task + 1):
        task_start = time.perf_counter()
        for split, code_list, label_list in (
                ("train", train_codes, train_labels), ("test", test_codes, test_labels)):
            h, labels = load_split(split, task)
            s = encoder.encode(h, config.encoder_batch_size)
            if not np.all(s.sum(axis=1) == config.k):
                raise AssertionError("non-exact k sparse code")
            code_list.append(sparse.csr_matrix(s, dtype=np.float32))
            label_list.append(labels)
            del h, s
        if all((task, tau) in done for tau in TAU_GRID) and all((task, split) in scale_done for split in
              ("train-current", "test-current", "test-old", "test-all")):
            print(json.dumps({"task": task, "status": "previously_complete"}), flush=True)
            continue
        wb = formal_wb(task, config.memory_slots, config.code_dim)
        wb_float_t = np.ascontiguousarray(wb.T, dtype=np.float32)
        wb_density = float(wb.mean())
        train_counts = [count_overlap(s, wb_float_t, config.k) for s in train_codes]
        test_counts = [count_overlap(s, wb_float_t, config.k) for s in test_codes]
        if task == 1:
            check = overlap(train_codes[0][:8].toarray().astype(np.uint8), wb)
            np.testing.assert_array_equal(train_counts[0][:8].astype(np.float64) / config.k, check)
        del wb, wb_float_t
        gc.collect()
        split_parts = {
            "train-current": train_counts[-1:], "test-current": test_counts[-1:],
            "test-old": test_counts[:-1], "test-all": test_counts,
        }
        for split, parts in split_parts.items():
            if (task, split) not in scale_done:
                if not parts:  # No old classes at Task1.
                    scale_rows.append({"task": task, "split": split, "WB_density": wb_density,
                                       "sample_count": 0, "element_count": 0,
                                       **{key: float("nan") for key in ("mean_a", "std_a", "p50_a", "p75_a", "p90_a", "p95_a", "p99_a", "p99_9_a")},
                                       "source": "current_WB_replay_oracle"})
                else:
                    scale_rows.append(a_scale_row(task, split, parts, wb_density))
                scale_done.add((task, split))
        write_csv(scale_path, scale_rows)
        test_hist = histogram(test_counts)
        test_elements = int(test_hist.sum())
        test_max = np.concatenate([part.max(axis=1) for part in test_counts])
        reference_zero: dict | None = next((r for r in sweep_rows if int(r["task"]) == task and float(r["tau"]) == 0.0), None)
        for tau in TAU_GRID:
            if (task, tau) in done:
                continue
            tau_start = time.perf_counter()
            table = np.maximum(0.0, COUNT_VALUES / config.k - tau)
            density = float(test_hist[table > 0].sum() / test_elements)
            fraction_nonzero = float(np.mean(table[test_max] > 0))
            mean_z = float(test_hist @ table / test_elements)
            p99_z = max(0.0, histogram_quantile(test_hist, 0.99) - tau)
            ridge = fit_ridge(train_counts, train_labels, tau, config.memory_slots,
                              config.k, config.gamma)
            result = evaluate(ridge, test_counts, test_labels, tau, config.k,
                              density, fraction_nonzero, mean_z, p99_z)
            del ridge
            if tau == 0:
                reference_zero = result
                archived = float(reference[task]["cumulative_accuracy"])
                if abs(result["cumulative_accuracy"] - archived) > 1e-12:
                    raise AssertionError(f"Task{task} tau0 oracle {result['cumulative_accuracy']} != aligned {archived}")
            if reference_zero is None:
                raise AssertionError("tau=0 result required before positive thresholds")
            sweep_rows.append({
                "task": task, "tau": tau, "method": "current_WB_replay_oracle",
                **result,
                "accuracy_delta_vs_tau0": result["cumulative_accuracy"] - float(reference_zero["cumulative_accuracy"]),
                "old_accuracy_delta_vs_tau0": result["old_accuracy"] - float(reference_zero["old_accuracy"]),
                "current_accuracy_delta_vs_tau0": result["current_accuracy"] - float(reference_zero["current_accuracy"]),
                "CA1_density_reduction_vs_tau0": float(reference_zero["CA1_element_density"]) - density,
                "WB_density": wb_density, "seen_classes": task * 5,
                "train_samples": sum(map(len, train_labels)),
                "test_samples": sum(map(len, test_labels)),
                "fit_eval_seconds": time.perf_counter() - tau_start,
                "source": "experimental_threshold_replay_not_exemplar_free",
            })
            done.add((task, tau))
            write_csv(sweep_path, sweep_rows)
            print(json.dumps({"task": task, "tau": tau,
                              "accuracy": result["cumulative_accuracy"],
                              "density": density,
                              "seconds": round(time.perf_counter() - tau_start, 2)}), flush=True)
        del train_counts, test_counts, test_hist, test_max
        gc.collect()
        print(json.dumps({"task": task, "completed": True,
                          "task_seconds": round(time.perf_counter() - task_start, 2)}), flush=True)
    print(json.dumps({"completed_tasks": args.max_task, "rows": len(sweep_rows),
                      "scale_rows": len(scale_rows), "total_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
