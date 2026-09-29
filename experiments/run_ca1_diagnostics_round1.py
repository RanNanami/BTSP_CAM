"""Five-task CA1 overlap, threshold, memory-drift, and write diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import yaml

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import pilot_class_order
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "btsp_cam_cifar100.yaml"
CACHE_DIR = ROOT / "data" / "feature_cache_btsp_cam"
OUTPUT_DIR = ROOT / "diagnostics"
WB_DIR = OUTPUT_DIR / "wb_checkpoints"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_cache(split: str, task: int) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE_DIR / f"{split}_task_{task:02d}.npz"
    with np.load(path, allow_pickle=False) as cache:
        features = cache["features"].copy()
        labels = cache["labels"].copy()
    if not np.all(np.isfinite(features)):
        raise ValueError(f"non-finite cached features: {path}")
    return features, labels


def overlap_from_codes(codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a = W_B s / k without applying the CA1 threshold."""

    s = np.asarray(codes)
    w = np.asarray(weights)
    if s.ndim != 2 or w.ndim != 2 or s.shape[1] != w.shape[1]:
        raise ValueError("codes and memory weights have incompatible shapes")
    active = s.sum(axis=1)
    if np.any(active == 0):
        raise ValueError("overlap normalization is undefined for an all-zero code")
    output = np.empty((s.shape[0], w.shape[0]), dtype=np.float32)
    for row_index, row in enumerate(s):
        indices = np.flatnonzero(row)
        counts = w[:, indices].sum(axis=1, dtype=np.int16)
        output[row_index] = counts / float(active[row_index])
    return output


def threshold_overlap(overlap: np.ndarray, tau: float) -> np.ndarray:
    return np.maximum(0.0, np.asarray(overlap) - float(tau)).astype(
        np.float32, copy=False
    )


def distribution_row(task: int, split: str, overlap: np.ndarray, tau: float) -> dict:
    values = np.asarray(overlap, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "task": task,
            "split": split,
            "sample_count": 0,
            "activation_count": 0,
            "mean_a": float("nan"),
            "std_a": float("nan"),
            "median_a": float("nan"),
            "p90_a": float("nan"),
            "p95_a": float("nan"),
            "p99_a": float("nan"),
            "max_a": float("nan"),
            "fraction_a_gt_tau": float("nan"),
            "fraction_a_gt_0": float("nan"),
            "fraction_a_gt_0.1": float("nan"),
            "fraction_a_gt_0.2": float("nan"),
            "fraction_a_gt_0.5": float("nan"),
        }
    quantiles = np.quantile(values, [0.5, 0.9, 0.95, 0.99])
    return {
        "task": task,
        "split": split,
        "sample_count": int(overlap.shape[0]),
        "activation_count": int(values.size),
        "mean_a": float(values.mean()),
        "std_a": float(values.std()),
        "median_a": float(quantiles[0]),
        "p90_a": float(quantiles[1]),
        "p95_a": float(quantiles[2]),
        "p99_a": float(quantiles[3]),
        "max_a": float(values.max()),
        "fraction_a_gt_tau": float(np.mean(values > tau)),
        "fraction_a_gt_0": float(np.mean(values > 0.0)),
        "fraction_a_gt_0.1": float(np.mean(values > 0.1)),
        "fraction_a_gt_0.2": float(np.mean(values > 0.2)),
        "fraction_a_gt_0.5": float(np.mean(values > 0.5)),
    }


def readout_row(
    task: int,
    method: str,
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict:
    logits = np.asarray(features, dtype=np.float64) @ weights
    predictions = logits.argmax(axis=1)
    new_start = (task - 1) * 5
    new_stop = task * 5
    new_mask = (labels >= new_start) & (labels < new_stop)
    old_mask = labels < new_start
    return {
        "task": task,
        "method": method,
        "current_accuracy": float(np.mean(predictions[new_mask] == labels[new_mask])),
        "cumulative_accuracy": float(np.mean(predictions == labels)),
        "old_accuracy": (
            float(np.mean(predictions[old_mask] == labels[old_mask]))
            if np.any(old_mask)
            else float("nan")
        ),
        "new_accuracy": float(np.mean(predictions[new_mask] == labels[new_mask])),
    }


def representation_geometry(features: np.ndarray, labels: np.ndarray) -> dict:
    """Exact pairwise cosine means without constructing an N by N matrix."""

    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    classes = np.unique(labels)
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    same_numerator = 0.0
    same_denominator = 0
    for label in classes:
        rows = normalized[labels == label]
        row_sum = rows.sum(axis=0)
        sums[int(label)] = row_sum
        counts[int(label)] = rows.shape[0]
        same_numerator += float(row_sum @ row_sum - np.einsum("ij,ij->", rows, rows))
        same_denominator += rows.shape[0] * (rows.shape[0] - 1)
    different_numerator = 0.0
    different_denominator = 0
    for left_index, left in enumerate(classes):
        for right in classes[left_index + 1 :]:
            different_numerator += float(sums[int(left)] @ sums[int(right)])
            different_denominator += counts[int(left)] * counts[int(right)]
    same = same_numerator / same_denominator
    different = different_numerator / different_denominator
    return {
        "same_class_cosine": same,
        "different_class_cosine": different,
        "margin": same - different,
        "nonzero_density": float(np.count_nonzero(x) / x.size),
        "zero_row_fraction": float(np.mean(norms[:, 0] == 0)),
    }


def quantile_fields(prefix: str, values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    q50, q90, q95, q99 = np.quantile(array, [0.5, 0.9, 0.95, 0.99])
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std()),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_median": float(q50),
        f"{prefix}_p90": float(q90),
        f"{prefix}_p95": float(q95),
        f"{prefix}_p99": float(q99),
        f"{prefix}_max": float(array.max()),
    }


def safe_row_cosines(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left64, axis=1) * np.linalg.norm(right64, axis=1)
    valid = denominator > 0
    cosine = np.full(left64.shape[0], np.nan, dtype=np.float64)
    cosine[valid] = np.einsum("ij,ij->i", left64[valid], right64[valid]) / denominator[valid]
    return cosine, float(valid.mean())


def drift_row(
    sample_task: int,
    feature_source_wb: int,
    eval_wb: int,
    baseline_a: np.ndarray,
    baseline_z: np.ndarray,
    eval_a: np.ndarray,
    eval_z: np.ndarray,
) -> dict:
    z_cosine, valid_fraction = safe_row_cosines(baseline_z, eval_z)
    a_cosine, a_valid_fraction = safe_row_cosines(baseline_a, eval_a)
    baseline_active = baseline_z > 0
    eval_active = eval_z > 0
    union = np.count_nonzero(baseline_active | eval_active)
    intersection = np.count_nonzero(baseline_active & eval_active)
    baseline_density = np.count_nonzero(baseline_active, axis=1) / baseline_z.shape[1]
    eval_density = np.count_nonzero(eval_active, axis=1) / eval_z.shape[1]
    return {
        "sample_task": sample_task,
        "feature_source_WB": feature_source_wb,
        "eval_WB": eval_wb,
        "mean_cosine": float(np.nanmean(z_cosine)) if np.any(np.isfinite(z_cosine)) else float("nan"),
        "median_cosine": float(np.nanmedian(z_cosine)) if np.any(np.isfinite(z_cosine)) else float("nan"),
        "nonzero_overlap": float(intersection / union) if union else float("nan"),
        "activity_difference": float(np.mean(np.abs(eval_density - baseline_density))),
        "valid_z_cosine_sample_fraction": valid_fraction,
        "baseline_z_nonzero_density": float(baseline_active.mean()),
        "eval_z_nonzero_density": float(eval_active.mean()),
        "a_mean_cosine": float(np.nanmean(a_cosine)),
        "a_median_cosine": float(np.nanmedian(a_cosine)),
        "valid_a_cosine_sample_fraction": a_valid_fraction,
        "mean_absolute_a_difference": float(np.mean(np.abs(eval_a - baseline_a))),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WB_DIR.mkdir(parents=True, exist_ok=True)
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = BTSPCAMConfig.from_mapping(raw_config)
    provenance = RNGProvenance.from_seed(config.seed)
    class_order = pilot_class_order(config.num_classes, config.seed)
    model = BTSPCAM(config, class_order=class_order, rng_provenance=provenance)
    overlap_ridge = StreamingRidge(config.memory_slots, 0, config.gamma)

    distribution_rows: list[dict] = []
    readout_rows: list[dict] = []
    geometry_rows: list[dict] = []
    memory_rows: list[dict] = []
    test_code_parts: list[np.ndarray] = []
    test_label_parts: list[np.ndarray] = []
    task1_test_codes: np.ndarray | None = None
    started = time.perf_counter()

    for task in range(1, 6):
        task_started = time.perf_counter()
        train_h, train_y = load_cache("train", task)
        test_h, test_y = load_cache("test", task)
        train_s = model.encoder.encode(train_h, config.encoder_batch_size)
        current_test_s = model.encoder.encode(test_h, config.encoder_batch_size)
        if task == 1:
            task1_test_codes = current_test_s.copy()
        test_code_parts.append(current_test_s)
        test_label_parts.append(test_y)
        cumulative_s = np.concatenate(test_code_parts)
        cumulative_y = np.concatenate(test_label_parts)

        logs = model.learn_task(train_h, train_y, task_id=task)
        wb_path = WB_DIR / f"WB_task{task:02d}.npz"
        np.savez_compressed(
            wb_path,
            weights=model.memory.weights,
            task=np.asarray(task, dtype=np.int64),
            seed=np.asarray(config.seed, dtype=np.int64),
            k=np.asarray(config.k, dtype=np.int64),
            tau=np.asarray(config.tau, dtype=np.float64),
        )

        train_a = overlap_from_codes(train_s, model.memory.weights)
        cumulative_a = overlap_from_codes(cumulative_s, model.memory.weights)
        train_z = threshold_overlap(train_a, config.tau)
        cumulative_z = threshold_overlap(cumulative_a, config.tau)
        current_count = current_test_s.shape[0]
        old_count = cumulative_s.shape[0] - current_count
        current_test_a = cumulative_a[old_count:]
        old_test_a = cumulative_a[:old_count]

        distribution_rows.extend(
            [
                distribution_row(task, "train", train_a, config.tau),
                distribution_row(task, "current_test", current_test_a, config.tau),
                distribution_row(task, "old_test", old_test_a, config.tau),
            ]
        )

        overlap_ridge.expand_classes(int(train_y.max()) + 1)
        overlap_ridge.update(train_a, train_y)
        overlap_ridge.solve()
        readout_rows.extend(
            [
                readout_row(
                    task,
                    "thresholded_CA1",
                    cumulative_z,
                    cumulative_y,
                    model.ridge.weights,
                ),
                readout_row(
                    task,
                    "unthresholded_overlap",
                    cumulative_a,
                    cumulative_y,
                    overlap_ridge.weights,
                ),
            ]
        )

        geometry_groups = [
            ("current_train", train_s, train_a, train_z, train_y),
            (
                "current_test",
                cumulative_s[old_count:],
                cumulative_a[old_count:],
                cumulative_z[old_count:],
                cumulative_y[old_count:],
            ),
        ]
        if old_count:
            geometry_groups.append(
                (
                    "old_test",
                    cumulative_s[:old_count],
                    cumulative_a[:old_count],
                    cumulative_z[:old_count],
                    cumulative_y[:old_count],
                )
            )
        for split, split_s, split_a, split_z, split_y in geometry_groups:
            for name, representation in (
                ("sparse_s", split_s),
                ("overlap_a", split_a),
                ("thresholded_z", split_z),
            ):
                geometry_rows.append(
                    {
                        "task": task,
                        "representation": name,
                        "split": split,
                        **representation_geometry(representation, split_y),
                    }
                )

        row_activation = model.memory.weights.mean(axis=1)
        column_activation = model.memory.weights.mean(axis=0)
        slot_plateaus = model.last_task_slot_plateau_counts
        flips = np.asarray(
            [entry["zero_to_one_flips"] + entry["one_to_zero_flips"] for entry in logs],
            dtype=np.float64,
        )
        plateau_counts = np.asarray([entry["plateau_count"] for entry in logs])
        memory_rows.append(
            {
                "task": task,
                "scope": "task",
                "incremental_class": "",
                "original_class": "",
                "sample_count": int(train_y.size),
                "memory_occupancy": model.memory.active_fraction,
                **quantile_fields("row_activation", row_activation),
                **quantile_fields("column_activation", column_activation),
                **quantile_fields("slot_plateau_count", slot_plateaus),
                "mean_write_count": float(flips.mean()),
                "total_write_count": int(flips.sum()),
                "mean_plateau_count": float(plateau_counts.mean()),
                "total_plateau_count": int(plateau_counts.sum()),
            }
        )
        for label in np.unique(train_y):
            mask = train_y == label
            memory_rows.append(
                {
                    "task": task,
                    "scope": "class",
                    "incremental_class": int(label),
                    "original_class": int(class_order[int(label)]),
                    "sample_count": int(mask.sum()),
                    "mean_write_count": float(flips[mask].mean()),
                    "total_write_count": int(flips[mask].sum()),
                    "mean_plateau_count": float(plateau_counts[mask].mean()),
                    "total_plateau_count": int(plateau_counts[mask].sum()),
                }
            )

        print(
            json.dumps(
                {
                    "task": task,
                    "thresholded_cumulative_accuracy": readout_rows[-2]["cumulative_accuracy"],
                    "unthresholded_cumulative_accuracy": readout_rows[-1]["cumulative_accuracy"],
                    "train_fraction_a_gt_tau": distribution_rows[-3]["fraction_a_gt_tau"],
                    "current_test_fraction_a_gt_tau": distribution_rows[-2]["fraction_a_gt_tau"],
                    "runtime_seconds": time.perf_counter() - task_started,
                }
            ),
            flush=True,
        )

    if task1_test_codes is None:
        raise RuntimeError("Task 1 test codes were not captured")
    with np.load(WB_DIR / "WB_task01.npz", allow_pickle=False) as saved:
        baseline_weights = saved["weights"]
    baseline_a = overlap_from_codes(task1_test_codes, baseline_weights)
    baseline_z = threshold_overlap(baseline_a, config.tau)
    drift_rows = []
    for eval_task in range(1, 6):
        with np.load(WB_DIR / f"WB_task{eval_task:02d}.npz", allow_pickle=False) as saved:
            eval_weights = saved["weights"]
        eval_a = overlap_from_codes(task1_test_codes, eval_weights)
        eval_z = threshold_overlap(eval_a, config.tau)
        drift_rows.append(
            drift_row(1, 1, eval_task, baseline_a, baseline_z, eval_a, eval_z)
        )

    write_csv(OUTPUT_DIR / "ca1_overlap_distribution.csv", distribution_rows)
    write_csv(OUTPUT_DIR / "ca1_readout_ablation.csv", readout_rows)
    write_csv(OUTPUT_DIR / "memory_drift.csv", drift_rows)
    write_csv(OUTPUT_DIR / "memory_geometry_before_after_ca1.csv", geometry_rows)
    write_csv(OUTPUT_DIR / "memory_statistics.csv", memory_rows)
    summary = {
        "tasks": 5,
        "parameter_changes": "none",
        "thresholded_final_cumulative_accuracy": readout_rows[-2]["cumulative_accuracy"],
        "unthresholded_final_cumulative_accuracy": readout_rows[-1]["cumulative_accuracy"],
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "ca1_diagnostics_round1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
