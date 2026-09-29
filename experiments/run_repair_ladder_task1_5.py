"""Isolated five-task repair ladder, without changing formal BTSP-CAM code.

REF is the previously validated sparse-ridge run on the same cached features.
R0/R1 use the original WB checkpoints; R2/R3 replay only the centered encoder.
All four BTSP readouts are newly fitted with the existing StreamingRidge class.
"""

from __future__ import annotations

import csv
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
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import pilot_class_order
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance

EXPERIMENT_CONFIG = Path(__file__).resolve().parent / "configs" / "repair_ladder_task1_5.yaml"
CACHE = ROOT / "data" / "feature_cache_btsp_cam"
WB_DIR = ROOT / "diagnostics" / "wb_checkpoints"
REF_CSV = ROOT / "diagnostics" / "representation_ablation.csv"
EXPECTED_BRANCHES = {
    "REF": "sparse_ridge_archived_reference",
    "R0": "raw_codes_evolving_wb",
    "R1": "raw_codes_wb_frozen_after_task1",
    "R2": "task1_centered_codes_evolving_wb",
    "R3": "task1_centered_codes_wb_frozen_after_task1",
}


def load_split(split: str, task: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(CACHE / f"{split}_task_{task:02d}.npz", allow_pickle=False) as data:
        h = data["features"].copy()
        y = data["labels"].copy()
    if not np.all(np.isfinite(h)):
        raise ValueError(f"nonfinite {split} features for task {task}")
    if set(np.unique(y).tolist()) != set(range((task - 1) * 5, task * 5)):
        raise ValueError(f"unexpected incremental labels for task {task}")
    return h, y


def original_wb(task: int, slots: int, dimensions: int) -> np.ndarray:
    with np.load(WB_DIR / f"WB_task{task:02d}.npz", allow_pickle=False) as data:
        weights = data["weights"].copy()
        if int(data["task"]) != task:
            raise ValueError("WB checkpoint task mismatch")
    if weights.shape != (slots, dimensions) or not np.all((weights == 0) | (weights == 1)):
        raise ValueError("WB checkpoint shape or binary invariant failed")
    return weights


def overlap(codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exactly sum active binary columns, then divide by per-row ||s||_1."""
    out = np.empty((len(codes), len(weights)), dtype=np.float64)
    for i, code in enumerate(codes):
        active = np.flatnonzero(code)
        if len(active) == 0:
            raise ValueError("empty sparse code")
        out[i] = weights[:, active].sum(axis=1, dtype=np.float64) / len(active)
    return out


def geometry(x: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Exact pairwise cosine means (exclude self-pairs)."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    unit = np.divide(x, norms[:, None], out=np.zeros_like(x), where=norms[:, None] > 0)
    class_sums = []
    same_sum = 0.0
    same_pairs = 0
    for label in np.unique(labels):
        rows = unit[labels == label]
        total = rows.sum(axis=0)
        class_sums.append((total, len(rows)))
        same_sum += float(total @ total - np.einsum("ij,ij->", rows, rows))
        same_pairs += len(rows) * (len(rows) - 1)
    diff_sum = 0.0
    diff_pairs = 0
    for i, (left, n_left) in enumerate(class_sums):
        for right, n_right in class_sums[i + 1 :]:
            diff_sum += float(left @ right)
            diff_pairs += n_left * n_right
    same = same_sum / same_pairs
    different = diff_sum / diff_pairs
    return same, different, same - different


def bit_and_collision_statistics(train_codes: list[np.ndarray], task: int) -> dict:
    current = train_codes[task - 1]
    cumulative_counts = np.zeros(current.shape[1], dtype=np.int64)
    for part in train_codes[:task]:
        cumulative_counts += part.sum(axis=0, dtype=np.int64)
    total = int(cumulative_counts.sum())
    top_n = int(np.ceil(0.01 * current.shape[1]))
    result = {
        "top1_bit_share": float(np.sort(cumulative_counts)[-top_n:].sum() / total),
        "zero_winner_bits": int(np.count_nonzero(cumulative_counts == 0)),
        "cross_task_mean_J": float("nan"),
        "P_J_ge_5": float("nan"),
        "cross_task_pair_count": 0,
    }
    if task == 1:
        return result
    old = sparse.csr_matrix(np.concatenate(train_codes[: task - 1]), dtype=np.uint8)
    new = sparse.csr_matrix(current, dtype=np.uint8)
    j_sum = 0
    ge5 = 0
    pair_count = 0
    for start in range(0, new.shape[0], 100):
        j = (new[start : start + 100] @ old.T).toarray()
        j_sum += int(j.sum(dtype=np.int64))
        ge5 += int(np.count_nonzero(j >= 5))
        pair_count += int(j.size)
    result.update(
        cross_task_mean_J=j_sum / pair_count,
        P_J_ge_5=ge5 / pair_count,
        cross_task_pair_count=pair_count,
    )
    return result


def accuracies(logits: np.ndarray, labels: np.ndarray, task: int) -> dict:
    current = (labels >= (task - 1) * 5) & (labels < task * 5)
    old = ~current
    prediction = logits.argmax(axis=1)
    five_way = logits[current, (task - 1) * 5 : task * 5].argmax(axis=1) + (task - 1) * 5
    return {
        "current_accuracy": float(np.mean(prediction[current] == labels[current])),
        "old_accuracy": float(np.mean(prediction[old] == labels[old])) if np.any(old) else float("nan"),
        "cumulative_accuracy": float(np.mean(prediction == labels)),
        "current_5way_accuracy": float(np.mean(five_way == labels[current])),
    }


def row(
    branch: str,
    task: int,
    scores: dict,
    sparse_geom: tuple[float, float, float],
    sparse_stats: dict,
    *,
    a_geom: tuple[float, float, float] | None = None,
    weights: np.ndarray | None = None,
    source: str,
) -> dict:
    column_occupancy = weights.mean(axis=0) if weights is not None else None
    return {
        "branch": branch,
        "task": task,
        **scores,
        "sparse_same_cos": sparse_geom[0],
        "sparse_diff_cos": sparse_geom[1],
        "sparse_margin": sparse_geom[2],
        "a_same_cos": a_geom[0] if a_geom else float("nan"),
        "a_diff_cos": a_geom[1] if a_geom else float("nan"),
        "a_margin": a_geom[2] if a_geom else float("nan"),
        "WB_density": float(weights.mean()) if weights is not None else float("nan"),
        "WB_column_occupancy_mean": float(column_occupancy.mean()) if column_occupancy is not None else float("nan"),
        "WB_column_occupancy_p99": float(np.quantile(column_occupancy, 0.99)) if column_occupancy is not None else float("nan"),
        "WB_column_occupancy_max": float(column_occupancy.max()) if column_occupancy is not None else float("nan"),
        **sparse_stats,
        "source": source,
    }


def update_ridge(ridge: StreamingRidge, features: np.ndarray, labels: np.ndarray) -> None:
    ridge.expand_classes(int(labels.max()) + 1)
    ridge.update(features, labels)
    ridge.solve()


def copy_ridge(ridge: StreamingRidge) -> StreamingRidge:
    copy = StreamingRidge(ridge.feature_dim, ridge.num_classes, ridge.gamma)
    copy.a = ridge.a.copy()
    copy.b = ridge.b.copy()
    copy.weights = ridge.weights.copy()
    copy.last_condition_number = ridge.condition_number
    copy.last_solve_status = ridge.last_solve_status
    return copy


def main() -> None:
    experiment = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    if experiment["tasks"] != 5 or experiment["branches"] != EXPECTED_BRANCHES:
        raise ValueError("this runner is restricted to the specified five-task ladder")
    if experiment["ca1"] != "unthresholded_overlap" or experiment["center"] != "task1_train_mean_frozen":
        raise ValueError("unexpected experimental intervention")
    config_path = (EXPERIMENT_CONFIG.parent / experiment["base_config"]).resolve()
    config = BTSPCAMConfig.from_mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    if config.seed != 1993 or config.k != 41 or config.memory_slots != 4096:
        raise ValueError("formal shared settings differ from expected configuration")
    out_path = (EXPERIMENT_CONFIG.parent / experiment["output_csv"]).resolve()
    if out_path != ROOT / "diagnostics" / "repair_ladder_task1_5.csv":
        raise ValueError("output must be the isolated diagnostic CSV")

    provenance = RNGProvenance.from_seed(config.seed)
    model = BTSPCAM(config, class_order=pilot_class_order(config.num_classes, config.seed), rng_provenance=provenance)
    with np.load(ROOT / "results" / "btsp_cam_cifar100" / "checkpoint_task1.npz", allow_pickle=False) as checkpoint:
        np.testing.assert_array_equal(model.encoder.projection, checkpoint["projection"])
        np.testing.assert_array_equal(model.encoder.bias, checkpoint["bias"])
    train = [load_split("train", t) for t in range(1, 6)]
    test = [load_split("test", t) for t in range(1, 6)]
    mu_base = np.asarray(train[0][0], dtype=np.float64).mean(axis=0)
    code_sets = {}
    for name, center in (("raw", None), ("centered", mu_base)):
        code_sets[name] = {
            "train": [model.encoder.encode(h if center is None else np.asarray(h, dtype=np.float64) - center, config.encoder_batch_size) for h, _ in train],
            "test": [model.encoder.encode(h if center is None else np.asarray(h, dtype=np.float64) - center, config.encoder_batch_size) for h, _ in test],
        }
        for codes in code_sets[name]["train"] + code_sets[name]["test"]:
            if not np.all(codes.sum(axis=1) == config.k):
                raise AssertionError("sparse encoder lost exact k")
        print(json.dumps({"encoded": name, "tasks": 5}), flush=True)

    with REF_CSV.open(newline="", encoding="utf-8") as handle:
        reference = {int(r["task"]): r for r in csv.DictReader(handle) if r["method"] == "sparse_ridge"}
    if set(reference) != set(range(1, 6)) or abs(float(reference[5]["cumulative_acc"]) - 0.6516) > 1e-12:
        raise ValueError("archived sparse-ridge reference differs from validated run")

    ridge: dict[str, StreamingRidge] = {}
    rows: list[dict] = []
    raw_wb1 = original_wb(1, config.memory_slots, config.code_dim)
    started = time.perf_counter()
    for task in range(1, 6):
        task_started = time.perf_counter()
        current_train_y = train[task - 1][1]
        current_test_y = test[task - 1][1]
        cumulative_test_y = np.concatenate([y for _, y in test[:task]])
        family: dict[str, dict] = {}
        for name in ("raw", "centered"):
            codes = code_sets[name]
            current_test_s = codes["test"][task - 1]
            sparse_geom = geometry(current_test_s, current_test_y)
            sparse_stats = bit_and_collision_statistics(codes["train"], task)
            family[name] = {
                "train_s": codes["train"][task - 1],
                "test_s": np.concatenate(codes["test"][:task]),
                "sparse_geom": sparse_geom,
                "sparse_stats": sparse_stats,
            }

        ref = reference[task]
        rows.append(row(
            "REF", task,
            {"current_accuracy": float(ref["current_acc"]), "old_accuracy": float(ref["old_acc"]),
             "cumulative_accuracy": float(ref["cumulative_acc"]), "current_5way_accuracy": float(ref["current_task_5way_acc"])},
            family["raw"]["sparse_geom"], family["raw"]["sparse_stats"],
            source="validated_archived_sparse_ridge",
        ))

        # The original (raw-code) WB trajectory is immutable input to R0/R1.
        raw_wb = original_wb(task, config.memory_slots, config.code_dim)
        if task == 1:
            np.testing.assert_array_equal(raw_wb, raw_wb1)

        # Re-run only the centered write trajectory, using the formal _learn_code
        # and identical plateau/synapse seeds and cached sample order.
        for code, label in zip(family["centered"]["train_s"], current_train_y):
            model._learn_code(code, int(label))
        centered_wb = model.memory.weights
        if task == 1:
            centered_wb1 = centered_wb.copy()

        branch_weights = {"R0": raw_wb, "R1": raw_wb1,
                          "R2": centered_wb, "R3": centered_wb1}
        for branch in ("R0", "R1", "R2", "R3"):
            name = "raw" if branch in ("R0", "R1") else "centered"
            s = family[name]
            weights = branch_weights[branch]
            if task == 1 and branch in ("R1", "R3"):
                twin = "R0" if branch == "R1" else "R2"
                ridge[branch] = copy_ridge(ridge[twin])
                a_test = family[name]["test_a_" + twin]
            else:
                train_a = overlap(s["train_s"], weights)
                if branch not in ridge:
                    ridge[branch] = StreamingRidge(config.memory_slots, 0, config.gamma)
                update_ridge(ridge[branch], train_a, current_train_y)
                a_test = overlap(s["test_s"], weights)
            family[name]["test_a_" + branch] = a_test
            scores = accuracies(ridge[branch].logits(a_test), cumulative_test_y, task)
            a_geom = geometry(a_test[-len(current_test_y):], current_test_y)
            rows.append(row(branch, task, scores, s["sparse_geom"], s["sparse_stats"],
                            a_geom=a_geom, weights=weights, source="experimental_repair_ladder"))

        task_rows = {r["branch"]: r for r in rows if r["task"] == task}
        print(json.dumps({
            "task": task,
            "cumulative": {branch: task_rows[branch]["cumulative_accuracy"] for branch in EXPECTED_BRANCHES},
            "seconds": round(time.perf_counter() - task_started, 1),
        }), flush=True)

    fields = list(rows[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(out_path), "rows": len(rows),
                      "seconds": round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
