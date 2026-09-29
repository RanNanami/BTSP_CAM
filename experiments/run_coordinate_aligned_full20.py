"""Experimental 20-task unthresholded C/D scaling test.

The aligned branch retains only aggregate C and D across tasks. Archived
formal W_B checkpoints reproduce the unchanged write trajectory. Historical
cache access is confined to evaluation and explicit replay-oracle checks.
"""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from scipy import sparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_repair_ladder_task1_5 import load_split, overlap
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder

CONFIG_PATH = ROOT / "experiments" / "configs" / "coordinate_aligned_full20.yaml"
ARCHIVED_LADDER = ROOT / "diagnostics" / "repair_ladder_task1_5.csv"
ARCHIVED_ALIGNED = ROOT / "diagnostics" / "coordinate_aligned_ridge.csv"
MIB = 1024 * 1024


def formal_wb(task: int, slots: int, code_dim: int) -> np.ndarray:
    path = ROOT / "results" / "btsp_cam_cifar100" / f"checkpoint_task{task}.npz"
    with np.load(path, allow_pickle=False) as state:
        w = state["memory_weights"].copy()
    if w.shape != (slots, code_dim) or w.dtype != np.uint8 or not np.all((w == 0) | (w == 1)):
        raise ValueError(f"invalid formal memory checkpoint at task {task}")
    return w


def fast_overlap(codes: np.ndarray, w_transposed_float: np.ndarray, k: int) -> np.ndarray:
    """Float32 accumulates at most 41 integer ones exactly; divide in float64."""
    counts = sparse.csr_matrix(codes, dtype=np.float32) @ w_transposed_float
    return np.asarray(counts, dtype=np.float64) / k


class CosineMoments:
    """Exact class-pair cosine means without retaining individual test rows."""

    def __init__(self, feature_dim: int):
        self.sums: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}
        self.squared_norm_sums: dict[int, float] = {}
        self.feature_dim = feature_dim

    def add(self, values: np.ndarray, labels: np.ndarray) -> None:
        norm = np.linalg.norm(values, axis=1)
        unit = np.divide(values, norm[:, None], out=np.zeros_like(values), where=norm[:, None] > 0)
        for label in np.unique(labels):
            rows = unit[labels == label]
            key = int(label)
            self.sums[key] = self.sums.get(key, np.zeros(self.feature_dim)) + rows.sum(axis=0)
            self.counts[key] = self.counts.get(key, 0) + len(rows)
            self.squared_norm_sums[key] = self.squared_norm_sums.get(key, 0.0) + float(np.einsum("ij,ij->", rows, rows))

    def values(self) -> tuple[float, float, float]:
        if len(self.sums) < 2:
            return float("nan"), float("nan"), float("nan")
        keys = sorted(self.sums)
        same_pairs = sum(self.counts[k] * (self.counts[k] - 1) for k in keys)
        same_sum = sum(float(self.sums[k] @ self.sums[k]) - self.squared_norm_sums[k] for k in keys)
        all_sum = sum((self.sums[k] for k in keys), np.zeros(self.feature_dim))
        total_n = sum(self.counts.values())
        diff_pairs = total_n * total_n - sum(self.counts[k] ** 2 for k in keys)
        diff_sum = float(all_sum @ all_sum) - sum(float(self.sums[k] @ self.sums[k]) for k in keys)
        same = same_sum / same_pairs
        different = diff_sum / diff_pairs
        return same, different, same - different


def output_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_early(task: int, scores: dict[str, dict]) -> None:
    if task > 5:
        return
    with ARCHIVED_LADDER.open(newline="", encoding="utf-8") as handle:
        prior = {(r["branch"], int(r["task"])): r for r in csv.DictReader(handle)}
    with ARCHIVED_ALIGNED.open(newline="", encoding="utf-8") as handle:
        aligned = {int(r["task"]): r for r in csv.DictReader(handle)}
    targets = {
        "official_evolving_unthresholded": float(prior[("R0", task)]["cumulative_accuracy"]),
        "frozen_WB1_unthresholded": float(prior[("R1", task)]["cumulative_accuracy"]),
        "coordinate_aligned_ridge": float(aligned[task]["coordinate_aligned_cumulative_accuracy"]),
    }
    for name, target in targets.items():
        got = scores[name]["cumulative_accuracy"]
        if abs(got - target) > 1e-12:
            raise AssertionError(f"archived Task{task} {name}: {got} != {target}")


def replay_oracle(task: int, config: BTSPCAMConfig, encoder: RandomSparseEncoder,
                  w_transposed_float: np.ndarray) -> StreamingRidge:
    """Diagnostic-only replay; never supplies old examples to C/D updates."""
    ridge = StreamingRidge(config.memory_slots, task * 5, config.gamma)
    for source_task in range(1, task + 1):
        h, y = load_split("train", source_task)
        s = encoder.encode(h, config.encoder_batch_size)
        z = fast_overlap(s, w_transposed_float, config.k)
        ridge.update(z, y)
        del h, s, z
    ridge.solve()
    ridge._last_cholesky = None
    return ridge


def main() -> None:
    experiment = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if (experiment["variant"], experiment["tasks"], experiment["readout"],
            experiment["oracle_checkpoints"]) != (
                "coordinate_aligned_ridge", 20, "unthresholded_overlap", [1, 5, 10, 15, 20]):
        raise ValueError("unexpected experimental variant")
    config_path = (CONFIG_PATH.parent / experiment["base_config"]).resolve()
    config = BTSPCAMConfig.from_mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    if (config.seed, config.code_dim, config.k, config.memory_slots, config.num_classes) != (1993, 8192, 41, 4096, 100):
        raise ValueError("formal model parameters changed")
    output_full = (CONFIG_PATH.parent / experiment["full20_csv"]).resolve()
    output_scale = (CONFIG_PATH.parent / experiment["scaling_csv"]).resolve()
    if output_full != ROOT / "diagnostics" / "coordinate_aligned_full20.csv" or output_scale != ROOT / "diagnostics" / "coordinate_aligned_scaling.csv":
        raise ValueError("unexpected experiment output location")
    encoder = RandomSparseEncoder(config.feature_dim, config.code_dim, config.k,
                                  RNGProvenance.from_seed(config.seed).projection_seed)
    with np.load(ROOT / "results" / "btsp_cam_cifar100" / "checkpoint_task1.npz", allow_pickle=False) as state:
        np.testing.assert_array_equal(encoder.projection, state["projection"])
        np.testing.assert_array_equal(encoder.bias, state["bias"])
    wb1 = formal_wb(1, config.memory_slots, config.code_dim)
    wb1_float_t = np.ascontiguousarray(wb1.T, dtype=np.float32)
    aligned = CoordinateAlignedRidge(config.code_dim, config.memory_slots, config.k, config.gamma)
    official = StreamingRidge(config.memory_slots, 0, config.gamma)
    frozen = StreamingRidge(config.memory_slots, 0, config.gamma)
    rows: list[dict] = []
    scales: list[dict] = []
    process = psutil.Process()
    total_start = time.perf_counter()

    for task in range(1, 21):
        start = time.perf_counter()
        wb = wb1 if task == 1 else formal_wb(task, config.memory_slots, config.code_dim)
        wt = wb1_float_t if task == 1 else np.ascontiguousarray(wb.T, dtype=np.float32)
        h, y = load_split("train", task)
        codes = encoder.encode(h, config.encoder_batch_size)
        del h
        if not np.all(codes.sum(axis=1) == config.k):
            raise AssertionError("sparse code is not exact k")
        if task == 1:
            direct = overlap(codes[:8], wb)
            np.testing.assert_array_equal(fast_overlap(codes[:8], wt, config.k), direct)
        aligned.update(codes, y)
        update_seconds = aligned.last_update["update_seconds"]
        current_dynamic = fast_overlap(codes, wt, config.k)
        current_frozen = current_dynamic if task == 1 else fast_overlap(codes, wb1_float_t, config.k)
        del codes
        official.expand_classes(task * 5)
        frozen.expand_classes(task * 5)
        official.update(current_dynamic, y)
        frozen.update(current_frozen, y)
        del current_dynamic, current_frozen, y
        official_solve_start = time.perf_counter()
        official.solve()
        official_solve_seconds = time.perf_counter() - official_solve_start
        official._last_cholesky = None
        frozen_solve_start = time.perf_counter()
        frozen.solve()
        frozen_solve_seconds = time.perf_counter() - frozen_solve_start
        frozen._last_cholesky = None
        a_aligned, b_aligned = aligned.aligned_statistics(wb)
        alignment_seconds = aligned.last_alignment["alignment_seconds"]
        aligned_ridge, aligned_solve_seconds = aligned.solve(a_aligned, b_aligned)
        aligned_ridge._last_cholesky = None

        check = task in experiment["oracle_checkpoints"]
        oracle = replay_oracle(task, config, encoder, wt) if check else None
        checks = {
            "oracle_checked": int(check), "A_max_abs_diff": float("nan"),
            "B_max_abs_diff": float("nan"), "ridge_weight_max_abs_diff": float("nan"),
            "logit_max_abs_diff": float("nan"), "oracle_prediction_equal": "",
            "oracle_cumulative_accuracy": float("nan"),
        }
        if oracle is not None:
            checks["A_max_abs_diff"] = float(np.max(np.abs(a_aligned - oracle.a)))
            checks["B_max_abs_diff"] = float(np.max(np.abs(b_aligned - oracle.b)))
            checks["ridge_weight_max_abs_diff"] = float(np.max(np.abs(aligned_ridge.weights - oracle.weights)))
        del a_aligned, b_aligned
        gc.collect()

        geometry = {"old": CosineMoments(config.memory_slots), "current": CosineMoments(config.memory_slots)}
        geometry_frozen = {"old": CosineMoments(config.memory_slots), "current": CosineMoments(config.memory_slots)}
        counts = {name: {"old": [0, 0], "current": [0, 0], "five_way": [0, 0]}
                  for name in ("official_evolving_unthresholded", "coordinate_aligned_ridge", "frozen_WB1_unthresholded")}
        oracle_correct = oracle_total = 0
        oracle_logit_diff = 0.0
        oracle_prediction_equal = True
        eval_start = time.perf_counter()
        for source_task in range(1, task + 1):
            h, labels = load_split("test", source_task)
            test_codes = encoder.encode(h, config.encoder_batch_size)
            del h
            dynamic = fast_overlap(test_codes, wt, config.k)
            stable = dynamic if task == 1 else fast_overlap(test_codes, wb1_float_t, config.k)
            del test_codes
            source = "current" if source_task == task else "old"
            geometry[source].add(dynamic, labels)
            geometry_frozen[source].add(stable, labels)
            logits_by_branch = {
                "official_evolving_unthresholded": official.logits(dynamic),
                "coordinate_aligned_ridge": aligned_ridge.logits(dynamic),
                "frozen_WB1_unthresholded": frozen.logits(stable),
            }
            for name, logits in logits_by_branch.items():
                correct, total = counts[name][source]
                counts[name][source] = [correct + int(np.count_nonzero(logits.argmax(axis=1) == labels)), total + len(labels)]
                if source == "current":
                    selected = logits[:, (task - 1) * 5: task * 5]
                    five_correct = int(np.count_nonzero(selected.argmax(axis=1) + (task - 1) * 5 == labels))
                    counts[name]["five_way"] = [five_correct, len(labels)]
            if oracle is not None:
                oracle_logits = oracle.logits(dynamic)
                aligned_logits = logits_by_branch["coordinate_aligned_ridge"]
                oracle_logit_diff = max(oracle_logit_diff, float(np.max(np.abs(aligned_logits - oracle_logits))))
                oracle_prediction_equal &= bool(np.array_equal(aligned_logits.argmax(axis=1), oracle_logits.argmax(axis=1)))
                oracle_correct += int(np.count_nonzero(oracle_logits.argmax(axis=1) == labels))
                oracle_total += len(labels)
            del dynamic, stable, logits_by_branch
        eval_seconds = time.perf_counter() - eval_start
        if oracle is not None:
            checks["logit_max_abs_diff"] = oracle_logit_diff
            checks["oracle_prediction_equal"] = int(oracle_prediction_equal)
            checks["oracle_cumulative_accuracy"] = oracle_correct / oracle_total
            if (checks["A_max_abs_diff"] > 1e-6 or checks["B_max_abs_diff"] > 1e-7
                    or checks["ridge_weight_max_abs_diff"] > 1e-5
                    or checks["logit_max_abs_diff"] > 1e-5 or not oracle_prediction_equal):
                raise AssertionError(f"oracle numerical closure failed at Task{task}: {checks}")
            del oracle
        metrics: dict[str, dict] = {}
        for name, count in counts.items():
            old_correct, old_total = count["old"]
            current_correct, current_total = count["current"]
            metrics[name] = {
                "current_accuracy": current_correct / current_total,
                "old_accuracy": old_correct / old_total if old_total else float("nan"),
                "cumulative_accuracy": (old_correct + current_correct) / (old_total + current_total),
                "current_5way_accuracy": count["five_way"][0] / count["five_way"][1],
            }
        validate_early(task, metrics)
        if check and abs(checks["oracle_cumulative_accuracy"] - metrics["coordinate_aligned_ridge"]["cumulative_accuracy"]) > 1e-12:
            raise AssertionError("oracle prediction parity failed")
        wb_density = float(wb.mean())
        for name, metric in metrics.items():
            family = geometry_frozen if name == "frozen_WB1_unthresholded" else geometry
            old_geom = family["old"].values()
            current_geom = family["current"].values()
            rows.append({
                "task": task, "branch": name, **metric,
                "WB_density": float(wb1.mean()) if name == "frozen_WB1_unthresholded" else wb_density,
                "old_a_same_cos": old_geom[0], "old_a_diff_cos": old_geom[1], "old_a_margin": old_geom[2],
                "current_a_same_cos": current_geom[0], "current_a_diff_cos": current_geom[1], "current_a_margin": current_geom[2],
                "oracle_checked": checks["oracle_checked"],
                "oracle_cumulative_accuracy": checks["oracle_cumulative_accuracy"] if name == "coordinate_aligned_ridge" else float("nan"),
                "readout": "unthresholded_overlap", "source": "experimental_full20_formal_WB_checkpoints",
            })

        storage = aligned.storage()
        rss = process.memory_info()
        scales.append({
            "task": task, "seen_classes": storage["seen_classes"], "seen_samples": storage["seen_samples"],
            "C_nnz": storage["C_nnz"], "C_density": storage["C_density"],
            "C_csr_float64_actual_MiB": storage["C_csr_float64_actual_bytes"] / MIB,
            "C_csr_float32_est_MiB": storage["C_csr_float32_est_bytes"] / MIB,
            "C_coo_float64_est_MiB": storage["C_coo_float64_est_bytes"] / MIB,
            "C_dense_float32_est_MiB": storage["C_dense_float32_bytes"] / MIB,
            "C_dense_float64_est_MiB": storage["C_dense_float64_bytes"] / MIB,
            "D_float64_actual_MiB": storage["D_float64_actual_bytes"] / MIB,
            "D_float32_est_MiB": storage["D_float32_est_bytes"] / MIB,
            "CSR_ge_dense_float32": int(storage["C_csr_float64_actual_bytes"] >= storage["C_dense_float32_bytes"]),
            "CSR_ge_dense_float64": int(storage["C_csr_float64_actual_bytes"] >= storage["C_dense_float64_bytes"]),
            "C_new_nnz": aligned.last_update["c_new_nnz"],
            "task_pair_contributions": aligned.last_update["pair_contributions"],
            "task_gram_nnz": aligned.last_update["task_gram_nnz"],
            "C_D_update_seconds": update_seconds, "WCWt_alignment_seconds": alignment_seconds,
            "aligned_ridge_solve_seconds": aligned_solve_seconds,
            "official_ridge_solve_seconds": official_solve_seconds,
            "frozen_ridge_solve_seconds": frozen_solve_seconds,
            "evaluation_seconds": eval_seconds, "task_total_seconds": time.perf_counter() - start,
            "process_working_set_end_MiB": rss.rss / MIB,
            "process_peak_working_set_to_date_MiB": getattr(rss, "peak_wset", rss.rss) / MIB,
            "host_available_RAM_end_MiB": psutil.virtual_memory().available / MIB,
            **checks, "C_storage_format": "CSR_float64", "variant_state": "C_D_only_no_historical_examples",
        })
        output_rows(output_full, rows)
        output_rows(output_scale, scales)
        print(json.dumps({
            "task": task, "cumulative": {name: metrics[name]["cumulative_accuracy"] for name in metrics},
            "C_nnz": storage["C_nnz"], "C_density": storage["C_density"],
            "C_CSR_MiB": scales[-1]["C_csr_float64_actual_MiB"],
            "task_seconds": round(scales[-1]["task_total_seconds"], 2),
            "oracle": checks["oracle_cumulative_accuracy"] if check else None,
        }), flush=True)
        del aligned_ridge, geometry, geometry_frozen, wt
        if task != 1:
            del wb
        gc.collect()
    print(json.dumps({"completed_tasks": 20, "seconds": time.perf_counter() - total_start,
                      "full20_csv": str(output_full), "scaling_csv": str(output_scale)}), flush=True)


if __name__ == "__main__":
    main()
