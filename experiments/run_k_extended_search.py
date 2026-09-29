"""Predeclared train-only extended k search and locked full20 confirmation.

Only experimental files are written. Formal BTSP-CAM code, YAML, and
checkpoints remain untouched. `select` reads validation CSV only; `full20`
refuses to run until a selection lock exists.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import pilot_class_order
from btsp_cam.rng import RNGProvenance
from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_coordinate_aligned_full20 import CosineMoments
from run_k_robustness_and_selection import development_split, score_sources, solve_aligned
from run_repair_ladder_task1_5 import load_split
from run_unresolved_parameter_screen import overlap, read_rows, write_rows

CONFIG = ROOT / "experiments/configs/k_extended_search.yaml"
OLD = ROOT / "diagnostics/k_train_only_selection.csv"
PROCESS = psutil.Process()


def protocol() -> tuple[dict, BTSPCAMConfig, dict[str, Path]]:
    exp = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg_path = (CONFIG.parent / exp["base_config"]).resolve()
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(cfg_path.read_text(encoding="utf-8")))
    if exp["grid"] != [128, 160, 192, 224, 256, 320, 384, 512, 768, 1024]:
        raise ValueError("predeclared grid changed")
    if (base.seed, base.code_dim, base.memory_slots, base.gamma,
            base.trace_decay, base.flip_probability) != (1993, 8192, 4096, 0.1, 0.01, 0.5):
        raise ValueError("formal baseline settings changed")
    if (exp["validation_split_seed"], exp["validation_fraction"],
            exp["class_order_seed"], exp["rf_seed"], exp["writer_seed"],
            exp["validation_tasks"], exp["tie_tolerance_percentage_points"],
            exp["tau"], exp["gamma"]) != (20260928, 0.2, 1993, 1993, 1993, 5, 0.10, 0.0, 0.1):
        raise ValueError("predeclared protocol changed")
    paths = {name: (CONFIG.parent / path).resolve() for name, path in exp["outputs"].items()}
    return exp, base, paths


def make_model(base: BTSPCAMConfig, k: int) -> BTSPCAM:
    cfg = replace(base, k=k, tau=0.0)
    rf = RNGProvenance.from_seed(1993)
    wr = RNGProvenance.from_seed(1993)
    model = BTSPCAM(cfg, class_order=pilot_class_order(100, 1993),
                    projection_seed=rf.projection_seed,
                    plateau_seed=wr.plateau_seed,
                    synapse_seed=wr.synapse_seed)
    model.ridge = None  # readout is the separate coordinate-aligned variant
    return model


def complete(rows: list[dict], k: int, tasks: int) -> bool:
    own = [row for row in rows if int(row["k"]) == k]
    return len(own) == tasks and {int(row["task"]) for row in own} == set(range(1, tasks + 1))


def record(path: Path, k: int, new: list[dict], grid: list[int]) -> None:
    old = read_rows(path) if path.exists() else []
    rows = [row for row in old if int(row["k"]) != k] + new
    order = {value: idx for idx, value in enumerate(grid)}
    rows.sort(key=lambda row: (order[int(row["k"])], int(row["task"])))
    write_rows(path, rows)


def class_overlap_geometry(codes: list[sparse.csr_matrix], labels: list[np.ndarray],
                           k: int, bit_dim: int) -> tuple[float, float, float, float, float]:
    """Exact ordered-pair J means; sparse cosine is J/k for exact-k codes."""
    sums = {}
    sizes = {}
    for s, y in zip(codes, labels):
        for label in np.unique(y):
            label = int(label)
            row = np.asarray(s[y == label].sum(axis=0)).ravel().astype(np.int64)
            sums[label] = sums.get(label, np.zeros(bit_dim, dtype=np.int64)) + row
            sizes[label] = sizes.get(label, 0) + int(np.count_nonzero(y == label))
    all_sum = np.zeros(bit_dim, dtype=np.int64)
    same_dot = same_pairs = self_dot = class_sum_sq = class_n_sq = 0
    total_n = sum(sizes.values())
    for label, v in sums.items():
        n = sizes[label]
        sq = int(v @ v)
        same_dot += sq - n * k
        same_pairs += n * (n - 1)
        class_sum_sq += sq
        class_n_sq += n * n
        all_sum += v
    diff_dot = int(all_sum @ all_sum) - class_sum_sq
    diff_pairs = total_n * total_n - class_n_sq
    mean_same = same_dot / same_pairs
    mean_diff = diff_dot / diff_pairs
    return mean_same / k, mean_diff / k, (mean_same - mean_diff) / k, mean_same, mean_diff


def representation(model: BTSPCAM, k: int, eval_s: list[sparse.csr_matrix],
                   eval_y: list[np.ndarray], dev_bit_counts: np.ndarray) -> dict:
    sparse_same, sparse_diff, sparse_margin, mean_same_j, mean_diff_j = (
        class_overlap_geometry(eval_s, eval_y, k, model.config.code_dim))
    weighted = dev_bit_counts.astype(np.float64)
    weighted /= weighted.sum()
    positive = weighted[weighted > 0]
    top = int(np.ceil(0.01 * len(weighted)))
    ca1_moments = CosineMoments(model.config.memory_slots)
    wt = np.ascontiguousarray(model.memory.weights.T, dtype=np.float32)
    a_sum = a_sq = 0.0
    a_elements = a_positive = 0
    for s, y in zip(eval_s, eval_y):
        a = overlap(s, wt, k)
        ca1_moments.add(a, y)
        a_sum += float(a.sum())
        a_sq += float(np.square(a).sum())
        a_elements += a.size
        a_positive += int(np.count_nonzero(a > 0))
    a_same, a_diff, a_margin = ca1_moments.values()
    a_mean = a_sum / a_elements
    return dict(sparse_same_cos=sparse_same, sparse_diff_cos=sparse_diff,
                sparse_margin=sparse_margin, mean_J_same=mean_same_j,
                mean_J_diff=mean_diff_j,
                effective_support_entropy=float(np.exp(-np.sum(positive * np.log(positive)))),
                top1_bit_activation_share=float(np.sort(weighted)[-top:].sum()),
                zero_winner_RF_bits=int(np.count_nonzero(dev_bit_counts == 0)),
                bit_frequency_cv=float(np.std(dev_bit_counts) / np.mean(dev_bit_counts)),
                a_same_cos=a_same, a_diff_cos=a_diff, a_margin=a_margin,
                a_mean=a_mean,
                a_std=float(np.sqrt(max(a_sq / a_elements - a_mean * a_mean, 0.0))),
                a_density=a_positive / a_elements)


def memory_stats(wb: np.ndarray) -> dict:
    row = wb.sum(axis=1, dtype=np.int64) / wb.shape[1]
    col_counts = wb.sum(axis=0, dtype=np.int64)
    col = col_counts / wb.shape[0]
    top = int(np.ceil(0.01 * len(col)))
    return dict(WB_density=float(wb.mean()), row_density_mean=float(row.mean()),
                row_density_p90=float(np.percentile(row, 90)), row_density_max=float(row.max()),
                column_occupancy_p50=float(np.percentile(col, 50)),
                column_occupancy_p90=float(np.percentile(col, 90)),
                column_occupancy_p99=float(np.percentile(col, 99)),
                column_occupancy_max=float(col.max()),
                top1_column_occupancy_share=float(np.sort(col_counts)[-top:].sum() / col_counts.sum()))


def writer_update(model: BTSPCAM, codes: np.ndarray, labels: np.ndarray) -> dict:
    plateaus = flips01 = flips10 = 0
    for code, label in zip(codes, labels):
        log = model._learn_code(code, int(label))
        plateaus += int(log["plateau_count"])
        flips01 += int(log["zero_to_one_flips"])
        flips10 += int(log["one_to_zero_flips"])
    gross = flips01 + flips10
    return dict(plateau_count_task=plateaus, flip_01_task=flips01,
                flip_10_task=flips10, gross_flips_task=gross,
                gross_flips_per_sample=gross / len(labels),
                gross_flips_per_plateau=gross / plateaus if plateaus else float("nan"))


def check_controls(rows: list[dict], exp: dict) -> None:
    old = [r for r in read_rows(OLD) if r["phase"] == "development_validation"]
    for k in exp["control_k"]:
        if not complete(rows, k, 5):
            raise ValueError(f"control k={k} not complete")
        for task in range(1, 6):
            got = next(r for r in rows if int(r["k"]) == k and int(r["task"]) == task)
            ref = next(r for r in old if int(r["k"]) == k and int(r["task"]) == task)
            for field, old_field in [("cumulative_accuracy", "cumulative_accuracy"),
                                     ("current_accuracy", "current_accuracy"),
                                     ("current_5way_accuracy", "current_5way_accuracy"),
                                     ("validation_average_incremental_to_task",
                                      "validation_average_incremental_to_task")]:
                if abs(float(got[field]) - float(ref[old_field])) > 1e-12:
                    raise AssertionError(f"control mismatch k={k} task={task} {field}: "
                                         f"{got[field]} vs {ref[old_field]}")
    print(json.dumps({"control_status": "exact_accuracy_match_all_tasks",
                      "control_k": exp["control_k"]}), flush=True)


def validate_k(exp: dict, base: BTSPCAMConfig, paths: dict[str, Path], k: int) -> None:
    rows = read_rows(paths["validation"]) if paths["validation"].exists() else []
    if complete(rows, k, 5):
        print(json.dumps({"phase": "validation", "k": k, "skipped": "complete"}), flush=True)
        return
    if k not in exp["control_k"]:
        check_controls(rows, exp)
    model = make_model(base, k)
    aligned = CoordinateAlignedRidge(base.code_dim, base.memory_slots, k, base.gamma)
    val_s: list[sparse.csr_matrix] = []
    val_y: list[np.ndarray] = []
    bit_counts = np.zeros(base.code_dim, dtype=np.int64)
    result = []
    running = 0.0
    cumulative_plateaus = cumulative_flips = 0
    for task in range(1, 6):
        start = time.perf_counter()
        h, y = load_split("train", task)
        dev_i, val_i = development_split(y, exp["validation_split_seed"],
                                          exp["validation_fraction"])
        if len(dev_i) != 2000 or len(val_i) != 500:
            raise AssertionError("per-class dev/validation split changed")
        s_dev = model.encoder.encode(h[dev_i], base.encoder_batch_size)
        s_val = model.encoder.encode(h[val_i], base.encoder_batch_size)
        del h
        bit_counts += s_dev.sum(axis=0, dtype=np.int64)
        writer = writer_update(model, s_dev, y[dev_i])
        aligned.update(s_dev, y[dev_i])
        cumulative_plateaus += writer["plateau_count_task"]
        cumulative_flips += writer["gross_flips_task"]
        val_s.append(sparse.csr_matrix(s_val))
        val_y.append(y[val_i].copy())
        del s_dev, s_val, y
        ridge = solve_aligned(aligned, model.memory.weights)
        score = score_sources(ridge, model.memory.weights, k, val_s, val_y, task)
        running += score["cumulative_accuracy"]
        stats = representation(model, k, val_s, val_y, bit_counts)
        wb_stats = memory_stats(model.memory.weights)
        row = dict(phase="train_only_validation", k=k, k_over_m=k / base.code_dim,
                   task=task, split_seed=exp["validation_split_seed"],
                   development_samples_per_class=400, validation_samples_per_class=100,
                   rf_seed=exp["rf_seed"], writer_seed=exp["writer_seed"],
                   gamma=base.gamma, tau=0.0,
                   **score, validation_average_incremental_to_task=running / task,
                   **writer, cumulative_plateau_count=cumulative_plateaus,
                   cumulative_gross_flips=cumulative_flips,
                   cumulative_gross_flips_per_sample=cumulative_flips / (task * 2000),
                   **stats, **wb_stats,
                   task_seconds=time.perf_counter() - start,
                   process_rss_mib=PROCESS.memory_info().rss / 2**20)
        result.append(row)
        record(paths["validation"], k, result, exp["grid"])
        print(json.dumps({"phase": "validation", "k": k, "task": task,
                          "cumulative": score["cumulative_accuracy"],
                          "mean": running / task,
                          "seconds": round(row["task_seconds"], 1),
                          "rss_mib": round(row["process_rss_mib"])}), flush=True)
        del ridge
        gc.collect()
    if k in exp["control_k"]:
        rows = read_rows(paths["validation"])
        if all(complete(rows, x, 5) for x in exp["control_k"]):
            check_controls(rows, exp)


def selection_classification(grid: list[int], averages: dict[int, float],
                             tolerance: float) -> str:
    best = max(averages.values())
    max_k = grid[-1]
    # A higher boundary with a nontrivial final step is not a bracketed optimum.
    if averages[max_k] >= best - tolerance and averages[max_k] - averages[grid[-2]] > tolerance:
        return "STILL_INCREASING_AT_BOUNDARY"
    peaks = [k for k in grid if averages[k] >= best - tolerance]
    if len(peaks) >= 2 and any(grid.index(b) == grid.index(a) + 1 for a, b in zip(peaks, peaks[1:])):
        return "PLATEAU"
    winner = min(peaks)
    if winner < max_k and sum(averages[k] < best - tolerance for k in grid if k > winner) >= 2:
        return "INTERIOR_PEAK"
    return "NOISY/INCONCLUSIVE"


def select(exp: dict, paths: dict[str, Path]) -> dict:
    if paths["selection_lock"].exists():
        raise ValueError("selection already locked; refusing to reselect")
    rows = read_rows(paths["validation"])
    check_controls(rows, exp)
    for k in exp["grid"]:
        if not complete(rows, k, 5):
            raise ValueError(f"validation incomplete for k={k}")
    if len(rows) != len(exp["grid"]) * 5:
        raise ValueError("duplicate or extra validation rows")
    averages = {k: float(next(r for r in rows if int(r["k"]) == k and int(r["task"]) == 5)
                         ["validation_average_incremental_to_task"]) for k in exp["grid"]}
    best = max(averages.values())
    tolerance = exp["tie_tolerance_percentage_points"] / 100.0
    selected = min(k for k in exp["grid"] if best - averages[k] <= tolerance + 1e-15)
    category = selection_classification(exp["grid"], averages, tolerance)
    index = exp["grid"].index(selected)
    if selected == 1024:
        neighbors = [768, 1024]
    elif selected == 128:
        neighbors = [128, 160]
    else:
        neighbors = exp["grid"][index - 1:index + 2]
    lock = {"selected_k": selected, "classification": category,
            "validation_mean_by_k": {str(k): averages[k] for k in exp["grid"]},
            "max_validation_mean": best,
            "tie_tolerance_percentage_points": exp["tie_tolerance_percentage_points"],
            "full20_k": neighbors,
            "selection_used_test_accuracy": False,
            "selection_metric": exp["primary_selection_metric"],
            "grid": exp["grid"], "config_path": str(CONFIG)}
    paths["selection_lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["selection_lock"].write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "locked_selection_before_test", **lock}), flush=True)
    return lock


def full20_k(exp: dict, base: BTSPCAMConfig, paths: dict[str, Path], k: int) -> None:
    if not paths["selection_lock"].exists():
        raise ValueError("train-only selection lock absent; no test access allowed")
    lock = json.loads(paths["selection_lock"].read_text(encoding="utf-8"))
    if k not in lock["full20_k"]:
        raise ValueError(f"k={k} not in preselected full20 confirmation set")
    existing = read_rows(paths["full20"]) if paths["full20"].exists() else []
    if complete(existing, k, 20):
        print(json.dumps({"phase": "fresh_full20", "k": k, "skipped": "complete"}), flush=True)
        return
    model = make_model(base, k)  # never reuse the development model
    aligned = CoordinateAlignedRidge(base.code_dim, base.memory_slots, k, base.gamma)
    test_s: list[sparse.csr_matrix] = []
    test_y: list[np.ndarray] = []
    bit_counts = np.zeros(base.code_dim, dtype=np.int64)
    result = []
    running = 0.0
    cumulative_plateaus = cumulative_flips = 0
    for task in range(1, 21):
        start = time.perf_counter()
        h, y = load_split("train", task)
        s = model.encoder.encode(h, base.encoder_batch_size)
        bit_counts += s.sum(axis=0, dtype=np.int64)
        writer = writer_update(model, s, y)
        aligned.update(s, y)
        cumulative_plateaus += writer["plateau_count_task"]
        cumulative_flips += writer["gross_flips_task"]
        del h, y, s
        ridge = solve_aligned(aligned, model.memory.weights)
        h, y = load_split("test", task)
        test_s.append(sparse.csr_matrix(model.encoder.encode(h, base.encoder_batch_size)))
        test_y.append(y)
        del h, y
        score = score_sources(ridge, model.memory.weights, k, test_s, test_y, task)
        running += score["cumulative_accuracy"]
        stats = representation(model, k, test_s, test_y, bit_counts)
        wb_stats = memory_stats(model.memory.weights)
        row = dict(phase="fresh_full20_locked_k", k=k, k_over_m=k / base.code_dim,
                   task=task, selected_k=lock["selected_k"],
                   role="selected" if k == lock["selected_k"] else "neighbor_confirmation",
                   rf_seed=exp["rf_seed"], writer_seed=exp["writer_seed"],
                   gamma=base.gamma, tau=0.0, full_train_samples=task * 2500,
                   **score, average_incremental_to_task=running / task,
                   **writer, cumulative_plateau_count=cumulative_plateaus,
                   cumulative_gross_flips=cumulative_flips,
                   cumulative_gross_flips_per_sample=cumulative_flips / (task * 2500),
                   **stats, **wb_stats,
                   test_evaluations_at_this_task=1,
                   task_seconds=time.perf_counter() - start,
                   process_rss_mib=PROCESS.memory_info().rss / 2**20)
        result.append(row)
        record(paths["full20"], k, result, exp["grid"])
        print(json.dumps({"phase": "fresh_full20", "k": k, "task": task,
                          "cumulative": score["cumulative_accuracy"],
                          "average": running / task,
                          "seconds": round(row["task_seconds"], 1),
                          "rss_mib": round(row["process_rss_mib"])}), flush=True)
        del ridge
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["validate", "select", "full20"])
    parser.add_argument("--k", type=int)
    args = parser.parse_args()
    exp, base, paths = protocol()
    if args.phase in ["validate", "full20"]:
        if args.k not in exp["grid"]:
            raise ValueError("--k must be from the predeclared grid")
    if args.phase == "validate":
        validate_k(exp, base, paths, args.k)
    elif args.phase == "select":
        if args.k is not None:
            raise ValueError("selection does not accept a k override")
        select(exp, paths)
    else:
        full20_k(exp, base, paths, args.k)


if __name__ == "__main__":
    main()
