"""Exploratory high-k stress test on the independent tau=0 aligned pipeline.

Validation is train-side only. The full20 phase is locked from validation CSVs
before any test cache is loaded. Nothing in the formal model/config is edited.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import pilot_class_order
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance
from run_k_extended_search import class_overlap_geometry, representation
from run_k_robustness_and_selection import development_split, score_sources
from run_repair_ladder_task1_5 import load_split
from run_unresolved_parameter_screen import read_rows, write_rows


CONFIG = ROOT / "experiments/configs/high_k_stress_test.yaml"
ARCHIVE = ROOT / "diagnostics/k_extended_validation.csv"
ARCHIVE_FULL20 = ROOT / "diagnostics/k_extended_full20.csv"


class DenseHighKAligned:
    """Exact C/D closure with dense float64 C; no historical sample storage.

    Integer-valued C/D and W products in these runs are far below 2**53, so
    float64 BLAS accumulation represents them exactly before k normalization.
    This is an experimental storage/kernel substitution, not a new readout.
    """

    def __init__(self, m: int, slots: int, k: int, gamma: float):
        self.m, self.slots, self.k, self.gamma = m, slots, k, gamma
        self.c = torch.zeros((m, m), dtype=torch.float64)
        self.d = np.zeros((m, 0), dtype=np.float64)

    def update(self, codes: np.ndarray, labels: np.ndarray) -> None:
        s = np.asarray(codes)
        y = np.asarray(labels, dtype=np.int64)
        if s.ndim != 2 or s.shape[1] != self.m or y.shape != (len(s),):
            raise ValueError("invalid current-task codes or labels")
        if not np.all(s.sum(axis=1, dtype=np.int64) == self.k):
            raise AssertionError("exact-k invariant failed")
        x = np.ascontiguousarray(s, dtype=np.float64)
        xt = torch.from_numpy(x)
        self.c.addmm_(xt.T, xt)
        count = int(y.max()) + 1
        if count > self.d.shape[1]:
            self.d = np.pad(self.d, ((0, 0), (0, count - self.d.shape[1])))
        for label in np.unique(y):
            self.d[:, label] += x[y == label].sum(axis=0)

    def solve(self, memory: np.ndarray) -> StreamingRidge:
        w = torch.from_numpy(np.ascontiguousarray(memory, dtype=np.float64))
        wc = w @ self.c
        a = ((wc @ w.T) / (self.k * self.k)).numpy()
        b = ((w @ torch.from_numpy(self.d)) / self.k).numpy()
        ridge = StreamingRidge(self.slots, b.shape[1], self.gamma)
        ridge.a, ridge.b = a, b
        ridge.solve()
        ridge._last_cholesky = None
        return ridge


def setup() -> tuple[dict, BTSPCAMConfig, dict[str, Path]]:
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if spec["initial_grid"] != [1024, 1536, 2048, 3072, 4096]:
        raise ValueError("initial high-k grid changed")
    if spec["conditional_extension"] != [5120, 6144, 7168]:
        raise ValueError("conditional high-k grid changed")
    if spec["conditions"] != ["FIXED_FBASE", "EQUAL_WRITE_APPROX"]:
        raise ValueError("write conditions changed")
    fixed = (spec["validation_tasks"], spec["validation_split_seed"],
             spec["validation_fraction"], spec["class_order_seed"],
             spec["rf_seed"], spec["writer_seed"],
             spec["tie_tolerance_percentage_points"], spec["base_f_base"],
             spec["equal_write_reference_k"], spec["tau"], spec["readout"])
    if fixed != (5, 20260928, 0.2, 1993, 1993, 1993, 0.1, 0.005,
                 1024, 0.0, "coordinate_aligned_ridge"):
        raise ValueError("locked validation/readout protocol changed")
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(
        (CONFIG.parent / spec["base_config"]).resolve().read_text(encoding="utf-8")))
    if (base.seed, base.code_dim, base.memory_slots, base.flip_probability,
            base.beta, base.trace_decay, base.gamma) != (
            1993, 8192, 4096, 0.5, 1.0, 0.01, 0.1):
        raise ValueError("formal baseline settings changed")
    paths = {key: (CONFIG.parent / value).resolve()
             for key, value in spec["outputs"].items()}
    return spec, base, paths


def f_base(spec: dict, k: int, condition: str) -> float:
    if condition == "FIXED_FBASE":
        return float(spec["base_f_base"])
    if condition == "EQUAL_WRITE_APPROX":
        return float(spec["base_f_base"] * spec["equal_write_reference_k"] / k)
    raise ValueError("unknown write condition")


def make_model(base: BTSPCAMConfig, k: int, plateau_base: float) -> BTSPCAM:
    cfg = replace(base, k=k, tau=0.0, base_plateau_probability=plateau_base)
    rf = RNGProvenance.from_seed(1993)
    wr = RNGProvenance.from_seed(1993)
    model = BTSPCAM(cfg, class_order=pilot_class_order(100, 1993),
                    projection_seed=rf.projection_seed,
                    plateau_seed=wr.plateau_seed,
                    synapse_seed=wr.synapse_seed)
    model.ridge = None
    return model


def writer_step(model: BTSPCAM, codes: np.ndarray, labels: np.ndarray) -> dict:
    probability_sum = plateau_events = flips01 = flips10 = 0
    for code, label in zip(codes, labels):
        entry = model._learn_code(code, int(label))
        probability_sum += entry["plateau_probability"]
        plateau_events += int(entry["plateau_count"])
        flips01 += int(entry["zero_to_one_flips"])
        flips10 += int(entry["one_to_zero_flips"])
    gross = flips01 + flips10
    n = len(labels)
    return dict(mean_f_eff=probability_sum / n,
                plateau_events=plateau_events,
                plateau_events_per_sample=plateau_events / n,
                gross_flips=gross, gross_flips_per_sample=gross / n,
                flip_01=flips01, flip_10=flips10)


def memory_stats(memory: np.ndarray) -> dict:
    column_count = memory.sum(axis=0, dtype=np.int64)
    occupancy = column_count / memory.shape[0]
    top = int(np.ceil(0.01 * len(column_count)))
    total = int(column_count.sum())
    return dict(WB_density=total / memory.size,
                column_occupancy_mean=float(occupancy.mean()),
                column_occupancy_std=float(occupancy.std()),
                column_occupancy_p95=float(np.percentile(occupancy, 95)),
                column_occupancy_p99=float(np.percentile(occupancy, 99)),
                column_occupancy_max=float(occupancy.max()),
                top1_column_occupancy_share=(
                    float(np.sort(column_count)[-top:].sum() / total)
                    if total else 0.0))


def sparse_stats(codes: list[sparse.csr_matrix], labels: list[np.ndarray],
                 bit_counts: np.ndarray, k: int, m: int) -> dict:
    same_cos, diff_cos, margin, same_j, diff_j = class_overlap_geometry(
        codes, labels, k, m)
    weighted = bit_counts.astype(np.float64) / bit_counts.sum()
    positive = weighted[weighted > 0]
    top = int(np.ceil(0.01 * m))
    packed = np.packbits(sparse.vstack(codes, format="csr").toarray(), axis=1)
    duplicate_fraction = 1.0 - np.unique(packed, axis=0).shape[0] / len(packed)
    return dict(activation_density=k / m,
                expected_random_J=k * k / m,
                expected_random_normalized_J=k / m,
                empirical_same_J=same_j, empirical_diff_J=diff_j,
                sparse_same_cos=same_cos, sparse_diff_cos=diff_cos,
                sparse_margin=margin,
                top1_bit_activation_share=float(np.sort(weighted)[-top:].sum()),
                effective_support_entropy=float(np.exp(-np.sum(
                    positive * np.log(positive)))),
                never_winning_RF_bits=int(np.count_nonzero(bit_counts == 0)),
                exact_duplicate_fraction=duplicate_fraction,
                exact_duplicate_count=int(round(duplicate_fraction * len(packed))),
                hamming_same_mean=2 * (k - same_j),
                hamming_diff_mean=2 * (k - diff_j),
                hamming_same_normalized=2 * (k - same_j) / m,
                hamming_diff_normalized=2 * (k - diff_j) / m)


def replace_key(path: Path, key: dict, new_rows: list[dict]) -> None:
    old = read_rows(path) if path.exists() else []
    kept = [row for row in old if not all(str(row.get(name)) == str(value)
                                      for name, value in key.items())]
    write_rows(path, kept + new_rows)


def complete(path: Path, k: int, condition: str, count: int) -> bool:
    if not path.exists():
        return False
    rows = [r for r in read_rows(path) if int(r["k"]) == k
            and r["condition"] == condition]
    return len(rows) == count and {int(r["task"]) for r in rows} == set(range(1, count + 1))


def control_check(task: int, row: dict) -> None:
    prior = next(r for r in read_rows(ARCHIVE)
                 if int(r["k"]) == 1024 and int(r["task"]) == task)
    fields = {"cumulative_accuracy": "cumulative_accuracy",
              "current_accuracy": "current_accuracy",
              "current_5way_accuracy": "current_5way_accuracy",
              "validation_mean_to_task": "validation_average_incremental_to_task",
              "gross_flips": "gross_flips_task",
              "WB_density": "WB_density"}
    for current, archived in fields.items():
        if abs(float(row[current]) - float(prior[archived])) > 1e-12:
            raise AssertionError(f"k=1024 archived control mismatch task={task} "
                                 f"field={current}: {row[current]} vs {prior[archived]}")


def validation_run(spec: dict, base: BTSPCAMConfig, paths: dict[str, Path],
                   k: int, condition: str) -> None:
    if complete(paths["validation"], k, condition, 5):
        print(json.dumps({"phase": "validation", "k": k, "condition": condition,
                          "skipped": "complete"}), flush=True)
        return
    if k != 1024:
        for prior_condition in spec["conditions"]:
            if not complete(paths["validation"], 1024, prior_condition, 5):
                raise ValueError("both k=1024 control runs must precede high-k runs")
    if k in spec["conditional_extension"] and not should_extend(spec, paths):
        raise ValueError("extension gate not met")
    model = make_model(base, k, f_base(spec, k, condition))
    aligned = DenseHighKAligned(base.code_dim, base.memory_slots, k, base.gamma)
    val_s: list[sparse.csr_matrix] = []
    val_y: list[np.ndarray] = []
    bit_counts = np.zeros(base.code_dim, dtype=np.int64)
    validation_rows, geometry_rows, writer_rows = [], [], []
    running = 0.0
    gross_cumulative = plateau_cumulative = 0
    for task in range(1, 6):
        started = time.perf_counter()
        h, y = load_split("train", task)
        dev_i, val_i = development_split(y, 20260928, 0.2)
        if len(dev_i) != 2000 or len(val_i) != 500:
            raise AssertionError("development split changed")
        s_dev = model.encoder.encode(h[dev_i], base.encoder_batch_size)
        s_val = model.encoder.encode(h[val_i], base.encoder_batch_size)
        bit_counts += s_dev.sum(axis=0, dtype=np.int64)
        writer = writer_step(model, s_dev, y[dev_i])
        aligned.update(s_dev, y[dev_i])
        gross_cumulative += writer["gross_flips"]
        plateau_cumulative += writer["plateau_events"]
        val_s.append(sparse.csr_matrix(s_val))
        val_y.append(y[val_i].copy())
        del h, s_dev, s_val, y
        ridge = aligned.solve(model.memory.weights)
        accuracy = score_sources(ridge, model.memory.weights, k, val_s, val_y, task)
        running += accuracy["cumulative_accuracy"]
        common = dict(phase="train_side_validation", k=k, condition=condition,
                      f_base=model.config.base_plateau_probability, task=task,
                      split_seed=20260928, train_per_class=400,
                      validation_per_class=100, rf_seed=1993, writer_seed=1993,
                      tau=0.0, gamma=base.gamma)
        memory = memory_stats(model.memory.weights)
        val_row = dict(**common, **accuracy, validation_mean_to_task=running / task,
                       WB_density=memory["WB_density"],
                       gross_flips=writer["gross_flips"],
                       task_seconds=time.perf_counter() - started)
        if k == 1024:
            control_check(task, val_row)
        validation_rows.append(val_row)
        writer_rows.append(dict(**common, **writer,
                                cumulative_gross_flips=gross_cumulative,
                                cumulative_plateau_events=plateau_cumulative,
                                **memory))
        sparse_geometry = sparse_stats(val_s, val_y, bit_counts, k, base.code_dim)
        ca1 = representation(model, k, val_s, val_y, bit_counts)
        geometry_rows.append(dict(**common, **sparse_geometry,
                                  a_mean=ca1["a_mean"], a_std=ca1["a_std"],
                                  a_positive_density=ca1["a_density"],
                                  a_same_cos=ca1["a_same_cos"],
                                  a_diff_cos=ca1["a_diff_cos"],
                                  a_margin=ca1["a_margin"]))
        key = {"k": k, "condition": condition}
        replace_key(paths["validation"], key, validation_rows)
        replace_key(paths["writer"], key, writer_rows)
        replace_key(paths["geometry"], key, geometry_rows)
        print(json.dumps({"phase": "validation", "k": k, "condition": condition,
                          "task": task, "cumulative": accuracy["cumulative_accuracy"],
                          "mean": running / task,
                          "seconds": round(val_row["task_seconds"], 1)}), flush=True)
        del ridge


def task5(path: Path, k: int, condition: str) -> dict:
    return next(row for row in read_rows(path) if int(row["k"]) == k
                and row["condition"] == condition and int(row["task"]) == 5)


def should_extend(spec: dict, paths: dict[str, Path]) -> bool:
    for condition in spec["conditions"]:
        if not complete(paths["validation"], 3072, condition, 5):
            return False
        if not complete(paths["validation"], 4096, condition, 5):
            return False
    geometry_3072 = task5(paths["geometry"], 3072, "FIXED_FBASE")
    geometry_4096 = task5(paths["geometry"], 4096, "FIXED_FBASE")
    earlier = float(geometry_3072["sparse_margin"])
    later = float(geometry_4096["sparse_margin"])
    geometry_ok = earlier > 0 and later > 0 and later >= (
        1 - spec["extension_max_sparse_margin_drop_fraction"]) * earlier
    enough_gain = any((float(task5(paths["validation"], 4096, condition)["validation_mean_to_task"])
                       - float(task5(paths["validation"], 3072, condition)["validation_mean_to_task"]))
                      * 100 > spec["extension_min_gain_percentage_points"]
                      for condition in spec["conditions"])
    return geometry_ok and enough_gain


def peak_status(grid: list[int], means: dict[int, float]) -> str:
    best_k = max(grid, key=lambda k: means[k])
    best = means[best_k]
    if best_k == grid[-1] and (means[grid[-1]] - means[grid[-2]]) * 100 > 0.20:
        return "PEAK_NOT_REACHED"
    later = grid[grid.index(best_k) + 1:]
    if len(later) >= 2 and all((best - means[k]) * 100 > 0.20 for k in later[:2]):
        return "PEAK_BRACKETED"
    return "PLATEAU_OR_NOISY"


def lock_selection(spec: dict, paths: dict[str, Path]) -> dict:
    if paths["selection_lock"].exists():
        raise ValueError("selection lock already exists")
    extend = should_extend(spec, paths)
    grid = spec["initial_grid"] + (spec["conditional_extension"] if extend else [])
    for condition in spec["conditions"]:
        for k in grid:
            if not complete(paths["validation"], k, condition, 5):
                raise ValueError(f"validation incomplete: k={k} {condition}")
    means = {condition: {k: float(task5(paths["validation"], k, condition)
                                   ["validation_mean_to_task"]) for k in grid}
             for condition in spec["conditions"]}
    primary = means[spec["full20_selection_condition"]]
    best = max(primary.values())
    tolerance = spec["tie_tolerance_percentage_points"] / 100
    selected = min(k for k in grid if best - primary[k] <= tolerance + 1e-15)
    points = {1024, selected, grid[-1]}
    if len(points) < 3 and len(grid) > 2:
        points.add(grid[-2])
    points = sorted(points)
    csv_hash = hashlib.sha256(paths["validation"].read_bytes()).hexdigest()
    lock = dict(grid=grid, extension_triggered=extend, means=means,
                peak_status={condition: peak_status(grid, value)
                             for condition, value in means.items()},
                selected_k=selected, full20_k=points,
                selection_condition=spec["full20_selection_condition"],
                tie_tolerance_percentage_points=spec["tie_tolerance_percentage_points"],
                selection_used_test_accuracy=False,
                validation_csv_sha256=csv_hash)
    paths["selection_lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["selection_lock"].write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "selection_locked_before_test", **lock}), flush=True)
    return lock


def full20_run(spec: dict, base: BTSPCAMConfig, paths: dict[str, Path],
               k: int, condition: str) -> None:
    if not paths["selection_lock"].exists():
        raise ValueError("validation selection lock is required before test access")
    lock = json.loads(paths["selection_lock"].read_text(encoding="utf-8"))
    if k not in lock["full20_k"]:
        raise ValueError("k not selected by train-side validation")
    if complete(paths["full20"], k, condition, 20):
        print(json.dumps({"phase": "full20", "k": k, "condition": condition,
                          "skipped": "complete"}), flush=True)
        return
    model = make_model(base, k, f_base(spec, k, condition))
    aligned = DenseHighKAligned(base.code_dim, base.memory_slots, k, base.gamma)
    test_s: list[sparse.csr_matrix] = []
    test_y: list[np.ndarray] = []
    rows = []
    running = 0.0
    for task in range(1, 21):
        started = time.perf_counter()
        h, y = load_split("train", task)
        s = model.encoder.encode(h, base.encoder_batch_size)
        writer = writer_step(model, s, y)
        aligned.update(s, y)
        del h, y, s
        ridge = aligned.solve(model.memory.weights)
        h_test, y_test = load_split("test", task)
        test_s.append(sparse.csr_matrix(model.encoder.encode(h_test, base.encoder_batch_size)))
        test_y.append(y_test)
        del h_test, y_test
        score = score_sources(ridge, model.memory.weights, k, test_s, test_y, task)
        running += score["cumulative_accuracy"]
        row = dict(phase="exploratory_repaired_full20", k=k, condition=condition,
                   f_base=model.config.base_plateau_probability, task=task,
                   rf_seed=1993, writer_seed=1993, tau=0.0, gamma=base.gamma,
                   **score, average_incremental_to_task=running / task,
                   WB_density=float(model.memory.weights.mean()),
                   **writer, test_evaluations_at_this_task=1,
                   task_seconds=time.perf_counter() - started)
        rows.append(row)
        replace_key(paths["full20"], {"k": k, "condition": condition}, rows)
        print(json.dumps({"phase": "full20", "k": k, "condition": condition,
                          "task": task, "cumulative": score["cumulative_accuracy"],
                          "average": running / task,
                          "seconds": round(row["task_seconds"], 1)}), flush=True)
        del ridge


def import_archived_1024_control(spec: dict, paths: dict[str, Path]) -> None:
    """Reuse the already completed identical full20 path; do not claim retraining."""
    if not paths["selection_lock"].exists():
        raise ValueError("validation selection lock is required")
    lock = json.loads(paths["selection_lock"].read_text(encoding="utf-8"))
    if 1024 not in lock["full20_k"]:
        raise ValueError("1024 was not selected as the boundary control")
    prior = [r for r in read_rows(ARCHIVE_FULL20) if int(r["k"]) == 1024]
    if len(prior) != 20 or {int(r["task"]) for r in prior} != set(range(1, 21)):
        raise ValueError("archived k=1024 full20 trajectory is incomplete")
    for condition in spec["conditions"]:
        if complete(paths["full20"], 1024, condition, 20):
            continue
        rows = []
        for old in sorted(prior, key=lambda r: int(r["task"])):
            if (int(old["rf_seed"]), int(old["writer_seed"]), float(old["tau"]),
                    float(old["gamma"])) != (1993, 1993, 0.0, 0.1):
                raise AssertionError("archived k=1024 protocol mismatch")
            rows.append(dict(phase="archived_identical_full20_control",
                             provenance="k_extended_full20.csv; no new retrain",
                             k=1024, condition=condition,
                             f_base=f_base(spec, 1024, condition),
                             task=int(old["task"]), rf_seed=1993, writer_seed=1993,
                             tau=0.0, gamma=0.1,
                             current_accuracy=old["current_accuracy"],
                             old_accuracy=old["old_accuracy"],
                             cumulative_accuracy=old["cumulative_accuracy"],
                             current_5way_accuracy=old["current_5way_accuracy"],
                             average_incremental_to_task=old["average_incremental_to_task"],
                             WB_density=old["WB_density"],
                             plateau_events=old["plateau_count_task"],
                             plateau_events_per_sample=(
                                 float(old["plateau_count_task"]) / 2500),
                             gross_flips=old["gross_flips_task"],
                             gross_flips_per_sample=old["gross_flips_per_sample"],
                             flip_01=old["flip_01_task"],
                             flip_10=old["flip_10_task"],
                             test_evaluations_at_this_task=old["test_evaluations_at_this_task"],
                             task_seconds=old["task_seconds"]))
        replace_key(paths["full20"], {"k": 1024, "condition": condition}, rows)
        print(json.dumps({"phase": "import_archived_control", "k": 1024,
                          "condition": condition, "rows": len(rows),
                          "task20": float(rows[-1]["cumulative_accuracy"])}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["validate", "extension-check", "select", "full20",
                                 "import-control"])
    parser.add_argument("--k", type=int)
    parser.add_argument("--condition", choices=["FIXED_FBASE", "EQUAL_WRITE_APPROX"])
    args = parser.parse_args()
    spec, base, paths = setup()
    grid = spec["initial_grid"] + spec["conditional_extension"]
    if args.phase in ["validate", "full20"]:
        if args.k not in grid or args.condition is None:
            raise ValueError("valid --k and --condition required")
    elif args.k is not None or args.condition is not None:
        raise ValueError("this phase takes no --k or --condition")
    if args.phase == "validate":
        validation_run(spec, base, paths, args.k, args.condition)
    elif args.phase == "extension-check":
        print(json.dumps({"extension_allowed": should_extend(spec, paths)}), flush=True)
    elif args.phase == "select":
        lock_selection(spec, paths)
    elif args.phase == "import-control":
        import_archived_1024_control(spec, paths)
    else:
        full20_run(spec, base, paths, args.k, args.condition)


if __name__ == "__main__":
    main()
