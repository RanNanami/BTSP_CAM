"""Experimental current-WB replay oracle for predeclared overlap-count q levels.

Rebuilds the archived k=128/160 writer trajectories without touching formal
checkpoints/configuration. All q>0 fits require explicit --phase sweep after
all twenty q=0 fits agree with archived metrics for both k values.
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
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.ridge import StreamingRidge
from run_coordinate_aligned_full20 import CosineMoments
from run_k_robustness_and_selection import make_model
from run_repair_ladder_task1_5 import load_split

BASE_CONFIG = ROOT / "configs/btsp_cam_cifar100.yaml"
ARCHIVE_SEED = ROOT / "diagnostics/k_seed_runs_taskwise.csv"
ARCHIVE_PATH = ROOT / "diagnostics/k_path_decomposition.csv"
Q_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32)
TASKS = 20
SEEDS = (1993, 1993)


def checkpoint_dir(k: int) -> Path:
    return ROOT / "diagnostics" / f"tau_oracle_k{k}"


def output_csv(k: int) -> Path:
    return ROOT / "diagnostics" / f"tau_q_oracle_k{k}.csv"


def rows_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def config() -> BTSPCAMConfig:
    cfg = BTSPCAMConfig.from_mapping(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")))
    assert (cfg.seed, cfg.k, cfg.code_dim, cfg.memory_slots, cfg.num_classes, cfg.gamma) == (
        1993, 41, 8192, 4096, 100, 0.1)
    return cfg


def archived(k: int) -> dict[int, dict]:
    if k == 128:
        selected = [r for r in rows_csv(ARCHIVE_SEED)
                    if (r["k"], r["rf_seed"], r["write_seed"]) == ("128", "1993", "1993")]
        path_selected = [r for r in rows_csv(ARCHIVE_PATH)
                         if r["run"] == "k128" and r["branch"] == "evolving_aligned"]
        if len(selected) != TASKS or len(path_selected) != TASKS:
            raise ValueError("missing archived k128 seed/path rows")
        seed_by_task = {int(r["task"]): r for r in selected}
        for row in path_selected:
            source = seed_by_task[int(row["task"])]
            if abs(float(row["cumulative_accuracy"]) - float(source["cumulative_accuracy"])) > 1e-12:
                raise AssertionError("k128 archived runs disagree")
        selected = path_selected
    elif k == 160:
        selected = [r for r in rows_csv(ARCHIVE_PATH)
                    if r["run"] == "k160" and r["branch"] == "evolving_aligned"]
    else:
        raise ValueError("only locked k values 128,160 are allowed")
    out = {int(r["task"]): r for r in selected}
    if set(out) != set(range(1, TASKS + 1)):
        raise ValueError(f"incomplete archived trajectory for k={k}")
    return out


def rebuild(k: int, cfg: BTSPCAMConfig) -> None:
    prior = archived(k)
    model = make_model(cfg, k, *SEEDS)
    target = checkpoint_dir(k)
    target.mkdir(parents=True, exist_ok=True)
    for task in range(1, TASKS + 1):
        start = time.perf_counter()
        h, y = load_split("train", task)
        codes = model.encoder.encode(h, cfg.encoder_batch_size)
        if not np.all(codes.sum(axis=1) == k):
            raise AssertionError("sparse encoder is not exact-k")
        plateau_count = gross_flips = 0
        for code, label in zip(codes, y):
            log = model._learn_code(code, int(label))
            plateau_count += int(log["plateau_count"])
            gross_flips += int(log["flips"])
        del h, y, codes
        expected = prior[task]
        density = float(model.memory.weights.mean())
        if abs(density - float(expected["WB_density"])) > 1e-15:
            raise AssertionError(f"k{k} task{task} WB density mismatch")
        if "plateau_count_task" in expected and plateau_count != int(expected["plateau_count_task"]):
            raise AssertionError(f"k{k} task{task} plateau count mismatch")
        if "gross_flips_task" in expected and gross_flips != int(expected["gross_flips_task"]):
            raise AssertionError(f"k{k} task{task} flip count mismatch")
        path = target / f"WB_task{task:02d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as state:
                np.testing.assert_array_equal(state["WB"], model.memory.weights)
                if int(state["k"]) != k or int(state["task"]) != task:
                    raise AssertionError("preexisting checkpoint metadata mismatch")
        else:
            np.savez_compressed(path, WB=model.memory.weights,
                                k=k, task=task, rf_seed=SEEDS[0], write_seed=SEEDS[1],
                                plateau_count=plateau_count, gross_flips=gross_flips)
        print(json.dumps({"phase": "rebuild", "k": k, "task": task,
                          "plateau_count": plateau_count, "gross_flips": gross_flips,
                          "WB_density": density, "seconds": round(time.perf_counter() - start, 2)}),
              flush=True)
    (target / "rebuild_complete.json").write_text(json.dumps({
        "k": k, "tasks": TASKS, "rf_seed": SEEDS[0], "write_seed": SEEDS[1],
        "archive": str(ARCHIVE_SEED if k == 128 else ARCHIVE_PATH),
        "validation": "plateau/gross/density taskwise exact",
    }, indent=2), encoding="utf-8")


def load_checkpoint(k: int, task: int, slots: int, code_dim: int) -> np.ndarray:
    path = checkpoint_dir(k) / f"WB_task{task:02d}.npz"
    with np.load(path, allow_pickle=False) as state:
        if int(state["k"]) != k or int(state["task"]) != task:
            raise AssertionError("checkpoint k/task mismatch")
        wb = state["WB"]
    if wb.shape != (slots, code_dim) or wb.dtype != np.uint8 or not np.all(wb <= 1):
        raise AssertionError("checkpoint WB has invalid shape/type/values")
    return wb


def sparse_sources(k: int, cfg: BTSPCAMConfig) -> tuple[list[sparse.csr_matrix], list[np.ndarray], list[sparse.csr_matrix], list[np.ndarray]]:
    encoder = make_model(cfg, k, *SEEDS).encoder
    train_s, train_y, test_s, test_y = [], [], [], []
    for task in range(1, TASKS + 1):
        for split, codes_out, labels_out in (("train", train_s, train_y), ("test", test_s, test_y)):
            h, y = load_split(split, task)
            s = encoder.encode(h, cfg.encoder_batch_size)
            if not np.all(s.sum(axis=1) == k):
                raise AssertionError("non-exact sparse code")
            codes_out.append(sparse.csr_matrix(s, dtype=np.float32))
            labels_out.append(y)
            del h, s
        print(json.dumps({"phase": "encode_cached_features", "k": k, "task": task}), flush=True)
    return train_s, train_y, test_s, test_y


def overlaps(codes: list[sparse.csr_matrix], w_t: np.ndarray, task: int, k: int) -> list[np.ndarray]:
    parts = []
    for s in codes[:task]:
        counts = np.asarray(s @ w_t)
        if counts.dtype != np.float32 or not np.all((counts >= 0) & (counts <= k) & (counts == np.floor(counts))):
            raise AssertionError("inexact binary overlap")
        parts.append(counts.astype(np.uint8))
    return parts


def quantile_from_hist(hist: np.ndarray, p: float) -> float:
    n = int(hist.sum())
    pos = (n - 1) * p
    csum = np.cumsum(hist)
    lo = int(np.searchsorted(csum, int(np.floor(pos)) + 1))
    hi = int(np.searchsorted(csum, int(np.ceil(pos)) + 1))
    return lo + (pos - np.floor(pos)) * (hi - lo)


def transformed_quantile_from_hist(hist: np.ndarray, lookup: np.ndarray, p: float) -> float:
    n = int(hist.sum())
    pos = (n - 1) * p
    csum = np.cumsum(hist)
    lo = int(np.searchsorted(csum, int(np.floor(pos)) + 1))
    hi = int(np.searchsorted(csum, int(np.ceil(pos)) + 1))
    return float(lookup[lo] + (pos - np.floor(pos)) * (lookup[hi] - lookup[lo]))


def fit_eval(k: int, task: int, q: int, gamma: float,
             train_counts: list[np.ndarray], train_y: list[np.ndarray],
             test_counts: list[np.ndarray], test_y: list[np.ndarray],
             wb_density: float) -> dict:
    start = time.perf_counter()
    tau = q / k
    lookup = np.maximum(np.arange(k + 1, dtype=np.float64) - q, 0.0) / k
    ridge = StreamingRidge(test_counts[0].shape[1], task * 5, gamma)
    for counts, labels in zip(train_counts, train_y[:task]):
        z = lookup[counts]
        ridge.update(z, labels)
        del z
    ridge.solve()
    ridge._last_cholesky = None
    same_cos = CosineMoments(test_counts[0].shape[1])
    old_correct = old_total = current_correct = current_total = five_correct = 0
    hist = np.zeros(k + 1, dtype=np.int64)
    active_samples = active_units = total_samples = 0
    for source, (counts, labels) in enumerate(zip(test_counts, test_y[:task]), 1):
        hist += np.bincount(counts.ravel(), minlength=k + 1)
        z = lookup[counts]
        logits = ridge.logits(z)
        pred = logits.argmax(axis=1)
        correct = int(np.count_nonzero(pred == labels))
        if source == task:
            current_correct += correct
            current_total += len(labels)
            local = logits[:, (task - 1) * 5:task * 5]
            five_correct += int(np.count_nonzero(
                local.argmax(axis=1) + (task - 1) * 5 == labels))
        else:
            old_correct += correct
            old_total += len(labels)
        active = counts > q
        active_samples += int(np.count_nonzero(active.any(axis=1)))
        active_units += int(np.count_nonzero(active))
        total_samples += len(labels)
        same_cos.add(z, labels)
        del z, logits, active
    same, different, margin = same_cos.values()
    element_count = int(hist.sum())
    counts_axis = np.arange(k + 1)
    z_axis = lookup
    j_mean = float(hist @ counts_axis / element_count)
    fields = {
        "k": k, "task": task, "q": q, "tau": tau,
        "method": "current_WB_replay_oracle_not_exemplar_free",
        "cumulative_accuracy": (old_correct + current_correct) / (old_total + current_total),
        "old_accuracy": old_correct / old_total if old_total else float("nan"),
        "current_accuracy": current_correct / current_total,
        "current_5way_accuracy": five_correct / current_total,
        "CA1_element_density": active_units / element_count,
        "fraction_nonzero_samples": active_samples / total_samples,
        "mean_active_units_per_sample": active_units / total_samples,
        "mean_z": float(hist @ z_axis / element_count),
        "p95_z": transformed_quantile_from_hist(hist, lookup, 0.95),
        "p99_z": transformed_quantile_from_hist(hist, lookup, 0.99),
        "J_mean": j_mean,
        "J_p50": quantile_from_hist(hist, 0.50),
        "J_p90": quantile_from_hist(hist, 0.90),
        "J_p95": quantile_from_hist(hist, 0.95),
        "J_p99": quantile_from_hist(hist, 0.99),
        "same_class_cosine": same,
        "different_class_cosine": different,
        "cosine_margin": margin,
        "WB_density": wb_density,
        "train_samples": sum(len(y) for y in train_y[:task]),
        "test_samples": total_samples,
        "fit_eval_seconds": time.perf_counter() - start,
    }
    for j in range(33):
        fields[f"P_J_{j}"] = hist[j] / element_count
    fields["P_J_gt32"] = hist[33:].sum() / element_count
    return fields


def require_trajectory(k: int) -> None:
    path = checkpoint_dir(k) / "rebuild_complete.json"
    if not path.exists():
        raise FileNotFoundError(f"k{k} rebuild incomplete")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if (meta["k"], meta["tasks"], meta["rf_seed"], meta["write_seed"]) != (k, TASKS, *SEEDS):
        raise AssertionError("trajectory metadata mismatch")
    for t in range(1, TASKS + 1):
        if not (checkpoint_dir(k) / f"WB_task{t:02d}.npz").exists():
            raise FileNotFoundError(f"k{k} task{t} checkpoint missing")


def oracle_pass(k: int, cfg: BTSPCAMConfig, *, zero_only: bool) -> None:
    require_trajectory(k)
    reference = archived(k)
    train_s, train_y, test_s, test_y = sparse_sources(k, cfg)
    output = output_csv(k)
    rows = rows_csv(output)
    seen = {(int(row["task"]), int(row["q"])) for row in rows}
    if not zero_only:
        for t in range(1, TASKS + 1):
            zrow = next((row for row in rows if int(row["task"]) == t and int(row["q"]) == 0), None)
            if zrow is None or abs(float(zrow["cumulative_accuracy"]) - float(reference[t]["cumulative_accuracy"])) > 1e-12:
                raise AssertionError(f"k{k} Task{t} q0 has not passed archived validation")
    levels = (0,) if zero_only else (*Q_LEVELS[1:], k // 2)
    for task in range(1, TASKS + 1):
        pending = [q for q in levels if (task, q) not in seen]
        if not pending:
            continue
        task_start = time.perf_counter()
        wb = load_checkpoint(k, task, cfg.memory_slots, cfg.code_dim)
        w_t = np.ascontiguousarray(wb.T, dtype=np.float32)
        train_counts = overlaps(train_s, w_t, task, k)
        test_counts = overlaps(test_s, w_t, task, k)
        for q in pending:
            result = fit_eval(k, task, q, cfg.gamma, train_counts, train_y,
                              test_counts, test_y, float(wb.mean()))
            if q == 0:
                expected = reference[task]
                for metric in ("cumulative_accuracy", "current_accuracy", "current_5way_accuracy"):
                    if abs(result[metric] - float(expected[metric])) > 1e-12:
                        raise AssertionError(f"k{k} Task{task} {metric} q0 mismatch: {result[metric]} vs {expected[metric]}")
                if task > 1 and abs(result["old_accuracy"] - float(expected["old_accuracy"])) > 1e-12:
                    raise AssertionError(f"k{k} Task{task} old accuracy q0 mismatch")
            rows.append(result)
            seen.add((task, q))
            write_csv(output, sorted(rows, key=lambda r: (int(r["task"]), int(r["q"]))))
            print(json.dumps({"phase": "q0_validation" if zero_only else "q_sweep",
                              "k": k, "task": task, "q": q,
                              "accuracy": result["cumulative_accuracy"],
                              "density": result["CA1_element_density"],
                              "seconds": round(result["fit_eval_seconds"], 2)}), flush=True)
        del wb, w_t, train_counts, test_counts
        gc.collect()
        print(json.dumps({"k": k, "task": task,
                          "task_seconds": round(time.perf_counter() - task_start, 2)}), flush=True)


def frontier() -> None:
    all_rows = {k: rows_csv(output_csv(k)) for k in (128, 160)}
    expected_n = TASKS * (len(Q_LEVELS) + 1)
    if any(len(rows) != expected_n for rows in all_rows.values()):
        raise ValueError(f"incomplete oracle CSVs: {[len(r) for r in all_rows.values()]}")
    summary = []
    for k, rows in all_rows.items():
        by = {(int(r["task"]), int(r["q"])): r for r in rows}
        for q in (*Q_LEVELS, k // 2):
            stages = [by[(t, q)] for t in range(1, TASKS + 1)]
            final = stages[-1]
            baseline = by[(TASKS, 0)]
            density_reduction = 1.0 - float(final["CA1_element_density"]) / float(baseline["CA1_element_density"])
            accuracy_drop_pp = 100 * (float(baseline["cumulative_accuracy"]) - float(final["cumulative_accuracy"]))
            summary.append({
                "k": k, "q": q, "tau": q / k,
                "average_incremental_accuracy": np.mean([float(r["cumulative_accuracy"]) for r in stages]),
                "task5_accuracy": stages[4]["cumulative_accuracy"],
                "task10_accuracy": stages[9]["cumulative_accuracy"],
                "task15_accuracy": stages[14]["cumulative_accuracy"],
                "task20_accuracy": final["cumulative_accuracy"],
                "task20_old_accuracy": final["old_accuracy"],
                "task20_current_accuracy": final["current_accuracy"],
                "task20_CA1_density": final["CA1_element_density"],
                "task20_density_reduction_vs_q0": density_reduction,
                "task20_accuracy_drop_pp_vs_q0": accuracy_drop_pp,
                "diagnostic_frontier_20pct_0_5pp": int(q > 0 and density_reduction >= 0.20 and accuracy_drop_pp <= 0.5),
                "source": "test_replay_oracle_not_model_selection",
            })
    write_csv(ROOT / "diagnostics/tau_q_frontier.csv", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("rebuild", "validate_zero", "sweep", "frontier"))
    parser.add_argument("--k", type=int, choices=(128, 160))
    args = parser.parse_args()
    cfg = config()
    if args.phase != "frontier" and args.k is None:
        parser.error("--k is required for this phase")
    if args.phase == "rebuild":
        rebuild(args.k, cfg)
    elif args.phase == "validate_zero":
        oracle_pass(args.k, cfg, zero_only=True)
    elif args.phase == "sweep":
        for k in (128, 160):
            require_trajectory(k)
            ref = archived(k)
            rows = rows_csv(output_csv(k))
            by_task = {int(r["task"]): r for r in rows if int(r["q"]) == 0}
            if set(by_task) != set(range(1, TASKS + 1)):
                raise AssertionError(f"k{k} q0 validation incomplete")
            for task in range(1, TASKS + 1):
                if abs(float(by_task[task]["cumulative_accuracy"]) - float(ref[task]["cumulative_accuracy"])) > 1e-12:
                    raise AssertionError(f"k{k} Task{task} q0 accuracy mismatch")
        oracle_pass(args.k, cfg, zero_only=False)
    else:
        frontier()


if __name__ == "__main__":
    main()
