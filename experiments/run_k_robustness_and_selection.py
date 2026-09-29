"""Independent tau=0 k diagnostics; never edits the formal configuration.

Run in phases: seeds, paths, selection. The YAML is the predeclared protocol.
Per-task CSVs make long stochastic trajectories auditable and restartable, but
an interrupted trajectory always restarts from Task1, never from stale state.
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
from scipy import sparse
from scipy.stats import t as student_t
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from btsp_cam.plateau import plateau_probability, sample_slot_plateaus
from btsp_cam.protocol import pilot_class_order
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance
from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_coordinate_aligned_full20 import CosineMoments
from run_repair_ladder_task1_5 import load_split
from run_unresolved_parameter_screen import aligned_statistics_blas, overlap, read_rows, write_rows

CFG_PATH = ROOT / "experiments" / "configs" / "k_robustness_and_selection.yaml"
DIAG = ROOT / "diagnostics"
SEED_TASKS = DIAG / "k_seed_runs_taskwise.csv"
SEED_SUMMARY = DIAG / "k_seed_robustness.csv"
PATH_ROWS = DIAG / "k_path_decomposition.csv"
SELECT_ROWS = DIAG / "k_train_only_selection.csv"


def protocol():
    exp = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    cfg = BTSPCAMConfig.from_mapping(yaml.safe_load(
        (CFG_PATH.parent / exp["base_config"]).resolve().read_text(encoding="utf-8")))
    assert (cfg.seed, cfg.k, cfg.code_dim, cfg.memory_slots) == (1993, 41, 8192, 4096)
    assert exp["rf_seeds"] == [1993, 3001, 7001]
    assert exp["write_seeds"] == [1993, 4001, 8001]
    assert exp["write_budget_scalar"] == 41 / 128
    return exp, cfg


def make_model(base: BTSPCAMConfig, k: int, rf_seed: int, write_seed: int) -> BTSPCAM:
    cfg = replace(base, k=k, tau=0.0)
    rf = RNGProvenance.from_seed(rf_seed)
    wr = RNGProvenance.from_seed(write_seed)
    model = BTSPCAM(cfg, class_order=pilot_class_order(100, 1993),
                    projection_seed=rf.projection_seed,
                    plateau_seed=wr.plateau_seed,
                    synapse_seed=wr.synapse_seed)
    model.ridge = None
    return model


def score_sources(ridge: StreamingRidge, wb: np.ndarray, k: int,
                  codes: list[sparse.csr_matrix], labels: list[np.ndarray],
                  task: int) -> dict[str, float]:
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    old_ok = old_n = current_ok = current_n = five_ok = 0
    for source, (s, y) in enumerate(zip(codes, labels), 1):
        logits = ridge.logits(overlap(s, wt, k))
        pred = logits.argmax(axis=1)
        ok = int(np.count_nonzero(pred == y))
        if source == task:
            current_ok += ok
            current_n += len(y)
            five = logits[:, (task - 1) * 5:task * 5].argmax(axis=1) + (task - 1) * 5
            five_ok += int(np.count_nonzero(five == y))
        else:
            old_ok += ok
            old_n += len(y)
    return dict(current_accuracy=current_ok / current_n,
                old_accuracy=old_ok / old_n if old_n else float("nan"),
                cumulative_accuracy=(old_ok + current_ok) / (old_n + current_n),
                current_5way_accuracy=five_ok / current_n)


def solve_aligned(aligned: CoordinateAlignedRidge, wb: np.ndarray) -> StreamingRidge:
    a, b = aligned_statistics_blas(aligned, wb)
    ridge = StreamingRidge(aligned.slots, b.shape[1], aligned.gamma)
    ridge.a, ridge.b = a, b
    ridge.solve()
    ridge._last_cholesky = None
    return ridge


def append_completed(path: Path, key: dict, rows: list[dict]) -> None:
    old = read_rows(path) if path.exists() else []
    retained = [r for r in old if not all(str(r.get(k)) == str(v) for k, v in key.items())]
    write_rows(path, retained + rows)


def seed_run(base: BTSPCAMConfig, k: int, rf_seed: int, write_seed: int,
             tasks: int = 20) -> None:
    key = dict(k=k, rf_seed=rf_seed, write_seed=write_seed)
    if SEED_TASKS.exists():
        prior = [r for r in read_rows(SEED_TASKS)
                 if all(str(r.get(a)) == str(b) for a, b in key.items())]
        if len(prior) == tasks and {int(r["task"]) for r in prior} == set(range(1, tasks + 1)):
            print(json.dumps({**key, "skipped": "complete"}), flush=True)
            return
    model = make_model(base, k, rf_seed, write_seed)
    aligned = CoordinateAlignedRidge(base.code_dim, base.memory_slots, k, base.gamma)
    test_s: list[sparse.csr_matrix] = []
    test_y: list[np.ndarray] = []
    rows = []
    running = 0.0
    for task in range(1, tasks + 1):
        start = time.perf_counter()
        h, y = load_split("train", task)
        s = model.encoder.encode(h, base.encoder_batch_size)
        for code, label in zip(s, y):
            model._learn_code(code, int(label))
        aligned.update(s, y)
        del h, y, s
        ridge = solve_aligned(aligned, model.memory.weights)
        h, y = load_split("test", task)
        test_s.append(sparse.csr_matrix(model.encoder.encode(h, base.encoder_batch_size)))
        test_y.append(y)
        del h
        metric = score_sources(ridge, model.memory.weights, k, test_s, test_y, task)
        running += metric["cumulative_accuracy"]
        row = {**key, "task": task, **metric,
               "average_incremental_to_task": running / task,
               "WB_density": float(model.memory.weights.mean()),
               "task_seconds": time.perf_counter() - start,
               "source": "independent_formal_trace_aligned_tau0"}
        rows.append(row)
        append_completed(SEED_TASKS, key, rows)
        print(json.dumps({**key, "task": task,
                          "cumulative": round(metric["cumulative_accuracy"], 6),
                          "seconds": round(row["task_seconds"], 2)}), flush=True)
        del ridge
        gc.collect()


def seed_summary(exp: dict) -> None:
    src = read_rows(SEED_TASKS)
    pair_rows = []
    for rf in exp["rf_seeds"]:
        for wr in exp["write_seeds"]:
            result = {}
            for k in exp["seed_pair_k"]:
                history = sorted((r for r in src if int(r["rf_seed"]) == rf
                                  and int(r["write_seed"]) == wr and int(r["k"]) == k),
                                 key=lambda r: int(r["task"]))
                if len(history) != exp["tasks"]:
                    raise ValueError(f"incomplete paired trajectory RF={rf} write={wr} k={k}")
                result[k] = dict(avg=float(history[-1]["average_incremental_to_task"]),
                                 final=float(history[-1]["cumulative_accuracy"]),
                                 old=float(history[-1]["old_accuracy"]),
                                 current=float(history[-1]["current_accuracy"]),
                                 five=float(history[-1]["current_5way_accuracy"]))
            pair_rows.append(dict(rf_seed=rf, write_seed=wr,
                                  k41_avg=result[41]["avg"], k128_avg=result[128]["avg"],
                                  k41_final=result[41]["final"], k128_final=result[128]["final"],
                                  k41_old=result[41]["old"], k128_old=result[128]["old"],
                                  k41_current=result[41]["current"], k128_current=result[128]["current"],
                                  k41_current_5way=result[41]["five"],
                                  k128_current_5way=result[128]["five"],
                                  delta_avg=result[128]["avg"] - result[41]["avg"],
                                  delta_final=result[128]["final"] - result[41]["final"],
                                  delta_avg_sign=np.sign(result[128]["avg"] - result[41]["avg"]),
                                  delta_final_sign=np.sign(result[128]["final"] - result[41]["final"])))
    write_rows(SEED_SUMMARY, pair_rows)
    for metric in ("delta_avg", "delta_final"):
        x = np.array([r[metric] for r in pair_rows], dtype=np.float64)
        sd = x.std(ddof=1)
        half = float(student_t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
        grid = x.reshape(len(exp["rf_seeds"]), len(exp["write_seeds"]))
        rng = np.random.default_rng(exp["crossed_bootstrap_seed"])
        reps = int(exp["crossed_bootstrap_replicates"])
        rf_draw = rng.integers(0, grid.shape[0], size=(reps, grid.shape[0]))
        wr_draw = rng.integers(0, grid.shape[1], size=(reps, grid.shape[1]))
        boot = grid[rf_draw[:, :, None], wr_draw[:, None, :]].mean(axis=(1, 2))
        boot_low, boot_high = np.quantile(boot, [0.025, 0.975])
        print(json.dumps({"metric": metric, "n": len(x), "mean": x.mean(),
                          "std": sd, "ci95_low": x.mean() - half,
                          "ci95_high": x.mean() + half,
                          "crossed_bootstrap_ci95_low": boot_low,
                          "crossed_bootstrap_ci95_high": boot_high,
                          "positive": int(np.count_nonzero(x > 0)),
                          "negative": int(np.count_nonzero(x < 0))}), flush=True)


def sparse_ridge_weights(aligned: CoordinateAlignedRidge) -> np.ndarray:
    """Same normal equations/gamma as formal ridge, in stable sparse coordinates."""
    gram = torch.from_numpy(aligned.c.toarray())
    gram.diagonal().add_(aligned.gamma)
    chol = torch.linalg.cholesky(gram)
    weights = torch.cholesky_solve(torch.from_numpy(aligned.d), chol).numpy().copy()
    del gram, chol
    return weights


def sparse_scores(weights: np.ndarray, test_s: list[sparse.csr_matrix],
                  test_y: list[np.ndarray], task: int) -> dict[str, float]:
    old_ok = old_n = current_ok = current_n = five_ok = 0
    for source, (s, y) in enumerate(zip(test_s, test_y), 1):
        logits = np.asarray(s @ weights)
        ok = int(np.count_nonzero(logits.argmax(axis=1) == y))
        if source == task:
            current_ok += ok
            current_n += len(y)
            five = logits[:, (task - 1) * 5:task * 5].argmax(axis=1) + (task - 1) * 5
            five_ok += int(np.count_nonzero(five == y))
        else:
            old_ok += ok
            old_n += len(y)
    return dict(current_accuracy=current_ok / current_n,
                old_accuracy=old_ok / old_n if old_n else float("nan"),
                cumulative_accuracy=(old_ok + current_ok) / (old_n + current_n),
                current_5way_accuracy=five_ok / current_n)


def geometry(wb: np.ndarray, k: int, test_s: list[sparse.csr_matrix],
             test_y: list[np.ndarray], bit_counts: np.ndarray) -> dict[str, float]:
    sparse_mom = CosineMoments(test_s[0].shape[1])
    overlap_mom = CosineMoments(wb.shape[0])
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    for s, y in zip(test_s, test_y):
        sparse_mom.add(s.toarray().astype(np.float64), y)
        overlap_mom.add(overlap(s, wt, k), y)
    ss, sd, sm = sparse_mom.values()
    aa, ad, am = overlap_mom.values()
    p = bit_counts.astype(np.float64)
    p /= p.sum()
    positive = p[p > 0]
    top = int(np.ceil(0.01 * len(bit_counts)))
    out = dict(sparse_same_cos=ss, sparse_diff_cos=sd, sparse_margin=sm,
               a_same_cos=aa, a_diff_cos=ad, a_margin=am,
               top1_activation_share=float(np.sort(p)[-top:].sum()),
               effective_support_entropy=float(np.exp(-(positive * np.log(positive)).sum())),
               zero_winner_bits=int(np.count_nonzero(bit_counts == 0)))
    if len(test_s) == 1:
        out.update(cross_task_mean_J=float("nan"), P_J_ge_5=float("nan"))
    else:
        current = test_s[-1]
        old = sparse.vstack(test_s[:-1], format="csr")
        cf = np.asarray(current.sum(axis=0)).ravel()
        of = np.asarray(old.sum(axis=0)).ravel()
        mean_j = float(cf @ of / (current.shape[0] * old.shape[0]))
        ge5 = 0
        pairs = 0
        for past in test_s[:-1]:
            hit = (current @ past.T).tocsr()
            ge5 += int(np.count_nonzero(hit.data >= 5))
            pairs += current.shape[0] * past.shape[0]
        out.update(cross_task_mean_J=mean_j, P_J_ge_5=ge5 / pairs)
    return out


def scaled_write(model: BTSPCAM, code: np.ndarray, label: int, scalar: float) -> tuple[int, int]:
    """B4-only globally scaled plateau probability; no formal-code mutation."""
    model.traces.update(code, label)
    signals = model.traces.signals(code, label, model.config.beta)
    prob = min(1.0, scalar * plateau_probability(model.config.base_plateau_probability, signals))
    plateaus = sample_slot_plateaus(prob, model.memory.slots, model.plateau_rng)
    flips = model.memory.update(code, plateaus, model.config.flip_probability, model.synapse_rng)
    return int(plateaus.sum()), int(flips)


def path_run(base: BTSPCAMConfig, k: int, tasks: int, budget_scalar: float | None = None) -> None:
    label = f"k{k}_budget41over128" if budget_scalar is not None else f"k{k}"
    branches = ["sparse_only", "fixed_WB1", "evolving_aligned"] if budget_scalar is None else ["evolving_budget_matched"]
    if PATH_ROWS.exists():
        prior = [r for r in read_rows(PATH_ROWS) if r["run"] == label]
        if len(prior) == tasks * len(branches) and {int(r["task"]) for r in prior} == set(range(1, tasks + 1)):
            print(json.dumps({"run": label, "skipped": "complete"}), flush=True)
            return
    model = make_model(base, k, 1993, 1993)
    aligned = CoordinateAlignedRidge(base.code_dim, base.memory_slots, k, base.gamma)
    fixed = StreamingRidge(base.memory_slots, 0, base.gamma) if budget_scalar is None else None
    wb1 = None
    test_s: list[sparse.csr_matrix] = []
    test_y: list[np.ndarray] = []
    bit_counts = np.zeros(base.code_dim, dtype=np.int64)
    gross_total = plateau_total = 0
    rows = []
    running = {branch: 0.0 for branch in branches}
    for task in range(1, tasks + 1):
        start = time.perf_counter()
        h, y = load_split("train", task)
        s = model.encoder.encode(h, base.encoder_batch_size)
        bit_counts += s.sum(axis=0, dtype=np.int64)
        plateau_task = gross_task = 0
        for code, label_y in zip(s, y):
            if budget_scalar is None:
                log = model._learn_code(code, int(label_y))
                plateau_task += int(log["plateau_count"])
                gross_task += int(log["flips"])
            else:
                pcount, fcount = scaled_write(model, code, int(label_y), budget_scalar)
                plateau_task += pcount
                gross_task += fcount
        plateau_total += plateau_task
        gross_total += gross_task
        if task == 1:
            wb1 = model.memory.weights.copy()
        aligned.update(s, y)
        if fixed is not None:
            assert wb1 is not None
            train_a = overlap(sparse.csr_matrix(s), np.ascontiguousarray(wb1.T, dtype=np.float32), k)
            fixed.expand_classes(int(y.max()) + 1)
            fixed.update(train_a, y)
            fixed.solve()
            fixed._last_cholesky = None
        del h, y, s
        ridge = solve_aligned(aligned, model.memory.weights)
        h, y = load_split("test", task)
        test_s.append(sparse.csr_matrix(model.encoder.encode(h, base.encoder_batch_size)))
        test_y.append(y)
        del h
        common = dict(run=label, k=k, task=task, rf_seed=1993, write_seed=1993,
                      plateau_scalar=budget_scalar if budget_scalar is not None else 1.0,
                      plateau_count_task=plateau_task, gross_flips_task=gross_task,
                      cumulative_plateau_count=plateau_total,
                      cumulative_gross_flips=gross_total,
                      WB_density=float(model.memory.weights.mean()))
        geom = geometry(model.memory.weights, k, test_s, test_y, bit_counts)
        run_rows = []
        if budget_scalar is None:
            weights = sparse_ridge_weights(aligned)
            metric = sparse_scores(weights, test_s, test_y, task)
            del weights
            running["sparse_only"] += metric["cumulative_accuracy"]
            sparse_geom = {**geom, "a_same_cos": float("nan"),
                           "a_diff_cos": float("nan"), "a_margin": float("nan")}
            run_rows.append({**common, "branch": "sparse_only", **metric, **sparse_geom,
                             "average_incremental_to_task": running["sparse_only"] / task,
                             "feature_WB_density": float("nan")})
            assert fixed is not None and wb1 is not None
            metric = score_sources(fixed, wb1, k, test_s, test_y, task)
            running["fixed_WB1"] += metric["cumulative_accuracy"]
            fixed_geom = geometry(wb1, k, test_s, test_y, bit_counts)
            run_rows.append({**common, "branch": "fixed_WB1", **metric, **fixed_geom,
                             "average_incremental_to_task": running["fixed_WB1"] / task,
                             "feature_WB_density": float(wb1.mean())})
        metric = score_sources(ridge, model.memory.weights, k, test_s, test_y, task)
        branch = "evolving_budget_matched" if budget_scalar is not None else "evolving_aligned"
        running[branch] += metric["cumulative_accuracy"]
        run_rows.append({**common, "branch": branch, **metric, **geom,
                         "average_incremental_to_task": running[branch] / task,
                         "feature_WB_density": float(model.memory.weights.mean())})
        for row in run_rows:
            row["task_seconds"] = time.perf_counter() - start
        rows.extend(run_rows)
        append_completed(PATH_ROWS, {"run": label}, rows)
        print(json.dumps({"run": label, "task": task,
                          "metrics": {r["branch"]: round(r["cumulative_accuracy"], 5) for r in run_rows},
                          "seconds": round(time.perf_counter() - start, 2)}), flush=True)
        del ridge
        gc.collect()


def development_split(y: np.ndarray, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed class-stratified split; no test data and no k-dependent randomness."""
    val_mask = np.zeros(len(y), dtype=bool)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng = np.random.default_rng(seed + int(label))
        chosen = rng.permutation(idx)[:round(len(idx) * fraction)]
        val_mask[chosen] = True
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def development_run(base: BTSPCAMConfig, k: int, exp: dict) -> list[dict]:
    model = make_model(base, k, 1993, 1993)
    aligned = CoordinateAlignedRidge(base.code_dim, base.memory_slots, k, base.gamma)
    val_s: list[sparse.csr_matrix] = []
    val_y: list[np.ndarray] = []
    running = 0.0
    rows = []
    for task in range(1, 6):
        start = time.perf_counter()
        h, y = load_split("train", task)
        train_i, val_i = development_split(y, exp["validation_split_seed"],
                                           exp["validation_fraction"])
        s_train = model.encoder.encode(h[train_i], base.encoder_batch_size)
        s_val = model.encoder.encode(h[val_i], base.encoder_batch_size)
        for code, label_y in zip(s_train, y[train_i]):
            model._learn_code(code, int(label_y))
        aligned.update(s_train, y[train_i])
        # The validation subset has no write/eligibility/C/D/readout update.
        val_s.append(sparse.csr_matrix(s_val))
        val_y.append(y[val_i])
        del h, y, s_train, s_val
        ridge = solve_aligned(aligned, model.memory.weights)
        score = score_sources(ridge, model.memory.weights, k, val_s, val_y, task)
        running += score["cumulative_accuracy"]
        rows.append(dict(phase="development_validation", k=k, task=task,
                         split_seed=exp["validation_split_seed"],
                         validation_fraction=exp["validation_fraction"],
                         development_samples_per_class=400,
                         validation_samples_per_class=100,
                         **score, validation_average_incremental_to_task=running / task,
                         selection_primary=exp["selection_primary"],
                         selected=0, task_seconds=time.perf_counter() - start))
        print(json.dumps({"phase": "validation", "k": k, "task": task,
                          "cumulative": score["cumulative_accuracy"],
                          "average": running / task}), flush=True)
        del ridge
        gc.collect()
    return rows


def selected_full_retrain(base: BTSPCAMConfig, k: int, tasks: int) -> dict:
    """From scratch on full train; only Task20 invokes test data."""
    model = make_model(base, k, 1993, 1993)
    aligned = CoordinateAlignedRidge(base.code_dim, base.memory_slots, k, base.gamma)
    start = time.perf_counter()
    for task in range(1, tasks + 1):
        h, y = load_split("train", task)
        s = model.encoder.encode(h, base.encoder_batch_size)
        for code, label_y in zip(s, y):
            model._learn_code(code, int(label_y))
        aligned.update(s, y)
        del h, y, s
        print(json.dumps({"phase": "selected_full_retrain", "k": k,
                          "task": task, "test_accessed": False}), flush=True)
    ridge = solve_aligned(aligned, model.memory.weights)
    test_s = []
    test_y = []
    for task in range(1, tasks + 1):
        h, y = load_split("test", task)
        test_s.append(sparse.csr_matrix(model.encoder.encode(h, base.encoder_batch_size)))
        test_y.append(y)
        del h
    score = score_sources(ridge, model.memory.weights, k, test_s, test_y, tasks)
    return dict(phase="final_single_test", k=k, task=tasks, **score,
                WB_density=float(model.memory.weights.mean()),
                test_evaluation_count=1, full_train_samples=tasks * 2500,
                total_seconds=time.perf_counter() - start)


def selection_run(base: BTSPCAMConfig, exp: dict) -> None:
    existing = read_rows(SELECT_ROWS) if SELECT_ROWS.exists() else []
    if any(r["phase"] == "final_single_test" for r in existing):
        print(json.dumps({"phase": "selection", "skipped": "final_test_already_recorded"}), flush=True)
        return
    validation = [r for r in existing if r["phase"] == "development_validation"]
    for k in exp["selection_k"]:
        prior = [r for r in validation if int(r["k"]) == k]
        if len(prior) == 5 and {int(r["task"]) for r in prior} == set(range(1, 6)):
            continue
        fresh = development_run(base, k, exp)
        validation = [r for r in validation if int(r["k"]) != k] + fresh
        write_rows(SELECT_ROWS, validation)
    if len(validation) != len(exp["selection_k"]) * 5:
        raise AssertionError("validation screen incomplete")
    endings = {k: next(r for r in validation if int(r["k"]) == k and int(r["task"]) == 5)
               for k in exp["selection_k"]}
    chosen = sorted(exp["selection_k"],
                    key=lambda k: (-float(endings[k]["validation_average_incremental_to_task"]), k))[0]
    for row in validation:
        row["selected"] = int(int(row["k"]) == chosen)
        row["selected_k"] = chosen
    # Persist locked selection *before* any test split is opened.
    write_rows(SELECT_ROWS, validation)
    print(json.dumps({"phase": "locked_selection_before_test", "selected_k": chosen,
                      "criterion": exp["selection_primary"]}), flush=True)
    result = selected_full_retrain(base, chosen, exp["tasks"])
    result.update(selected=1, selected_k=chosen,
                  selection_primary=exp["selection_primary"],
                  selection_tie_break=exp["selection_tie_break"])
    write_rows(SELECT_ROWS, validation + [result])
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("seeds", "seed-summary", "paths", "selection"), required=True)
    parser.add_argument("--rf-seed", type=int)
    parser.add_argument("--write-seed", type=int)
    parser.add_argument("--k", type=int)
    args = parser.parse_args()
    exp, base = protocol()
    if args.phase == "seeds":
        for rf in exp["rf_seeds"]:
            for wr in exp["write_seeds"]:
                for k in exp["seed_pair_k"]:
                    if args.rf_seed is not None and rf != args.rf_seed:
                        continue
                    if args.write_seed is not None and wr != args.write_seed:
                        continue
                    if args.k is not None and k != args.k:
                        continue
                    seed_run(base, k, rf, wr, exp["tasks"])
    elif args.phase == "seed-summary":
        seed_summary(exp)
    elif args.phase == "paths":
        for k in exp["path_k"]:
            if args.k is None or k == args.k:
                path_run(base, k, exp["tasks"])
        if args.k is None or args.k == 128:
            path_run(base, 128, exp["tasks"], exp["write_budget_scalar"])
    elif args.phase == "selection":
        selection_run(base, exp)
    else:
        raise AssertionError(args.phase)


if __name__ == "__main__":
    main()
