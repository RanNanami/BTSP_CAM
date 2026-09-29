"""Readout-only gamma audit on archived W_B trajectories (experimental).

This script never writes W_B or changes the formal reproduction.  Its
coordinate-aligned tau=0 statistics are reconstructed from current-task
cached ViT features and aggregate sparse-code C/D.  Gamma=0 uses a stable
spectral pseudoinverse solely as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
import sys
import time

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder
from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_repair_ladder_task1_5 import load_split
from run_unresolved_parameter_screen import aligned_statistics_blas, overlap
from run_coordinate_aligned_full20 import formal_wb

CONFIG = ROOT / "experiments/configs/gamma_spectral_audit.yaml"
DIAG = ROOT / "diagnostics"
SPECTRUM = DIAG / "gamma_spectrum.csv"
FIXED = DIAG / "gamma_fixed_task10.csv"
NORMALIZED = DIAG / "gamma_normalized_full20.csv"
NUMERICS = DIAG / "ridge_numerics.csv"
ARCHIVE = DIAG / "k_path_decomposition.csv"
SLOTS = 4096


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open(newline="", encoding="utf-8") as handle:
            fields = list(csv.DictReader(handle).fieldnames or [])
        if list(row) != fields:
            raise ValueError(f"CSV schema changed: {path}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def wb_checkpoint(k: int, task: int) -> np.ndarray:
    if k == 41:
        w = formal_wb(task, SLOTS, 8192)
    else:
        path = DIAG / f"tau_oracle_k{k}" / f"WB_task{task:02d}.npz"
        with np.load(path, allow_pickle=False) as state:
            w = state["WB"].copy()
    if w.shape != (4096, 8192) or not np.all((w == 0) | (w == 1)):
        raise ValueError(f"invalid W_B checkpoint for k={k}, task={task}")
    return w


def baseline_lookup(k: int) -> dict[int, float]:
    out = {}
    for row in read_csv(ARCHIVE):
        if int(row["k"]) == k and row["branch"] == "evolving_aligned":
            out[int(row["task"])] = float(row["cumulative_accuracy"])
    if len(out) != 20:
        raise ValueError(f"archived baseline missing k={k}")
    return out


def eig_summary(a: np.ndarray, exact: bool, vectors: bool):
    if vectors:
        eig, vec = linalg.eigh(a, check_finite=False, overwrite_a=False)
    elif exact:
        eig = linalg.eigh(a, eigvals_only=True, check_finite=False, overwrite_a=False)
        vec = None
    else:
        eig = None
        vec = None
    return eig, vec


def leading_eigenvalue(a: np.ndarray, eig: np.ndarray | None) -> float:
    if eig is not None:
        return float(eig[-1])
    val = eigsh(a, k=1, which="LA", return_eigenvectors=False, tol=1e-7)[0]
    return float(val)


def spectral_fields(eig: np.ndarray | None, trace: float, lam_max: float,
                    gamma0: float, d: int) -> dict:
    result = {
        "eigensolver": "exact_eigh" if eig is not None else "eigsh_largest_only",
        "eig_min": float("nan"), "eig_max": lam_max,
        "eig_min_nonzero": float("nan"), "rank": float("nan"),
        "effective_rank": float("nan"), "condition_A": float("nan"),
        "condition_A_positive_subspace": float("nan"),
        "condition_A_plus_gamma0": float("nan"),
        "eig_p1": float("nan"), "eig_p5": float("nan"),
        "eig_p25": float("nan"), "eig_p50": float("nan"),
        "eig_p75": float("nan"), "eig_p95": float("nan"),
        "eig_p99": float("nan"),
        "r_med": float("nan"),
    }
    if eig is None:
        return result
    tol = d * np.finfo(np.float64).eps * max(abs(lam_max), 1.0)
    if eig[0] < -10 * tol:
        raise ValueError(f"A has a substantial negative eigenvalue {eig[0]}, tolerance {tol}")
    pos = eig[eig > tol]
    rank = len(pos)
    p = pos / pos.sum() if rank else np.array([])
    quant = np.percentile(eig, [1, 5, 25, 50, 75, 95, 99])
    result.update({
        "eig_min": float(eig[0]),
        "eig_min_nonzero": float(pos[0]) if rank else float("nan"),
        "rank": rank,
        "effective_rank": float(np.exp(-np.sum(p * np.log(p)))) if rank else 0.0,
        "condition_A": float(lam_max / eig[0]) if rank == d and eig[0] > 0 else float("inf"),
        "condition_A_positive_subspace": float(lam_max / pos[0]) if rank else float("inf"),
        "condition_A_plus_gamma0": float((lam_max + gamma0) / (max(eig[0], 0.0) + gamma0)),
        "r_med": float(gamma0 / np.median(pos)) if rank else float("nan"),
        **{f"eig_p{x}": float(v) for x, v in zip([1, 5, 25, 50, 75, 95, 99], quant)},
    })
    return result


def solve_positive(a: np.ndarray, b: np.ndarray, gamma: float) -> tuple[np.ndarray, float, str]:
    ridge = StreamingRidge(a.shape[0], b.shape[1], gamma)
    ridge.a = a
    ridge.b = b
    ridge.solve()
    weights = ridge.weights
    condition_1 = ridge.condition_number
    status = ridge.last_solve_status
    ridge._last_cholesky = None
    return weights, condition_1, status


def solve_zero(eig: np.ndarray, vec: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    tol = len(eig) * np.finfo(np.float64).eps * max(abs(eig[-1]), 1.0)
    keep = eig > tol
    projected = vec.T @ b
    projected[keep] /= eig[keep, None]
    projected[~keep] = 0.0
    return vec @ projected, int(np.count_nonzero(keep))


def evaluate(weights_by_name: dict[str, np.ndarray], test_codes: list[sparse.csr_matrix],
             test_labels: list[np.ndarray], wb: np.ndarray, k: int, task: int) -> dict:
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    counts = {name: {"old": [0, 0], "current": [0, 0], "five": [0, 0],
                     "logit_sq": 0.0, "logit_sum": 0.0, "logit_n": 0}
              for name in weights_by_name}
    for source, (codes, labels) in enumerate(zip(test_codes, test_labels), 1):
        z = overlap(codes, wt, k)
        region = "current" if source == task else "old"
        for name, weights in weights_by_name.items():
            logits = z @ weights
            item = counts[name]
            item[region][0] += int(np.count_nonzero(np.argmax(logits, axis=1) == labels))
            item[region][1] += len(labels)
            item["logit_sq"] += float(np.sum(logits ** 2))
            item["logit_sum"] += float(np.sum(logits))
            item["logit_n"] += logits.size
            if source == task:
                start = (task - 1) * 5
                pred5 = start + np.argmax(logits[:, start:start + 5], axis=1)
                item["five"] = [int(np.count_nonzero(pred5 == labels)), len(labels)]
        del z
    out = {}
    for name, item in counts.items():
        old_hit, old_n = item["old"]
        cur_hit, cur_n = item["current"]
        five_hit, five_n = item["five"]
        logit_mean = item["logit_sum"] / item["logit_n"]
        out[name] = {
            "cumulative_accuracy": (old_hit + cur_hit) / (old_n + cur_n),
            "old_accuracy": old_hit / old_n if old_n else float("nan"),
            "current_accuracy": cur_hit / cur_n,
            "current_5way_accuracy": five_hit / five_n,
            "logit_l2_rms_per_sample": float(np.sqrt(item["logit_sq"] / (old_n + cur_n))),
            "logit_std": float(np.sqrt(max(item["logit_sq"] / item["logit_n"] - logit_mean ** 2, 0.0))),
        }
    return out


def run(k: int, exp: dict, alpha_trace: float | None, alpha_max: float | None,
        max_task: int) -> tuple[float | None, float | None]:
    cfg_path = (CONFIG.parent / exp["base_config"]).resolve()
    config = BTSPCAMConfig.from_mapping(yaml.safe_load(cfg_path.read_text(encoding="utf-8")))
    if (config.seed, config.code_dim, config.memory_slots, config.gamma,
            config.flip_probability, config.trace_decay) != (1993, 8192, 4096, 0.1, 0.5, 0.01):
        raise ValueError("baseline config drift")
    gamma0 = float(exp["baseline_gamma"])
    encoder = RandomSparseEncoder(config.feature_dim, config.code_dim, k,
                                  RNGProvenance.from_seed(config.seed).projection_seed)
    aligned = CoordinateAlignedRidge(config.code_dim, config.memory_slots, k, gamma0)
    archive = baseline_lookup(k)
    test_codes: list[sparse.csr_matrix] = []
    test_labels: list[np.ndarray] = []
    exact_tasks = set(exp["spectrum_tasks"])
    normalized = k in exp["normalized_k"]
    fixed_grid = k == exp["fixed_gamma_k"]
    spectrum_prior = {(int(r["k"]), int(r["task"])) for r in read_csv(SPECTRUM)}
    fixed_prior = {(int(r["k"]), int(r["task"]), float(r["gamma"])) for r in read_csv(FIXED)}
    norm_prior = {(int(r["k"]), int(r["task"]), r["method"]) for r in read_csv(NORMALIZED)}
    num_prior = {(int(r["k"]), int(r["task"])) for r in read_csv(NUMERICS)}

    for task in range(1, max_task + 1):
        started = time.perf_counter()
        h, y = load_split("train", task)
        codes = encoder.encode(h, config.encoder_batch_size)
        del h
        aligned.update(codes, y)
        del codes, y
        h, y = load_split("test", task)
        test_codes.append(sparse.csr_matrix(encoder.encode(h, config.encoder_batch_size)))
        test_labels.append(y)
        del h, y
        wb = wb_checkpoint(k, task)
        a, b = aligned_statistics_blas(aligned, wb)
        trace = float(np.trace(a))
        fro = float(np.linalg.norm(a, "fro"))
        diag = np.diag(a)
        symmetry = float(np.max(np.abs(a - a.T)))
        if symmetry > 1e-8:
            raise ValueError(f"A symmetry error {symmetry}")
        exact = task in exact_tasks or (fixed_grid and task <= exp["fixed_gamma_tasks"])
        eig, vec = eig_summary(a, exact, fixed_grid and task <= exp["fixed_gamma_tasks"])
        lam_max = leading_eigenvalue(a, eig)
        if k == 128 and task == 1:
            alpha_trace = gamma0 / (trace / SLOTS)
            alpha_max = gamma0 / lam_max
        if alpha_trace is None or alpha_max is None:
            raise ValueError("k128 Task1 reference must be run before other k")
        eig_info = spectral_fields(eig, trace, lam_max, gamma0, SLOTS)
        # The first moment is recoverable without retaining historical codes:
        # each column's mean is W @ D @ 1 / (kN).
        n = aligned.sample_count
        col_mean = (wb.astype(np.float64) @ aligned.d.sum(axis=1)) / (k * n)
        mean_z = float(col_mean.mean())
        std_z = float(np.sqrt(max(trace / (n * SLOTS) - mean_z ** 2, 0.0)))
        col_var = np.maximum(diag / n - col_mean ** 2, 0.0)
        # Nonzero count/magnitude require actual features, sampled from current
        # task test rather than old train examples; label the statistic clearly.
        test_current_z = overlap(test_codes[-1], np.ascontiguousarray(wb.T, dtype=np.float32), k)
        nonzero = test_current_z > 0
        active_mean = float(test_current_z[nonzero].mean()) if np.any(nonzero) else 0.0
        current_density = float(np.mean(nonzero))
        del test_current_z, nonzero
        if (k, task) not in spectrum_prior and task in exact_tasks:
            row = {
                "k": k, "task": task, "N": n, "feature_dim": SLOTS,
                "gamma0": gamma0, "trace_A": trace, "trace_A_per_dim": trace / SLOTS,
                "frobenius_A": fro, "spectral_norm_A": lam_max,
                **eig_info,
                "r_trace": gamma0 / (trace / SLOTS),
                "r_max": gamma0 / lam_max,
                "Z_mean_all_train_elements": mean_z,
                "Z_std_all_train_elements": std_z,
                "Z_mean_l2_per_train_sample": float("nan"),
                "Z_rms_l2_per_train_sample": float(np.sqrt(trace / n)),
                "Z_mean_active_magnitude_current_test": active_mean,
                "Z_density_current_test": current_density,
                "column_variance_p5": float(np.percentile(col_var, 5)),
                "column_variance_p50": float(np.percentile(col_var, 50)),
                "column_variance_p95": float(np.percentile(col_var, 95)),
                "alpha_trace_ref_k128_task1": alpha_trace,
                "alpha_max_ref_k128_task1": alpha_max,
            }
            append_row(SPECTRUM, row)
        gammas = {}
        if normalized:
            gammas.update({"FIXED_GAMMA": gamma0,
                           "TRACE_NORMALIZED": alpha_trace * trace / SLOTS,
                           "MAX_NORMALIZED": alpha_max * lam_max})
        elif task == 20:
            # k41 is a mechanism-only Task20 comparison, not a full sweep.
            gammas.update({"FIXED_GAMMA": gamma0,
                           "TRACE_NORMALIZED": alpha_trace * trace / SLOTS,
                           "MAX_NORMALIZED": alpha_max * lam_max})
        if fixed_grid and task <= exp["fixed_gamma_tasks"]:
            for multiplier in exp["fixed_gamma_multipliers"]:
                gammas[f"GRID_{multiplier}"] = gamma0 * float(multiplier)
        weights_by_gamma = {}
        condition_by_gamma = {}
        status_by_gamma = {}
        for gamma in sorted(set(gammas.values())):
            if gamma == 0.0:
                if eig is None or vec is None:
                    raise ValueError("gamma=0 needs exact eigenvectors")
                weights, rank = solve_zero(eig, vec, b)
                condition, status = float("nan"), f"spectral_pinv_rank_{rank}"
            elif vec is not None and eig is not None and gamma != gamma0:
                weights = vec @ ((vec.T @ b) / (eig[:, None] + gamma))
                condition = float("nan")
                status = "exact_spectral_ridge"
            else:
                weights, condition, status = solve_positive(a, b, gamma)
            weights_by_gamma[gamma] = weights
            condition_by_gamma[gamma] = condition
            status_by_gamma[gamma] = status
        if weights_by_gamma:
            # Evaluate each distinct gamma only once even if methods coincide.
            results = evaluate({str(g): w for g, w in weights_by_gamma.items()},
                               test_codes, test_labels, wb, k, task)
        else:
            results = {}
        for method, gamma in gammas.items():
            metrics = results[str(gamma)]
            weights = weights_by_gamma[gamma]
            solve_resid = float(np.linalg.norm(a @ weights + gamma * weights - b, "fro") /
                                max(np.linalg.norm(b, "fro"), 1e-30))
            common = {
                "k": k, "task": task, "N": n, "method": method, "gamma": gamma,
                "gamma0": gamma0, "gamma_over_trace_per_dim": gamma / (trace / SLOTS),
                "gamma_over_lambda_max": gamma / lam_max,
                "trace_A_per_dim": trace / SLOTS, "lambda_max_A": lam_max,
                "condition_A_plus_gamma_1norm_est": condition_by_gamma[gamma],
                "condition_A_plus_gamma_spectral": (
                    ((lam_max + gamma) / (max(float(eig[0]), 0.0) + gamma)
                     if gamma > 0 else (float("inf") if status_by_gamma[gamma] != "spectral_pinv_rank_4096"
                                         else lam_max / float(eig[0])))
                    if eig is not None else float("nan")),
                "min_eigenvalue_after_regularization": (float(eig[0]) + gamma if eig is not None else float("nan")),
                "ridge_weight_frobenius": float(np.linalg.norm(weights, "fro")),
                "train_ridge_residual_mse": float((np.sum(weights * (a @ weights)) -
                                                     2 * np.sum(weights * b) + n) / n),
                "solve_relative_residual": solve_resid,
                "solve_status": status_by_gamma[gamma],
                **metrics,
            }
            if method.startswith("GRID_") and (k, task, gamma) not in fixed_prior:
                append_row(FIXED, common)
            elif not method.startswith("GRID_") and (k, task, method) not in norm_prior:
                append_row(NORMALIZED, common)
            if method == "FIXED_GAMMA":
                target = archive[task]
                if abs(metrics["cumulative_accuracy"] - target) > 1e-12:
                    raise AssertionError(f"baseline mismatch k={k} task={task}: {metrics['cumulative_accuracy']} vs {target}")
        if task in exact_tasks and (k, task) not in num_prior:
            gamma = gamma0
            weights = weights_by_gamma[gamma] if gamma in weights_by_gamma else solve_positive(a, b, gamma)[0]
            residual = float(np.linalg.norm(a @ weights + gamma * weights - b, "fro") /
                             max(np.linalg.norm(b, "fro"), 1e-30))
            # Sum-vs-mean algebraic equivalence: mean-form gamma0 == sum-form N*gamma0.
            mean_weights, _, mean_status = solve_positive(a, b, n * gamma0)
            row = {"k": k, "task": task, "N": n, "A_symmetry_max_abs": symmetry,
                   "A_min_eigenvalue": float(eig[0]), "A_max_eigenvalue": lam_max,
                   "A_rank": eig_info["rank"], "A_plus_gamma0_spectral_condition": eig_info["condition_A_plus_gamma0"],
                   "float_dtype": str(a.dtype), "cholesky_status_gamma0": "ok",
                   "solve_relative_residual_gamma0": residual,
                   "mean_form_equivalent_gamma_sum": n * gamma0,
                   "mean_sum_weight_max_abs_diff": float(np.max(np.abs(mean_weights -
                                                                        solve_positive(a / n, b / n, gamma0)[0]))),
                   "mean_form_cholesky_status": mean_status}
            append_row(NUMERICS, row)
        elapsed = time.perf_counter() - started
        print(f"k={k} task={task:02d} N={n} tr/d={trace/SLOTS:.5g} eigmax={lam_max:.5g} "
              f"gammas={len(weights_by_gamma)} elapsed={elapsed:.1f}s", flush=True)
        del a, b, wb, eig, vec, weights_by_gamma, results
        gc.collect()
    return alpha_trace, alpha_max


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, choices=[41, 128, 160], required=True)
    ap.add_argument("--max-task", type=int, default=20)
    args = ap.parse_args()
    exp = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    alpha_trace = alpha_max = None
    if args.k != 128:
        ref = [r for r in read_csv(SPECTRUM) if int(r["k"]) == 128 and int(r["task"]) == 1]
        if len(ref) != 1:
            raise ValueError("run k=128 first to anchor normalized gamma")
        alpha_trace = float(ref[0]["alpha_trace_ref_k128_task1"])
        alpha_max = float(ref[0]["alpha_max_ref_k128_task1"])
    if not 1 <= args.max_task <= 20:
        raise ValueError("max-task must be in 1..20")
    run(args.k, exp, alpha_trace, alpha_max, args.max_task)


if __name__ == "__main__":
    main()
