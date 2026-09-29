"""Independent, resumable one-factor BTSP parameter screen (tau=0).

Neither this script nor its YAML changes the formal reproduction.  Every
non-gamma candidate replays the same cached frozen-ViT stream and RNG seeds.
Gamma candidates reuse the baseline W_B/C/D trajectory exactly.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import gc
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
from btsp_cam.eligibility import ClassEligibilityTraces, GatingSignals
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import pilot_class_order
from btsp_cam.ridge import StreamingRidge
from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_coordinate_aligned_full20 import CosineMoments, formal_wb
from run_repair_ladder_task1_5 import load_split

CFG_PATH = ROOT / "experiments" / "configs" / "unresolved_parameter_screen.yaml"
ARCHIVE = ROOT / "diagnostics" / "coordinate_aligned_full20.csv"


class SplitDecayTraces(ClassEligibilityTraces):
    """Experimental Eq.(4) state-dependent decay family; not author code."""

    def __init__(self, classes: int, bits: int, eta: float, current: float, inactive: float):
        super().__init__(classes, bits, eta, inactive)
        self.current_decay = current
        self.inactive_decay = inactive

    def update(self, code: np.ndarray, label: int) -> None:
        old = self.values[label].copy()
        self.values *= 1.0 - self.inactive_decay
        self.values[label] = (1.0 - self.current_decay) * old + self.eta * code


class FastTraces(ClassEligibilityTraces):
    """Algebraically identical cosine signals, sparse dot and cached norms.

    This is experimental acceleration only. The baseline run must reproduce
    every archived W_B bit and aligned accuracy before other candidates use it.
    """

    def __init__(self, classes: int, bits: int, eta: float, current: float, inactive: float):
        super().__init__(classes, bits, eta, inactive)
        self.current_decay = float(current)
        self.inactive_decay = float(inactive)
        self.norm_sq = np.zeros(classes, dtype=np.float64)

    def update(self, code: np.ndarray, label: int) -> None:
        active = np.flatnonzero(code)
        old_row = self.values[label].copy()
        old_norm = self.norm_sq[label]
        self.values *= 1.0 - self.inactive_decay
        self.norm_sq *= (1.0 - self.inactive_decay) ** 2
        self.values[label] = (1.0 - self.current_decay) * old_row
        before_add = self.values[label, active].sum(dtype=np.float64)
        self.values[label, active] += self.eta
        self.norm_sq[label] = ((1.0 - self.current_decay) ** 2 * old_norm
                               + 2.0 * self.eta * before_add + self.eta ** 2 * len(active))

    def signals(self, code: np.ndarray, label: int, beta: float) -> GatingSignals:
        active = np.flatnonzero(code)
        dots = self.values[:, active].sum(axis=1, dtype=np.float64)
        denom = np.sqrt(len(active) * self.norm_sq)
        similarities = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
        familiarity = float(np.clip(similarities[label], 0.0, 1.0))
        similarities[label] = 0.0
        max_rival = float(similarities.max())
        return GatingSignals(familiarity, 1.0 - familiarity,
                             float(np.exp(-beta * max_rival)), max_rival)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    k: int = 41
    p: float = 0.5
    f_q_base: float = 0.005
    beta: float = 1.0
    gamma: float = 0.1
    lambda_current: float = 0.01
    lambda_inactive: float = 0.01

    def row(self) -> dict:
        return dict(candidate=self.name, family=self.family, k=self.k, p=self.p,
                    f_q_base=self.f_q_base, beta=self.beta, gamma=self.gamma,
                    lambda_current=self.lambda_current, lambda_inactive=self.lambda_inactive)


BASE = Candidate("baseline", "baseline")


def candidates(exp: dict) -> list[Candidate]:
    out = [BASE]
    for family, field in (("k", "k"), ("p", "p"), ("f_q_base", "f_q_base"), ("beta", "beta")):
        for value in exp["families"][family]:
            if float(value) != float(getattr(BASE, field)):
                out.append(replace(BASE, name=f"{family}_{value}", family=family, **{field: value}))
    for name, decay in exp["families"]["lambda"].items():
        current, inactive = float(decay["current"]), float(decay["inactive"])
        if (current, inactive) != (BASE.lambda_current, BASE.lambda_inactive):
            out.append(replace(BASE, name=name, family="lambda",
                               lambda_current=current, lambda_inactive=inactive))
    # L1 and L2 duplicate the current baseline numerically; represent as aliases
    # in the report, never waste a second stochastic trajectory on them.
    for entry in exp.get("joint_candidates", []):
        out.append(Candidate(**entry))
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finalize_formal_screen_csv(path: Path) -> None:
    """Label exact duplicate lambda controls without a fictitious retrain."""
    rows = read_rows(path)
    if any(r["candidate"] in ("L1", "L2") for r in rows):
        raise ValueError("lambda alias rows are already present")
    baseline = [r for r in rows if r["candidate"] == "baseline"]
    if len(baseline) != 10 or len(rows) != 300:
        raise ValueError("formal screen is incomplete; refuse to finalize aliases")
    for row in rows:
        row["trace_backend"] = "formal_original"
        row["trajectory_status"] = "independently_run_or_shared_gamma"
    for alias in ("L1", "L2"):
        for source in baseline:
            row = dict(source)
            row["candidate"] = alias
            row["family"] = "lambda"
            row["source"] = "exact_alias_of_baseline_no_retraining"
            row["trajectory_status"] = "alias_of_baseline"
            rows.append(row)
    write_rows(path, rows)


def overlap(codes: sparse.csr_matrix, wt: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(codes.astype(np.float32) @ wt, dtype=np.float64) / k


def aligned_statistics_blas(aligned: CoordinateAlignedRidge,
                            wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact float64 integer products, dense BLAS order; C/D remain the state."""
    c = torch.from_numpy(aligned.c.toarray())
    w = torch.from_numpy(np.asarray(wb, dtype=np.float64))
    middle = w @ c
    a = ((middle @ w.T) / (aligned.k ** 2)).numpy()
    b = (w @ torch.from_numpy(aligned.d)).numpy() / aligned.k
    return a, b


def evaluate(ridges: dict[float, StreamingRidge], test_codes: list[sparse.csr_matrix],
             test_labels: list[np.ndarray], wb: np.ndarray, k: int, task: int,
             first_a: np.ndarray | None) -> tuple[dict[float, dict], dict, np.ndarray]:
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    counts = {g: {"old": [0, 0], "current": [0, 0], "five": [0, 0]} for g in ridges}
    a_moments = CosineMoments(wb.shape[0])
    s_moments = CosineMoments(test_codes[0].shape[1])
    old_codes = None
    new_codes = test_codes[-1]
    if task > 1:
        old_codes = sparse.vstack(test_codes[:-1], format="csr")
        # Cross-task E[J] is exact via bit frequencies; tail P(J>=5) uses a
        # deterministic 500x500 probe per previous source task.
        old_freq = np.asarray(old_codes.sum(axis=0)).ravel().astype(np.float64)
        new_freq = np.asarray(new_codes.sum(axis=0)).ravel().astype(np.float64)
        mean_j = float(old_freq @ new_freq / (old_codes.shape[0] * new_codes.shape[0]))
        ge5 = 0
        pairs = 0
        for part in test_codes[:-1]:
            hit = (new_codes @ part.T).tocsr()
            ge5 += int(np.count_nonzero(hit.data >= 5))
            pairs += new_codes.shape[0] * part.shape[0]
        p_j_ge5 = ge5 / pairs
    else:
        mean_j = p_j_ge5 = float("nan")
    current_a1 = None
    for source, (codes, y) in enumerate(zip(test_codes, test_labels), 1):
        a = overlap(codes, wt, k)
        if source == 1:
            current_a1 = a.copy()
        a_moments.add(a, y)
        s_moments.add(codes.toarray().astype(np.float64), y)
        region = "current" if source == task else "old"
        for gamma, ridge in ridges.items():
            logits = ridge.logits(a)
            correct = int(np.count_nonzero(logits.argmax(axis=1) == y))
            counts[gamma][region][0] += correct
            counts[gamma][region][1] += len(y)
            if source == task:
                five = logits[:, (task - 1) * 5: task * 5].argmax(axis=1) + (task - 1) * 5
                counts[gamma]["five"] = [int(np.count_nonzero(five == y)), len(y)]
        del a
    assert current_a1 is not None
    if first_a is None:
        first_a = current_a1.copy()
    dot = np.einsum("ij,ij->i", current_a1, first_a)
    denom = np.linalg.norm(current_a1, axis=1) * np.linalg.norm(first_a, axis=1)
    drift_cos = float(np.mean(np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)))
    normalized_drift = float(np.mean(np.linalg.norm(current_a1 - first_a, axis=1) /
                                     np.maximum(np.linalg.norm(first_a, axis=1), 1e-12)))
    out = {}
    for gamma, c in counts.items():
        old_correct, old_total = c["old"]
        new_correct, new_total = c["current"]
        out[gamma] = dict(current_accuracy=new_correct / new_total,
                          old_accuracy=old_correct / old_total if old_total else float("nan"),
                          cumulative_accuracy=(old_correct + new_correct) / (old_total + new_total),
                          current_5way_accuracy=c["five"][0] / c["five"][1])
    sparse_same, sparse_diff, sparse_margin = s_moments.values()
    a_same, a_diff, a_margin = a_moments.values()
    aux = dict(sparse_same_cos=sparse_same, sparse_diff_cos=sparse_diff,
               sparse_margin=sparse_margin, a_same_cos=a_same, a_diff_cos=a_diff,
               a_margin=a_margin, cross_task_mean_J=mean_j, P_J_ge_5=p_j_ge5,
               task1_a_cosine_to_initial=drift_cos,
               task1_a_normalized_l2_drift=normalized_drift)
    return out, aux, first_a


def run_candidate(candidate: Candidate, tasks: int, output: Path,
                  base_config: BTSPCAMConfig, gamma_values: list[float] | None = None,
                  validate_baseline: bool = False, fast_traces: bool = True) -> None:
    existing = read_rows(output) if output.exists() else []
    prior_tasks = {int(r["task"]) for r in existing if r["candidate"] == candidate.name}
    if prior_tasks == set(range(1, tasks + 1)):
        print(json.dumps({"candidate": candidate.name, "skipped": "already in CSV"}), flush=True)
        return
    if prior_tasks:
        # A crashed run is replayed from the initial RNG state; partial rows
        # cannot be used to resume a memory trajectory.
        related = {candidate.name}
        if gamma_values:
            related.update(f"gamma_{g}" for g in gamma_values if g != candidate.gamma)
        existing = [r for r in existing if r["candidate"] not in related]
        write_rows(output, existing)
    cfg = replace(base_config, k=candidate.k, flip_probability=candidate.p,
                  base_plateau_probability=candidate.f_q_base, beta=candidate.beta,
                  gamma=candidate.gamma, trace_decay=candidate.lambda_inactive, tau=0.0)
    model = BTSPCAM(cfg, class_order=pilot_class_order(cfg.num_classes, cfg.seed))
    # The formal model allocates a ridge in __init__; this independent runner
    # uses only its exact writer/encoder and C/D readout.
    model.ridge = None
    if fast_traces:
        model.traces = FastTraces(cfg.num_classes, cfg.code_dim, cfg.eta,
                                 candidate.lambda_current, candidate.lambda_inactive)
    elif candidate.lambda_current != candidate.lambda_inactive:
        model.traces = SplitDecayTraces(cfg.num_classes, cfg.code_dim, cfg.eta,
                                        candidate.lambda_current, candidate.lambda_inactive)
    aligned = CoordinateAlignedRidge(cfg.code_dim, cfg.memory_slots, cfg.k, cfg.gamma)
    test_codes: list[sparse.csr_matrix] = []
    test_labels: list[np.ndarray] = []
    train_counts = np.zeros(cfg.code_dim, dtype=np.int64)
    first_a = None
    rows = []
    cumulative_sum = {g: 0.0 for g in (gamma_values or [candidate.gamma])}
    archived = {(int(r["task"]), r["branch"]): r for r in read_rows(ARCHIVE)} if validate_baseline else {}
    start_all = time.perf_counter()
    for task in range(1, tasks + 1):
        started = time.perf_counter()
        h, y = load_split("train", task)
        codes = model.encoder.encode(h, cfg.encoder_batch_size)
        del h
        encoded_at = time.perf_counter()
        if not np.all(codes.sum(axis=1) == cfg.k):
            raise AssertionError("encoder violated exact-k")
        train_counts += codes.sum(axis=0, dtype=np.int64)
        logs = [model._learn_code(code, int(label)) for code, label in zip(codes, y)]
        learned_at = time.perf_counter()
        aligned.update(codes, y)
        del codes, y
        updated_at = time.perf_counter()
        wb = model.memory.weights
        if validate_baseline:
            np.testing.assert_array_equal(wb, formal_wb(task, cfg.memory_slots, cfg.code_dim))
        a, b = aligned_statistics_blas(aligned, wb)
        if validate_baseline and task == 1:
            old_a, old_b = aligned.aligned_statistics(wb)
            np.testing.assert_array_equal(a, old_a)
            np.testing.assert_array_equal(b, old_b)
            del old_a, old_b
        aligned_at = time.perf_counter()
        gammas = gamma_values or [candidate.gamma]
        ridges = {}
        for gamma in gammas:
            ridge = StreamingRidge(cfg.memory_slots, b.shape[1], gamma)
            ridge.a = a
            ridge.b = b
            ridge.solve()
            ridge._last_cholesky = None
            ridges[gamma] = ridge
        del a, b
        solved_at = time.perf_counter()
        h_test, y_test = load_split("test", task)
        test_codes.append(sparse.csr_matrix(model.encoder.encode(h_test, cfg.encoder_batch_size)))
        test_labels.append(y_test)
        del h_test
        test_encoded_at = time.perf_counter()
        metrics, aux, first_a = evaluate(ridges, test_codes, test_labels, wb, cfg.k, task, first_a)
        evaluated_at = time.perf_counter()
        storage = aligned.storage()
        top = int(np.ceil(0.01 * cfg.code_dim))
        top_share = float(np.sort(train_counts)[-top:].sum() / (task * 2500 * cfg.k))
        log_names = ("familiarity", "novelty", "interference", "plateau_probability", "plateau_count")
        log_stats = {f"{key}_mean": float(np.mean([rec[key] for rec in logs])) for key in log_names}
        log_stats.update({f"{key}_std": float(np.std([rec[key] for rec in logs]))
                          for key in ("familiarity", "novelty", "interference")})
        for gamma in gammas:
            cumulative_sum[gamma] += metrics[gamma]["cumulative_accuracy"]
            row_candidate = candidate if gamma == candidate.gamma else replace(
                candidate, name=f"gamma_{gamma}", family="gamma", gamma=gamma)
            row = {**row_candidate.row(), "task": task, **metrics[gamma],
                   "average_cumulative_to_task": cumulative_sum[gamma] / task,
                   "WB_density": float(wb.mean()),
                   "top1_bit_share": top_share,
                   "zero_winner_bits": int(np.count_nonzero(train_counts == 0)),
                   **aux, **log_stats, "C_nnz": storage["C_nnz"],
                   "task_seconds": time.perf_counter() - started,
                   "train_encode_seconds": encoded_at - started,
                   "WB_write_seconds": learned_at - encoded_at,
                   "C_D_update_seconds": updated_at - learned_at,
                   "alignment_seconds": aligned_at - updated_at,
                   "ridge_solve_seconds": solved_at - aligned_at,
                   "test_encode_seconds": test_encoded_at - solved_at,
                   "evaluation_seconds": evaluated_at - test_encoded_at,
                   "protocol": "CIFAR100_B0Inc5_seed1993_tau0_aligned",
                   "source": "independent_parameter_screen"}
            if validate_baseline and gamma == BASE.gamma:
                expected = float(archived[(task, "coordinate_aligned_ridge")]["cumulative_accuracy"])
                if abs(row["cumulative_accuracy"] - expected) > 1e-12:
                    raise AssertionError(f"baseline Task{task}: {row['cumulative_accuracy']} != {expected}")
            rows.append(row)
        write_rows(output, existing + rows)
        print(json.dumps({"candidate": candidate.name, "task": task,
                          "accuracy": metrics[candidate.gamma]["cumulative_accuracy"],
                          "seconds": round(time.perf_counter() - started, 1)}), flush=True)
        del ridges, wb, logs
        gc.collect()
    print(json.dumps({"candidate": candidate.name, "tasks": tasks,
                      "elapsed_seconds": round(time.perf_counter() - start_all, 1)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "full"), required=True)
    parser.add_argument("--candidate", help="Run a single named candidate; otherwise run phase list")
    parser.add_argument("--formal-screen", action="store_true",
                        help="Use the unaccelerated formal trace for the entire Task1-10 screen")
    args = parser.parse_args()
    if args.formal_screen and args.phase != "screen":
        raise ValueError("--formal-screen applies only to the screening phase")
    exp = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    base_path = (CFG_PATH.parent / exp["base_config"]).resolve()
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(base_path.read_text(encoding="utf-8")))
    if (base.seed, base.code_dim, base.memory_slots, base.k, base.gamma) != (1993, 8192, 4096, 41, 0.1):
        raise AssertionError("formal baseline configuration changed")
    all_candidates = candidates(exp)
    if args.phase == "screen":
        selected = all_candidates
        tasks = int(exp["screen_tasks"])
        path = (CFG_PATH.parent / exp["outputs"]["screen_csv"]).resolve()
    else:
        if not exp["selected_full20"]:
            raise ValueError("select full20 candidates after reviewing Task1-10 screen")
        if len(exp["selected_full20"]) > 5:
            raise ValueError("full20 stage allows at most five candidate trajectories")
        names = set(exp["selected_full20"])
        selected = [c for c in all_candidates if c.name in names]
        if len(selected) != len(names):
            raise ValueError(f"unknown selected candidates: {names - {c.name for c in selected}}")
        tasks = int(exp["full_tasks"])
        path = (CFG_PATH.parent / exp["outputs"]["full_csv"]).resolve()
    if args.candidate:
        selected = [c for c in selected if c.name == args.candidate]
        if not selected:
            raise ValueError(f"unknown/ineligible candidate: {args.candidate}")
    for candidate in selected:
        gamma_values = [float(x) for x in exp["families"]["gamma"]] if args.phase == "screen" and candidate == BASE else None
        run_candidate(candidate, tasks, path, base, gamma_values,
                      validate_baseline=(candidate == BASE),
                      fast_traces=(args.phase == "screen" and not args.formal_screen))


if __name__ == "__main__":
    main()
