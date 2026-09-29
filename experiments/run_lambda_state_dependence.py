"""Independent lambda-family diagnostic. No formal config/checkpoints are changed.

All families share the same cached ViT stream, sparse encoder and writer RNG.
The readout is experimental coordinate-aligned, unthresholded ridge (tau=0).
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from btsp_cam.eligibility import GatingSignals
from btsp_cam.protocol import pilot_class_order
from btsp_cam.ridge import StreamingRidge
from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_coordinate_aligned_full20 import CosineMoments
from run_repair_ladder_task1_5 import load_split
from run_unresolved_parameter_screen import FastTraces, aligned_statistics_blas, overlap

CONFIG = ROOT / "experiments" / "configs" / "lambda_state_dependence.yaml"
OUT10 = ROOT / "diagnostics" / "lambda_family_task10.csv"
OUT20 = ROOT / "diagnostics" / "lambda_family_full20.csv"


class StableFastTraces(FastTraces):
    """Use direct formal norms when cached squared norms become subnormal.

    Repeatedly multiplying norm_sq by (1-decay)^2 loses subnormal precision
    earlier than the formal implementation's norm of the actual trace row.
    This late-task fallback is required for bitwise baseline trajectory checks.
    """

    def __init__(self, classes: int, bits: int, eta: float, current: float, inactive: float):
        super().__init__(classes, bits, eta, current, inactive)
        self.ever_seen = np.zeros(classes, dtype=bool)

    def update(self, code: np.ndarray, label: int) -> None:
        super().update(code, label)
        self.ever_seen[label] = True

    def signals(self, code: np.ndarray, label: int, beta: float) -> GatingSignals:
        active = np.flatnonzero(code)
        dots = self.values[:, active].sum(axis=1, dtype=np.float64)
        # Keep the original accelerated arithmetic bit-for-bit outside the
        # subnormal fallback so the Task1-10 screen and full20 pair exactly.
        denom = np.sqrt(len(active) * np.maximum(self.norm_sq, 0.0))
        tiny = self.ever_seen & (self.norm_sq < 1e-260)
        if np.any(tiny):
            denom[tiny] = np.sqrt(len(active)) * np.linalg.norm(self.values[tiny], axis=1)
        similarities = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
        familiarity = float(np.clip(similarities[label], 0.0, 1.0))
        similarities[label] = 0.0
        max_rival = float(similarities.max())
        return GatingSignals(familiarity, 1.0 - familiarity,
                             float(np.exp(-beta * max_rival)), max_rival)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preencode(config: BTSPCAMConfig, model: BTSPCAM, tasks: int):
    train, labels, test, test_labels = [], [], [], []
    for task in range(1, tasks + 1):
        h, y = load_split("train", task)
        train.append(model.encoder.encode(h, config.encoder_batch_size))
        labels.append(y)
        h, y = load_split("test", task)
        test.append(sparse.csr_matrix(model.encoder.encode(h, config.encoder_batch_size)))
        test_labels.append(y)
        print(json.dumps({"preencoded_task": task, "k": config.k}), flush=True)
    return train, labels, test, test_labels


def test_geometry(test_codes, test_labels, wb, k, task, ridge):
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    groups = {name: CosineMoments(wb.shape[0]) for name in ("all", "old", "current")}
    sparse_groups = {name: CosineMoments(test_codes[0].shape[1]) for name in ("all", "old", "current")}
    totals = {name: [0, 0] for name in ("all", "old", "current", "five")}
    for source, (codes, labels) in enumerate(zip(test_codes[:task], test_labels[:task]), 1):
        a = overlap(codes, wt, k)
        s = codes.toarray().astype(np.float64)
        name = "current" if source == task else "old"
        for bucket in ("all", name):
            groups[bucket].add(a, labels)
            sparse_groups[bucket].add(s, labels)
        logits = ridge.logits(a)
        correct = int(np.count_nonzero(logits.argmax(axis=1) == labels))
        totals[name][0] += correct
        totals[name][1] += len(labels)
        totals["all"][0] += correct
        totals["all"][1] += len(labels)
        if source == task:
            five = logits[:, (task - 1) * 5 : task * 5].argmax(axis=1) + (task - 1) * 5
            totals["five"] = [int(np.count_nonzero(five == labels)), len(labels)]
    result = {
        "cumulative_accuracy": totals["all"][0] / totals["all"][1],
        "old_accuracy": totals["old"][0] / totals["old"][1] if task > 1 else float("nan"),
        "current_accuracy": totals["current"][0] / totals["current"][1],
        "current_5way_accuracy": totals["five"][0] / totals["five"][1],
    }
    for name in groups:
        for prefix, moments in (("a", groups[name]), ("sparse", sparse_groups[name])):
            same, diff, margin = moments.values()
            result[f"{name}_{prefix}_same_cos"] = same
            result[f"{name}_{prefix}_diff_cos"] = diff
            result[f"{name}_{prefix}_margin"] = margin
    return result


def trace_summary(traces: FastTraces, last_seen: np.ndarray, sample_index: int,
                  seen_count: int) -> dict:
    values = traces.values[:seen_count]
    # Task-end reporting is cheap enough to use the actual vectors. Cached
    # norm_sq can stick at a false ~1e-162 floor after deep underflow.
    norms = np.linalg.norm(values, axis=1)
    ages = sample_index - last_seen[:seen_count]
    old = slice(0, max(0, seen_count - 5))
    current = slice(max(0, seen_count - 5), seen_count)
    result = {}
    for name, selector in (("all", slice(None)), ("old", old), ("current", current)):
        n = norms[selector]
        v = values[selector]
        age = ages[selector]
        result[f"trace_{name}_norm_mean"] = float(np.mean(n)) if n.size else float("nan")
        result[f"trace_{name}_norm_median"] = float(np.median(n)) if n.size else float("nan")
        result[f"trace_{name}_element_density_eps1e_12"] = float(np.mean(v > 1e-12)) if v.size else float("nan")
        result[f"trace_{name}_age_mean_samples"] = float(np.mean(age)) if age.size else float("nan")
    return result


def run_family(config: BTSPCAMConfig, family: str, current_decay: float,
               other_decay: float, tasks: int, precomputed, output: Path,
               recompute: bool = False) -> None:
    previous = read_rows(output)
    if not recompute and len([r for r in previous if r["family"] == family and int(r["k"]) == config.k]) == tasks:
        print(json.dumps({"family": family, "k": config.k, "skipped": "complete"}), flush=True)
        return
    previous = [r for r in previous if not (r["family"] == family and int(r["k"]) == config.k)]
    model = BTSPCAM(config, class_order=pilot_class_order(config.num_classes, config.seed))
    model.ridge = None
    model.traces = StableFastTraces(config.num_classes, config.code_dim, config.eta,
                              current_decay, other_decay)
    aligned = CoordinateAlignedRidge(config.code_dim, config.memory_slots, config.k, config.gamma)
    train, labels, test, test_labels = precomputed
    last_seen = np.full(config.num_classes, -1, dtype=np.int64)
    rows = []
    running_accuracy = 0.0
    sample_index = 0
    for task in range(1, tasks + 1):
        started = time.perf_counter()
        logs = []
        for code, y in zip(train[task - 1], labels[task - 1]):
            rec = model._learn_code(code, int(y))
            logs.append(rec)
            last_seen[int(y)] = sample_index
            sample_index += 1
        aligned.update(train[task - 1], labels[task - 1])
        wb = model.memory.weights
        if family == "L_FIXED_01":
            with np.load(ROOT / "diagnostics" / f"tau_oracle_k{config.k}" /
                         f"WB_task{task:02d}.npz", allow_pickle=False) as check:
                np.testing.assert_array_equal(wb, check["WB"])
        a, b = aligned_statistics_blas(aligned, wb)
        ridge = StreamingRidge(config.memory_slots, b.shape[1], config.gamma)
        ridge.a, ridge.b = a, b
        ridge.solve()
        ridge._last_cholesky = None
        metrics = test_geometry(test, test_labels, wb, config.k, task, ridge)
        if family == "L_FIXED_01":
            reference = next(r for r in read_rows(ROOT / "diagnostics" /
                "k_path_decomposition.csv") if int(r["k"]) == config.k and
                r["branch"] == "evolving_aligned" and int(r["task"]) == task)
            if abs(metrics["cumulative_accuracy"] - float(reference["cumulative_accuracy"])) > 1e-12:
                raise AssertionError(f"k128 baseline accuracy mismatch at task {task}")
        running_accuracy += metrics["cumulative_accuracy"]
        f = np.array([r["familiarity"] for r in logs], dtype=np.float64)
        n = np.array([r["novelty"] for r in logs], dtype=np.float64)
        interference = np.array([r["interference"] for r in logs], dtype=np.float64)
        gfn = 4.0 * f * n * (1.0 - n)
        plateau = np.array([r["plateau_count"] for r in logs], dtype=np.float64)
        flips = np.array([r["flips"] for r in logs], dtype=np.float64)
        seen_count = task * 5
        row = {
            "family": family, "k": config.k, "task": task,
            "lambda_current": current_decay, "lambda_other": other_decay,
            "readout": "coordinate_aligned_tau0", "status": "DIAGNOSTIC_COUNTERFACTUAL",
            **metrics, "average_cumulative_to_task": running_accuracy / task,
            **trace_summary(model.traces, last_seen, sample_index - 1, seen_count),
            "F_mean": float(f.mean()), "F_std": float(f.std()),
            "N_mean": float(n.mean()), "N_std": float(n.std()),
            "I_mean": float(interference.mean()), "I_std": float(interference.std()),
            "G_FN_mean": float(gfn.mean()), "G_FN_std": float(gfn.std()),
            "f_eff_mean": float(np.mean([r["plateau_probability"] for r in logs])),
            "plateau_per_sample_mean": float(plateau.mean()),
            "plateau_count_task": int(plateau.sum()),
            "gross_flips_task": int(flips.sum()),
            "WB_density": float(wb.mean()),
            "runtime_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        write_rows(output, previous + rows)
        print(json.dumps({"family": family, "k": config.k, "task": task,
                          "cumulative": metrics["cumulative_accuracy"],
                          "seconds": round(row["runtime_seconds"], 2)}), flush=True)
        del a, b, ridge, logs
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, choices=(10, 20), required=True)
    parser.add_argument("--k", type=int, choices=(128, 160), default=128)
    parser.add_argument("--family", action="append")
    parser.add_argument("--recompute", action="store_true",
                        help="Replace only selected experimental family rows after a failed numerical audit")
    args = parser.parse_args()
    exp = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base_path = (CONFIG.parent / exp["base_config"]).resolve()
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(base_path.read_text(encoding="utf-8")))
    if (base.seed, base.code_dim, base.memory_slots, base.gamma, base.trace_decay) != (1993, 8192, 4096, 0.1, 0.01):
        raise AssertionError("formal baseline configuration changed")
    cfg = replace(base, k=args.k, tau=0.0)
    families = exp["families"]
    chosen = list(families) if not args.family else args.family
    if any(name not in families for name in chosen):
        raise ValueError("unknown family")
    if args.tasks == 20 and any(name not in exp["full20_families"] for name in chosen):
        raise ValueError("full20 family not predeclared")
    output = OUT10 if args.tasks == 10 else OUT20
    seed_model = BTSPCAM(cfg, class_order=pilot_class_order(cfg.num_classes, cfg.seed))
    precomputed = preencode(cfg, seed_model, args.tasks)
    del seed_model
    for name in chosen:
        decays = families[name]
        run_family(cfg, name, float(decays["current"]), float(decays["other"]),
                   args.tasks, precomputed, output, args.recompute)


if __name__ == "__main__":
    main()
