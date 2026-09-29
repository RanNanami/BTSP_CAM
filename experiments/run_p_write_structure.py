"""Independent p audit. Formal config, model implementation and checkpoints stay untouched."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.binary_memory import BinaryMemory
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import pilot_class_order
from btsp_cam.ridge import StreamingRidge
from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_lambda_state_dependence import StableFastTraces, preencode, read_rows, test_geometry, write_rows
from run_unresolved_parameter_screen import aligned_statistics_blas, overlap

EXPERIMENT = ROOT / "experiments" / "configs" / "p_write_structure.yaml"
FIXED10 = ROOT / "diagnostics" / "p_fixed_fbase_task10.csv"
EQUAL10 = ROOT / "diagnostics" / "p_equal_budget_task10.csv"
FULL20 = ROOT / "diagnostics" / "p_full20.csv"


class FunctionalTracingMemory(BinaryMemory):
    """Original XOR update with read-only event attribution to fixed Task1 queries."""

    def __init__(self, slots: int, code_dim: int):
        super().__init__(slots, code_dim)
        self.probe_bit_counts = None
        self.probe_samples = 0
        self.gross_probe_movement = 0.0
        self.ltp_probe_movement = 0.0
        self.ltd_probe_movement = 0.0

    def reset_task_attribution(self) -> None:
        self.gross_probe_movement = 0.0
        self.ltp_probe_movement = 0.0
        self.ltd_probe_movement = 0.0

    def update(self, code, plateaus, flip_probability, rng):
        if self.probe_bit_counts is None:
            return super().update(code, plateaus, flip_probability, rng)
        selected = np.flatnonzero(plateaus)
        active = np.flatnonzero(code)
        if selected.size == 0 or active.size == 0:
            return super().update(code, plateaus, flip_probability, rng)
        target = np.ix_(selected, active)
        before = self.weights[target].copy()
        count = super().update(code, plateaus, flip_probability, rng)
        changed = before != self.weights[target]
        weights = self.probe_bit_counts[active][None, :]
        scale = float(self.probe_samples * len(active) * self.slots)
        self.gross_probe_movement += float(np.sum(changed * weights) / scale)
        self.ltp_probe_movement += float(np.sum(((before == 0) & changed) * weights) / scale)
        self.ltd_probe_movement += float(np.sum(((before == 1) & changed) * weights) / scale)
        return count


def occupancy(weights: np.ndarray) -> dict:
    row = weights.mean(axis=1, dtype=np.float64)
    col = weights.mean(axis=0, dtype=np.float64)
    degrees = weights.sum(axis=0, dtype=np.int64)
    top_n = max(1, int(np.ceil(0.01 * degrees.size)))
    total = int(degrees.sum())
    out = {"WB_density": float(weights.mean())}
    for name, values in (("row_occupancy", row), ("column_occupancy", col)):
        out.update({f"{name}_mean": float(values.mean()),
                    f"{name}_median": float(np.median(values)),
                    f"{name}_p90": float(np.quantile(values, 0.9)),
                    f"{name}_p99": float(np.quantile(values, 0.99)),
                    f"{name}_max": float(values.max())})
    out["top1pct_column_occupancy_share"] = (
        float(np.sort(degrees)[-top_n:].sum() / total) if total else float("nan"))
    return out


def check_baseline(k: int, task: int, wb: np.ndarray, accuracy: float) -> None:
    if k != 128:
        return
    with np.load(ROOT / "diagnostics" / "tau_oracle_k128" /
                 f"WB_task{task:02d}.npz", allow_pickle=False) as check:
        np.testing.assert_array_equal(wb, check["WB"])
    source = read_rows(ROOT / "diagnostics" / "k_path_decomposition.csv")
    reference = next(r for r in source if int(r["k"]) == k and
                     r["branch"] == "evolving_aligned" and int(r["task"]) == task)
    if abs(accuracy - float(reference["cumulative_accuracy"])) > 1e-12:
        raise AssertionError(f"baseline accuracy mismatch task {task}")


def run_one(base: BTSPCAMConfig, k: int, p: float, mode: str, tasks: int,
            precomputed, output: Path) -> None:
    previous = read_rows(output)
    own = [r for r in previous if int(r["k"]) == k and float(r["p"]) == p and r["mode"] == mode]
    if len(own) == tasks:
        print(json.dumps({"k": k, "p": p, "mode": mode, "skipped": "complete"}), flush=True)
        return
    previous = [r for r in previous if not (int(r["k"]) == k and
                float(r["p"]) == p and r["mode"] == mode)]
    f_base = base.base_plateau_probability * (0.5 / p if mode == "equal_budget" else 1.0)
    cfg = replace(base, k=k, tau=0.0, flip_probability=p,
                  base_plateau_probability=f_base)
    model = BTSPCAM(cfg, class_order=pilot_class_order(cfg.num_classes, cfg.seed))
    model.ridge = None
    model.traces = StableFastTraces(cfg.num_classes, cfg.code_dim, cfg.eta,
                                   cfg.trace_decay, cfg.trace_decay)
    model.memory = FunctionalTracingMemory(cfg.memory_slots, cfg.code_dim)
    aligned = CoordinateAlignedRidge(cfg.code_dim, cfg.memory_slots, cfg.k, cfg.gamma)
    train, labels, test, test_labels = precomputed
    model.memory.probe_bit_counts = np.asarray(test[0].sum(axis=0)).ravel().astype(np.int64)
    model.memory.probe_samples = test[0].shape[0]
    rows = []
    previous_wb = None
    for task in range(1, tasks + 1):
        started = time.perf_counter()
        model.memory.reset_task_attribution()
        logs = [model._learn_code(code, int(y))
                for code, y in zip(train[task - 1], labels[task - 1])]
        aligned.update(train[task - 1], labels[task - 1])
        wb = model.memory.weights
        a, b = aligned_statistics_blas(aligned, wb)
        ridge = StreamingRidge(cfg.memory_slots, b.shape[1], cfg.gamma)
        ridge.a, ridge.b = a, b
        ridge.solve()
        ridge._last_cholesky = None
        metrics = test_geometry(test, test_labels, wb, k, task, ridge)
        if p == 0.5 and f_base == base.base_plateau_probability:
            check_baseline(k, task, wb, metrics["cumulative_accuracy"])
        plateaus = int(sum(x["plateau_count"] for x in logs))
        flips_01 = int(sum(x["zero_to_one_flips"] for x in logs))
        flips_10 = int(sum(x["one_to_zero_flips"] for x in logs))
        gross = flips_01 + flips_10
        if gross != int(sum(x["flips"] for x in logs)):
            raise AssertionError("gross flip accounting failed")
        gate = {}
        for field, key in (("F", "familiarity"), ("N", "novelty"),
                           ("I", "interference"), ("f_eff", "plateau_probability")):
            values = np.array([x[key] for x in logs], dtype=np.float64)
            gate[f"{field}_mean"] = float(values.mean())
            gate[f"{field}_std"] = float(values.std())
        if previous_wb is None:
            probe_drift = probe_l2 = net_changed = float("nan")
        else:
            probe = test[0]
            current_a = overlap(probe, np.ascontiguousarray(wb.T, dtype=np.float32), k)
            prior_a = overlap(probe, np.ascontiguousarray(previous_wb.T, dtype=np.float32), k)
            delta = current_a - prior_a
            probe_drift = float(np.mean(np.abs(delta)))
            probe_l2 = float(np.mean(np.linalg.norm(delta, axis=1)))
            net_changed = float(np.count_nonzero(wb != previous_wb))
        trace = model.memory
        row = {
            "status": "DIAGNOSTIC_COUNTERFACTUAL", "mode": mode, "k": k,
            "p": p, "f_base": f_base, "f_base_times_p": f_base * p,
            "task": task, "readout": "coordinate_aligned_tau0",
            **metrics, **occupancy(wb), **gate,
            "plateau_count": plateaus, "gross_flips": gross,
            "flips_01": flips_01, "flips_10": flips_10,
            "net_added_bits": flips_01 - flips_10,
            "destructive_ratio": flips_10 / gross if gross else float("nan"),
            "mean_flips_per_plateau": gross / plateaus if plateaus else float("nan"),
            "task1_probe_gross_functional_movement": trace.gross_probe_movement,
            "task1_probe_ltp_functional_movement": trace.ltp_probe_movement,
            "task1_probe_ltd_functional_movement": trace.ltd_probe_movement,
            "task1_probe_net_mean_abs_delta_a": probe_drift,
            "task1_probe_net_mean_l2_delta_a": probe_l2,
            "net_synapses_changed_since_previous_task": net_changed,
            "runtime_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        write_rows(output, previous + rows)
        print(json.dumps({"mode": mode, "k": k, "p": p, "task": task,
                          "acc": metrics["cumulative_accuracy"],
                          "density": row["WB_density"],
                          "seconds": round(row["runtime_seconds"], 2)}), flush=True)
        previous_wb = wb.copy()
        del a, b, ridge, logs
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, choices=(10, 20), required=True)
    parser.add_argument("--k", type=int, choices=(128, 160), default=128)
    parser.add_argument("--mode", choices=("fixed_fbase", "equal_budget"))
    parser.add_argument("--p", type=float)
    args = parser.parse_args()
    exp = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    source = (EXPERIMENT.parent / exp["base_config"]).resolve()
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(source.read_text(encoding="utf-8")))
    if (base.seed, base.k, base.code_dim, base.memory_slots, base.flip_probability,
        base.base_plateau_probability, base.trace_decay, base.gamma) != (
            1993, 41, 8192, 4096, 0.5, 0.005, 0.01, 0.1):
        raise AssertionError("formal baseline changed; stop rather than silently adapt")
    if args.tasks == 20:
        if args.p is None or args.mode is None or args.p not in (0.5, exp["full20_contrast_p"]):
            raise ValueError("full20 accepts only predeclared p=0.5 and contrast p")
        if args.p == exp["full20_contrast_p"] and args.mode != exp["full20_contrast_mode"]:
            raise ValueError("full20 contrast must use predeclared equal-budget mode")
        choices = [(args.mode, args.p, FULL20)]
    elif args.p is not None or args.mode is not None:
        if args.p is None or args.mode is None or args.p not in exp["task10_p"]:
            raise ValueError("task10 p/mode must be a predeclared pair")
        choices = [(args.mode, args.p, FIXED10 if args.mode == "fixed_fbase" else EQUAL10)]
    else:
        choices = [("fixed_fbase", p, FIXED10) for p in (0.5, 0.1, 0.25, 0.75, 1.0)]
        choices += [("equal_budget", p, EQUAL10) for p in (0.1, 0.25, 0.5, 0.75, 1.0)]
    cfg = replace(base, k=args.k, tau=0.0)
    encoder = BTSPCAM(cfg, class_order=pilot_class_order(cfg.num_classes, cfg.seed))
    precomputed = preencode(cfg, encoder, args.tasks)
    del encoder
    for mode, p, output in choices:
        run_one(base, args.k, float(p), mode, args.tasks, precomputed, output)


if __name__ == "__main__":
    main()
