"""Isolated Task1-5 f_base/beta sensitivity using existing BTSP writer."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from btsp_cam.protocol import pilot_class_order
from btsp_cam.rng import RNGProvenance
from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_high_k_mechanism_decomposition import eval_memory, load_codes, load_wb
from run_k_extended_search import class_overlap_geometry
from run_k_robustness_and_selection import solve_aligned
from run_unresolved_parameter_screen import read_rows, write_rows

SPEC_PATH = ROOT / "experiments/configs/fbase_beta_sensitivity.yaml"


def setup() -> tuple[dict, BTSPCAMConfig, Path, Path, dict[str, Path]]:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    fixed = (spec["task_count"], spec["class_order_seed"], spec["rf_seed"],
             spec["writer_seed"], spec["k"], spec["code_dim"],
             spec["memory_slots"], spec["p"], spec["lambda"],
             spec["gamma"], spec["tau"], spec["baseline_f_base"],
             spec["baseline_beta"])
    if fixed != (5, 1993, 1993, 1993, 96, 8192, 4096, 0.5, 0.01,
                 0.1, 0.0, 0.005, 1.0):
        raise ValueError("locked local protocol changed")
    if (spec["f_base_grid"] != [0.00125, 0.0025, 0.005, 0.01, 0.02]
            or spec["beta_grid"] != [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
            or spec["grid_f_base"] != [0.00125, 0.005, 0.02]
            or spec["grid_beta"] != [0.0, 1.0, 4.0]):
        raise ValueError("predeclared grid changed")
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(
        (SPEC_PATH.parent / spec["base_config"]).resolve().read_text(encoding="utf-8")))
    actual = (base.seed, base.code_dim, base.memory_slots, base.k,
              base.flip_probability, base.trace_decay, base.gamma,
              base.base_plateau_probability, base.beta)
    if actual != (1993, 8192, 4096, 41, 0.5, 0.01, 0.1, 0.005, 1.0):
        raise ValueError("formal config changed")
    cache = (SPEC_PATH.parent / spec["cache_dir"]).resolve()
    archive = (SPEC_PATH.parent / spec["archive_csv"]).resolve()
    paths = {name: (SPEC_PATH.parent / path).resolve()
             for name, path in spec["outputs"].items()}
    for task in range(1, 6):
        for split in ("train", "test"):
            if not (cache / f"{split}_task{task:02d}.npz").exists():
                raise FileNotFoundError(f"missing k=96 sparse code cache {split} {task}")
    if not archive.exists():
        raise FileNotFoundError("missing archived k=96 trajectory")
    return spec, base, cache, archive, paths


def make_model(base: BTSPCAMConfig, f_base: float, beta: float) -> BTSPCAM:
    cfg = replace(base, k=96, tau=0.0,
                  base_plateau_probability=f_base, beta=beta)
    rf = RNGProvenance.from_seed(1993)
    wr = RNGProvenance.from_seed(1993)
    model = BTSPCAM(cfg, class_order=pilot_class_order(100, 1993),
                    projection_seed=rf.projection_seed,
                    plateau_seed=wr.plateau_seed,
                    synapse_seed=wr.synapse_seed)
    model.ridge = None
    return model


def condition_rows(path: Path, f_base: float, beta: float) -> list[dict]:
    return ([r for r in read_rows(path)
             if float(r["f_base"]) == f_base and float(r["beta"]) == beta]
            if path.exists() else [])


def replace_condition(path: Path, f_base: float, beta: float,
                      new_rows: list[dict]) -> None:
    previous = read_rows(path) if path.exists() else []
    keep = [r for r in previous
            if not (float(r["f_base"]) == f_base and float(r["beta"]) == beta)]
    combined = keep + new_rows
    combined.sort(key=lambda r: (float(r["f_base"]), float(r["beta"]), int(r["task"])))
    write_rows(path, combined)


def archived_baseline_check(archive: Path, cache: Path, task: int,
                            row: dict, memory: np.ndarray) -> None:
    found = [r for r in read_rows(archive)
             if r["run"] == "k96" and r["branch"] == "evolving_aligned"
             and int(r["task"]) == task]
    if len(found) != 1:
        raise AssertionError(f"missing archived k=96 task {task}")
    ref = found[0]
    for field in ("cumulative_accuracy", "old_accuracy", "current_accuracy",
                  "current_5way_accuracy", "WB_density", "plateau_count_task",
                  "gross_flips_task"):
        got, expected = float(row[field]), float(ref[field])
        if not (np.isnan(got) and np.isnan(expected)) and abs(got - expected) > 1e-12:
            raise AssertionError(f"baseline mismatch task={task} {field}: {got} != {expected}")
    checkpoint = cache / f"WB_task{task:02d}.npz"
    if checkpoint.exists() and not np.array_equal(memory, load_wb(checkpoint)):
        raise AssertionError(f"baseline WB differs bitwise at task {task}")


def task_dynamics(logs: list[dict], slots: int, f_base: float) -> dict:
    def arr(name: str) -> np.ndarray:
        return np.fromiter((float(entry[name]) for entry in logs),
                           dtype=np.float64, count=len(logs))
    f, n, interference = arr("familiarity"), arr("novelty"), arr("interference")
    effective, collision = arr("plateau_probability"), arr("max_rival_similarity")
    plateaus = arr("plateau_count")
    flips01, flips10 = arr("zero_to_one_flips"), arr("one_to_zero_flips")
    gfn = 4.0 * n * (1.0 - n) * f
    if not np.allclose(effective, f_base * gfn * interference, atol=1e-12):
        raise AssertionError("logged gate components do not reproduce f_eff")
    order = np.argsort(collision, kind="stable")
    quartile = np.empty(len(logs), dtype=np.uint8)
    for q, indices in enumerate(np.array_split(order, 4), 1):
        quartile[indices] = q
    gross = flips01.sum() + flips10.sum()
    output = dict(mean_F=float(f.mean()), mean_N=float(n.mean()),
                  mean_I=float(interference.mean()), mean_G_FN=float(gfn.mean()),
                  mean_f_eff=float(effective.mean()),
                  median_f_eff=float(np.median(effective)),
                  p90_f_eff=float(np.percentile(effective, 90)),
                  p99_f_eff=float(np.percentile(effective, 99)),
                  rival_collision_mean=float(collision.mean()),
                  rival_collision_median=float(np.median(collision)),
                  rival_collision_p75=float(np.percentile(collision, 75)),
                  rival_collision_p90=float(np.percentile(collision, 90)),
                  rival_collision_p99=float(np.percentile(collision, 99)),
                  plateau_count_task=int(plateaus.sum()),
                  plateau_per_sample=float(plateaus.mean()),
                  gross_flips_task=int(gross),
                  gross_flips_per_sample=float(gross / len(logs)),
                  gross_flips_per_plateau=float(gross / plateaus.sum())
                  if plateaus.sum() else float("nan"),
                  flip_01_task=int(flips01.sum()), flip_10_task=int(flips10.sum()))
    rates = []
    for q in range(1, 5):
        selected = quartile == q
        rate = float(plateaus[selected].sum() / (selected.sum() * slots))
        output.update({f"q{q}_samples": int(selected.sum()),
                       f"q{q}_collision_min": float(collision[selected].min()),
                       f"q{q}_collision_max": float(collision[selected].max()),
                       f"q{q}_mean_I": float(interference[selected].mean()),
                       f"q{q}_mean_f_eff": float(effective[selected].mean()),
                       f"q{q}_plateau_rate": rate})
        rates.append(rate)
    output["q4_q1_plateau_rate_ratio"] = rates[3] / rates[0] if rates[0] else float("nan")
    return output


def run_condition(base: BTSPCAMConfig, cache: Path, archive: Path,
                  path: Path, family: str, f_base: float, beta: float) -> None:
    prior = condition_rows(path, f_base, beta)
    if len(prior) == 5 and {int(r["task"]) for r in prior} == set(range(1, 6)):
        print(json.dumps({"family": family, "f_base": f_base, "beta": beta,
                          "skipped": "complete"}), flush=True)
        return
    model = make_model(base, f_base, beta)
    aligned = CoordinateAlignedRidge(8192, 4096, 96, 0.1)
    test_codes: list[sparse.csr_matrix] = []
    test_labels: list[np.ndarray] = []
    rows: list[dict] = []
    accuracy_sum = 0.0
    baseline = f_base == 0.005 and beta == 1.0
    for task in range(1, 6):
        start = time.perf_counter()
        codes, labels = load_codes(cache / f"train_task{task:02d}.npz", 96)
        logs = [model._learn_code(code, int(label))
                for code, label in zip(codes, labels)]
        dynamics = task_dynamics(logs, model.memory.slots, f_base)
        del logs
        aligned.update(codes, labels)
        del codes, labels
        current_codes, current_labels = load_codes(
            cache / f"test_task{task:02d}.npz", 96)
        test_codes.append(sparse.csr_matrix(current_codes, dtype=np.float32))
        test_labels.append(current_labels)
        del current_codes
        ridge = solve_aligned(aligned, model.memory.weights)
        scores, a_cos = eval_memory(ridge, model.memory.weights, 96,
                                    test_codes, test_labels, task)
        accuracy_sum += scores["cumulative_accuracy"]
        if task == 5:
            s_cos = class_overlap_geometry(test_codes, test_labels, 96, 8192)[:3]
        else:
            s_cos = (float("nan"),) * 3
            a_cos = (float("nan"),) * 3
        row = dict(family=family, f_base=f_base, beta=beta, task=task,
                   source="fresh_writer_run", k=96, rf_seed=1993, writer_seed=1993,
                   class_order_seed=1993, p=0.5, trace_decay=0.01,
                   gamma=0.1, tau=0.0, train_samples=2500,
                   **scores, average_incremental_to_task=accuracy_sum / task,
                   **dynamics, WB_density=float(model.memory.weights.mean()),
                   sparse_same_cos=s_cos[0], sparse_diff_cos=s_cos[1],
                   sparse_margin=s_cos[2], a_same_cos=a_cos[0],
                   a_diff_cos=a_cos[1], a_margin=a_cos[2],
                   sparse_geometry_scope="all_seen_test",
                   a_geometry_scope="current_task_first_20_test_per_class",
                   seconds=time.perf_counter() - start)
        if baseline:
            archived_baseline_check(archive, cache, task, row, model.memory.weights)
        rows.append(row)
        replace_condition(path, f_base, beta, rows)
        print(json.dumps({"family": family, "f_base": f_base, "beta": beta,
                          "task": task, "cumulative": scores["cumulative_accuracy"],
                          "WB_density": row["WB_density"],
                          "gross_flips": row["gross_flips_task"],
                          "seconds": round(row["seconds"], 1)}), flush=True)
        del ridge


def copy_condition(source: Path, target: Path, family: str,
                   f_base: float, beta: float) -> bool:
    if len(condition_rows(target, f_base, beta)) == 5:
        return True
    old = sorted(condition_rows(source, f_base, beta), key=lambda r: int(r["task"]))
    if len(old) != 5:
        return False
    rows = [dict(r, family=family, source=f"reused_from_{source.name}") for r in old]
    replace_condition(target, f_base, beta, rows)
    return True


def sweep(spec: dict, base: BTSPCAMConfig, cache: Path, archive: Path,
          paths: dict[str, Path], family: str) -> None:
    if family == "fbase":
        for f_base in [0.005] + [value for value in spec["f_base_grid"]
                                  if value != 0.005]:
            run_condition(base, cache, archive, paths["fbase"],
                          "f_base_sweep", float(f_base), 1.0)
    elif family == "beta":
        for beta in spec["beta_grid"]:
            beta = float(beta)
            if beta == 1.0 and copy_condition(paths["fbase"], paths["beta"],
                                              "beta_sweep", 0.005, 1.0):
                continue
            run_condition(base, cache, archive, paths["beta"],
                          "beta_sweep", 0.005, beta)
    else:
        raise ValueError(family)


def relative_change(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return float("nan")
    if reference == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return abs(value / reference - 1.0)


def grid_triggers(spec: dict, paths: dict[str, Path]) -> dict[str, float]:
    f_rows = [r for r in read_rows(paths["fbase"]) if int(r["task"]) == 5]
    b_rows = [r for r in read_rows(paths["beta"]) if int(r["task"]) == 5]
    if (len(f_rows) != len(spec["f_base_grid"])
            or len(b_rows) != len(spec["beta_grid"])):
        raise ValueError("complete both one-factor sweeps before grid decision")
    f0 = next(r for r in f_rows if float(r["f_base"]) == 0.005)
    b0 = next(r for r in b_rows if float(r["beta"]) == 1.0)
    fields = ("cumulative_accuracy", "WB_density", "gross_flips_per_sample",
              "q4_q1_plateau_rate_ratio", "a_margin")
    if any(abs(float(f0[name]) - float(b0[name])) > 1e-12 for name in fields):
        raise AssertionError("baseline control differs across CSVs")
    return dict(
        fbase_task5_accuracy_pp=max(abs(float(r["cumulative_accuracy"]) -
                                        float(f0["cumulative_accuracy"])) * 100 for r in f_rows),
        fbase_task5_density_pp=max(abs(float(r["WB_density"]) -
                                       float(f0["WB_density"])) * 100 for r in f_rows),
        fbase_gross_flips_relative=max(relative_change(
            float(r["gross_flips_per_sample"]), float(f0["gross_flips_per_sample"]))
                                         for r in f_rows),
        beta_task5_accuracy_pp=max(abs(float(r["cumulative_accuracy"]) -
                                       float(b0["cumulative_accuracy"])) * 100 for r in b_rows),
        beta_q4_q1_relative=max(relative_change(
            float(r["q4_q1_plateau_rate_ratio"]),
            float(b0["q4_q1_plateau_rate_ratio"])) for r in b_rows),
        beta_a_margin_relative=max(relative_change(
            float(r["a_margin"]), float(b0["a_margin"])) for r in b_rows))


def grid_is_triggered(spec: dict, measures: dict[str, float]) -> bool:
    gate = spec["grid_trigger"]
    return (measures["fbase_task5_accuracy_pp"] >= gate["task5_accuracy_pp"]
            or measures["fbase_task5_density_pp"] >= gate["task5_wb_density_pp"]
            or measures["fbase_gross_flips_relative"] >= gate["gross_flips_per_sample_relative"]
            or measures["beta_task5_accuracy_pp"] >= gate["task5_accuracy_pp"]
            or measures["beta_q4_q1_relative"] >= gate["q4_q1_plateau_ratio_relative"]
            or measures["beta_a_margin_relative"] >= gate["task5_a_margin_relative"])


def grid(spec: dict, base: BTSPCAMConfig, cache: Path, archive: Path,
         paths: dict[str, Path]) -> None:
    measures = grid_triggers(spec, paths)
    triggered = grid_is_triggered(spec, measures)
    print(json.dumps({"grid_triggered": triggered, **measures}), flush=True)
    if not triggered:
        return
    for f_base in spec["grid_f_base"]:
        for beta in spec["grid_beta"]:
            f_base, beta = float(f_base), float(beta)
            if beta == 1.0 and copy_condition(paths["fbase"], paths["grid"],
                                              "3x3_mechanism_grid", f_base, beta):
                continue
            if f_base == 0.005 and copy_condition(paths["beta"], paths["grid"],
                                                  "3x3_mechanism_grid", f_base, beta):
                continue
            run_condition(base, cache, archive, paths["grid"],
                          "3x3_mechanism_grid", f_base, beta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fbase", "beta", "grid", "all"),
                        default="all")
    args = parser.parse_args()
    spec, base, cache, archive, paths = setup()
    for phase in (("fbase", "beta", "grid") if args.phase == "all"
                  else (args.phase,)):
        if phase == "grid":
            grid(spec, base, cache, archive, paths)
        else:
            sweep(spec, base, cache, archive, paths, phase)


if __name__ == "__main__":
    main()
