"""Isolated CIFAR-100 k=1024 paper-style component ablations.

Full uses validated existing high-k W_B checkpoints. A1/A2/A3 write fresh,
independent memories. All readouts are current-W_B historical TRAIN replay.
Never edits formal source, configs, caches, or archived result files.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.plateau import plateau_probability, sample_slot_plateaus
from btsp_cam.ridge import StreamingRidge
from run_coordinate_aligned_full20 import CosineMoments
from run_high_k_mechanism_decomposition import (
    code_path, load_codes, load_wb, model_for_k, prepare_memory, wb_path,
)

K, M, SLOTS, Q, GAMMA, TASKS = 1024, 8192, 4096, 4, .1, 20
TAU = Q / K
FULL = "K1024_FULL"
A1 = "A1_TRACE_FIXED_FBASE"
A2 = "A2_NO_COLLISION_I1"
A3 = "A3_ONLY_LTP"
A4A = "A4A_NO_THRESHOLD"
A4B = "A4B_RAW_WBS_SCALE_CONFOUNDED"
WRITER_BRANCHES = (A1, A2, A3)
BRANCHES = (FULL, A1, A2, A3, A4A, A4B)
DIAG = ROOT / "diagnostics"
CACHE = DIAG / "high_k_mechanism_cache"
NEW_CACHE = DIAG / "cifar_k1024_ablation_cache"
ARCHIVE = DIAG / "high_k_stress_full20.csv"
OLD_ABLATION = DIAG / "paper_style_ablation_full20.csv"
FULL20 = DIAG / "cifar_k1024_ablation_full20.csv"
MECHANISMS = DIAG / "cifar_k1024_ablation_mechanisms.csv"
DISTANCE = DIAG / "cifar_k1024_ablation_writer_distance.csv"
COMPARE = DIAG / "cifar_k96_vs_k1024_ablation_compare.csv"
SUMMARY = DIAG / "cifar_k1024_ablation_summary.json"
VERIFICATION = DIAG / "cifar_k1024_full_writer_verification.json"


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def setup():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from project root: {ROOT}")
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(
        (ROOT / "configs/btsp_cam_cifar100.yaml").read_text(encoding="utf-8")))
    assert (base.seed, base.code_dim, base.memory_slots, base.base_plateau_probability,
            base.beta, base.trace_decay, base.eta, base.flip_probability, base.gamma) == (
            1993, M, SLOTS, .005, 1., .01, 1., .5, GAMMA)
    assert TAU == .00390625
    source = CACHE / f"k{K}"
    for task in range(1, TASKS + 1):
        assert all(code_path(CACHE, K, split, task).exists()
                   for split in ("train", "test"))
        assert wb_path(CACHE, K, task).exists()
    return base


def archive_full() -> dict[int, dict]:
    rows = [r for r in csv_rows(ARCHIVE) if int(r["k"]) == K
            and r["condition"] == "FIXED_FBASE"]
    result = {int(r["task"]): r for r in rows}
    assert len(rows) == len(result) == TASKS
    return result


def cache_pair(split: str, task: int) -> tuple[np.ndarray, np.ndarray]:
    path = code_path(CACHE, K, split, task)
    s, y = load_codes(path, K)
    assert s.shape == (2500 if split == "train" else 500, M)
    assert np.unique(y).tolist() == list(range((task - 1) * 5, task * 5))
    return s, y


def verify_full_writer(base) -> None:
    if VERIFICATION.exists():
        raise FileExistsError(VERIFICATION)
    # prepare_memory replays the original writer on existing train codes and
    # checks archived gross flips, occupancy, and every existing WB bit.
    assert all(wb_path(CACHE, K, t).exists() for t in range(1, TASKS + 1))
    prepare_memory(base, CACHE, K)
    archived = archive_full()
    checks = []
    for task in range(1, TASKS + 1):
        wb = load_wb(wb_path(CACHE, K, task))
        row = archived[task]
        density = float(wb.mean())
        assert abs(density - float(row["WB_density"])) < 1e-12
        checks.append(dict(task=task, WB_sha256=sha(wb), WB_density=density,
                           archived_gross_flips=int(float(row["gross_flips"])),
                           archived_plateau_count=int(float(row["plateau_events"])),
                           checkpoint_bitwise_writer_replay_equal=True))
    VERIFICATION.write_text(json.dumps(dict(status="PASS", k=K,
                                            existing_checkpoint_count=TASKS,
                                            checks=checks), indent=2), encoding="utf-8")


def writer_step(model, code: np.ndarray, label: int, branch: str) -> dict:
    """Same draw order across branches; A3 restores sampled 1->0 after XOR."""
    model.traces.update(code, int(label))
    signals = model.traces.signals(code, int(label), model.config.beta)
    if branch == A1:
        p_eff = model.config.base_plateau_probability
        interference = signals.interference
    elif branch == A2:
        p_eff = plateau_probability(model.config.base_plateau_probability,
                                    replace(signals, interference=1.0))
        interference = 1.0
    elif branch == A3:
        p_eff = plateau_probability(model.config.base_plateau_probability, signals)
        interference = signals.interference
    else:
        raise ValueError(branch)
    plateaus = sample_slot_plateaus(p_eff, model.memory.slots, model.plateau_rng)
    if branch == A3:
        selected, active = np.flatnonzero(plateaus), np.flatnonzero(code)
        index = np.ix_(selected, active) if selected.size and active.size else None
        before = model.memory.weights[index].copy() if index is not None else None
    attempts = model.memory.update(code, plateaus, model.config.flip_probability,
                                   model.synapse_rng)
    sampled_ltd = int(model.memory.last_one_to_zero)
    if branch == A3:
        if index is not None:
            after = model.memory.weights[index]
            after[before == 1] = 1
            model.memory.weights[index] = after
        model.memory.last_one_to_zero = 0
    actual_ltp = int(model.memory.last_zero_to_one)
    actual_ltd = int(model.memory.last_one_to_zero)
    assert attempts == actual_ltp + sampled_ltd
    return dict(f_eff=float(p_eff), I_effective=float(interference),
                plateau_count=int(plateaus.sum()), LTP_count=actual_ltp,
                LTD_count=actual_ltd, sampled_flip_attempts=int(attempts),
                sampled_LTD_attempts=sampled_ltd)


def save_new_wb(branch: str, task: int, wb: np.ndarray) -> str:
    path = NEW_CACHE / branch / f"WB_task{task:02d}.npz"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert wb.shape == (SLOTS, M) and wb.dtype == np.uint8
    digest = sha(wb)
    with path.open("xb") as stream:
        np.savez_compressed(stream, packed=np.packbits(wb, axis=1),
                            sha256=np.array(digest), task=np.array(task, dtype=np.int16))
    return digest


def load_new_wb(branch: str, task: int) -> np.ndarray:
    path = NEW_CACHE / branch / f"WB_task{task:02d}.npz"
    with np.load(path, allow_pickle=False) as data:
        wb = np.unpackbits(data["packed"], axis=1, count=M)
        assert int(data["task"]) == task and sha(wb) == str(data["sha256"])
    return wb


def writer_rows(base) -> None:
    if MECHANISMS.exists():
        raise FileExistsError(MECHANISMS)
    assert VERIFICATION.exists()
    archived = archive_full()
    rows = []
    for task in range(1, TASKS + 1):
        wb = load_wb(wb_path(CACHE, K, task))
        old = archived[task]
        rows.append(dict(variant=FULL, task=task, train_samples=2500,
                         plateau_count=int(float(old["plateau_events"])),
                         LTP_count=int(float(old["flip_01"])),
                         LTD_count=int(float(old["flip_10"])),
                         gross_flips=int(float(old["gross_flips"])),
                         sampled_flip_attempts=int(float(old["gross_flips"])),
                         sampled_LTD_attempts=int(float(old["flip_10"])),
                         W_B_density=float(wb.mean()), W_B_sha256=sha(wb),
                         mean_f_eff=float("nan"), mean_I_effective=float("nan"),
                         writer_seconds=0.0,
                         source="ARCHIVED_FULL_BITWISE_REPLAY_VERIFIED"))
    for branch in WRITER_BRANCHES:
        model = model_for_k(base, K)
        assert model.config.tau == 0.0  # writer ignores tau; readout uses fixed Q/K
        assert model.memory.weights.sum() == 0
        for task in range(1, TASKS + 1):
            started = time.perf_counter()
            s, y = cache_pair("train", task)
            totals = dict(plateau_count=0, LTP_count=0, LTD_count=0,
                          sampled_flip_attempts=0, sampled_LTD_attempts=0)
            eff_sum = interference_sum = 0.0
            for code, label in zip(s, y):
                log = writer_step(model, code, int(label), branch)
                for key in totals:
                    totals[key] += log[key]
                eff_sum += log["f_eff"]
                interference_sum += log["I_effective"]
                if branch == A1:
                    assert log["f_eff"] == .005
                if branch == A2:
                    assert log["I_effective"] == 1.0
                if branch == A3:
                    assert log["LTD_count"] == 0
            if branch == A3:
                assert totals["plateau_count"] == int(float(archived[task]["plateau_events"]))
                assert totals["sampled_flip_attempts"] == int(float(archived[task]["gross_flips"]))
            digest = save_new_wb(branch, task, model.memory.weights)
            rows.append(dict(variant=branch, task=task, train_samples=len(y),
                             plateau_count=totals["plateau_count"],
                             LTP_count=totals["LTP_count"], LTD_count=totals["LTD_count"],
                             gross_flips=totals["LTP_count"] + totals["LTD_count"],
                             sampled_flip_attempts=totals["sampled_flip_attempts"],
                             sampled_LTD_attempts=totals["sampled_LTD_attempts"],
                             W_B_density=float(model.memory.weights.mean()),
                             W_B_sha256=digest, mean_f_eff=eff_sum/len(y),
                             mean_I_effective=interference_sum/len(y),
                             writer_seconds=time.perf_counter()-started,
                             source="FRESH_INDEPENDENT_WRITER"))
            print(json.dumps(dict(phase="writer", variant=branch, task=task,
                                  density=rows[-1]["W_B_density"],
                                  plateau=totals["plateau_count"],
                                  LTP=totals["LTP_count"], LTD=totals["LTD_count"])), flush=True)
            del s, y
            gc.collect()
    assert len(rows) == 80
    write_csv_new(MECHANISMS, rows)


def wb_for(branch: str, task: int) -> np.ndarray:
    if branch in (FULL, A4A, A4B):
        return load_wb(wb_path(CACHE, K, task))
    return load_new_wb(branch, task)


def features(counts: np.ndarray, branch: str) -> np.ndarray:
    j = np.asarray(counts, dtype=np.float64)
    assert np.all(j == np.floor(j)) and np.all((j >= 0) & (j <= K))
    if branch == A4B:
        return j
    a = j / K
    if branch == A4A:
        return a
    return np.maximum(a - TAU, 0.0)


def fit_evaluate(branch: str, task: int, wb: np.ndarray,
                 train_cache: dict, test_cache: dict) -> dict:
    started = time.perf_counter()
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    ridge = StreamingRidge(SLOTS, task * 5, GAMMA)
    for source in range(1, task + 1):
        s, y = train_cache[source]
        assert source <= task and len(y) == 2500
        j = sparse.csr_matrix(s, dtype=np.float32) @ wt
        ridge.update(features(j, branch), y)
    ridge.solve()
    condition = ridge.condition_number
    ridge._last_cholesky = None
    cosine = CosineMoments(SLOTS)
    old_ok = old_n = current_ok = current_n = five_ok = 0
    active = zero_rows = elements = 0
    norm_sum = 0.0
    for source in range(1, task + 1):
        s, y = test_cache[source]
        assert len(y) == 500
        z = features(sparse.csr_matrix(s, dtype=np.float32) @ wt, branch)
        logits = ridge.logits(z)
        pred = logits.argmax(axis=1)
        ok = int(np.count_nonzero(pred == y))
        if source == task:
            current_ok += ok
            current_n += len(y)
            restricted = logits[:, (task - 1) * 5:task * 5].argmax(axis=1) + (task - 1) * 5
            five_ok += int(np.count_nonzero(restricted == y))
        else:
            old_ok += ok
            old_n += len(y)
        active += int(np.count_nonzero(z))
        zero_rows += int(np.count_nonzero(~np.any(z > 0, axis=1)))
        elements += z.size
        norm_sum += float(np.linalg.norm(z, axis=1).sum())
        cosine.add(z, y)
    same, different, margin = cosine.values()
    n = old_n + current_n
    result = dict(variant=branch, task=task, seen_classes=task*5,
                  cumulative_accuracy=(old_ok+current_ok)/n,
                  old_accuracy=old_ok/old_n if old_n else float("nan"),
                  current_accuracy=current_ok/current_n,
                  restricted_current_5way_accuracy=five_ok/current_n,
                  CA1_density=active/elements, zero_row_fraction=zero_rows/n,
                  mean_z_norm=norm_sum/n,
                  same_class_cosine=same, different_class_cosine=different,
                  cosine_margin=margin, ridge_condition=condition,
                  W_B_density=float(wb.mean()), W_B_sha256=sha(wb),
                  readout="CURRENT_WB_REPLAY_REFIT",
                  tau=0.0 if branch in (A4A, A4B) else TAU,
                  normalization="NONE" if branch == A4B else "DIVIDE_BY_K",
                  evaluation_seconds=time.perf_counter()-started)
    del ridge
    gc.collect()
    return result


def writer_distance() -> None:
    if DISTANCE.exists():
        raise FileExistsError(DISTANCE)
    rows = []
    for task in (1, 5, 10, 20):
        full = wb_for(FULL, task)
        full_row = full.sum(axis=1, dtype=np.int64)
        full_col = full.sum(axis=0, dtype=np.int64)
        for branch in WRITER_BRANCHES:
            other = wb_for(branch, task)
            row = other.sum(axis=1, dtype=np.int64)
            col = other.sum(axis=0, dtype=np.int64)
            diff = int(np.count_nonzero(full != other))
            rows.append(dict(task=task, variant=branch, full_WB_sha256=sha(full),
                             branch_WB_sha256=sha(other), hamming_bits=diff,
                             differing_bit_fraction=diff/full.size,
                             full_density=float(full.mean()), branch_density=float(other.mean()),
                             row_degree_abs_diff_mean=float(np.abs(row-full_row).mean()),
                             row_degree_abs_diff_p90=float(np.percentile(np.abs(row-full_row),90)),
                             row_occupancy_mean=float(row.mean()/M),
                             row_occupancy_p90=float(np.percentile(row/M,90)),
                             row_occupancy_p99=float(np.percentile(row/M,99)),
                             column_degree_abs_diff_mean=float(np.abs(col-full_col).mean()),
                             column_degree_abs_diff_p90=float(np.percentile(np.abs(col-full_col),90)),
                             column_occupancy_mean=float(col.mean()/SLOTS),
                             column_occupancy_p90=float(np.percentile(col/SLOTS,90)),
                             column_occupancy_p99=float(np.percentile(col/SLOTS,99))))
    write_csv_new(DISTANCE, rows)


def finalize(full_rows: list[dict]) -> None:
    if COMPARE.exists() or SUMMARY.exists():
        raise FileExistsError("comparison or summary already exists")
    mechanisms = csv_rows(MECHANISMS)
    by_branch = {branch: [r for r in full_rows if r["variant"] == branch]
                 for branch in BRANCHES}
    stats = {}
    for branch, rows in by_branch.items():
        assert len(rows) == TASKS
        writer_key = FULL if branch in (A4A, A4B) else branch
        writer_rows = [r for r in mechanisms if r["variant"] == writer_key]
        assert len(writer_rows) == TASKS
        stats[branch] = dict(Abar=float(np.mean([r["cumulative_accuracy"] for r in rows])),
                             Final=rows[-1]["cumulative_accuracy"],
                             Old_Task20=rows[-1]["old_accuracy"],
                             Current_Task20=rows[-1]["current_accuracy"],
                             Restricted5_Task20=rows[-1]["restricted_current_5way_accuracy"],
                             WB_density=rows[-1]["W_B_density"],
                             Plateau_per_task=sum(int(r["plateau_count"]) for r in writer_rows)/TASKS,
                             LTP_total=sum(int(r["LTP_count"]) for r in writer_rows),
                             LTD_total=sum(int(r["LTD_count"]) for r in writer_rows),
                             CA1_density_Task20=rows[-1]["CA1_density"],
                             Cosine_margin_Task20=rows[-1]["cosine_margin"])
    for branch in BRANCHES:
        stats[branch]["Delta_Abar_vs_Full_pp"] = 100*(stats[branch]["Abar"]-stats[FULL]["Abar"])
        stats[branch]["Delta_Final_vs_Full_pp"] = 100*(stats[branch]["Final"]-stats[FULL]["Final"])
    prior = csv_rows(OLD_ABLATION)
    name96 = {FULL:"FULL", A1:"A1_FIXED_GATE", A2:"A2_NO_COLLISION",
              A3:"A3_ONLY_LTP", A4A:"A4A_NO_THRESHOLD", A4B:"A4B_NO_CA1_OPERATOR"}
    old = {key:[r for r in prior if r["variant"] == value]
           for key,value in name96.items()}
    assert all(len(v)==20 for v in old.values())
    old_stats = {key:dict(Abar=float(np.mean([float(r["cumulative_accuracy"]) for r in rows])),
                          Final=float(rows[-1]["cumulative_accuracy"]))
                 for key,rows in old.items()}
    compare = []
    for branch in BRANCHES:
        delta96_avg = 100*(old_stats[branch]["Abar"]-old_stats[FULL]["Abar"])
        delta96_final = 100*(old_stats[branch]["Final"]-old_stats[FULL]["Final"])
        delta1024_avg = stats[branch]["Delta_Abar_vs_Full_pp"]
        delta1024_final = stats[branch]["Delta_Final_vs_Full_pp"]
        compare.append(dict(variant=branch, k96_Delta_Abar_pp=delta96_avg,
                            k1024_Delta_Abar_pp=delta1024_avg,
                            difference_in_Abar_ablation_effect_pp=delta1024_avg-delta96_avg,
                            k96_Delta_Final_pp=delta96_final,
                            k1024_Delta_Final_pp=delta1024_final,
                            difference_in_Final_ablation_effect_pp=delta1024_final-delta96_final))
    write_csv_new(COMPARE, compare)
    output = dict(status="PASS", protocol="CIFAR100_B0_INC5_K1024_Q4_CURRENT_WB_REPLAY_REFIT",
                  paper_exact=False, k=K, q=Q, tau=TAU, gamma=GAMMA,
                  variants=stats, k96_comparison=compare,
                  full_checkpoint_verification=str(VERIFICATION.name),
                  A4_reuses_full_WB=True, A5="NOT_REPRODUCIBLE_FROM_PUBLIC_DESCRIPTION",
                  tests_are_used_only_for_evaluation=True,
                  readout="LOCAL_EXPERIMENTAL_COORDINATE_ALIGNED_READOUT",
                  historical_train_replay=True)
    SUMMARY.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(phase="final", status="PASS", variants=stats)), flush=True)


def readouts() -> None:
    if FULL20.exists():
        raise FileExistsError(FULL20)
    assert VERIFICATION.exists() and MECHANISMS.exists()
    mechanism = {(r["variant"], int(r["task"])):r for r in csv_rows(MECHANISMS)}
    archived = archive_full()
    train_cache, test_cache = {}, {}
    for task in range(1, TASKS+1):
        train_cache[task] = cache_pair("train", task)
        test_cache[task] = cache_pair("test", task)
    rows = []
    for branch in BRANCHES:
        for task in range(1, TASKS+1):
            wb = wb_for(branch, task)
            row = fit_evaluate(branch, task, wb, train_cache, test_cache)
            writer_key = FULL if branch in (A4A, A4B) else branch
            log = mechanism[writer_key, task]
            assert row["W_B_sha256"] == log["W_B_sha256"]
            for key in ("plateau_count", "LTP_count", "LTD_count", "gross_flips",
                        "sampled_flip_attempts", "sampled_LTD_attempts"):
                row[key] = int(log[key])
            row["running_Abar"] = float(np.mean([r["cumulative_accuracy"] for r in rows
                                                  if r["variant"] == branch] +
                                                 [row["cumulative_accuracy"]]))
            if branch == A4A:
                expected = float(archived[task]["cumulative_accuracy"])
                assert abs(row["cumulative_accuracy"]-expected) < 1e-12, (task, row["cumulative_accuracy"], expected)
            if branch in (A4A, A4B):
                assert row["W_B_sha256"] == mechanism[FULL,task]["W_B_sha256"]
            rows.append(row)
            print(json.dumps(dict(phase="readout", variant=branch, task=task,
                                  accuracy=row["cumulative_accuracy"],
                                  WB_density=row["W_B_density"])), flush=True)
    assert len(rows) == 120
    write_csv_new(FULL20, rows)
    writer_distance()
    finalize(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("verify", "writers", "readouts"), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    base = setup()
    if args.phase == "verify":
        verify_full_writer(base)
    elif args.phase == "writers":
        writer_rows(base)
    else:
        readouts()
    print(json.dumps(dict(phase=args.phase, status="DONE",
                          seconds=time.perf_counter()-started)), flush=True)


if __name__ == "__main__":
    main()
