"""Local Section-6 component operations on the frozen working baseline.

All modifications live here: formal source/config/checkpoints are read-only.
Nonlinear CA1 readouts use exact current-WB historical sparse-code replay.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
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
from btsp_cam.plateau import plateau_probability, sample_slot_plateaus
from btsp_cam.ridge import StreamingRidge
from run_coordinate_aligned_full20 import CosineMoments
from run_k_robustness_and_selection import make_model
from run_k_tau_joint_selection import (load_codes, load_wb as load_frozen_wb,
                                        wb_file as frozen_wb_file)

SPEC = ROOT / "experiments/configs/paper_style_ablation.yaml"
ARCHIVED_WRITER = ROOT / "diagnostics/k_path_decomposition.csv"
VARIANTS = ("FULL", "A1_FIXED_GATE", "A2_NO_COLLISION", "A3_ONLY_LTP",
            "A4A_NO_THRESHOLD", "A4B_NO_CA1_OPERATOR")
WRITERS = ("A1_FIXED_GATE", "A2_NO_COLLISION", "A3_ONLY_LTP")


def setup():
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    lock_path = (SPEC.parent / spec["frozen_config"]).resolve()
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if (lock["status"], lock["k"], lock["q"], lock["tau"], lock["f_base"],
            lock["beta"], lock["lambda"], lock["p"], lock["gamma"], lock["readout_mode"]) != (
            "LOCAL_WORKING_ABLATION_BASELINE", 96, 4, 4/96, .005, 1.0,
            .01, .5, .1, "CURRENT_WB_REPLAY_REFIT"):
        raise AssertionError("frozen working baseline changed")
    for key, filename in (("validation_csv_sha256", "k_tau_validation_grid.csv"),
                          ("overlap_csv_sha256", "k_tau_overlap_statistics.csv")):
        if hashlib.sha256((ROOT / "diagnostics" / filename).read_bytes()).hexdigest() != lock[key]:
            raise AssertionError(f"frozen selection source changed: {filename}")
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(
        (SPEC.parent / spec["formal_config"]).resolve().read_text(encoding="utf-8")))
    if (base.seed, base.code_dim, base.memory_slots, base.base_plateau_probability,
            base.beta, base.trace_decay, base.flip_probability, base.gamma) != (
            1993, 8192, 4096, .005, 1.0, .01, .5, .1):
        raise AssertionError("formal config drifted")
    paths = {key: (SPEC.parent / spec[key]).resolve() for key in (
        "cache_dir", "full_csv", "mechanisms_csv", "familiarity_csv", "partial_cue_csv",
        "baseline_confirmation_csv", "baseline_cache")}
    return spec, lock, base, paths


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    if Path(path).exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def writer_step_variant(model, code, label, variant):
    """Preserve the original writer RNG draws; only the named operation changes."""
    model.traces.update(code, label)
    signals = model.traces.signals(code, label, model.config.beta)
    if variant == "A1_FIXED_GATE":
        probability = model.config.base_plateau_probability
    elif variant == "A2_NO_COLLISION":
        probability = plateau_probability(model.config.base_plateau_probability,
                                          replace(signals, interference=1.0))
    else:
        probability = plateau_probability(model.config.base_plateau_probability, signals)
    plateaus = sample_slot_plateaus(probability, model.memory.slots, model.plateau_rng)
    if variant == "A3_ONLY_LTP":
        selected = np.flatnonzero(plateaus)
        active = np.flatnonzero(code)
        if selected.size and active.size:
            target = np.ix_(selected, active)
            before = model.memory.weights[target].copy()
        else:
            target = None
            before = None
        model.memory.update(code, plateaus, model.config.flip_probability, model.synapse_rng)
        if target is not None:
            after = model.memory.weights[target]
            after[before == 1] = 1
            model.memory.weights[target] = after
        model.memory.last_one_to_zero = 0
        flips = model.memory.last_zero_to_one
    else:
        flips = model.memory.update(code, plateaus, model.config.flip_probability,
                                    model.synapse_rng)
    return dict(f_eff=probability, I_effective=1.0 if variant == "A2_NO_COLLISION"
                else signals.interference, F=signals.familiarity,
                plateau_count=int(plateaus.sum()), flip_01=model.memory.last_zero_to_one,
                flip_10=model.memory.last_one_to_zero, gross_flips=flips)


def experimental_wb(cache: Path, variant: str, task: int) -> Path:
    return cache / variant / f"WB_task{task:02d}.npz"


def save_experimental_wb(path, weights):
    if path.exists():
        np.testing.assert_array_equal(load_experimental_wb(path), weights)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=np.packbits(weights, axis=1))


def load_experimental_wb(path):
    with np.load(path, allow_pickle=False) as data:
        weights = np.unpackbits(data["packed"], axis=1, count=8192)
    if weights.shape != (4096, 8192):
        raise AssertionError("ablation WB shape")
    return weights


def code_pair(paths, split, task):
    path = paths["baseline_cache"] / f"{split}_task{task:02d}.npz"
    return load_codes(path, 96)


def train_writer_variant(base, paths, variant, tasks):
    model = make_model(base, 96, 1993, 1993)
    rows = []
    for task in range(1, tasks+1):
        start = time.perf_counter()
        s, y = code_pair(paths, "full_train", task)
        totals = dict(plateau_count=0, flip_01=0, flip_10=0, gross_flips=0)
        f_sum = F_sum = I_sum = 0.0
        for code, label in zip(s, y):
            log = writer_step_variant(model, code, int(label), variant)
            for key in totals:
                totals[key] += log[key]
            f_sum += log["f_eff"]
            F_sum += log["F"]
            I_sum += log["I_effective"]
            if variant == "A1_FIXED_GATE" and log["f_eff"] != .005:
                raise AssertionError("fixed gate deviated from f_base")
            if variant == "A2_NO_COLLISION" and log["I_effective"] != 1.0:
                raise AssertionError("collision term not removed")
            if variant == "A3_ONLY_LTP" and log["flip_10"] != 0:
                raise AssertionError("LTD occurred in only-LTP variant")
        save_experimental_wb(experimental_wb(paths["cache_dir"], variant, task),
                             model.memory.weights)
        rows.append(dict(variant=variant, task=task, train_samples=len(y),
                         plateau_count=totals["plateau_count"],
                         plateau_per_sample=totals["plateau_count"]/len(y),
                         gross_flips=totals["gross_flips"],
                         LTP_count=totals["flip_01"], LTD_count=totals["flip_10"],
                         WB_density=float(model.memory.weights.mean()),
                         mean_f_eff=f_sum/len(y), mean_F_train=F_sum/len(y),
                         mean_I_effective=I_sum/len(y),
                         writer_seconds=time.perf_counter()-start))
        print(json.dumps(dict(phase="writer", variant=variant, task=task,
                              plateau=totals["plateau_count"],
                              LTP=totals["flip_01"], LTD=totals["flip_10"])), flush=True)
    return rows, model.traces.values.copy()


def full_writer_rows(paths, tasks):
    archived = {int(r["task"]): r for r in read_csv(ARCHIVED_WRITER)
                if r.get("run") == "k96" and r.get("branch") == "evolving_aligned"}
    if set(archived) != set(range(1, 21)):
        raise AssertionError("archived Full writer trajectory incomplete")
    rows = []
    previous_ones = 0
    for task in range(1, tasks+1):
        wb = load_frozen_wb(frozen_wb_file(paths["baseline_cache"].parent, 96,
                                           task, development=False), 96, task)
        old = archived[task]
        if float(wb.mean()) != float(old["WB_density"]):
            raise AssertionError("Full WB archive mismatch")
        gross = int(old["gross_flips_task"])
        ones = int(wb.sum())
        net = ones-previous_ones
        if (gross+net) % 2 or (gross-net) % 2:
            raise AssertionError("Full flip balance mismatch")
        ltp, ltd = (gross+net)//2, (gross-net)//2
        rows.append(dict(variant="FULL", task=task, train_samples=2500,
                         plateau_count=int(old["plateau_count_task"]),
                         plateau_per_sample=int(old["plateau_count_task"])/2500,
                         gross_flips=gross, LTP_count=ltp, LTD_count=ltd,
                         WB_density=float(wb.mean()), mean_f_eff=float("nan"),
                         mean_F_train=float("nan"), mean_I_effective=float("nan"),
                         writer_seconds=0.0))
        previous_ones = ones
    return rows


def wb_for(paths, variant, task):
    if variant in ("FULL", "A4A_NO_THRESHOLD", "A4B_NO_CA1_OPERATOR"):
        return load_frozen_wb(frozen_wb_file(paths["baseline_cache"].parent, 96,
                                             task, development=False), 96, task)
    return load_experimental_wb(experimental_wb(paths["cache_dir"], variant, task))


def features(j, variant, active_count=96):
    x = np.asarray(j, dtype=np.float64)
    if variant == "A4B_NO_CA1_OPERATOR":
        return x
    a = x/active_count
    if variant == "A4A_NO_THRESHOLD":
        return a
    return np.maximum(a-4/96, 0.0)


def class_proto_overlap(class_sums, class_n):
    centroids = class_sums / class_n[:, None]
    norm = np.linalg.norm(centroids, axis=1)
    unit = np.divide(centroids, norm[:, None], out=np.zeros_like(centroids),
                     where=norm[:, None] > 0)
    sim = unit @ unit.T
    mask = ~np.eye(len(class_n), dtype=bool)
    return float(sim[mask].mean())


def fit_task(paths, variant, task, gamma, *, save_weights=False):
    start = time.perf_counter()
    wb = wb_for(paths, variant, task)
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    ridge = StreamingRidge(4096, task*5, gamma)
    for source in range(1, task+1):
        s, y = code_pair(paths, "full_train", source)
        j = np.asarray(sparse.csr_matrix(s, dtype=np.float32) @ wt)
        if np.any(j != np.floor(j)):
            raise AssertionError("noninteger binary overlap")
        ridge.update(features(j, variant), y)
    ridge.solve()
    ridge._last_cholesky = None
    if save_weights:
        out = paths["cache_dir"] / variant / "ridge_task20.npz"
        if out.exists():
            raise FileExistsError(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, weights=ridge.weights)
    cosine = CosineMoments(4096)
    class_sums = np.zeros((task*5, 4096), dtype=np.float64)
    class_n = np.zeros(task*5, dtype=np.int64)
    source_acc = []
    old_to_new = 0
    old_n = 0
    current_acc = current_five = float("nan")
    active = total = 0
    z_sum = z_sq = norm_sum = 0.0
    for source in range(1, task+1):
        s, y = code_pair(paths, "full_test", source)
        j = np.asarray(sparse.csr_matrix(s, dtype=np.float32) @ wt)
        z = features(j, variant)
        logits = ridge.logits(z)
        pred = logits.argmax(axis=1)
        source_acc.append(float(np.mean(pred == y)))
        if source < task:
            old_to_new += int(np.count_nonzero(pred >= (task-1)*5))
            old_n += len(y)
        else:
            current_acc = source_acc[-1]
            current_five = float(np.mean(
                logits[:, (task-1)*5:task*5].argmax(axis=1)+(task-1)*5 == y))
        active += int(np.count_nonzero(z))
        total += z.size
        z_sum += float(z.sum())
        z_sq += float(np.einsum("ij,ij->", z, z))
        norm_sum += float(np.linalg.norm(z, axis=1).sum())
        cosine.add(z, y)
        for label in np.unique(y):
            class_sums[label] += z[y == label].sum(axis=0)
            class_n[label] += int(np.count_nonzero(y == label))
    same, diff, margin = cosine.values()
    trace = float(np.trace(ridge.a))
    row = dict(variant=variant, task=task,
               cumulative_accuracy=float(np.mean(source_acc)),
               old_accuracy=float(np.mean(source_acc[:-1])) if task > 1 else float("nan"),
               current_accuracy=current_acc, current_5way_accuracy=current_five,
               CA1_density=active/total, z_mean=z_sum/total,
               z_std=float(np.sqrt(max(0,z_sq/total-(z_sum/total)**2))),
               feature_l2_mean=norm_sum/(task*500),
               gram_trace=trace, gram_trace_per_sample=trace/(task*500),
               gram_max_diagonal=float(np.diag(ridge.a).max()),
               same_class_cosine=same, different_class_cosine=diff,
               cosine_margin=margin, prototype_overlap_local=class_proto_overlap(class_sums,class_n),
               old_to_new_local=old_to_new/old_n if old_n else float("nan"),
               source_task_accuracies=json.dumps(source_acc),
               WB_density=float(wb.mean()), evaluation_seconds=time.perf_counter()-start)
    return row


def smoke(spec, lock, base, paths):
    if paths["cache_dir"].joinpath("smoke_pass.json").exists():
        raise FileExistsError("smoke already completed")
    prior = read_csv(paths["baseline_confirmation_csv"])
    if len(prior) != 20:
        raise AssertionError("frozen Full confirmation missing")
    for task in range(1, 6):
        actual = fit_task(paths, "FULL", task, lock["gamma"])
        expected = float(prior[task-1]["cumulative_validation_accuracy"])
        if abs(actual["cumulative_accuracy"]-expected) > 1e-12:
            raise AssertionError(f"Full Task{task} mismatch {actual['cumulative_accuracy']} != {expected}")
        for variant in ("A4A_NO_THRESHOLD", "A4B_NO_CA1_OPERATOR"):
            probe = fit_task(paths, variant, task, lock["gamma"])
            if not np.isfinite(probe["cumulative_accuracy"]):
                raise AssertionError("nonfinite A4 smoke accuracy")
    for variant in WRITERS:
        rows, _ = train_writer_variant(base, paths, variant, 5)
        for task in range(1, 6):
            probe = fit_task(paths, variant, task, lock["gamma"])
            if not np.isfinite(probe["cumulative_accuracy"]):
                raise AssertionError(f"nonfinite smoke accuracy {variant}")
        if variant == "A3_ONLY_LTP" and any(r["LTD_count"] for r in rows):
            raise AssertionError("A3 LTD smoke count nonzero")
    marker = paths["cache_dir"] / "smoke_pass.json"
    marker.write_text(json.dumps({"status":"PASS", "tasks":5,
                                  "variants":list(VARIANTS)}, indent=2),encoding="utf-8")
    print(json.dumps({"phase":"smoke", "status":"PASS"}),flush=True)


def full(spec, lock, base, paths):
    if not (paths["cache_dir"] / "smoke_pass.json").exists():
        raise AssertionError("smoke test must pass before full20")
    if paths["full_csv"].exists() or paths["mechanisms_csv"].exists():
        raise FileExistsError("full20 outputs exist; refusing overwrite")
    writer_rows = full_writer_rows(paths, 20)
    final_traces = None
    for variant in WRITERS:
        rows, traces = train_writer_variant(base, paths, variant, 20)
        writer_rows.extend(rows)
        if variant == "A1_FIXED_GATE":
            final_traces = traces
    # All operations update eligibility identically; the A1 final traces are
    # therefore also the Full/A2/A3 final traces, independent of writer RNG.
    if final_traces is None:
        raise AssertionError("missing eligibility traces")
    trace_path = paths["cache_dir"] / "final_traces.npz"
    np.savez_compressed(trace_path, values=final_traces)
    for variant in ("A4A_NO_THRESHOLD", "A4B_NO_CA1_OPERATOR"):
        writer_rows.extend([dict(r, variant=variant) for r in writer_rows if r["variant"] == "FULL"])
    out = []
    prior = read_csv(paths["baseline_confirmation_csv"])
    for variant in VARIANTS:
        for task in range(1, 21):
            row = fit_task(paths, variant, task, lock["gamma"], save_weights=task==20)
            if variant == "FULL" and abs(row["cumulative_accuracy"]-float(prior[task-1]["cumulative_validation_accuracy"])) > 1e-12:
                raise AssertionError(f"frozen Full Task{task} mismatch")
            row["average_incremental_to_task"] = float(np.mean(
                [r["cumulative_accuracy"] for r in out if r["variant"] == variant]
                + [row["cumulative_accuracy"]]))
            out.append(row)
            print(json.dumps(dict(phase="readout",variant=variant,task=task,
                                  accuracy=row["cumulative_accuracy"])),flush=True)
    for variant in VARIANTS:
        trajectory = [r for r in out if r["variant"] == variant]
        source_matrix = [json.loads(r["source_task_accuracies"]) for r in trajectory]
        forgetting = float(np.mean([max(source_matrix[t][source] for t in range(source,20))
                                    -source_matrix[19][source] for source in range(19)]))*100
        for row in trajectory:
            row["Fgt_local_pp"] = forgetting
    # Attach paired writer stats; A4 variants inherit Full exactly.
    writer_by = {(r["variant"],r["task"]):r for r in writer_rows}
    for row in out:
        log=writer_by[row["variant"],row["task"]]
        for key in ("plateau_count","plateau_per_sample","gross_flips","LTP_count",
                    "LTD_count","mean_f_eff","mean_F_train","mean_I_effective"):
            row[key]=log[key]
    write_csv(paths["mechanisms_csv"],writer_rows)
    write_csv(paths["full_csv"],out)


def partial_cue(spec, lock, paths):
    # Corrected seed namespace: the original seed+task formula re-used RNG
    # streams across distinct (seed, task) pairs. Preserve the old CSV and
    # write only a separately named diagnostic.
    output = ROOT / "diagnostics/partial_cue_robustness_corrected.csv"
    if output.exists():
        raise FileExistsError("corrected partial-cue output exists")
    full_rows = read_csv(paths["full_csv"])
    if len(full_rows) != 120:
        raise AssertionError("full20 ablation table incomplete")
    old_sources = [code_pair(paths,"full_test",task) for task in range(1,20)]
    variants=("FULL","A4A_NO_THRESHOLD","A4B_NO_CA1_OPERATOR","A3_ONLY_LTP")
    readouts={variant:np.load(paths["cache_dir"]/variant/"ridge_task20.npz")["weights"]
              for variant in variants}
    wts={variant:np.ascontiguousarray(wb_for(paths,variant,20).T,dtype=np.float32)
         for variant in variants}
    rows=[]
    for ratio in spec["partial_cue_drop_ratios"]:
        for seed in spec["partial_cue_seeds"]:
            counters={variant:[0,0] for variant in variants}
            for task,(s,y) in enumerate(old_sources,1):
                rng=np.random.default_rng(np.random.SeedSequence([seed,task]))
                cue=s.copy()
                n_drop=round(96*ratio)
                if n_drop:
                    active=np.argwhere(s==1).reshape(len(y),96,2)[:,:,1]
                    picks=np.argpartition(rng.random((len(y),96)),n_drop-1,axis=1)[:,:n_drop]
                    cue[np.arange(len(y))[:,None],np.take_along_axis(active,picks,axis=1)]=0
                remaining=96-n_drop
                cue_sparse=sparse.csr_matrix(cue,dtype=np.float32)
                for variant in variants:
                    j=np.asarray(cue_sparse@wts[variant])
                    z=features(j,variant,remaining)
                    pred=(z@readouts[variant]).argmax(axis=1)
                    counters[variant][0]+=int(np.count_nonzero(pred==y))
                    counters[variant][1]+=len(y)
            for variant,(correct,total) in counters.items():
                rows.append(dict(variant=variant,drop_ratio=ratio,seed=seed,
                                 old_class_accuracy=correct/total,
                                 correct=correct,samples=total,
                                 dropped_bits=n_drop,remaining_bits=remaining))
            print(json.dumps(dict(phase="partial_cue",ratio=ratio,seed=seed)),flush=True)
    for variant in variants:
        source=next(r for r in full_rows if r["variant"]==variant and int(r["task"])==20)
        expected=float(source["old_accuracy"])
        for r in rows:
            if r["variant"]==variant and r["drop_ratio"]==0.0 and abs(r["old_class_accuracy"]-expected)>1e-12:
                raise AssertionError(f"drop0 did not match old accuracy: {variant}")
    write_csv(output,rows)


def familiarity_audit(paths):
    """Final-trace test F separated by correct/error; local Table-2 statistic."""
    if paths["familiarity_csv"].exists():
        raise FileExistsError("familiarity output exists")
    with np.load(paths["cache_dir"] / "final_traces.npz", allow_pickle=False) as data:
        traces = data["values"]
    norm = np.linalg.norm(traces, axis=1)
    variants = VARIANTS
    wts = {v: np.ascontiguousarray(wb_for(paths,v,20).T,dtype=np.float32)
           for v in variants}
    readouts = {v:np.load(paths["cache_dir"]/v/"ridge_task20.npz")["weights"]
                for v in variants}
    totals = {v:dict(correct_count=0,error_count=0,F_correct_sum=0.0,
                     F_error_sum=0.0) for v in variants}
    # Old-class traces can underflow to exact zero after tens of thousands of
    # per-sample decay steps. Restrict this local Table-2 proxy to Task20's
    # current five classes, whose final traces remain well defined.
    for task in (20,):
        s,y = code_pair(paths,"full_test",task)
        if np.any(norm[y] == 0):
            raise AssertionError("current-task trace unexpectedly vanished")
        F = np.einsum("ij,ij->i",s.astype(np.float64),traces[y]) / (np.sqrt(96)*norm[y])
        F = np.clip(F,0,1)
        s_csr=sparse.csr_matrix(s,dtype=np.float32)
        for variant in variants:
            j=np.asarray(s_csr@wts[variant])
            pred=(features(j,variant)@readouts[variant]).argmax(axis=1)
            good=pred==y
            record=totals[variant]
            record["correct_count"]+=int(good.sum())
            record["error_count"]+=int((~good).sum())
            record["F_correct_sum"]+=float(F[good].sum())
            record["F_error_sum"]+=float(F[~good].sum())
    rows=[]
    for variant in variants:
        r=totals[variant]
        rows.append(dict(variant=variant,
                         F_correct=r["F_correct_sum"]/r["correct_count"],
                         F_error=r["F_error_sum"]/r["error_count"],
                         F_gap=r["F_correct_sum"]/r["correct_count"]-
                               r["F_error_sum"]/r["error_count"],
                         correct_count=r["correct_count"],error_count=r["error_count"],
                         trace_snapshot="final_Task20_true_class_eligibility",
                         evaluation_split="Task20_current_five_class_test_only"))
    write_csv(paths["familiarity_csv"],rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",required=True,choices=("smoke","full","partial_cue","familiarity"))
    args=parser.parse_args()
    spec,lock,base,paths=setup()
    if args.phase=="smoke":
        smoke(spec,lock,base,paths)
    elif args.phase=="full":
        full(spec,lock,base,paths)
    elif args.phase=="partial_cue":
        partial_cue(spec,lock,paths)
    else:
        familiarity_audit(paths)


if __name__=="__main__":
    main()
