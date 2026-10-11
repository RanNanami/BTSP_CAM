"""Exploratory high-k mechanism controls on the frozen feature cache.

The formal configuration, runtime, checkpoints and repaired baseline artifacts
are only read. All new trajectories and sparse-code caches live in diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from run_coordinate_aligned_full20 import CosineMoments
from run_high_k_stress_test import DenseHighKAligned, writer_step
from run_k_extended_search import class_overlap_geometry
from run_repair_ladder_task1_5 import load_split
from run_unresolved_parameter_screen import read_rows, write_rows

SPEC_PATH = ROOT / "experiments/configs/high_k_mechanism_decomposition.yaml"
ARCHIVE_HIGH = ROOT / "diagnostics/high_k_stress_full20.csv"
ARCHIVE_K96 = ROOT / "diagnostics/k_path_decomposition.csv"
M = 8192
SLOTS = 4096
TASKS = 20


def setup() -> tuple[dict, BTSPCAMConfig, Path, dict[str, Path]]:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    if (spec["ks"], spec["memory_ks"], spec["tau_ks"],
            spec["control_seeds"], spec["tau_grid"]) != (
            [96, 1024, 4096, 6144, 7168], [96, 1024, 6144],
            [96, 1024, 6144],
            [2026101001, 2026101002, 2026101003, 2026101004, 2026101005],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5]):
        raise ValueError("exploratory grid or random seeds changed")
    if (spec["class_order_seed"], spec["rf_seed"], spec["writer_seed"],
            spec["f_base"], spec["gamma"]) != (1993, 1993, 1993, 0.005, 0.1):
        raise ValueError("locked protocol changed")
    base_path = (SPEC_PATH.parent / spec["base_config"]).resolve()
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(base_path.read_text(encoding="utf-8")))
    if (base.seed, base.code_dim, base.memory_slots, base.flip_probability,
            base.beta, base.trace_decay, base.gamma,
            base.base_plateau_probability) != (
            1993, M, SLOTS, 0.5, 1.0, 0.01, 0.1, 0.005):
        raise ValueError("formal baseline settings changed")
    cache = (SPEC_PATH.parent / spec["cache_dir"]).resolve()
    output = {key: (SPEC_PATH.parent / value).resolve()
              for key, value in spec["outputs"].items()}
    return spec, base, cache, output


def model_for_k(base: BTSPCAMConfig, k: int) -> BTSPCAM:
    cfg = replace(base, k=k, tau=0.0)
    rf = RNGProvenance.from_seed(1993)
    wr = RNGProvenance.from_seed(1993)
    model = BTSPCAM(cfg, class_order=pilot_class_order(100, 1993),
                    projection_seed=rf.projection_seed,
                    plateau_seed=wr.plateau_seed,
                    synapse_seed=wr.synapse_seed)
    model.ridge = None
    return model


def code_path(cache: Path, k: int, split: str, task: int) -> Path:
    return cache / f"k{k}" / f"{split}_task{task:02d}.npz"


def wb_path(cache: Path, k: int, task: int) -> Path:
    return cache / f"k{k}" / f"WB_task{task:02d}.npz"


def write_codes(path: Path, codes: np.ndarray, labels: np.ndarray, k: int) -> None:
    if codes.dtype != np.uint8 or codes.shape[1] != M:
        raise AssertionError("sparse-code dtype/shape changed")
    if not np.all(codes.sum(axis=1, dtype=np.int64) == k):
        raise AssertionError("exact top-k invariant failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=np.packbits(codes, axis=1),
                        labels=np.asarray(labels, dtype=np.int16))


def load_codes(path: Path, k: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        packed = item["packed"]
        y = item["labels"].astype(np.int64)
    s = np.unpackbits(packed, axis=1, count=M)
    if s.shape[0] != len(y) or not np.all(s.sum(axis=1, dtype=np.int64) == k):
        raise AssertionError(f"corrupt sparse cache: {path}")
    return s, y


def write_wb(path: Path, weights: np.ndarray) -> None:
    if weights.shape != (SLOTS, M) or weights.dtype != np.uint8:
        raise AssertionError("memory shape/dtype changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=np.packbits(weights, axis=1))


def load_wb(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as item:
        packed = item["packed"]
    w = np.unpackbits(packed, axis=1, count=M)
    if w.shape != (SLOTS, M):
        raise AssertionError(f"corrupt memory cache: {path}")
    return w


def expected_writer(k: int, task: int) -> dict:
    if k == 96:
        matches = [r for r in read_rows(ARCHIVE_K96)
                   if r["run"] == "k96" and int(r["task"]) == task
                   and r["branch"] == "evolving_aligned"]
        if len(matches) != 1:
            raise AssertionError("missing archived k=96 trajectory")
        return matches[0]
    matches = [r for r in read_rows(ARCHIVE_HIGH)
               if int(r["k"]) == k and int(r["task"]) == task
               and r["condition"] == "FIXED_FBASE"]
    if len(matches) != 1:
        raise AssertionError(f"missing archived k={k} trajectory")
    return matches[0]


def prepared(cache: Path, k: int) -> bool:
    return all(code_path(cache, k, split, task).exists()
               for split in ("train", "test") for task in range(1, TASKS + 1))


def prepare_codes(base: BTSPCAMConfig, cache: Path, k: int) -> None:
    if prepared(cache, k):
        print(json.dumps({"phase": "codes", "k": k, "skipped": "cached"}), flush=True)
        return
    model = model_for_k(base, k)
    for task in range(1, TASKS + 1):
        start = time.perf_counter()
        for split in ("train", "test"):
            path = code_path(cache, k, split, task)
            if path.exists():
                continue
            h, y = load_split(split, task)
            s = model.encoder.encode(h, base.encoder_batch_size)
            write_codes(path, s, y, k)
            del h, y, s
        print(json.dumps({"phase": "codes", "k": k, "task": task,
                          "seconds": round(time.perf_counter() - start, 2)}), flush=True)


def prepare_memory(base: BTSPCAMConfig, cache: Path, k: int) -> None:
    if not prepared(cache, k):
        raise ValueError("prepare sparse codes before rebuilding memory")
    model = model_for_k(base, k)
    for task in range(1, TASKS + 1):
        start = time.perf_counter()
        s, y = load_codes(code_path(cache, k, "train", task), k)
        writer = writer_step(model, s, y)
        reference = expected_writer(k, task)
        target_flip = float(reference.get("gross_flips", reference.get("gross_flips_task")))
        target_density = float(reference["WB_density"])
        if writer["gross_flips"] != target_flip or abs(float(model.memory.weights.mean()) - target_density) > 1e-12:
            raise AssertionError(f"replayed writer trajectory mismatch k={k} task={task}")
        path = wb_path(cache, k, task)
        if path.exists():
            if not np.array_equal(load_wb(path), model.memory.weights):
                raise AssertionError(f"existing experimental WB differs k={k} task={task}")
        else:
            write_wb(path, model.memory.weights)
        print(json.dumps({"phase": "memory", "k": k, "task": task,
                          "density": target_density,
                          "gross_flips": writer["gross_flips"],
                          "seconds": round(time.perf_counter() - start, 2)}), flush=True)


def replace_rows(path: Path, key: dict, rows: list[dict]) -> None:
    old = read_rows(path) if path.exists() else []
    keep = [row for row in old if not all(str(row.get(k)) == str(v)
                                           for k, v in key.items())]
    write_rows(path, keep + rows)


def complete_rows(path: Path, key: dict) -> bool:
    if not path.exists():
        return False
    matches = [r for r in read_rows(path)
               if all(str(r.get(a)) == str(b) for a, b in key.items())]
    return len(matches) == TASKS and {int(r["task"]) for r in matches} == set(range(1, TASKS + 1))


def fisher_stats(gram: np.ndarray, class_sums: np.ndarray,
                 class_counts: np.ndarray) -> tuple[float, float, float]:
    n = float(class_counts.sum())
    centroids = class_sums.T / class_counts[:, None]
    global_mean = class_sums.sum(axis=1) / n
    centroid_sq = np.einsum("ij,ij->i", centroids, centroids)
    within = float(np.trace(gram) / n - class_counts @ centroid_sq / n)
    between = float(np.sum(class_counts[:, None] * (centroids - global_mean) ** 2) / n)
    return within, between, between / within if within > 0 else float("nan")


def task_metrics_from_logits(logits: np.ndarray, y: np.ndarray, task: int) -> dict:
    old_n = (task - 1) * 500
    pred = logits.argmax(axis=1)
    current = y[old_n:]
    current_logits = logits[old_n:]
    local = current_logits[:, (task - 1) * 5:task * 5]
    return dict(cumulative_accuracy=float(np.mean(pred == y)),
                old_accuracy=float(np.mean(pred[:old_n] == y[:old_n])) if old_n else float("nan"),
                current_accuracy=float(np.mean(pred[old_n:] == current)),
                current_5way_accuracy=float(np.mean(
                    local.argmax(axis=1) + (task - 1) * 5 == current)))


def sparse_only(base: BTSPCAMConfig, cache: Path, path: Path, k: int) -> None:
    key = {"representation": "sparse_s", "k": k}
    if complete_rows(path, key):
        print(json.dumps({"phase": "sparse_ridge", "k": k, "skipped": "complete"}), flush=True)
        return
    if not prepared(cache, k):
        raise ValueError("prepare sparse codes before sparse ridge")
    aligned = DenseHighKAligned(M, M, k, base.gamma)
    test_s: list[sparse.csr_matrix] = []
    test_y: list[np.ndarray] = []
    counts = np.zeros(M, dtype=np.int64)
    rows: list[dict] = []
    total_acc = 0.0
    for task in range(1, TASKS + 1):
        start = time.perf_counter()
        s, y = load_codes(code_path(cache, k, "train", task), k)
        counts += s.sum(axis=0, dtype=np.int64)
        aligned.update(s, y)
        del s, y
        st, yt = load_codes(code_path(cache, k, "test", task), k)
        test_s.append(sparse.csr_matrix(st, dtype=np.float32))
        test_y.append(yt)
        del st
        gram = aligned.c.clone()
        gram.diagonal().add_(base.gamma)
        chol = torch.linalg.cholesky(gram)
        weights = torch.cholesky_solve(torch.from_numpy(aligned.d), chol).numpy().copy()
        del gram, chol
        logits = np.vstack([np.asarray(part @ weights) for part in test_s])
        metrics = task_metrics_from_logits(logits, np.concatenate(test_y), task)
        total_acc += metrics["cumulative_accuracy"]
        same, diff, margin, same_j, diff_j = class_overlap_geometry(test_s, test_y, k, M)
        p = counts / counts.sum()
        positive = p[p > 0]
        class_counts = np.full(task * 5, 500, dtype=np.float64)
        within, between, fisher = fisher_stats(aligned.c.numpy(), aligned.d, class_counts)
        row = dict(representation="sparse_s", k=k, task=task, gamma=base.gamma,
                   **metrics, average_incremental_to_task=total_acc / task,
                   sparse_same_cos=same, sparse_diff_cos=diff, sparse_margin=margin,
                   hamming_same_mean=2 * (k - same_j),
                   hamming_diff_mean=2 * (k - diff_j),
                   effective_rf_support=float(np.exp(-np.sum(positive * np.log(positive)))),
                   top1_bit_share=float(np.sort(p)[-int(np.ceil(0.01 * M)):].sum()),
                   within_class_cov_trace=within, between_class_scatter=between,
                   fisher_like_ratio=fisher,
                   seconds=time.perf_counter() - start)
        rows.append(row)
        replace_rows(path, key, rows)
        print(json.dumps({"phase": "sparse_ridge", "k": k, "task": task,
                          "acc": metrics["cumulative_accuracy"],
                          "seconds": round(row["seconds"], 2)}), flush=True)


def vit_reference(base: BTSPCAMConfig, path: Path) -> None:
    key = {"representation": "vit_h", "k": "reference"}
    if complete_rows(path, key):
        return
    ridge = StreamingRidge(base.feature_dim, 0, base.gamma)
    test_h: list[np.ndarray] = []
    test_y: list[np.ndarray] = []
    rows = []
    total_acc = 0.0
    for task in range(1, TASKS + 1):
        h, y = load_split("train", task)
        ridge.expand_classes(task * 5)
        ridge.update(h, y)
        ridge.solve()
        ht, yt = load_split("test", task)
        test_h.append(ht)
        test_y.append(yt)
        logits = np.vstack([ridge.logits(part) for part in test_h])
        metrics = task_metrics_from_logits(logits, np.concatenate(test_y), task)
        total_acc += metrics["cumulative_accuracy"]
        class_counts = np.full(task * 5, 500, dtype=np.float64)
        within, between, fisher = fisher_stats(ridge.a, ridge.b, class_counts)
        rows.append(dict(representation="vit_h", k="reference", task=task,
                         gamma=base.gamma, **metrics,
                         average_incremental_to_task=total_acc / task,
                         within_class_cov_trace=within,
                         between_class_scatter=between, fisher_like_ratio=fisher))
        replace_rows(path, key, rows)
        print(json.dumps({"phase": "vit_ridge", "task": task,
                          "acc": metrics["cumulative_accuracy"]}), flush=True)


def memory_control_matrix(kind: str, learned: np.ndarray, seed: int,
                          k: int, task: int) -> np.ndarray:
    if kind == "LEARNED_WB":
        return learned
    rng = np.random.default_rng(seed + 100_000 * k + task)
    if kind == "BERNOULLI_MATCHED":
        return (rng.random(learned.shape, dtype=np.float32) < learned.mean()).astype(np.uint8)
    if kind == "ROW_SHUFFLED":
        shuffled = rng.permuted(learned, axis=1)
        if not np.array_equal(shuffled.sum(axis=1), learned.sum(axis=1)):
            raise AssertionError("row shuffle failed to preserve row degrees")
        return shuffled
    raise ValueError(kind)


def eval_memory(ridge: StreamingRidge, weights: np.ndarray, k: int,
                test_s: list[sparse.csr_matrix], test_y: list[np.ndarray],
                task: int) -> tuple[dict, tuple[float, float, float]]:
    # Associativity yields the identical linear readout while avoiding an
    # all-seen-sample 4096-D overlap matrix for every random control.
    effective = np.asarray(weights.T, dtype=np.float64) @ ridge.weights / k
    moments = CosineMoments(SLOTS)
    old_ok = old_n = cur_ok = cur_n = five_ok = 0
    for source, (s, y) in enumerate(zip(test_s, test_y), 1):
        logits = np.asarray(s @ effective)
        ok = int(np.count_nonzero(logits.argmax(axis=1) == y))
        if source == task:
            wt = np.ascontiguousarray(weights.T, dtype=np.float32)
            probe = np.concatenate([np.flatnonzero(y == label)[:20]
                                    for label in np.unique(y)])
            if len(probe) != 100:
                raise AssertionError("expected 20 fixed test probes per current class")
            a = np.asarray(s[probe] @ wt, dtype=np.float64) / k
            moments.add(a, y[probe])
            cur_ok += ok
            cur_n += len(y)
            local = logits[:, (task - 1) * 5:task * 5]
            five_ok += int(np.count_nonzero(
                local.argmax(axis=1) + (task - 1) * 5 == y))
        else:
            old_ok += ok
            old_n += len(y)
    metrics = dict(cumulative_accuracy=(old_ok + cur_ok) / (old_n + cur_n),
                   old_accuracy=old_ok / old_n if old_n else float("nan"),
                   current_accuracy=cur_ok / cur_n,
                   current_5way_accuracy=five_ok / cur_n)
    return metrics, moments.values()


def memory_controls(base: BTSPCAMConfig, cache: Path, path: Path,
                    spec: dict, k: int) -> None:
    if not all(wb_path(cache, k, t).exists() for t in range(1, TASKS + 1)):
        raise ValueError("prepare memory trajectory first")
    aligned = DenseHighKAligned(M, SLOTS, k, base.gamma)
    test_s: list[sparse.csr_matrix] = []
    test_y: list[np.ndarray] = []
    cases = [("LEARNED_WB", 0)] + [
        (kind, seed) for kind in ("BERNOULLI_MATCHED", "ROW_SHUFFLED")
        for seed in spec["control_seeds"]]
    for task in range(1, TASKS + 1):
        s, y = load_codes(code_path(cache, k, "train", task), k)
        aligned.update(s, y)
        del s, y
        st, yt = load_codes(code_path(cache, k, "test", task), k)
        test_s.append(sparse.csr_matrix(st, dtype=np.float32))
        test_y.append(yt)
        del st
        learned = load_wb(wb_path(cache, k, task))
        for kind, seed in cases:
            key = {"k": k, "task": task, "control": kind, "seed": seed}
            if any(all(str(r.get(f)) == str(v) for f, v in key.items())
                   and r.get("cosine_scope") == "current_task_20_per_class_probe"
                   for r in (read_rows(path) if path.exists() else [])):
                continue
            start = time.perf_counter()
            w = memory_control_matrix(kind, learned, seed, k, task)
            ridge = aligned.solve(w)
            metrics, cosine = eval_memory(ridge, w, k, test_s, test_y, task)
            if kind == "LEARNED_WB":
                reference = expected_writer(k, task)
                if abs(metrics["cumulative_accuracy"] - float(reference["cumulative_accuracy"])) > 1e-12:
                    raise AssertionError(f"learned WB baseline mismatch k={k} task={task}")
            class_counts = np.full(task * 5, 500, dtype=np.float64)
            within, between, fisher = fisher_stats(ridge.a, ridge.b, class_counts)
            n = 2500 * task
            sum_a = ridge.b.sum()
            mean_a = sum_a / (n * SLOTS)
            std_a = np.sqrt(max(np.trace(ridge.a) / (n * SLOTS) - mean_a ** 2, 0.0))
            row = dict(k=k, task=task, control=kind, seed=seed,
                       random_seed_rule="seed + 100000*k + task",
                       **metrics, gamma=base.gamma, tau=0.0,
                       wb_density=float(w.mean()), target_density=float(learned.mean()),
                       row_degrees_preserved=int(np.array_equal(
                           w.sum(axis=1), learned.sum(axis=1))),
                       a_mean=float(mean_a), a_std=float(std_a),
                       a_same_cos=cosine[0], a_diff_cos=cosine[1],
                       a_margin=cosine[2],
                       cosine_scope="current_task_20_per_class_probe",
                       within_class_cov_trace=within,
                       between_class_scatter=between,
                       fisher_like_ratio=fisher,
                       seconds=time.perf_counter() - start)
            replace_rows(path, key, [row])
            print(json.dumps({"phase": "memory_control", "k": k,
                              "task": task, "control": kind, "seed": seed,
                              "acc": metrics["cumulative_accuracy"],
                              "seconds": round(row["seconds"], 2)}), flush=True)
            del ridge, w


def merge_controls(path: Path, ks: list[int], seeds: list[int]) -> None:
    rows = []
    for k in ks:
        source = path.with_name(f"{path.stem}_k{k}{path.suffix}")
        part = read_rows(source)
        if len(part) != TASKS * (1 + 2 * len(seeds)):
            raise ValueError(f"incomplete control result for k={k}")
        keys = {(int(r["task"]), r["control"], int(r["seed"])) for r in part}
        if len(keys) != len(part):
            raise AssertionError(f"duplicate control rows for k={k}")
        for kind in ("LEARNED_WB", "BERNOULLI_MATCHED", "ROW_SHUFFLED"):
            kind_rows = [r for r in part if r["control"] == kind]
            for seed in sorted({int(r["seed"]) for r in kind_rows}):
                trajectory = sorted((r for r in kind_rows if int(r["seed"]) == seed),
                                    key=lambda r: int(r["task"]))
                if len(trajectory) != TASKS:
                    raise AssertionError("incomplete control-seed trajectory")
                running = 0.0
                for row in trajectory:
                    running += float(row["cumulative_accuracy"])
                    row["average_incremental_to_task"] = running / int(row["task"])
        for task in range(1, TASKS + 1):
            current = [r for r in part if int(r["task"]) == task]
            learned = next(float(r["cumulative_accuracy"]) for r in current
                           if r["control"] == "LEARNED_WB")
            random_mean = float(np.mean([float(r["cumulative_accuracy"])
                                         for r in current if r["control"] == "BERNOULLI_MATCHED"]))
            shuffled_mean = float(np.mean([float(r["cumulative_accuracy"])
                                           for r in current if r["control"] == "ROW_SHUFFLED"]))
            for row in current:
                row["delta_learned_minus_bernoulli_mean"] = learned - random_mean
                row["delta_learned_minus_row_shuffled_mean"] = learned - shuffled_mean
        rows.extend(part)
    write_rows(path, rows)
    print(json.dumps({"phase": "merge_controls", "rows": len(rows)}), flush=True)


def overlap_counts(cache: Path, k: int, split: str, task: int,
                   wt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s, y = load_codes(code_path(cache, k, split, task), k)
    # At 75% activation density, dense SGEMM is far cheaper than a nominally
    # sparse traversal. All integer overlap counts remain exactly representable
    # in float32 because k <= 6144 < 2**24.
    if k >= 4096:
        counts = np.asarray(s, dtype=np.float32) @ wt
    else:
        counts = np.asarray(sparse.csr_matrix(s, dtype=np.float32) @ wt)
    if (counts.min() < 0 or counts.max() > k
            or not np.all(counts == np.floor(counts))):
        raise AssertionError("binary overlap must be an exactly represented integer")
    return counts.astype(np.uint16), y


def tau_replay(base: BTSPCAMConfig, cache: Path, path: Path,
               spec: dict, k: int) -> None:
    """Current-WB replay oracle; nonlinear tau never uses the C/D closure."""
    controls_path = (SPEC_PATH.parent / spec["outputs"]["memory"]).resolve()
    if (not controls_path.exists() or
            len([r for r in read_rows(controls_path)
                 if int(r["k"]) == k and r["control"] == "LEARNED_WB"]) != TASKS):
        controls_path = controls_path.with_name(
            f"{controls_path.stem}_k{k}{controls_path.suffix}")
    if not controls_path.exists():
        raise ValueError("finish learned-WB controls before tau replay")
    controls = {(int(r["k"]), int(r["task"])): r for r in read_rows(controls_path)
                if r["control"] == "LEARNED_WB"}
    if not all((k, t) in controls for t in range(1, TASKS + 1)):
        raise ValueError("learned-WB trajectory incomplete")
    for task in range(1, TASKS + 1):
        if all(any(int(r["k"]) == k and int(r["task"]) == task
                   and float(r["tau"]) == tau
                   for r in (read_rows(path) if path.exists() else []))
               for tau in spec["tau_grid"]):
            print(json.dumps({"phase": "tau_oracle", "k": k, "task": task,
                              "skipped": "complete"}), flush=True)
            continue
        w = load_wb(wb_path(cache, k, task))
        wt = np.ascontiguousarray(w.T, dtype=np.float32)
        train_parts = [overlap_counts(cache, k, "train", source, wt)
                       for source in range(1, task + 1)]
        test_parts = [overlap_counts(cache, k, "test", source, wt)
                      for source in range(1, task + 1)]
        all_test_counts = np.concatenate([part[0].ravel() for part in test_parts])
        a_mean = float(all_test_counts.mean() / k)
        a_std = float(all_test_counts.std() / k)
        del all_test_counts
        for tau in spec["tau_grid"]:
            key = {"k": k, "task": task, "tau": tau}
            if any(all(str(r.get(f)) == str(v) for f, v in key.items())
                   for r in (read_rows(path) if path.exists() else [])):
                continue
            start = time.perf_counter()
            ridge = StreamingRidge(SLOTS, task * 5, base.gamma)
            for counts, labels in train_parts:
                z = np.maximum(counts.astype(np.float64) / k - tau, 0.0)
                ridge.update(z, labels)
                del z
            ridge.solve()
            moments = CosineMoments(SLOTS)
            old_ok = old_n = cur_ok = cur_n = five_ok = 0
            sum_z = sum_z2 = 0.0
            positives = elements = 0
            for source, (counts, labels) in enumerate(test_parts, 1):
                z = np.maximum(counts.astype(np.float64) / k - tau, 0.0)
                moments.add(z, labels)
                logits = ridge.logits(z)
                ok = int(np.count_nonzero(logits.argmax(axis=1) == labels))
                if source == task:
                    cur_ok += ok
                    cur_n += len(labels)
                    local = logits[:, (task - 1) * 5:task * 5]
                    five_ok += int(np.count_nonzero(
                        local.argmax(axis=1) + (task - 1) * 5 == labels))
                else:
                    old_ok += ok
                    old_n += len(labels)
                sum_z += float(z.sum())
                sum_z2 += float(np.square(z).sum())
                positives += int(np.count_nonzero(z))
                elements += z.size
                del z, logits
            cumulative = (old_ok + cur_ok) / (old_n + cur_n)
            if tau == 0.0:
                reference = controls[(k, task)]
                if abs(cumulative - float(reference["cumulative_accuracy"])) > 1e-12:
                    raise AssertionError(f"replay oracle and C/D differ k={k} task={task}")
            previous = [r for r in (read_rows(path) if path.exists() else [])
                        if int(r["k"]) == k and float(r["tau"]) == tau
                        and int(r["task"]) < task]
            if len(previous) != task - 1:
                raise AssertionError("tau average requires every prior task")
            same, diff, margin = moments.values()
            z_mean = sum_z / elements
            row = dict(k=k, task=task, tau=tau,
                       method="CURRENT_WB_REPLAY_ORACLE_NOT_EXEMPLAR_FREE",
                       gamma=base.gamma,
                       cumulative_accuracy=cumulative,
                       old_accuracy=old_ok / old_n if old_n else float("nan"),
                       current_accuracy=cur_ok / cur_n,
                       current_5way_accuracy=five_ok / cur_n,
                       average_incremental_to_task=(
                           sum(float(r["cumulative_accuracy"]) for r in previous)
                           + cumulative) / task,
                       ca1_positive_density=positives / elements,
                       a_mean=a_mean, a_std=a_std, z_mean=z_mean,
                       z_std=float(np.sqrt(max(sum_z2 / elements - z_mean ** 2, 0.0))),
                       z_same_cos=same, z_diff_cos=diff, z_margin=margin,
                       wb_density=float(w.mean()),
                       seconds=time.perf_counter() - start)
            old = read_rows(path) if path.exists() else []
            write_rows(path, old + [row])
            print(json.dumps({"phase": "tau_oracle", "k": k,
                              "task": task, "tau": tau, "acc": cumulative,
                              "density": row["ca1_positive_density"],
                              "seconds": round(row["seconds"], 2)}), flush=True)
            del ridge
        del train_parts, test_parts, w, wt


def merge_tau(path: Path, ks: list[int], tau_grid: list[float]) -> None:
    rows = []
    for k in ks:
        source = path.with_name(f"{path.stem}_k{k}{path.suffix}")
        part = read_rows(source)
        keys = {(int(r["task"]), float(r["tau"])) for r in part}
        if (len(part) != TASKS * len(tau_grid) or len(keys) != len(part)
                or keys != {(task, tau) for task in range(1, TASKS + 1)
                            for tau in tau_grid}):
            raise ValueError(f"incomplete tau replay k={k}")
        rows.extend(part)
    write_rows(path, rows)
    print(json.dumps({"phase": "merge_tau", "rows": len(rows)}), flush=True)


def decomposition(output: dict[str, Path]) -> None:
    sparse_rows = read_rows(output["sparse"])
    control_rows = read_rows(output["memory"])
    rows = []
    vit = [r for r in sparse_rows if r["representation"] == "vit_h"
           and int(r["task"]) == 20]
    if len(vit) != 1:
        raise ValueError("missing full20 ViT reference")
    vit_avg = float(vit[0]["average_incremental_to_task"])
    vit_final = float(vit[0]["cumulative_accuracy"])
    for k in (96, 1024, 6144):
        sparse_final = [r for r in sparse_rows if r["representation"] == "sparse_s"
                        and int(r["k"]) == k and int(r["task"]) == 20]
        if len(sparse_final) != 1:
            raise ValueError(f"missing sparse full20 k={k}")
        controls = [r for r in control_rows if int(r["k"]) == k
                    and int(r["task"]) == 20]
        if len(controls) != 11:
            raise ValueError(f"missing WB controls k={k}")
        chain = [("vit_h", [vit[0]]), ("sparse_s", sparse_final),
                 ("bernoulli_matched", [r for r in controls
                                        if r["control"] == "BERNOULLI_MATCHED"]),
                 ("row_shuffled", [r for r in controls
                                   if r["control"] == "ROW_SHUFFLED"]),
                 ("learned_WB", [r for r in controls
                                  if r["control"] == "LEARNED_WB"])]
        previous_avg = previous_final = float("nan")
        for representation, group in chain:
            if not group:
                raise AssertionError("empty decomposition link")
            mean_avg = float(np.mean([float(r["average_incremental_to_task"])
                                      for r in group])) if representation in ("vit_h", "sparse_s") else float("nan")
            if np.isnan(mean_avg):
                per_seed = [r for r in control_rows if int(r["k"]) == k
                            and r["control"] == group[0]["control"]]
                seed_means = []
                for seed in sorted({r["seed"] for r in per_seed}):
                    track = [r for r in per_seed if r["seed"] == seed]
                    if len(track) != TASKS:
                        raise ValueError("incomplete control trajectory")
                    seed_means.append(np.mean([float(r["cumulative_accuracy"])
                                               for r in track]))
                mean_avg = float(np.mean(seed_means))
            final_values = np.array([float(r["cumulative_accuracy"])
                                     for r in group])
            final = float(final_values.mean())
            rows.append(dict(representation=representation, k=k,
                             n_memory_seeds=len(group),
                             avg_incremental=mean_avg, final=final,
                             final_seed_std=float(final_values.std(ddof=1))
                             if len(group) > 1 else 0.0,
                             delta_from_previous_avg=mean_avg - previous_avg,
                             delta_from_previous_final=final - previous_final,
                             delta_from_vit_avg=mean_avg - vit_avg,
                             delta_from_vit_final=final - vit_final,
                             interpretation="descriptive sequential contrast, not causal attribution"))
            previous_avg, previous_final = mean_avg, final
    write_rows(output["decomposition"], rows)
    print(json.dumps({"phase": "decomposition", "rows": len(rows)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["codes", "memory", "sparse", "vit", "controls",
                                 "merge-controls", "tau", "merge-tau",
                                 "decomposition"])
    parser.add_argument("--k", type=int)
    args = parser.parse_args()
    spec, base, cache, output = setup()
    if args.phase not in ("vit", "merge-controls", "merge-tau",
                          "decomposition") and args.k not in spec["ks"]:
        raise ValueError("k not in locked diagnostic grid")
    if args.phase in ("memory", "controls") and args.k not in spec["memory_ks"]:
        raise ValueError("memory control k not in locked grid")
    if args.phase == "tau" and args.k not in spec["tau_ks"]:
        raise ValueError("tau k not in locked grid")
    if args.phase == "codes":
        prepare_codes(base, cache, args.k)
    elif args.phase == "memory":
        prepare_memory(base, cache, args.k)
    elif args.phase == "sparse":
        sparse_only(base, cache, output["sparse"], args.k)
    elif args.phase == "vit":
        vit_reference(base, output["sparse"])
    elif args.phase == "controls":
        path = output["memory"].with_name(
            f"{output['memory'].stem}_k{args.k}{output['memory'].suffix}")
        memory_controls(base, cache, path, spec, args.k)
    elif args.phase == "merge-controls":
        merge_controls(output["memory"], spec["memory_ks"], spec["control_seeds"])
    elif args.phase == "tau":
        path = output["tau"].with_name(
            f"{output['tau'].stem}_k{args.k}{output['tau'].suffix}")
        tau_replay(base, cache, path, spec, args.k)
    elif args.phase == "merge-tau":
        merge_tau(output["tau"], spec["tau_ks"], spec["tau_grid"])
    elif args.phase == "decomposition":
        decomposition(output)


if __name__ == "__main__":
    main()
