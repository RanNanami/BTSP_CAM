"""Read-only multi-k writer-state and Task1 pre-threshold drift audit.

Only existing feature/code caches, writer checkpoints, and archived CSV logs are
read. Missing k128/k160 Task1 codes are reconstructed in RAM from the frozen
feature cache and the archived RF seed; no model, writer, ridge, or ViT runs.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from btsp_cam.sparse_encoder import RandomSparseEncoder

DIAG = ROOT / "diagnostics"
FEATURE = ROOT / "data/feature_cache_btsp_cam/test_task_01.npz"
PATH_LOG = DIAG / "k_path_decomposition.csv"
HIGH_LOG = DIAG / "high_k_stress_full20.csv"
M = 8192
SLOTS = 4096
BITS = M * SLOTS
TASKS = 20
KS = (41, 64, 96, 128, 160, 1024, 6144)
OUTPUT = {
    "summary": DIAG / "k_writer_drift_readonly_summary.csv",
    "class": DIAG / "k_task1_to_task20_prototype_drift.csv",
    "trajectory": DIAG / "k_task1_drift_trajectory.csv",
    "state": DIAG / "k_writer_state_change.csv",
    "plateau": DIAG / "k_plateau_count_semantics.csv",
}
WB_DIRS = {
    96: DIAG / "k_tau_joint_cache/k96/full20",
    128: DIAG / "tau_oracle_k128",
    160: DIAG / "tau_oracle_k160",
    1024: DIAG / "high_k_mechanism_cache/k1024",
    6144: DIAG / "high_k_mechanism_cache/k6144",
}
CODE_PATHS = {
    96: DIAG / "k_tau_joint_cache/k96/full_test_task01.npz",
    1024: DIAG / "high_k_mechanism_cache/k1024/test_task01.npz",
    6144: DIAG / "high_k_mechanism_cache/k6144/test_task01.npz",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if not rows:
        raise ValueError(f"empty output {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archived_writer() -> dict[tuple[int, int], dict]:
    out = {}
    for row in read_csv(PATH_LOG):
        if row["branch"] != "evolving_aligned" or row["run"] not in {
            "k41", "k64", "k96", "k128", "k160"
        }:
            continue
        k, task = int(row["k"]), int(row["task"])
        out[(k, task)] = dict(
            density=float(row["WB_density"]),
            plateau=int(row["plateau_count_task"]),
            gross=int(row["gross_flips_task"]),
            flip01=None,
            flip10=None,
            source="k_path_decomposition.csv:evolving_aligned",
        )
    for row in read_csv(HIGH_LOG):
        if row["condition"] != "FIXED_FBASE" or int(row["k"]) not in {1024, 6144}:
            continue
        k, task = int(row["k"]), int(row["task"])
        out[(k, task)] = dict(
            density=float(row["WB_density"]),
            plateau=int(row["plateau_events"]),
            gross=int(row["gross_flips"]),
            flip01=int(row["flip_01"]),
            flip10=int(row["flip_10"]),
            source="high_k_stress_full20.csv:FIXED_FBASE",
        )
    for k in KS:
        if {task for kk, task in out if kk == k} != set(range(1, TASKS + 1)):
            raise ValueError(f"incomplete archived writer logs: k{k}")
    return out


def checkpoint_path(k: int, task: int) -> Path:
    return WB_DIRS[k] / f"WB_task{task:02d}.npz"


def checkpoint_complete(k: int) -> bool:
    return k in WB_DIRS and all(checkpoint_path(k, t).is_file()
                                for t in range(1, TASKS + 1))


def load_wb(k: int, task: int) -> tuple[np.ndarray, dict]:
    with np.load(checkpoint_path(k, task), allow_pickle=False) as state:
        if "packed" in state:
            packed = state["packed"]
            if packed.shape != (SLOTS, M // 8):
                raise AssertionError("packed WB shape")
            wb = np.unpackbits(packed, axis=1, count=M)
        else:
            wb = state["WB"]
        meta = {name: int(state[name]) for name in
                ("k", "task", "rf_seed", "write_seed", "gross_flips", "plateau_count")
                if name in state}
    if wb.shape != (SLOTS, M) or wb.dtype != np.uint8 or not np.all(wb <= 1):
        raise AssertionError("WB binary invariant")
    if "k" in meta and meta["k"] != k:
        raise AssertionError("WB k metadata mismatch")
    if "task" in meta and meta["task"] != task:
        raise AssertionError("WB task metadata mismatch")
    for key in ("rf_seed", "write_seed"):
        if key in meta and meta[key] != 1993:
            raise AssertionError("unexpected writer/RF seed")
    return wb, meta


def fixed_task1_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(FEATURE, allow_pickle=False) as item:
        h = item["features"]
        labels = item["labels"].astype(np.int64)
        indices = item["indices"].astype(np.int64)
    if h.shape != (500, 768) or labels.shape != (500,) or indices.shape != (500,):
        raise AssertionError("unexpected Task1 feature cache")
    if [int(np.sum(labels == c)) for c in range(5)] != [100] * 5:
        raise AssertionError("Task1 class labels/counts")
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("Task1 sample ID duplicate")
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    return h, labels, indices, digest


def reconstructed_codes(h: np.ndarray, ks: tuple[int, ...]) -> dict[int, np.ndarray]:
    # Mirrors RNGProvenance.from_seed(1993).projection_seed and the frozen
    # RandomSparseEncoder, without constructing BTSPCAM or a training writer.
    projection_seed = int(np.random.SeedSequence(1993).spawn(5)[2]
                          .generate_state(1, dtype=np.uint32)[0])
    encoder = RandomSparseEncoder(768, M, 96, projection_seed)
    ranks = np.empty((len(h), M), dtype=np.uint16)
    for start in range(0, len(h), 128):
        end = min(start + 128, len(h))
        scores = np.asarray(h[start:end], dtype=np.float64) @ encoder.projection.T + encoder.bias
        ranks[start:end] = np.argsort(-scores, axis=1, kind="stable")
    codes = {}
    for k in ks:
        s = np.zeros((len(h), M), dtype=np.uint8)
        np.put_along_axis(s, ranks[:, :k], 1, axis=1)
        if not np.all(s.sum(axis=1, dtype=np.int64) == k):
            raise AssertionError(f"not exact-k: {k}")
        codes[k] = s
    return codes


def validated_codes(h: np.ndarray, labels: np.ndarray, ks: tuple[int, ...]) -> tuple[dict, dict]:
    generated = reconstructed_codes(h, ks)
    sources = {}
    for k in ks:
        if k not in CODE_PATHS:
            sources[k] = "IN_MEMORY_FROM_FROZEN_VIT_CACHE;NO_SAVED_SPARSE_CACHE"
            continue
        with np.load(CODE_PATHS[k], allow_pickle=False) as item:
            cached = np.unpackbits(item["packed"], axis=1, count=M)
            cached_labels = item["labels"].astype(np.int64)
        if not np.array_equal(cached_labels, labels):
            raise AssertionError(f"Task1 labels/order mismatch k{k}")
        if not np.array_equal(cached, generated[k]):
            raise AssertionError(f"reconstructed sparse code mismatch k{k}")
        generated[k] = cached
        sources[k] = str(CODE_PATHS[k].relative_to(ROOT)) + ";BITWISE_REENCODE_VERIFIED"
    return generated, sources


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", a, b, dtype=np.float64)
    na = np.sqrt(np.einsum("ij,ij->i", a, a, dtype=np.float64))
    nb = np.sqrt(np.einsum("ij,ij->i", b, b, dtype=np.float64))
    if np.any(na == 0) or np.any(nb == 0):
        raise AssertionError("zero a-feature; cosine undefined")
    return numerator / (na * nb)


def overlap(s: np.ndarray, k: int, wb: np.ndarray) -> np.ndarray:
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    if k >= 1024:
        result = np.asarray(s, dtype=np.float32) @ wt
    else:
        result = sparse.csr_matrix(s, dtype=np.float32) @ wt
    if not (np.all(result == np.floor(result)) and result.min() >= 0 and result.max() <= k):
        raise AssertionError("overlap count not exact integer")
    return result / k


def prototype(a: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.stack([a[labels == c].mean(axis=0, dtype=np.float64)
                     for c in range(5)], axis=0)


def audit() -> dict:
    start = time.perf_counter()
    for path in OUTPUT.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    logs = archived_writer()
    h, labels, indices, sample_ids_sha = fixed_task1_rows()
    available = tuple(k for k in KS if checkpoint_complete(k))
    codes, code_sources = validated_codes(h, labels, available)
    del h
    class_rows, trajectory_rows, state_rows, plateau_rows, summary_rows = [], [], [], [], []
    for k in KS:
        present = k in available
        base_wb = None
        base_a = base_p = None
        first_density = 0.0
        proto20 = sample20 = None
        class20 = None
        changed20 = None
        density20 = None
        total_plateau = total_gross = total01 = total10 = 0
        cumulative_gross = 0
        prior_count = 0
        for task in range(1, TASKS + 1):
            log = logs[(k, task)]
            wb = None
            meta = {}
            active_count = round(log["density"] * BITS)
            if abs(active_count / BITS - log["density"]) > 1e-12:
                raise AssertionError("archived density cannot be converted to bit count")
            if present:
                wb, meta = load_wb(k, task)
                real_count = int(wb.sum(dtype=np.int64))
                if real_count != active_count:
                    raise AssertionError(f"k{k} task{task} checkpoint/log density mismatch")
                active_count = real_count
                if "gross_flips" in meta and meta["gross_flips"] != log["gross"]:
                    raise AssertionError(f"k{k} task{task} checkpoint/log gross mismatch")
                if "plateau_count" in meta and meta["plateau_count"] != log["plateau"]:
                    raise AssertionError(f"k{k} task{task} checkpoint/log plateau mismatch")
            net = active_count - prior_count
            if (log["gross"] + net) % 2 or abs(net) > log["gross"]:
                raise AssertionError(f"k{k} task{task} gross/net flip parity")
            flip01 = (log["gross"] + net) // 2
            flip10 = (log["gross"] - net) // 2
            if log["flip01"] is not None and (flip01, flip10) != (log["flip01"], log["flip10"]):
                raise AssertionError(f"k{k} task{task} archived LTP/LTD mismatch")
            total_plateau += log["plateau"]
            total_gross += log["gross"]
            total01 += flip01
            total10 += flip10
            cumulative_gross += log["gross"]
            if present:
                if task == 1:
                    base_wb = wb.copy()
                    first_density = active_count / BITS
                    base_a = overlap(codes[k], k, wb)
                    base_p = prototype(base_a, labels)
                changed = int(np.count_nonzero(base_wb != wb))
                a = base_a if task == 1 else overlap(codes[k], k, wb)
                p = base_p if task == 1 else prototype(a, labels)
                pc = cosine_rows(base_p, p)
                sc = cosine_rows(base_a, a)
                if task == 1 and (max(abs(pc - 1)) > 1e-12 or max(abs(sc - 1)) > 1e-12):
                    raise AssertionError("Task1 self-cosine not one")
                trajectory_rows.append(dict(
                    k=k, active_ratio=k / M, task=task,
                    prototype_cosine_a_mean=float(pc.mean()),
                    prototype_cosine_a_median=float(np.median(pc)),
                    sample_cosine_a_mean=float(sc.mean()),
                    WB_1_to_task_changed_bits=changed,
                    WB_1_to_task_fraction_changed=changed / BITS,
                    WB_density=active_count / BITS,
                    sample_count=len(labels), sample_ids_sha256=sample_ids_sha,
                    code_source=code_sources[k]))
                if task == 20:
                    proto20 = float(pc.mean())
                    sample20 = float(sc.mean())
                    class20 = pc
                    changed20 = changed
                    density20 = active_count / BITS
                    for c in range(5):
                        class_sc = sc[labels == c]
                        class_rows.append(dict(
                            k=k, class_id=c, source_task=1, eval_task=20,
                            sample_count=int(np.sum(labels == c)),
                            sample_ids_sha256=hashlib.sha256(indices[labels == c].tobytes()).hexdigest(),
                            prototype_cosine_a=float(pc[c]),
                            sample_cosine_a_mean=float(class_sc.mean()),
                            sample_cosine_a_median=float(np.median(class_sc)),
                            feature="PRE_THRESHOLD_A", code_source=code_sources[k]))
                del a, p, pc, sc
            state_rows.append(dict(
                k=k, task=task, checkpoint_status="AVAILABLE" if present else "CHECKPOINT_NOT_AVAILABLE",
                WB_density=active_count / BITS, WB_active_bits=active_count,
                WB_1_to_task_hamming_bits=changed if present else "",
                WB_1_to_task_fraction_changed=changed / BITS if present else "",
                plateau_slot_events=log["plateau"],
                gross_flips=log["gross"], gross_flip_01=flip01, gross_flip_10=flip10,
                net_flip_01_minus_10=net, cumulative_gross_flips=cumulative_gross,
                flip_direction_source="DERIVED_FROM_GROSS_AND_CHECKPOINT_NET" if log["flip01"] is None
                else "ARCHIVED_AND_DERIVED_AGREE", writer_log_source=log["source"]))
            plateau_rows.append(dict(
                k=k, task=task, train_samples=2500,
                slot_plateau_events=log["plateau"],
                slot_plateau_events_per_sample=log["plateau"] / 2500,
                samples_with_any_plateau="", mean_plateau_slots_per_positive_sample="",
                fraction_samples_with_any_plateau="",
                sample_level_status="PLATEAU_DENOMINATOR_NOT_RECOVERABLE_FROM_EXISTING_LOGS",
                source=log["source"]))
            prior_count = active_count
            if task in (1, 5, 10, 15, 20):
                print(f"k={k} task={task} density={active_count / BITS:.6f} "
                      f"prototype={float(proto20) if task == 20 and present else 'pending'}",
                      flush=True)
            del wb
        summary_rows.append(dict(
            k=k, active_ratio=k / M, checkpoint_status="AVAILABLE" if present else "CHECKPOINT_NOT_AVAILABLE",
            code_source=code_sources[k] if present else "",
            Task1_WB_density=first_density if present else logs[(k, 1)]["density"],
            Task20_WB_density=density20 if present else logs[(k, 20)]["density"],
            WB_1_to_20_hamming_bits=changed20 if present else "",
            WB_1_to_20_fraction_changed=changed20 / BITS if present else "",
            gross_flips_total=total_gross, flip_01_total=total01, flip_10_total=total10,
            plateau_slot_events_total=total_plateau,
            mean_slot_plateau_events_per_task=total_plateau / TASKS,
            Task1_to_20_prototype_cosine_a_mean=proto20 if present else "",
            Task1_to_20_prototype_cosine_a_median=float(np.median(class20)) if present else "",
            Task1_to_20_prototype_cosine_a_min=float(class20.min()) if present else "",
            Task1_to_20_prototype_cosine_a_max=float(class20.max()) if present else "",
            Task1_to_20_prototype_cosine_a_std=float(class20.std()) if present else "",
            fraction_prototypes_gt_090=float((class20 > 0.9).mean()) if present else "",
            fraction_prototypes_gt_095=float((class20 > 0.95).mean()) if present else "",
            fraction_prototypes_gt_099=float((class20 > 0.99).mean()) if present else "",
            Task1_to_20_sample_cosine_a_mean=sample20 if present else "",
            sample_count=500 if present else "", sample_ids_sha256=sample_ids_sha if present else "",
            plateau_sample_denominator_status="PLATEAU_DENOMINATOR_NOT_RECOVERABLE_FROM_EXISTING_LOGS"))
        del base_wb, base_a, base_p
    for name, rows in (("summary", summary_rows), ("class", class_rows),
                       ("trajectory", trajectory_rows), ("state", state_rows),
                       ("plateau", plateau_rows)):
        write_csv_new(OUTPUT[name], rows)
    return dict(runtime_seconds=time.perf_counter() - start,
                sample_ids_sha256=sample_ids_sha, available=available,
                summary=summary_rows, trajectory=trajectory_rows,
                outputs={name: str(path) for name, path in OUTPUT.items()})


if __name__ == "__main__":
    result = audit()
    print(f"runtime_seconds={result['runtime_seconds']:.3f}")
    print(f"available_k={result['available']}")
    print(f"sample_ids_sha256={result['sample_ids_sha256']}")
    for row in result["summary"]:
        print(f"k={row['k']} proto={row['Task1_to_20_prototype_cosine_a_mean']} "
              f"sample={row['Task1_to_20_sample_cosine_a_mean']} "
              f"changed={row['WB_1_to_20_fraction_changed']}")
