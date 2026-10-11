"""Read-only Figure-4-style CA1 prototype drift diagnostic.

Uses frozen CIFAR-100 and CUB-200 k=96/q=4 sparse caches and W_B
checkpoints. It never trains, updates W_B, fits a readout, or edits existing
results. The metric is a local operationalization because the paper does not
fully specify its prototype construction.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "diagnostics"
K, Q, TAU, M, SLOTS, TASKS = 96, 4, 4 / 96, 8192, 4096, 20
CLASSWISE = DIAG / "figure4_style_prototype_drift_classwise.csv"
SUMMARY = DIAG / "figure4_style_prototype_drift_summary.csv"
VS_SAMPLE = DIAG / "figure4_style_prototype_vs_sample.csv"
RELATION = DIAG / "figure4_style_prototype_readout_relation.csv"
SUMMARY_JSON = DIAG / "figure4_style_prototype_drift_summary.json"
OUTPUTS = (CLASSWISE, SUMMARY, VS_SAMPLE, RELATION, SUMMARY_JSON)
METHOD = "FIGURE4_STYLE_LOCAL_OPERATIONALIZATION"
EPS = 1e-12


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_array(value: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(value).view(np.uint8).tobytes())


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError(f"empty output: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def unpack_codes(path: Path, *, metadata: bool) -> tuple[sparse.csr_matrix, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if metadata:
            meta = json.loads(data["metadata"].tobytes().decode())
            assert meta["k"] == K and meta["density"] == K / M
        codes = np.unpackbits(data["packed"], axis=1, count=M)
        labels = data["labels"].astype(np.int64)
    assert codes.dtype == np.uint8 and codes.shape[0] == len(labels)
    assert np.all(codes.sum(axis=1) == K)
    return sparse.csr_matrix(codes, dtype=np.float32), labels


def load_wb(path: Path, task: int, *, cub: bool) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        assert int(data["task"]) == task
        if not cub:
            assert int(data["k"]) == K
        wb = np.unpackbits(data["packed"], axis=1, count=M)
        expected = str(data["sha256"]) if cub else None
    assert wb.shape == (SLOTS, M) and wb.dtype == np.uint8
    digest = sha_array(wb)
    if expected is not None:
        assert digest == expected
    return wb, digest


def cifar_ids(task: int, count: int) -> list[str]:
    return [f"CIFAR100_TEST_TASK{task:02d}_CACHE_ROW{row:04d}" for row in range(count)]


def cub_ids() -> dict[int, list[str]]:
    result = {task: [] for task in range(1, TASKS + 1)}
    path = DIAG / "cub200_cache_row_provenance.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == "test":
                result[int(row["task"])].append(row)
    output = {}
    for task, rows in result.items():
        rows.sort(key=lambda row: row["sparse_row_index"])
        assert [r["sparse_row_index"] for r in rows] == list(range(len(rows)))
        output[task] = [r["sample_id"] for r in rows]
    return output


def dataset_inputs(name: str):
    if name == "CIFAR100":
        cache_root = DIAG / "k_tau_joint_cache" / "k96"
        wb_root = cache_root / "full20"
        increment = 5
        codes, labels, ids = {}, {}, {}
        for task in range(1, TASKS + 1):
            codes[task], labels[task] = unpack_codes(
                cache_root / f"full_test_task{task:02d}.npz", metadata=False)
            ids[task] = cifar_ids(task, len(labels[task]))
            assert np.unique(labels[task]).tolist() == list(range((task - 1) * increment,
                                                                  task * increment))
        wb_loader = lambda task: load_wb(wb_root / f"WB_task{task:02d}.npz", task, cub=False)
    elif name == "CUB200":
        cache_root = DIAG / "cub200_sparse_cache"
        wb_root = DIAG / "cub200_readout_wb_checkpoints"
        increment = 10
        ids = cub_ids()
        codes, labels = {}, {}
        for task in range(1, TASKS + 1):
            codes[task], labels[task] = unpack_codes(
                cache_root / f"test_task{task:02d}.npz", metadata=True)
            assert len(ids[task]) == len(labels[task])
            assert np.unique(labels[task]).tolist() == list(range((task - 1) * increment,
                                                                  task * increment))
        wb_loader = lambda task: load_wb(wb_root / f"WB_task{task:02d}.npz", task, cub=True)
    else:
        raise ValueError(name)
    assert not set().union(*(set(ids[t]) for t in ids)) == set()
    assert len(set().union(*(set(ids[t]) for t in ids))) == sum(map(len, ids.values()))
    return increment, codes, labels, ids, wb_loader


def row_unit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(x, axis=1)
    unit = np.divide(x, norm[:, None], out=np.zeros_like(x, dtype=np.float32),
                     where=norm[:, None] > 0)
    return unit, norm


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx == 0 and ny == 0:
        return 1.0
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def extract(codes: sparse.csr_matrix, wt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j = np.asarray(codes @ wt, dtype=np.float32)
    assert np.all(j == np.floor(j)) and float(j.min()) >= 0 and float(j.max()) <= K
    a = j / K
    z = np.maximum(a - TAU, 0.0)
    return a, z


def digest_ids(values: list[str]) -> str:
    return sha_bytes("\n".join(values).encode("utf-8"))


def analyze_dataset(dataset: str) -> tuple[list[dict], dict]:
    increment, codes, labels, ids, wb_loader = dataset_inputs(dataset)
    intro = {}
    class_rows = []
    wb_hashes = {}
    started = time.perf_counter()
    for checkpoint in range(1, TASKS + 1):
        wb, wb_hashes[checkpoint] = wb_loader(checkpoint)
        wt = np.ascontiguousarray(wb.T, dtype=np.float32)
        for source in range(1, checkpoint + 1):
            a, z = extract(codes[source], wt)
            z_unit, z_norm = row_unit(z)
            for class_id in range((source - 1) * increment, source * increment):
                mask = labels[source] == class_id
                sample_ids = [value for value, keep in zip(ids[source], mask) if keep]
                assert int(mask.sum()) == len(sample_ids) > 0
                p_z = z[mask].mean(axis=0, dtype=np.float64)
                p_a = a[mask].mean(axis=0, dtype=np.float64)
                p_zn = z_unit[mask].mean(axis=0, dtype=np.float64)
                if checkpoint == source:
                    intro[class_id] = dict(p_z=p_z.copy(), p_a=p_a.copy(),
                                           p_zn=p_zn.copy(), sample_unit=z_unit[mask].copy(),
                                           ids_sha256=digest_ids(sample_ids), count=len(sample_ids))
                base = intro[class_id]
                assert base["ids_sha256"] == digest_ids(sample_ids)
                current_unit = z_unit[mask]
                sample_cosines = np.einsum("ij,ij->i", base["sample_unit"], current_unit)
                # row_unit maps a zero vector to zero. Treat two zeros as identical,
                # matching the existing project drift diagnostic convention.
                initial_zero = np.linalg.norm(base["sample_unit"], axis=1) == 0
                current_zero = z_norm[mask] == 0
                sample_cosines[initial_zero & current_zero] = 1.0
                raw = cosine(base["p_z"], p_z)
                normalized_mean = cosine(base["p_zn"], p_zn)
                pre = cosine(base["p_a"], p_a)
                if checkpoint == source:
                    assert abs(raw - 1.0) < 1e-12
                    assert abs(normalized_mean - 1.0) < 1e-12
                    assert abs(pre - 1.0) < 1e-12
                    assert abs(float(sample_cosines.mean()) - 1.0) < 2e-6
                class_rows.append(dict(
                    dataset=dataset, metric_status=METHOD, source_task=source,
                    checkpoint_task=checkpoint, class_id=class_id,
                    exposure_tasks=checkpoint - source, sample_count=len(sample_ids),
                    sample_ids_sha256=base["ids_sha256"], WB_source_sha256=wb_hashes[source],
                    WB_checkpoint_sha256=wb_hashes[checkpoint], k=K, q=Q, tau=TAU,
                    feature="THRESHOLDED_Z_PRIMARY",
                    prototype_cosine_rawmean=raw,
                    prototype_cosine_sample_norm_mean=normalized_mean,
                    prototype_cosine_a=pre,
                    sample_cosine_mean=float(sample_cosines.mean()),
                    prototype_minus_sample_cosine=raw - float(sample_cosines.mean()),
                    initial_zero_sample_fraction=float(initial_zero.mean()),
                    current_zero_sample_fraction=float(current_zero.mean())))
            del a, z, z_unit, z_norm
        print(json.dumps(dict(phase="prototype_drift", dataset=dataset,
                              checkpoint=checkpoint, rows=len(class_rows))), flush=True)
        del wb, wt
        gc.collect()
    return class_rows, dict(runtime_seconds=time.perf_counter() - started,
                            WB_sha256={str(k): v for k, v in wb_hashes.items()},
                            test_samples=sum(len(v) for v in ids.values()),
                            class_count=TASKS * increment)


def aggregate(rows: list[dict], *, scope: str, dataset: str,
              checkpoint: int, source_start: int, source_end: int) -> dict:
    selected = [r for r in rows if r["dataset"] == dataset
                and r["checkpoint_task"] == checkpoint
                and source_start <= r["source_task"] <= source_end]
    if not selected:
        raise ValueError((scope, dataset, checkpoint, source_start, source_end))
    proto = np.array([r["prototype_cosine_rawmean"] for r in selected], dtype=np.float64)
    proto_norm = np.array([r["prototype_cosine_sample_norm_mean"] for r in selected])
    proto_a = np.array([r["prototype_cosine_a"] for r in selected])
    sample = np.array([r["sample_cosine_mean"] for r in selected])
    return dict(summary_scope=scope, dataset=dataset, metric_status=METHOD,
                source_task_start=source_start, source_task_end=source_end,
                checkpoint_task=checkpoint, exposure_min=checkpoint-source_end,
                exposure_max=checkpoint-source_start, class_count=len(selected),
                sample_count=sum(int(r["sample_count"]) for r in selected),
                prototype_cosine_mean=float(proto.mean()),
                prototype_cosine_median=float(np.median(proto)),
                prototype_cosine_min=float(proto.min()),
                prototype_cosine_max=float(proto.max()),
                prototype_cosine_std=float(proto.std()),
                fraction_prototype_gt_0_90=float(np.mean(proto > .90)),
                fraction_prototype_gt_0_95=float(np.mean(proto > .95)),
                fraction_prototype_gt_0_99=float(np.mean(proto > .99)),
                prototype_sample_norm_mean_cosine_mean=float(proto_norm.mean()),
                prototype_a_cosine_mean=float(proto_a.mean()),
                sample_cosine_mean=float(sample.mean()),
                prototype_minus_sample_cosine=float(proto.mean()-sample.mean()),
                key_checkpoint=checkpoint in (1, 5, 10, 15, 20))


def build_summaries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    pair_rows = []
    for dataset in ("CIFAR100", "CUB200"):
        for source in range(1, TASKS + 1):
            for checkpoint in range(source, TASKS + 1):
                pair_rows.append(aggregate(rows, scope="SOURCE_TASK_TO_CHECKPOINT",
                                           dataset=dataset, checkpoint=checkpoint,
                                           source_start=source, source_end=source))
    summaries = list(pair_rows)
    for dataset in ("CIFAR100", "CUB200"):
        summaries.append(aggregate(rows, scope="TASK1_CLASSES_TO_TASK20", dataset=dataset,
                                   checkpoint=20, source_start=1, source_end=1))
        for lo, hi in ((1, 5), (6, 10), (11, 15)):
            summaries.append(aggregate(rows, scope="SOURCE_TASK_GROUP_TO_TASK20",
                                       dataset=dataset, checkpoint=20,
                                       source_start=lo, source_end=hi))
        # Do not pool Task16-20: each has a different exposure duration.
        for source in range(16, 21):
            summaries.append(aggregate(rows, scope="LATE_SOURCE_TASK_TO_TASK20",
                                       dataset=dataset, checkpoint=20,
                                       source_start=source, source_end=source))
    return summaries, pair_rows


def readout_relation(class_rows: list[dict]) -> tuple[list[dict], dict]:
    taskwise = csv_rows(DIAG / "cub200_readout_decomposition_taskwise.csv")
    r0 = {int(r["task"]): float(r["cumulative_accuracy"]) for r in taskwise
          if r["branch"] == "R0_CURRENT_WB_REPLAY_REFIT"}
    r1 = {int(r["task"]): float(r["cumulative_accuracy"]) for r in taskwise
          if r["branch"] == "R1_LITERAL_STREAMING_EQ8"}
    assert len(r0) == len(r1) == TASKS
    output = []
    for checkpoint in range(1, TASKS + 1):
        old = [r for r in class_rows if r["dataset"] == "CUB200"
               and r["checkpoint_task"] == checkpoint and r["source_task"] < checkpoint]
        proto = float(np.mean([r["prototype_cosine_rawmean"] for r in old])) if old else float("nan")
        sample = float(np.mean([r["sample_cosine_mean"] for r in old])) if old else float("nan")
        output.append(dict(task=checkpoint, metric_status=METHOD,
                           old_class_count=len(old),
                           prototype_similarity_mean_old_classes=proto,
                           sample_similarity_mean_old_classes=sample,
                           R0_current_WB_replay_refit_accuracy=r0[checkpoint],
                           R1_literal_streaming_accuracy=r1[checkpoint],
                           R0_minus_R1_gap=r0[checkpoint]-r1[checkpoint],
                           correlation_status="DESCRIPTIVE_CORRELATION_ONLY"))
    usable = output[1:]
    gaps = np.array([r["R0_minus_R1_gap"] for r in usable])
    proto = np.array([r["prototype_similarity_mean_old_classes"] for r in usable])
    sample = np.array([r["sample_similarity_mean_old_classes"] for r in usable])
    corr = dict(task_range="2-20", status="DESCRIPTIVE_CORRELATION_ONLY",
                prototype_vs_readout_gap_pearson=float(np.corrcoef(proto, gaps)[0, 1]),
                sample_vs_readout_gap_pearson=float(np.corrcoef(sample, gaps)[0, 1]))
    for row in output:
        row.update(corr)
    return output, corr


def interpretation(summary_rows: list[dict], relation: list[dict]) -> tuple[str, str, str]:
    cub = next(r for r in summary_rows if r["dataset"] == "CUB200"
               and r["summary_scope"] == "TASK1_CLASSES_TO_TASK20")
    proto, sample = cub["prototype_cosine_mean"], cub["sample_cosine_mean"]
    gap = relation[-1]["R0_minus_R1_gap"]
    if proto > .9 and sample < .9 and gap > .1:
        case = "P1_PROTOTYPE_STABILITY_DOES_NOT_IMPLY_COORDINATE_CONSISTENCY"
    elif proto < .9:
        case = "P2_LOCAL_WRITER_DRIFT_REGIME_STRONGER_THAN_PAPER_REPORTED"
    elif proto > .9 and sample >= .9 and gap > .1:
        case = "P3_NEED_HIGHER_ORDER_STATISTICS_EXPLANATION"
    else:
        case = "P4_SLOW_DRIFT_NARRATIVE_HOLDS_FOR_THIS_TRAJECTORY"
    if proto > .9:
        slowly = "YES"
    elif cub["fraction_prototype_gt_0_90"] > 0:
        slowly = "MIXED"
    else:
        slowly = "NO"
    guarantee = "NO" if case.startswith("P1_") else "NOT_DETERMINED"
    return case, slowly, guarantee


def main() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from project root: {ROOT}")
    for path in OUTPUTS:
        if path.exists():
            raise FileExistsError(path)
    started = time.perf_counter()
    class_rows, dataset_meta = [], {}
    for dataset in ("CIFAR100", "CUB200"):
        rows, meta = analyze_dataset(dataset)
        class_rows.extend(rows)
        dataset_meta[dataset] = meta
    summaries, pair_rows = build_summaries(class_rows)
    relation, correlations = readout_relation(class_rows)
    case, slowly, guarantee = interpretation(summaries, relation)
    write_csv_new(CLASSWISE, class_rows)
    write_csv_new(SUMMARY, summaries)
    write_csv_new(VS_SAMPLE, pair_rows)
    write_csv_new(RELATION, relation)
    key = {dataset: next(r for r in summaries if r["dataset"] == dataset
                         and r["summary_scope"] == "TASK1_CLASSES_TO_TASK20")
           for dataset in ("CIFAR100", "CUB200")}
    payload = dict(status="PASS", metric_status=METHOD,
                   paper_claim="prototype similarity > 0.9",
                   paper_wording="slowly drifting CA1 space",
                   exact_paper_metric_formula="NOT_FULLY_SPECIFIED",
                   paper_metric_exactly_reproducible=False,
                   primary_local_operationalization=(
                       "class-mean thresholded CA1 test feature; same sample IDs; "
                       "cosine between introduction checkpoint and later checkpoint"),
                   parameters=dict(k=K, q=Q, tau=TAU),
                   task1_to_task20=key, correlations=correlations,
                   interpretation_case=case,
                   does_our_prototype_look_slowly_drifting=slowly,
                   does_prototype_stability_guarantee_readout_coordinate_consistency=guarantee,
                   dataset_metadata=dataset_meta,
                   row_counts=dict(classwise=len(class_rows), summary=len(summaries),
                                   prototype_vs_sample=len(pair_rows), relation=len(relation)),
                   runtime_seconds=time.perf_counter()-started)
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(status="PASS", case=case, key=key,
                          runtime_seconds=payload["runtime_seconds"])), flush=True)


if __name__ == "__main__":
    main()
