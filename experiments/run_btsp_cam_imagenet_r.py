"""ImageNet-R cross-dataset transfer, with one writer and two explicit readouts.

Main: EXPERIMENTAL_CURRENT_WB_REPLAY_REFIT (historical *train* sparse-code replay).
Control: LITERAL_STREAMING_EQ8 (no historical feature replay or A/B transport).
Neither branch changes the formal CIFAR core/default. No ImageNet-R tuning.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from btsp_cam.backbone import PilotFrozenViT
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.feature_cache import extract_or_load_features
from btsp_cam.imagenet_r import CLASSES, INCREMENT, SEED, TASKS, ImageNetRProtocol
from btsp_cam.model import BTSPCAM
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance


MAIN = "EXPERIMENTAL_CURRENT_WB_REPLAY_REFIT"
CONTROL = "LITERAL_STREAMING_EQ8"
FIXED = dict(code_dim=8192, memory_slots=4096, k=96, q=4, tau=4 / 96,
             f_q_base=.005, beta=1., trace_decay=.01, synaptic_p=.5,
             eta=1., gamma=.1, seed=1993)


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def config_from_file(path: Path) -> tuple[dict, BTSPCAMConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = raw["protocol"]
    if (protocol["dataset"], protocol["classes"], protocol["init_cls"],
            protocol["increment"], protocol["tasks"], protocol["class_order_seed"],
            protocol["pilot_commit"]) != (
            "imagenetr", CLASSES, INCREMENT, INCREMENT, TASKS, SEED,
            "7a6e904c5bc5cb7a4e1823b3434020be27469b63"):
        raise ValueError("ImageNet-R pinned protocol changed")
    if (raw["readouts"]["main"], raw["readouts"]["control"]) != (MAIN, CONTROL):
        raise ValueError("readout identity changed")
    parsed = {key: entry["value"] for key, entry in raw["parameters"].items()}
    if parsed != FIXED or any(entry["status"] != "CROSS_DATASET_FIXED"
                              for key, entry in raw["parameters"].items()
                              if key not in ("code_dim", "memory_slots", "seed")):
        raise ValueError("frozen working parameters changed")
    cfg = BTSPCAMConfig(feature_dim=768, code_dim=FIXED["code_dim"], k=FIXED["k"],
                        memory_slots=FIXED["memory_slots"], num_classes=CLASSES,
                        base_plateau_probability=FIXED["f_q_base"], beta=FIXED["beta"],
                        flip_probability=FIXED["synaptic_p"], eta=FIXED["eta"],
                        trace_decay=FIXED["trace_decay"], tau=FIXED["tau"],
                        gamma=FIXED["gamma"], seed=SEED)
    return raw, cfg


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def comparison_rows(rows: list[dict]) -> list[dict]:
    pairs = []
    for index in range(0, len(rows), 2):
        main, control = rows[index:index + 2]
        if (main["readout"], control["readout"], main["task_id"]) != (
                MAIN, CONTROL, control["task_id"]):
            raise ValueError("readout rows are not paired on one writer task")
        pairs.append(dict(task_id=main["task_id"], seen_classes=main["seen_classes"],
                          replay_cumulative_accuracy=main["cumulative_accuracy"],
                          literal_cumulative_accuracy=control["cumulative_accuracy"],
                          replay_minus_literal_accuracy=(main["cumulative_accuracy"]
                                                         - control["cumulative_accuracy"]),
                          wb_density=main["wb_density"], gross_flips=main["gross_flips"]))
    return pairs


def sparse_metadata(protocol: ImageNetRProtocol, split: str, task: int,
                    feature_metadata: dict) -> dict:
    return dict(schema=1, dataset="imagenetr", split=split, task=task,
                sample_ids=[item.relative_path for item in protocol.task_samples(split, task)],
                labels=[item.remapped_label for item in protocol.task_samples(split, task)],
                feature_manifest_sha256=digest(feature_metadata),
                projection_seed=RNGProvenance.from_seed(SEED).projection_seed,
                k=FIXED["k"], code_dim=FIXED["code_dim"])


def save_codes(path: Path, codes: np.ndarray, labels: np.ndarray, metadata: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sparse cache: {path}")
    if (codes.dtype != np.uint8 or codes.shape != (len(labels), FIXED["code_dim"])
            or not np.all(codes.sum(axis=1) == FIXED["k"])):
        raise ValueError("sparse code is not exact-96 binary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, packed=np.packbits(codes, axis=1),
                            labels=np.asarray(labels, dtype=np.int16),
                            metadata=np.frombuffer(canonical_bytes(metadata), dtype=np.uint8))


def load_codes(path: Path, expected: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(cache["metadata"].tobytes().decode("utf-8"))
        if ({key: value for key, value in metadata.items()
             if key not in ("sample_ids", "labels")} !=
            {key: value for key, value in expected.items()
             if key not in ("sample_ids", "labels")} or
            sorted(zip(metadata["sample_ids"], metadata["labels"])) !=
            sorted(zip(expected["sample_ids"], expected["labels"]))):
            raise ValueError(f"sparse cache provenance mismatch: {path}")
        codes = np.unpackbits(cache["packed"], axis=1, count=FIXED["code_dim"])
        labels = cache["labels"].astype(np.int64)
    if codes.shape[0] != len(expected["sample_ids"]) or not np.all(codes.sum(axis=1) == FIXED["k"]):
        raise ValueError("sparse cache shape or exact-k mismatch")
    if not np.array_equal(labels, metadata["labels"]):
        raise ValueError("sparse cache label mismatch")
    return codes, labels


def feature_and_sparse(protocol: ImageNetRProtocol, split: str, task: int,
                       backbone, encoder, raw: dict, root: Path,
                       feature_dir: Path, replay_dir: Path,
                       test_code_dir: Path, *,
                       dataset_manifest_sha256: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    from torch.utils.data import DataLoader

    samples = protocol.task_samples(split, task)
    provenance = RNGProvenance.from_seed(SEED)
    order_seed = provenance.train_order_seed + task - 1 if split == "train" else provenance.train_order_seed
    aug_seed = provenance.augmentation_seed + task - 1 if split == "train" else None
    transform = raw["backbone"][f"{split}_transform"]
    if dataset_manifest_sha256 is None:
        dataset_manifest_sha256 = digest(protocol.manifest())
    feature_meta = dict(schema=1, dataset="imagenetr", split=split, task=task,
                        dataset_manifest_sha256=dataset_manifest_sha256,
                        sample_ids=[item.relative_path for item in samples],
                        labels=[item.remapped_label for item in samples],
                        transform=transform, normalization="NONE",
                        backbone=backbone.model_name,
                        pretrained_cfg=json.loads(canonical_bytes(backbone.pretrained_cfg)),
                        timm_version=backbone.timm_version,
                        output_point="final_LayerNorm_CLS_768D",
                        feature_dim=768, cache_dtype=raw["runtime"]["feature_dtype"],
                        pilot_commit=raw["protocol"]["pilot_commit"],
                        class_order=protocol.order.tolist(), sample_order_seed=int(order_seed),
                        augmentation_seed=int(aug_seed) if aug_seed is not None else None,
                        cache_policy="one_seeded_train_view_per_image" if split == "train"
                                     else "deterministic_test")
    sparse_meta = sparse_metadata(protocol, split, task, feature_meta)
    code_path = ((replay_dir if split == "train" else test_code_dir)
                 / f"{split}_task{task:02d}.npz")
    feature_path = feature_dir / f"{split}_task{task:02d}.npz"
    if code_path.exists():
        with np.load(code_path, allow_pickle=False) as cache:
            stored_meta = json.loads(cache["metadata"].tobytes().decode("utf-8"))
        return (*load_codes(code_path, sparse_meta), stored_meta)
    # Only the current task's train dataset is instantiated. Old train tasks
    # are accessed only as EXPERIMENTAL_ANALYSIS_STORAGE sparse-code files.
    if aug_seed is not None:
        torch.manual_seed(int(aug_seed))
    loader = DataLoader(protocol.dataset(split, task), batch_size=raw["runtime"]["batch_size"],
                        shuffle=split == "train", num_workers=raw["runtime"]["num_workers"],
                        generator=torch.Generator().manual_seed(int(order_seed)))
    features, labels, indices, _ = extract_or_load_features(
        backbone, loader, feature_path, feature_meta)
    if (features.shape != (len(samples), 768) or features.dtype != np.float32
            or len(np.unique(indices)) != len(samples)
            or not np.array_equal(labels, np.asarray(feature_meta["labels"])[indices])):
        raise AssertionError("feature cache sample/label provenance mismatch")
    codes = encoder.encode(features, 128)
    # Reorder metadata labels to DataLoader order for cache validation.
    sparse_meta["sample_ids"] = [sparse_meta["sample_ids"][i] for i in indices]
    sparse_meta["labels"] = labels.tolist()
    save_codes(code_path, codes, labels, sparse_meta)
    return codes, labels, sparse_meta


def ca1(codes: np.ndarray, memory_weights: np.ndarray, k: int, tau: float) -> np.ndarray:
    from scipy import sparse

    matrix = sparse.csr_matrix(codes, dtype=np.float32)
    overlap = np.asarray(matrix @ np.asarray(memory_weights.T, dtype=np.float32), dtype=np.float64)
    return np.maximum(overlap / k - tau, 0.0)


def fit_replay(model: BTSPCAM, task: int, load_training) -> StreamingRidge:
    """Explicitly non-exemplar-free historical *train sparse-code* replay."""
    ridge = StreamingRidge(model.config.memory_slots, task * INCREMENT, model.config.gamma)
    for source in range(1, task + 1):
        codes, labels = load_training(source)
        if np.any((labels < (source - 1) * INCREMENT) | (labels >= source * INCREMENT)):
            raise ValueError("replay source contains non-current/future labels")
        ridge.update(ca1(codes, model.memory.weights, model.config.k, model.config.tau), labels)
    ridge.solve()
    return ridge


def advance_task(model: BTSPCAM, task: int, codes: np.ndarray, labels: np.ndarray,
                 load_training) -> tuple[StreamingRidge, dict]:
    """One writer pass, then literal and replay readouts on the same W_B."""
    if task != model.current_task_index + 1:
        raise ValueError("task index is not the next task")
    if np.any((labels < (task - 1) * INCREMENT) | (labels >= task * INCREMENT)):
        raise ValueError("writer received labels outside current task")
    plateau = flips = up = down = 0
    for code, label in zip(codes, labels):
        log = model._learn_code(code, int(label))
        plateau += int(log["plateau_count"])
        flips += int(log["flips"])
        up += int(log["zero_to_one_flips"])
        down += int(log["one_to_zero_flips"])
    z_current = ca1(codes, model.memory.weights, model.config.k, model.config.tau)
    model.ridge.expand_classes(task * INCREMENT)
    model.ridge.update(z_current, labels)
    model.ridge.solve()
    replay = fit_replay(model, task, load_training)
    model.current_task_index = task
    model.seen_classes = np.arange(task * INCREMENT, dtype=np.int64)
    model.train_order_state = {"next_task_index": task}
    return replay, dict(task_id=task, train_samples=len(labels), plateau_events=plateau,
                        gross_flips=flips, flips_01=up, flips_10=down,
                        wb_density=float(model.memory.weights.mean()))


def evaluate(ridge: StreamingRidge, memory_weights: np.ndarray, test_sources,
             task: int, k: int, tau: float) -> dict:
    old_ok = old_n = current_ok = current_n = restricted_ok = nonzero = elements = 0
    for source, (codes, labels) in enumerate(test_sources, 1):
        if source > task:
            raise ValueError("future task test data passed to evaluation")
        z = ca1(codes, memory_weights, k, tau)
        logits = ridge.logits(z)
        pred = logits.argmax(axis=1)
        correct = int(np.count_nonzero(pred == labels))
        nonzero += int(np.count_nonzero(z))
        elements += int(z.size)
        if source == task:
            current_ok += correct
            current_n += len(labels)
            subset = logits[:, (task - 1) * INCREMENT:task * INCREMENT]
            restricted_ok += int(np.count_nonzero(subset.argmax(axis=1)
                                                   + (task - 1) * INCREMENT == labels))
        else:
            old_ok += correct
            old_n += len(labels)
    return dict(cumulative_accuracy=(old_ok + current_ok) / (old_n + current_n),
                old_class_accuracy=old_ok / old_n if old_n else None,
                current_task_accuracy=current_ok / current_n,
                current_task_restricted_accuracy=restricted_ok / current_n,
                ca1_density=nonzero / elements,
                ridge_condition=ridge.condition_number,
                test_samples=old_n + current_n)


def _memory_usage() -> dict:
    import psutil
    import torch

    info = psutil.Process().memory_info()
    return dict(process_rss_mib=info.rss / 2**20,
                cuda_peak_mib=torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available() else None)


def run(args) -> None:
    import torch

    config_path = resolve(ROOT, args.config)
    raw, cfg = config_from_file(config_path)
    data_root = resolve(ROOT, args.data_root or raw["protocol"]["dataset_root"])
    protocol = ImageNetRProtocol(data_root)
    manifest = protocol.manifest()
    manifest_hash = digest(manifest)
    output = resolve(ROOT, args.output_dir)
    if args.check_data:
        print(json.dumps(dict(status="PASS", train=len(protocol.samples["train"]),
                              test=len(protocol.samples["test"]),
                              manifest_sha256=manifest_hash)))
        return
    if not 1 <= args.max_tasks <= TASKS:
        raise ValueError(f"--max-tasks must be 1..{TASKS}")
    feature_dir = resolve(ROOT, raw["runtime"]["feature_cache_dir"])
    replay_dir = resolve(ROOT, raw["runtime"]["sparse_replay_dir"])
    test_code_dir = feature_dir.parent / "test_sparse"
    output.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    rows: list[dict] = []
    writer_rows: list[dict] = []
    start_task = 1
    if args.resume:
        checkpoint = resolve(ROOT, args.resume)
        state = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
        if state["config_sha256"] != config_hash or state["manifest_sha256"] != manifest_hash:
            raise ValueError("resume config/dataset provenance mismatch")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != state["checkpoint_sha256"]:
            raise ValueError("resume checkpoint hash mismatch")
        model = BTSPCAM.load(checkpoint)
        if not np.array_equal(model.class_order, protocol.order):
            raise ValueError("resume class order mismatch")
        rows = state["task_metrics"]
        writer_rows = state["writer_stats"]
        start_task = model.current_task_index + 1
        if len(rows) != start_task - 1 or len(writer_rows) != start_task - 1:
            raise ValueError("resume task history mismatch")
    else:
        if (output / "task_metrics.csv").exists() or list((output / "checkpoints").glob("task_*.npz")):
            raise FileExistsError("existing output requires --resume; refusing overwrite")
        model = BTSPCAM(cfg, class_order=protocol.order,
                        rng_provenance=RNGProvenance.from_seed(SEED))
    if start_task > args.max_tasks:
        raise ValueError("resume checkpoint already reached --max-tasks")
    # PILOT frozen backbone; no separate ViT implementation.
    backbone = PilotFrozenViT(device="cuda" if torch.cuda.is_available() else "cpu", pretrained=True)
    if backbone.output_dim != cfg.feature_dim:
        raise ValueError("backbone feature dimension mismatch")
    model.attach_backbone(backbone)
    if not (output / "config_snapshot.yaml").exists():
        (output / "config_snapshot.yaml").write_bytes(config_path.read_bytes())
    save_json(output / "protocol.json", dict(pilot_commit=raw["protocol"]["pilot_commit"],
               split_source="PILOT_SUPPLIED_FOLDERS_UNVERIFIED_ORIGIN",
               manifest_sha256=manifest_hash, class_order=protocol.order.tolist(),
               train_count=len(protocol.samples["train"]), test_count=len(protocol.samples["test"])))
    replay_metadata: dict[int, dict] = {}
    # Reconstruct expected cache metadata for old training sources on resume.
    for source in range(1, start_task):
        path = replay_dir / f"train_task{source:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing EXPERIMENTAL_ANALYSIS_STORAGE: {path}")
        with np.load(path, allow_pickle=False) as cache:
            replay_metadata[source] = json.loads(cache["metadata"].tobytes().decode("utf-8"))
        expected = protocol.task_samples("train", source)
        meta = replay_metadata[source]
        if (meta.get("split") != "train" or meta.get("task") != source
                or meta.get("k") != FIXED["k"]
                or sorted(zip(meta["sample_ids"], meta["labels"])) !=
                   sorted((item.relative_path, item.remapped_label) for item in expected)):
            raise ValueError("replay storage provenance is not current-task train-only")
    runtime_start = time.perf_counter()
    for task in range(start_task, args.max_tasks + 1):
        task_start = time.perf_counter()
        train_codes, train_labels, train_meta = feature_and_sparse(
            protocol, "train", task, backbone, model.encoder, raw, ROOT,
            feature_dir, replay_dir, test_code_dir,
            dataset_manifest_sha256=manifest_hash)
        replay_metadata[task] = train_meta

        def training_source(source: int):
            if not 1 <= source <= task:
                raise ValueError("future train source requested")
            return load_codes(replay_dir / f"train_task{source:02d}.npz",
                              replay_metadata[source])

        replay, writer = advance_task(model, task, train_codes, train_labels, training_source)
        test_sources = []
        for source in range(1, task + 1):
            test_codes, test_labels, _ = feature_and_sparse(
                protocol, "test", source, backbone, model.encoder, raw, ROOT,
                feature_dir, replay_dir, test_code_dir,
                dataset_manifest_sha256=manifest_hash)
            test_sources.append((test_codes, test_labels))
        for mode, ridge in ((MAIN, replay), (CONTROL, model.ridge)):
            metrics = evaluate(ridge, model.memory.weights, test_sources, task,
                               cfg.k, cfg.tau)
            previous = [row["cumulative_accuracy"] for row in rows
                        if row["readout"] == mode]
            row = dict(task_id=task, readout=mode, seen_classes=task * INCREMENT,
                       **metrics, average_incremental_accuracy=float(np.mean(
                           previous + [metrics["cumulative_accuracy"]])),
                       wb_density=writer["wb_density"],
                       plateau_events=writer["plateau_events"],
                       gross_flips=writer["gross_flips"],
                       runtime_seconds=time.perf_counter() - task_start,
                       **_memory_usage())
            rows.append(row)
        writer_rows.append(writer)
        model.task_metrics_history = rows
        checkpoint = output / "checkpoints" / f"task_{task:02d}.npz"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        model.save(checkpoint, compressed=True)
        save_json(checkpoint.with_suffix(".json"),
                  dict(task=task, config_sha256=config_hash,
                       manifest_sha256=manifest_hash,
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       readout_main=MAIN, readout_control=CONTROL,
                       replay_storage="EXPERIMENTAL_ANALYSIS_STORAGE",
                       task_metrics=rows, writer_stats=writer_rows))
        save_csv(output / "task_metrics.csv", rows)
        save_csv(output / "writer_stats.csv", writer_rows)
        save_csv(output / "readout_comparison.csv", comparison_rows(rows))
        main_row = rows[-2]
        print(json.dumps(dict(task=task, main_accuracy=main_row["cumulative_accuracy"],
                              control_accuracy=rows[-1]["cumulative_accuracy"])), flush=True)
    summary = dict(status="COMPLETE" if args.max_tasks == TASKS else "PARTIAL",
                   dataset="ImageNet-R", tasks_completed=args.max_tasks,
                   full_training_run=args.max_tasks == TASKS,
                   main_readout=MAIN, control_readout=CONTROL,
                   not_strict_exemplar_free=True, not_paper_algorithm=True,
                   manifest_sha256=manifest_hash,
                   runtime_this_invocation_seconds=time.perf_counter() - runtime_start,
                   **{mode: dict(average_incremental_accuracy=next(
                           row for row in reversed(rows) if row["readout"] == mode
                       )["average_incremental_accuracy"],
                       final=next(row for row in reversed(rows) if row["readout"] == mode))
                      for mode in (MAIN, CONTROL)})
    save_json(output / "summary.json", summary)
    save_json(output / "runtime.json", dict(**_memory_usage(),
               invocation_seconds=time.perf_counter() - runtime_start))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/btsp_cam_imagenet_r_k96.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="results/imagenet_r_k96")
    parser.add_argument("--resume", default=None, help="exact task_XX.npz checkpoint")
    parser.add_argument("--max-tasks", type=int, default=TASKS)
    parser.add_argument("--check-data", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
