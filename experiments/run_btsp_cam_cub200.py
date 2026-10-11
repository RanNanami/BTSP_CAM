"""Frozen CIFAR-working BTSP-CAM transfer to PILOT's CUB ImageFolder protocol.

This is an isolated experimental runner. It never changes formal CIFAR files.
No CUB-specific tuning or component ablation is implemented here.
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
import psutil
from scipy import sparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upstream/LAMDA-PILOT"))

from btsp_cam.backbone import PilotFrozenViT
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.feature_cache import extract_or_load_features
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import pilot_class_order
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance
from utils.data_manager import DataManager

CONFIG = ROOT / "configs/btsp_cam_cub200_frozen.yaml"
DIAG = ROOT / "diagnostics"
INCREMENT = 10


def load_config(path: Path = CONFIG):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    frozen = yaml.safe_load((path.parent / raw["source_frozen_lock"]).resolve().read_text(encoding="utf-8"))
    formal = yaml.safe_load((path.parent / raw["source_formal_config"]).resolve().read_text(encoding="utf-8"))
    p = raw["parameters"]
    expected = {"m": frozen["m"], "slots": frozen["slots"], "k": frozen["k"],
                "q": frozen["q"], "tau": frozen["tau"], "f_base": frozen["f_base"],
                "beta": frozen["beta"], "lambda": frozen["lambda"], "p": frozen["p"],
                "gamma": frozen["gamma"], "eta": formal["parameters"]["eta"]["value"],
                "rf_seed": frozen["rf_seed"], "writer_seed": frozen["writer_seed"]}
    if p != expected or frozen["status"] != "LOCAL_WORKING_ABLATION_BASELINE":
        raise AssertionError("CUB parameters do not exactly match frozen CIFAR working lock")
    protocol = raw["protocol"]
    if (protocol["dataset"], protocol["classes"], protocol["init_cls"],
            protocol["increment"], protocol["tasks"], protocol["shuffle"],
            protocol["class_order_seed"], protocol["pilot_commit"]) != (
            "cub", 200, 10, 10, 20, True, 1993,
            "7a6e904c5bc5cb7a4e1823b3434020be27469b63"):
        raise AssertionError("CUB protocol changed")
    if raw["readout"] != "CURRENT_WB_REPLAY_REFIT" or raw["readout_paper_exact"]:
        raise AssertionError("CUB readout changed")
    base = BTSPCAMConfig.from_mapping(formal)
    config = replace(base, num_classes=200, k=p["k"], tau=p["tau"])
    if (config.code_dim, config.memory_slots, config.base_plateau_probability,
            config.beta, config.trace_decay, config.eta, config.flip_probability,
            config.gamma, config.seed) != (8192, 4096, .005, 1., .01, 1., .5, .1, 1993):
        raise AssertionError("core frozen config changed")
    return raw, config


def resolve(path: Path, value: str) -> Path:
    return (path.parent / value).resolve()


def pilot_class_tasks() -> tuple[np.ndarray, list[np.ndarray]]:
    order = pilot_class_order(200, 1993)
    tasks = [order[i:i + INCREMENT] for i in range(0, 200, INCREMENT)]
    assert len(tasks) == 20 and all(len(x) == 10 for x in tasks)
    return order, tasks


def inventory(data_root: Path) -> dict:
    """Hash bytes and paths, check PILOT ImageFolder layout without rewriting it."""
    train_root, test_root = data_root / "train", data_root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"PILOT CUB folders required: {train_root} and {test_root}")
    train, test = ImageFolder(train_root), ImageFolder(test_root)
    if len(train.classes) != 200 or train.classes != test.classes:
        raise ValueError("CUB must have 200 identical sorted class folders in train/test")
    digest = hashlib.sha256()
    hashes: dict[str, set[str]] = {"train": set(), "test": set()}
    counts = {}
    for split, dataset in (("train", train), ("test", test)):
        per_class = np.bincount(dataset.targets, minlength=200)
        if np.any(per_class == 0):
            raise ValueError(f"CUB {split} has an empty class")
        counts[split] = dict(total=len(dataset), per_class=per_class.tolist())
        for image_path, label in dataset.samples:
            source = Path(image_path)
            rel = source.relative_to(data_root).as_posix()
            image_hash = hashlib.sha256()
            with source.open("rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    image_hash.update(block)
            content = image_hash.hexdigest()
            hashes[split].add(content)
            digest.update(f"{rel}\0{label}\0{content}\n".encode("utf-8"))
    if hashes["train"] & hashes["test"]:
        raise ValueError("CUB train/test have duplicate image content")
    return dict(dataset_sha256=digest.hexdigest(), train=counts["train"],
                test=counts["test"], class_names=train.classes,
                split_source="PILOT_ImageFolder_supplied_folders_not_resplit")


def make_manager(raw: dict, data_root: Path) -> DataManager:
    expected = (ROOT / "data/cub").resolve()
    if data_root.resolve() != expected:
        raise ValueError(f"pinned PILOT CUB loader hardcodes {expected}; configure/copy data there")
    if Path.cwd().resolve() != ROOT:
        raise ValueError(f"run from project root because pinned PILOT uses ./data/cub: {ROOT}")
    return DataManager("cub", True, 1993, 10, 10, {"dataset": "cub"})


def make_model(config: BTSPCAMConfig) -> BTSPCAM:
    provenance = RNGProvenance.from_seed(1993)
    model = BTSPCAM(config, class_order=pilot_class_order(200, 1993),
                    projection_seed=provenance.projection_seed,
                    plateau_seed=provenance.plateau_seed,
                    synapse_seed=provenance.synapse_seed)
    model.ridge = None  # replay-refit creates a new readout at each task
    return model


def save_codes(path: Path, codes: np.ndarray, labels: np.ndarray, metadata: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    if codes.dtype != np.uint8 or codes.shape[1] != 8192 or not np.all(codes.sum(axis=1) == 96):
        raise AssertionError("CUB sparse code is not exact-96 binary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:
        np.savez_compressed(fh, packed=np.packbits(codes, axis=1),
                            labels=np.asarray(labels, dtype=np.int16),
                            metadata=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8))


def load_codes(path: Path, expected: dict) -> tuple[sparse.csr_matrix, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        meta = json.loads(cache["metadata"].tobytes().decode())
        if meta != expected:
            raise ValueError(f"sparse cache metadata mismatch: {path}")
        codes = np.unpackbits(cache["packed"], axis=1, count=8192)
        labels = cache["labels"].astype(np.int64)
    if not np.all(codes.sum(axis=1) == 96):
        raise AssertionError(f"non-exact-96 cache: {path}")
    return sparse.csr_matrix(codes, dtype=np.float32), labels


def ca1(codes: sparse.csr_matrix, wt: np.ndarray, k: int = 96, tau: float = 4 / 96):
    j = np.asarray(codes @ wt, dtype=np.float64)
    if np.any(j != np.floor(j)) or np.any(j > k):
        raise AssertionError("invalid binary overlap")
    a = j / k
    return a, np.maximum(a - tau, 0.0)


def fit_replay(wb: np.ndarray, task: int, gamma: float, load_training):
    """Current W_B coordinate refit; loader is invoked only for seen training sources."""
    ridge = StreamingRidge(wb.shape[0], task * INCREMENT, gamma)
    wt = np.ascontiguousarray(wb.T, dtype=np.float32)
    for source in range(1, task + 1):
        codes, labels = load_training(source)
        if labels.min() < (source - 1) * INCREMENT or labels.max() >= source * INCREMENT:
            raise AssertionError("training labels outside source task")
        _, z = ca1(codes, wt)
        ridge.update(z, labels)
    ridge.solve()
    return ridge, wt


def extract_and_encode(split: str, task: int, manager: DataManager, backbone: PilotFrozenViT,
                       encoder, raw: dict, manifest: dict, order: np.ndarray):
    runtime = raw["runtime"]
    provenance = RNGProvenance.from_seed(1993)
    indices = np.arange((task - 1) * INCREMENT, task * INCREMENT)
    mode = "train" if split == "train" else "test"
    dataset = manager.get_dataset(indices, source=split, mode=mode)
    sample_seed = provenance.train_order_seed + task - 1 if split == "train" else provenance.train_order_seed
    aug_seed = provenance.augmentation_seed + task - 1 if split == "train" else None
    if aug_seed is not None:
        torch.manual_seed(int(aug_seed))
    generator = torch.Generator().manual_seed(int(sample_seed))
    loader = DataLoader(dataset, batch_size=runtime["batch_size"], shuffle=split == "train",
                        num_workers=runtime["num_workers"], generator=generator)
    transform = "PILOT_CUB_RRC224_FLIP_TOTENSOR" if split == "train" else "PILOT_CUB_RESIZE256_BICUBIC_CENTERCROP224_TOTENSOR"
    meta = dict(schema=1, dataset="PILOT_CUB_ImageFolder", dataset_sha256=manifest["dataset_sha256"],
                dataset_sample_count=manifest[split]["total"], split=split, task=task,
                class_order=order.tolist(), original_classes=order[indices].tolist(),
                transform=transform, pilot_commit=raw["protocol"]["pilot_commit"],
                backbone=backbone.model_name, timm_version=backbone.timm_version,
                output_point="final_LayerNorm_CLS_768D", sample_order_seed=int(sample_seed),
                augmentation_seed=int(aug_seed) if aug_seed is not None else None,
                cache_policy="one_seeded_train_view_per_image" if split == "train" else "deterministic_test")
    feature_path = resolve(CONFIG, runtime["feature_cache_dir"]) / f"{split}_task{task:02d}.npz"
    features, labels, source_indices, hit = extract_or_load_features(backbone, loader, feature_path, meta)
    if labels.shape != (len(dataset),) or np.unique(source_indices).size != len(dataset):
        raise AssertionError("feature cache sample count/index mismatch")
    if not np.array_equal(np.unique(labels), indices):
        raise AssertionError("PILOT remapped labels mismatch")
    if features.shape != (len(dataset), 768) or not np.all(np.isfinite(features)):
        raise AssertionError("ViT feature shape/finite check failed")
    sparse_meta = dict(dataset_sha256=manifest["dataset_sha256"], split=split, task=task,
                       sample_count=len(dataset), class_order=order.tolist(),
                       backbone=backbone.model_name, transform=transform,
                       projection_seed=provenance.projection_seed, k=96,
                       density=96 / 8192, source_feature_cache=str(feature_path))
    sparse_path = resolve(CONFIG, runtime["sparse_cache_dir"]) / f"{split}_task{task:02d}.npz"
    if sparse_path.exists():
        codes, saved_labels = load_codes(sparse_path, sparse_meta)
        if not np.array_equal(saved_labels, labels):
            raise AssertionError("sparse cache label mismatch")
    else:
        dense = encoder.encode(features, 128)
        save_codes(sparse_path, dense, labels, sparse_meta)
        codes = sparse.csr_matrix(dense, dtype=np.float32)
    return codes, labels, sparse_meta, hit


def evaluate(ridge, wt, task: int, test_sources):
    old_ok = old_n = current_ok = current_n = five_ok = 0
    geometry = []
    for source, (codes, labels) in enumerate(test_sources, 1):
        a, z = ca1(codes, wt)
        logits = ridge.logits(z)
        pred = logits.argmax(axis=1)
        if np.any(pred >= task * INCREMENT):
            raise AssertionError("prediction outside seen classes")
        ok = int(np.count_nonzero(pred == labels))
        geometry.append(dict(source_task=source, samples=len(labels),
                             mean_a=float(a.mean()), std_a=float(a.std()),
                             P_a_le_tau=float(np.mean(a <= 4 / 96)),
                             P_a_gt_tau=float(np.mean(a > 4 / 96)),
                             CA1_nonzero_density=float(np.count_nonzero(z) / z.size),
                             CA1_zero_row_fraction=float(np.mean(~np.any(z > 0, axis=1)))))
        if source == task:
            current_ok += ok
            current_n += len(labels)
            five = logits[:, (task - 1) * INCREMENT:task * INCREMENT].argmax(axis=1)
            five_ok += int(np.count_nonzero(five + (task - 1) * INCREMENT == labels))
        else:
            old_ok += ok
            old_n += len(labels)
    total = old_n + current_n
    weighted = lambda key: sum(g[key] * g["samples"] for g in geometry) / total
    mean_a = weighted("mean_a")
    mean_a2 = sum((g["std_a"] ** 2 + g["mean_a"] ** 2) * g["samples"] for g in geometry) / total
    return dict(task=task, seen_classes=task * INCREMENT,
                test_samples_cumulative=total,
                cumulative_accuracy=(old_ok + current_ok) / total,
                old_class_accuracy=old_ok / old_n if old_n else float("nan"),
                current_task_accuracy=current_ok / current_n,
                restricted_current_task_accuracy=five_ok / current_n,
                CA1_nonzero_density=weighted("CA1_nonzero_density"),
                CA1_zero_row_fraction=weighted("CA1_zero_row_fraction"),
                mean_a=mean_a, std_a=float(np.sqrt(max(0.0, mean_a2 - mean_a**2))),
                P_a_le_tau=weighted("P_a_le_tau"), P_a_gt_tau=weighted("P_a_gt_tau")), geometry


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(max_tasks: int, *, smoke: bool):
    raw, config = load_config()
    data_root = resolve(CONFIG, raw["runtime"]["data_root"])
    manifest = inventory(data_root)
    order, tasks = pilot_class_tasks()
    manager = make_manager(raw, data_root)
    if manager.nb_tasks != 20 or not np.array_equal(manager._class_order, order):
        raise AssertionError("PILOT task count/order mismatch")
    if len(manager._train_targets) != manifest["train"]["total"] or len(manager._test_targets) != manifest["test"]["total"]:
        raise AssertionError("PILOT sample count mismatch")
    output = resolve(CONFIG, raw["runtime"]["smoke_dir"]) if smoke else DIAG
    output.mkdir(parents=True, exist_ok=True)
    if not smoke:
        marker = resolve(CONFIG, raw["runtime"]["smoke_dir"]) / "smoke_pass.json"
        if not marker.exists() or json.loads(marker.read_text(encoding="utf-8"))["dataset_sha256"] != manifest["dataset_sha256"]:
            raise AssertionError("same-dataset real CUB Task1-2 smoke must pass before Full")
    name = "cub200_smoke" if smoke else "cub200_full"
    paths = [output / f"{name}_{suffix}" for suffix in
             ("taskwise.csv", "geometry.csv", "writer.csv", "summary.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite existing CUB result")
    backbone = PilotFrozenViT(device="cuda" if torch.cuda.is_available() else "cpu", pretrained=True)
    model = make_model(config)
    rows, geometry, writer_rows = [], [], []
    test_sources = []
    sparse_meta = {}
    start = time.perf_counter()
    for task in range(1, max_tasks + 1):
        task_start = time.perf_counter()
        train_codes, train_y, train_meta, _ = extract_and_encode("train", task, manager, backbone,
                                                                 model.encoder, raw, manifest, order)
        test_codes, test_y, _, _ = extract_and_encode("test", task, manager, backbone,
                                                     model.encoder, raw, manifest, order)
        sparse_meta[task] = train_meta
        test_sources.append((test_codes, test_y))
        totals = dict(plateau_count=0, LTP_count=0, LTD_count=0)
        for row, label in zip(train_codes.toarray(), train_y):
            log = model._learn_code(row, int(label))
            totals["plateau_count"] += int(log["plateau_count"])
            totals["LTP_count"] += int(log["zero_to_one_flips"])
            totals["LTD_count"] += int(log["one_to_zero_flips"])
        def load_training(source):
            path = resolve(CONFIG, raw["runtime"]["sparse_cache_dir"]) / f"train_task{source:02d}.npz"
            return load_codes(path, sparse_meta[source])
        ridge, wt = fit_replay(model.memory.weights, task, config.gamma, load_training)
        metrics, source_geometry = evaluate(ridge, wt, task, test_sources)
        if metrics["CA1_nonzero_density"] <= 0 or ridge.last_solve_status != "ok":
            raise AssertionError("CA1 collapsed or ridge failed")
        row = {**metrics, "current_task_classes":INCREMENT,
               "train_samples_current_task":len(train_y),
               "W_B_density":float(model.memory.weights.mean()), **totals,
               "ridge_condition_number":ridge.condition_number,
               "ridge_solve_status":ridge.last_solve_status,
               "runtime_seconds":time.perf_counter() - task_start}
        rows.append(row)
        writer_rows.append(dict(task=task, train_samples=len(train_y), **totals,
                                W_B_density=row["W_B_density"]))
        geometry.extend([dict(task=task, **g) for g in source_geometry])
        print(json.dumps({"phase": "smoke" if smoke else "full", "task": task,
                          "cumulative_accuracy": row["cumulative_accuracy"],
                          "samples":row["test_samples_cumulative"]}), flush=True)
    summary = dict(status="SMOKE_PASS" if smoke else "FULL_COMPLETED", dataset="CUB-200",
                   dataset_sha256=manifest["dataset_sha256"], sample_counts={
                       "train":manifest["train"]["total"], "test":manifest["test"]["total"]},
                   readout="LOCAL_CURRENT_WB_REPLAY_REFIT_NOT_PAPER_EXACT",
                   frozen_parameters=raw["parameters"], class_order=order.tolist(),
                   tasks=max_tasks, average_incremental_accuracy=float(np.mean([r["cumulative_accuracy"] for r in rows])),
                   final_accuracy=rows[-1]["cumulative_accuracy"], runtime_seconds=time.perf_counter() - start,
                   peak_ram_mib=getattr(psutil.Process().memory_info(), "peak_wset",
                                        psutil.Process().memory_info().rss) / 2**20,
                   peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None)
    write_csv_new(paths[0], rows)
    write_csv_new(paths[1], geometry)
    write_csv_new(paths[2], writer_rows)
    with paths[3].open("x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    if smoke:
        marker = output / "smoke_pass.json"
        with marker.open("x", encoding="utf-8") as fh:
            json.dump(dict(dataset_sha256=manifest["dataset_sha256"], tasks=2), fh)
        smoke_report = DIAG / "cub200_smoke_task1_2.json"
        with smoke_report.open("x", encoding="utf-8") as fh:
            json.dump(dict(status="PASS", dataset_sha256=manifest["dataset_sha256"],
                           tasks=[dict(task=r["task"], seen_classes=r["seen_classes"],
                                       train_samples_current_task=r["train_samples_current_task"],
                                       test_samples_cumulative=r["test_samples_cumulative"],
                                       plateau_count=r["plateau_count"], LTP_count=r["LTP_count"],
                                       LTD_count=r["LTD_count"], CA1_nonzero_density=r["CA1_nonzero_density"],
                                       ridge_solve_status=r["ridge_solve_status"]) for r in rows],
                           readout="LOCAL_CURRENT_WB_REPLAY_REFIT_NOT_PAPER_EXACT"), fh, indent=2)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("inventory", "smoke", "full"), required=True)
    args = parser.parse_args()
    raw, _ = load_config()
    if args.phase == "inventory":
        data_root = resolve(CONFIG, raw["runtime"]["data_root"])
        print(json.dumps(inventory(data_root), ensure_ascii=False), flush=True)
    else:
        run(2 if args.phase == "smoke" else 20, smoke=args.phase == "smoke")


if __name__ == "__main__":
    main()
