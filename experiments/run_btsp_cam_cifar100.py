"""Run the checkpointable BTSP-CAM core on the fixed CIFAR-100 PILOT protocol."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
import yaml

from btsp_cam.backbone import PilotFrozenViT
from btsp_cam.cifar100 import IndexedRemappedSubset, load_cifar100
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.feature_cache import extract_or_load_features
from btsp_cam.metrics import average_incremental_accuracy
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import equal_increment_tasks, pilot_class_order
from btsp_cam.rng import RNGProvenance


ROOT = Path(__file__).resolve().parents[1]


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, workers: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        generator=generator,
        persistent_workers=False,
    )


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def validate_fixed_protocol(config: dict) -> None:
    protocol = config["protocol"]
    expected = {
        "dataset": "cifar224",
        "init_cls": 5,
        "increment": 5,
        "tasks": 20,
        "shuffle": True,
        "seed": 1993,
        "memory_size": 0,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"fixed PILOT protocol requires {key}={value!r}")
    backbone = config["backbone"]
    if (
        backbone.get("model") != "vit_base_patch16_224.augreg_in21k"
        or backbone.get("pretrained") is not True
        or backbone.get("frozen") is not True
        or backbone.get("preprocessing") != "PILOT_CIFAR224_NO_NORMALIZATION"
    ):
        raise ValueError("the frozen PILOT backbone/preprocessing contract changed")


def feature_metadata(
    *,
    split: str,
    task_number: int,
    original_classes: np.ndarray,
    sample_seed: int,
    augmentation_seed: int | None,
    backbone: PilotFrozenViT,
    pilot_commit: str,
) -> dict:
    metadata = {
        "schema": 1,
        "dataset": "torchvision.CIFAR100",
        "split": split,
        "task": int(task_number),
        "original_classes": [int(value) for value in original_classes],
        "sample_order_seed": int(sample_seed),
        "backbone": backbone.model_name,
        "timm_version": backbone.timm_version,
        "pilot_commit": pilot_commit,
        "output_dim": backbone.output_dim,
    }
    if split == "train":
        metadata.update(
            augmentation_seed=int(augmentation_seed),
            transform="PILOT_RANDOM_RESIZED_CROP_FLIP_TO_TENSOR",
            cache_policy="ENGINEERING_CHOICE_FIXED_ONE_AUGMENTATION_PER_IMAGE_AND_TASK",
        )
    else:
        metadata.update(
            augmentation_seed=None,
            transform="PILOT_RESIZE_CENTER_CROP_TO_TENSOR",
            cache_policy="PROTOCOL_PRESERVING_DETERMINISTIC_EVALUATION",
        )
    return metadata


def extract_task_split(
    *,
    dataset,
    split: str,
    task_index: int,
    tasks: list[np.ndarray],
    class_order: np.ndarray,
    backbone: PilotFrozenViT,
    cache_dir: Path,
    runtime: dict,
    provenance: RNGProvenance,
    pilot_commit: str,
):
    task_number = task_index + 1
    classes = tasks[task_index]
    subset = IndexedRemappedSubset(dataset, classes, class_order)
    train = split == "train"
    sample_seed = (
        provenance.train_order_seed + task_index if train else provenance.train_order_seed
    )
    augmentation_seed = provenance.augmentation_seed + task_index if train else None
    if train:
        torch.manual_seed(int(augmentation_seed))
    loader = make_loader(
        subset,
        batch_size=runtime["batch_size"],
        shuffle=train,
        seed=sample_seed,
        workers=runtime["num_workers"],
    )
    metadata = feature_metadata(
        split=split,
        task_number=task_number,
        original_classes=classes,
        sample_seed=sample_seed,
        augmentation_seed=augmentation_seed,
        backbone=backbone,
        pilot_commit=pilot_commit,
    )
    cache_path = cache_dir / f"{split}_task_{task_number:02d}.npz"
    features, labels, indices, cache_hit = extract_or_load_features(
        backbone, loader, cache_path, metadata
    )
    expected = 2500 if train else 500
    if labels.size != expected or np.unique(indices).size != expected:
        raise AssertionError(f"{split} task {task_number} does not contain {expected} unique samples")
    expected_labels = np.arange(task_index * 5, task_number * 5)
    if not np.array_equal(np.unique(labels), expected_labels):
        raise AssertionError(f"{split} task labels do not match the PILOT interval")
    return features, labels, indices, cache_hit


def existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def peak_working_set_mb() -> float:
    info = psutil.Process().memory_info()
    value = getattr(info, "peak_wset", info.rss)
    return float(value / (1024**2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "btsp_cam_cifar100.yaml")
    )
    parser.add_argument("--max-tasks", type=int, choices=range(1, 21), default=2)
    parser.add_argument("--resume", type=str)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_fixed_protocol(raw_config)
    core_config = BTSPCAMConfig.from_mapping(raw_config)
    runtime = raw_config["runtime"]
    result_dir = resolve_from_root(runtime["result_dir"])
    cache_dir = resolve_from_root(runtime["feature_cache_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    protocol = raw_config["protocol"]
    provenance = RNGProvenance.from_seed(protocol["seed"])
    class_order = pilot_class_order(100, protocol["seed"])
    tasks = equal_increment_tasks(100, protocol["increment"], protocol["seed"])
    if len(tasks) != 20 or any(len(task) != 5 for task in tasks):
        raise AssertionError("B0-Inc5 must contain twenty five-class tasks")

    data_root = resolve_from_root(runtime["data_root"])
    train_dataset = load_cifar100(str(data_root), train=True, download=True)
    test_dataset = load_cifar100(str(data_root), train=False, download=True)
    if len(train_dataset) != 50_000 or len(test_dataset) != 10_000:
        raise AssertionError("official CIFAR-100 sample counts do not match")

    backbone = PilotFrozenViT(device="cpu", pretrained=True)
    if args.resume:
        model = BTSPCAM.load(Path(args.resume).resolve(), backbone=backbone)
        if model.config.to_dict() != core_config.to_dict():
            raise ValueError("resume checkpoint model config differs from --config")
        if not np.array_equal(model.class_order, class_order):
            raise ValueError("resume checkpoint class order differs from PILOT parity order")
    else:
        model = BTSPCAM(
            core_config,
            backbone=backbone,
            class_order=class_order,
            rng_provenance=provenance,
        )
    start_task = model.current_task_index
    if args.max_tasks <= start_task:
        raise ValueError(
            f"checkpoint already completed {start_task} tasks; --max-tasks must be larger"
        )

    (result_dir / "class_order.json").write_text(
        json.dumps(class_order.tolist(), indent=2), encoding="utf-8"
    )
    resolved = dict(raw_config)
    resolved["runtime_environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "timm": backbone.timm_version,
        "scipy": __import__("scipy").__version__,
        "rng_provenance": provenance.as_dict(),
        "backbone_pretrained_cfg": backbone.pretrained_cfg,
    }
    (result_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    metrics_path = result_dir / "task_metrics.csv"
    rows = [row for row in existing_rows(metrics_path) if int(row["task"]) <= start_task]
    stage_accuracies = [
        float(item["cumulative_accuracy"]) for item in model.task_metrics_history
    ]
    run_log = result_dir / "run.log"
    if start_task == 0:
        run_log.write_text("", encoding="utf-8")

    for task_index in range(start_task, args.max_tasks):
        task_number = task_index + 1
        task_started = time.perf_counter()
        train_started = time.perf_counter()
        train_features, train_labels, _, train_cache_hit = extract_task_split(
            dataset=train_dataset,
            split="train",
            task_index=task_index,
            tasks=tasks,
            class_order=class_order,
            backbone=backbone,
            cache_dir=cache_dir,
            runtime=runtime,
            provenance=provenance,
            pilot_commit=protocol["pilot_commit"],
        )
        train_feature_seconds = time.perf_counter() - train_started

        test_parts = []
        test_cache_hits = []
        test_started = time.perf_counter()
        for seen_task_index in range(task_number):
            features, labels, _, hit = extract_task_split(
                dataset=test_dataset,
                split="test",
                task_index=seen_task_index,
                tasks=tasks,
                class_order=class_order,
                backbone=backbone,
                cache_dir=cache_dir,
                runtime=runtime,
                provenance=provenance,
                pilot_commit=protocol["pilot_commit"],
            )
            test_parts.append((features, labels))
            test_cache_hits.append(hit)
        test_features = np.concatenate([part[0] for part in test_parts])
        test_labels = np.concatenate([part[1] for part in test_parts])
        test_feature_seconds = time.perf_counter() - test_started

        learn_started = time.perf_counter()
        model.learn_task(train_features, train_labels, task_id=task_number)
        learning_seconds = time.perf_counter() - learn_started
        model.train_order_state = {
            "next_task_index": task_number,
            "last_train_order_seed": provenance.train_order_seed + task_index,
            "last_augmentation_seed": provenance.augmentation_seed + task_index,
        }

        evaluation = model.evaluate(test_features, test_labels)
        cumulative_accuracy = evaluation["accuracy"]
        current_interval = np.arange(task_index * 5, task_number * 5)
        current_mask = np.isin(test_labels, current_interval)
        current_accuracy = float(
            (evaluation["predictions"][current_mask] == test_labels[current_mask]).mean()
        )
        stage_accuracies.append(cumulative_accuracy)
        history_entry = {
            "task": task_number,
            "cumulative_accuracy": cumulative_accuracy,
        }
        model.task_metrics_history.append(history_entry)

        checkpoint = result_dir / f"checkpoint_task{task_number}.npz"
        model.save(checkpoint, compressed=bool(runtime["checkpoint_compressed"]))
        restored = BTSPCAM.load(checkpoint)
        restored_predictions = restored.predict(test_features, restored.seen_classes)
        prediction_match = bool(
            np.array_equal(evaluation["predictions"], restored_predictions)
        )
        if not prediction_match:
            raise AssertionError("save/load changed predictions")
        del restored
        gc.collect()

        summary = model.last_task_summary
        row = {
            "task": task_number,
            "seen_classes": " ".join(map(str, model.seen_classes.tolist())),
            "original_classes": " ".join(map(str, tasks[task_index].tolist())),
            "train_samples": int(train_labels.size),
            "test_samples": int(test_labels.size),
            "current_accuracy": current_accuracy,
            "cumulative_accuracy": cumulative_accuracy,
            "average_incremental_accuracy": average_incremental_accuracy(stage_accuracies),
            "mean_plateau_probability": summary["mean_plateau_probability"],
            "plateau_rate": summary["plateau_rate"],
            "memory_active_fraction": summary["memory_active_fraction"],
            "ca1_nonzero_density": summary["ca1_nonzero_density"],
            "ridge_solve_status": summary["ridge_solve_status"],
            "ridge_condition_number": summary["ridge_condition_number"],
            "train_feature_cache_hit": train_cache_hit,
            "all_test_feature_caches_hit": all(test_cache_hits),
            "train_feature_seconds": train_feature_seconds,
            "test_feature_seconds": test_feature_seconds,
            "learning_seconds": learning_seconds,
            "runtime_seconds": time.perf_counter() - task_started,
            "peak_working_set_mb": peak_working_set_mb(),
            "checkpoint": str(checkpoint),
            "checkpoint_prediction_match": prediction_match,
            "old_train_replay_samples": 0,
            "task_id_used_for_prediction": False,
        }
        rows.append(row)
        write_rows(metrics_path, rows)
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    final = {
        "status": "BTSP-CAM core model is runnable",
        "completed_tasks": model.current_task_index,
        "checkpoint": str(
            result_dir / f"checkpoint_task{model.current_task_index}.npz"
        ),
        "paper_number_parity": False,
    }
    (result_dir / "final_status.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
