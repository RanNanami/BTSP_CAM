"""Two-task full-sample CIFAR-100 protocol validation.

This is Layer A and is always labeled NOT PAPER-COMPARABLE.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from btsp_cam.backbone import PilotFrozenViT
from btsp_cam.cifar100 import IndexedRemappedSubset, load_cifar100
from btsp_cam.metrics import accuracy, average_incremental_accuracy
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import equal_increment_tasks, pilot_class_order
from btsp_cam.rng import RNGProvenance


ROOT = Path(__file__).resolve().parents[1]


def summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def tensor_batch_summary(images: torch.Tensor) -> dict:
    """Record the actual post-transform tensor entering the frozen backbone."""
    images = images.detach().cpu()
    return {
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "mean": float(images.mean()),
        "std": float(images.std(unbiased=False)),
        "min": float(images.min()),
        "max": float(images.max()),
    }


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=False,
    )


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "cifar100_protocol_smoke.json")
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config["result_label"] != "NOT PAPER-COMPARABLE":
        raise ValueError("protocol smoke must remain NOT PAPER-COMPARABLE")

    result_dir = ROOT / "results" / "cifar100_protocol_smoke"
    result_dir.mkdir(parents=True, exist_ok=True)
    provenance = RNGProvenance.from_seed(config["seed"])
    order = pilot_class_order(100, provenance.class_order_seed)
    tasks = equal_increment_tasks(100, config["increment"], config["seed"])
    if len(tasks) != 20 or any(len(task) != 5 for task in tasks):
        raise AssertionError("B0-Inc5 must yield twenty five-class tasks")

    data_root = str((ROOT / config["data_root"]).resolve())
    train_data = load_cifar100(data_root, train=True, download=config["download"])
    test_data = load_cifar100(data_root, train=False, download=config["download"])
    if len(train_data) != 50_000 or len(test_data) != 10_000:
        raise AssertionError("official CIFAR-100 sample counts do not match")

    backbone = PilotFrozenViT(device="cpu", pretrained=True)
    model = BTSPCAM(
        feature_dim=config["feature_dim"],
        code_dim=config["code_dim"],
        k=config["k"],
        memory_slots=config["memory_slots"],
        num_classes=100,
        base_plateau_probability=config["base_plateau_probability"],
        beta=config["beta"],
        flip_probability=config["flip_probability"],
        eta=config["eta"],
        trace_decay=config["trace_decay"],
        tau=config["tau"],
        gamma=config["gamma"],
        seed=config["seed"],
        projection_seed=provenance.projection_seed,
        plateau_seed=provenance.plateau_seed,
        synapse_seed=provenance.synapse_seed,
    )

    feature_stats, sparse_stats, plateau_stats = [], [], []
    memory_stats, ca1_stats, ridge_stats, metric_rows = [], [], [], []
    stage_accuracies: list[float] = []
    run_lines: list[dict] = []

    for task_index in range(config["tasks_to_run"]):
        task_number = task_index + 1
        current_original_classes = tasks[task_index]
        seen_original_classes = np.concatenate(tasks[:task_number])
        train_subset = IndexedRemappedSubset(train_data, current_original_classes, order)
        test_subset = IndexedRemappedSubset(test_data, seen_original_classes, order)
        expected_train = 500 * config["increment"]
        expected_test = 100 * config["increment"] * task_number
        if len(train_subset) != expected_train or len(test_subset) != expected_test:
            raise AssertionError("task sample counts violate the CIFAR protocol")

        train_seed = provenance.train_order_seed + task_index
        augmentation_seed = provenance.augmentation_seed + task_index
        torch.manual_seed(augmentation_seed)
        inspection_loader = make_loader(
            train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            seed=train_seed,
            num_workers=config["num_workers"],
        )
        _, input_images, _ = next(iter(inspection_loader))
        input_batch_stats = tensor_batch_summary(input_images)

        # Recreate the loader from the same named seeds so inspection does not
        # alter either augmentation draws or the shuffled sample order.
        torch.manual_seed(augmentation_seed)
        train_loader = make_loader(
            train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            seed=train_seed,
            num_workers=config["num_workers"],
        )
        train_features, train_labels, train_indices = backbone.extract_loader(train_loader)

        test_loader = make_loader(
            test_subset,
            batch_size=config["batch_size"],
            shuffle=False,
            seed=provenance.train_order_seed,
            num_workers=config["num_workers"],
        )
        test_features, test_labels, test_indices = backbone.extract_loader(test_loader)

        write_logs = model.learn_task(train_features, train_labels)
        seen_incremental_classes = np.arange(config["increment"] * task_number)
        predictions = model.predict(test_features, seen_incremental_classes)
        cumulative_accuracy = accuracy(test_labels, predictions)
        stage_accuracies.append(cumulative_accuracy)
        current_mask = np.isin(test_labels, np.arange(task_index * 5, task_number * 5))
        current_accuracy = accuracy(test_labels[current_mask], predictions[current_mask])

        train_codes = model.encoder.encode(train_features)
        z = model.ca1(train_features)
        signal_arrays = {
            key: np.asarray([entry[key] for entry in write_logs])
            for key in ("familiarity", "novelty", "interference", "plateau_probability", "plateau_rate")
        }
        feature_stats.append(
            {
                "task": task_number,
                "train_shape": list(train_features.shape),
                "test_shape": list(test_features.shape),
                "train": summary(train_features),
                "test": summary(test_features),
                "input_batch": input_batch_stats,
                "dtype": str(train_features.dtype),
                "unique_train_indices": int(np.unique(train_indices).size),
                "unique_test_indices": int(np.unique(test_indices).size),
            }
        )
        sparse_stats.append(
            {
                "task": task_number,
                "shape": list(train_codes.shape),
                "active_count": summary(train_codes.sum(axis=1)),
                "density": float(train_codes.mean()),
                "all_exact_k": bool(np.all(train_codes.sum(axis=1) == config["k"])),
            }
        )
        plateau_stats.append(
            {"task": task_number, **{key: summary(value) for key, value in signal_arrays.items()}}
        )
        memory_stats.append(
            {
                "task": task_number,
                "shape": list(model.memory.weights.shape),
                "active_fraction": model.memory.active_fraction,
                "binary": bool(set(np.unique(model.memory.weights)).issubset({0, 1})),
            }
        )
        ca1_stats.append(
            {
                "task": task_number,
                "shape": list(z.shape),
                "values": summary(z),
                "activity_density": float((z > 0).mean()),
            }
        )
        ridge_stats.append(
            {
                "task": task_number,
                "a_shape": list(model.ridge.a.shape),
                "b_shape": list(model.ridge.b.shape),
                "weights_shape": list(model.ridge.weights.shape),
                "condition_number": model.ridge.condition_number,
                "solve_status": getattr(model.ridge, "last_solve_status", "unknown"),
            }
        )
        row = {
            "task": task_number,
            "original_classes": " ".join(map(str, current_original_classes.tolist())),
            "seen_class_count": int(seen_incremental_classes.size),
            "train_sample_count": len(train_subset),
            "test_sample_count": len(test_subset),
            "current_accuracy": current_accuracy,
            "cumulative_accuracy": cumulative_accuracy,
            "average_incremental_accuracy": average_incremental_accuracy(stage_accuracies),
            "old_train_replay_samples": 0,
            "task_id_used_for_prediction": False,
        }
        metric_rows.append(row)
        run_lines.append({"event": "task_complete", **row})
        print(json.dumps(run_lines[-1]))

    write_json(result_dir / "class_order.json", order.tolist())
    config_with_runtime = dict(config)
    config_with_runtime["rng_provenance"] = provenance.as_dict()
    config_with_runtime["pilot_commit"] = "7a6e904c5bc5cb7a4e1823b3434020be27469b63"
    config_with_runtime["timm_runtime_version"] = backbone.timm_version
    config_with_runtime["backbone_pretrained_cfg"] = backbone.pretrained_cfg
    write_json(result_dir / "config.json", config_with_runtime)
    write_json(result_dir / "feature_stats.json", feature_stats)
    write_json(result_dir / "sparse_stats.json", sparse_stats)
    write_json(result_dir / "plateau_stats.json", plateau_stats)
    write_json(result_dir / "memory_stats.json", memory_stats)
    write_json(result_dir / "ca1_stats.json", ca1_stats)
    write_json(result_dir / "ridge_stats.json", ridge_stats)
    with (result_dir / "task_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    with (result_dir / "run.log").open("w", encoding="utf-8") as handle:
        for line in run_lines:
            handle.write(json.dumps(line) + "\n")
    (result_dir / "environment.txt").write_text(
        f"python={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\n"
        f"numpy={np.__version__}\ntimm={backbone.timm_version}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "result_dir": str(result_dir)}, indent=2))


if __name__ == "__main__":
    main()
