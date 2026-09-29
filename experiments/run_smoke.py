"""Small synthetic pipeline test; its accuracy is not a paper comparison."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from btsp_cam.metrics import accuracy, average_incremental_accuracy
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import equal_increment_tasks, select_classes


ROOT = Path(__file__).resolve().parents[1]


def synthetic_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config["seed"])
    centers = rng.normal(size=(config["num_classes"], config["feature_dim"]))
    train_x, train_y, test_x, test_y = [], [], [], []
    for label, center in enumerate(centers):
        train_x.append(center + 0.35 * rng.normal(size=(config["train_samples_per_class"], config["feature_dim"])))
        test_x.append(center + 0.35 * rng.normal(size=(config["test_samples_per_class"], config["feature_dim"])))
        train_y.extend([label] * config["train_samples_per_class"])
        test_y.extend([label] * config["test_samples_per_class"])
    return np.vstack(train_x), np.asarray(train_y), np.vstack(test_x), np.asarray(test_y)


def main() -> None:
    config = json.loads((ROOT / "configs" / "smoke.json").read_text(encoding="utf-8"))
    model = BTSPCAM(
        feature_dim=config["feature_dim"],
        code_dim=config["code_dim"],
        k=config["k"],
        memory_slots=config["memory_slots"],
        num_classes=config["num_classes"],
        base_plateau_probability=config["base_plateau_probability"],
        beta=config["beta"],
        flip_probability=config["flip_probability"],
        eta=config["eta"],
        trace_decay=config["trace_decay"],
        tau=config["tau"],
        gamma=config["gamma"],
        seed=config["seed"],
    )
    train_x, train_y, test_x, test_y = synthetic_data(config)
    tasks = equal_increment_tasks(config["num_classes"], config["classes_per_task"], config["seed"])
    result_dir = ROOT / "results" / "smoke"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows, stage_accuracies, accuracy_matrix = [], [], []
    seen = np.array([], dtype=np.int64)
    for task_index, classes in enumerate(tasks, start=1):
        current_x, current_y = select_classes(train_x, train_y, classes)
        write_logs = model.learn_task(current_x, current_y)
        seen = np.concatenate([seen, classes])
        seen_x, seen_y = select_classes(test_x, test_y, seen)
        current_test_x, current_test_y = select_classes(test_x, test_y, classes)
        cumulative = accuracy(seen_y, model.predict(seen_x, seen))
        current = accuracy(current_test_y, model.predict(current_test_x, seen))
        stage_accuracies.append(cumulative)
        per_task = []
        for prior_classes in tasks[:task_index]:
            tx, ty = select_classes(test_x, test_y, prior_classes)
            per_task.append(accuracy(ty, model.predict(tx, seen)))
        accuracy_matrix.append(per_task)
        z = model.ca1(current_x)
        rows.append(
            {
                "task": task_index,
                "seen_classes": " ".join(map(str, seen.tolist())),
                "current_accuracy": current,
                "cumulative_accuracy": cumulative,
                "average_incremental_accuracy": average_incremental_accuracy(stage_accuracies),
                "sparse_code_density": config["k"] / config["code_dim"],
                "active_weight_fraction": model.memory.active_fraction,
                "plateau_rate": float(np.mean([row["plateau_rate"] for row in write_logs])),
                "eligibility_mean": float(model.traces.values.mean()),
                "familiarity_mean": float(np.mean([row["familiarity"] for row in write_logs])),
                "novelty_mean": float(np.mean([row["novelty"] for row in write_logs])),
                "interference_mean": float(np.mean([row["interference"] for row in write_logs])),
                "ca1_activity_density": float((z > 0).mean()),
                "ridge_shape": f"{model.ridge.a.shape[0]}x{model.ridge.a.shape[1]}",
                "ridge_condition": model.ridge.condition_number,
            }
        )

    (result_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (result_dir / "task_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (result_dir / "accuracy_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index, row in enumerate(accuracy_matrix, start=1):
            writer.writerow([index, *row])
    final = {
        "smoke_only": True,
        "final_accuracy": stage_accuracies[-1],
        "average_incremental_accuracy": average_incremental_accuracy(stage_accuracies),
        "tasks": len(tasks),
    }
    (result_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (result_dir / "seed.txt").write_text(str(config["seed"]) + "\n", encoding="utf-8")
    (result_dir / "environment.txt").write_text(
        f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\n",
        encoding="utf-8",
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "NOT_A_GIT_REPOSITORY"
    (result_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (result_dir / "parameter_audit_snapshot.md").write_text(
        (ROOT / "docs" / "PARAMETER_AUDIT.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with (result_dir / "run.log").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"result_dir": str(result_dir), "final": final, "last_task": rows[-1]}, indent=2))


if __name__ == "__main__":
    main()
