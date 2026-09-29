"""Read-only Task1-5 CA1 threshold and memory-drift diagnostics.

Only diagnostic CSV files are written. Formal checkpoints, configuration,
memory, model implementation, and feature caches are never changed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import yaml

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder
from run_ca1_diagnostics_round1 import overlap_from_codes, threshold_overlap


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "btsp_cam_cifar100.yaml"
CACHE = ROOT / "data" / "feature_cache_btsp_cam"
DIAGNOSTICS = ROOT / "diagnostics"
WB_DIR = DIAGNOSTICS / "wb_checkpoints"
FORMAL = ROOT / "results" / "btsp_cam_cifar100"
TAUS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
TASKS = range(1, 6)


def read_cache(split: str, task: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(CACHE / f"{split}_task_{task:02d}.npz", allow_pickle=False) as data:
        return data["features"].copy(), data["labels"].copy()


def read_memory(task: int, config: BTSPCAMConfig) -> tuple[np.ndarray, np.ndarray]:
    with np.load(WB_DIR / f"WB_task{task:02d}.npz", allow_pickle=False) as data:
        wb = data["weights"].copy()
        assert int(data["task"]) == task
        assert int(data["seed"]) == config.seed
        assert int(data["k"]) == config.k
    with np.load(FORMAL / f"checkpoint_task{task}.npz", allow_pickle=False) as data:
        assert np.array_equal(wb, data["memory_weights"])
        ridge_weights = data["ridge_weights"].copy()
    assert wb.shape == (config.memory_slots, config.code_dim)
    assert ridge_weights.shape == (config.memory_slots, task * 5)
    return wb, ridge_weights


def accuracy_row(
    labels: np.ndarray, logits: np.ndarray, task: int
) -> dict[str, float]:
    predictions = np.asarray(logits).argmax(axis=1)
    new = (labels >= (task - 1) * 5) & (labels < task * 5)
    old = labels < (task - 1) * 5
    assert np.count_nonzero(new) == 500
    return {
        # Identical by design in this CSV: both are seen-class accuracy on
        # current-task test examples, matching the previous diagnostic CSV.
        "current_accuracy": float(np.mean(predictions[new] == labels[new])),
        "cumulative_accuracy": float(np.mean(predictions == labels)),
        "old_accuracy": (
            float(np.mean(predictions[old] == labels[old]))
            if np.any(old)
            else float("nan")
        ),
        "new_accuracy": float(np.mean(predictions[new] == labels[new])),
    }


def activation_stats(z: np.ndarray) -> dict[str, float]:
    # Density is over [test samples, slots]; fraction_nonzero is the fraction
    # of test samples with at least one active CA1 slot.
    return {
        "ca1_density": float(np.mean(z > 0)),
        "mean_z": float(np.mean(z)),
        "p99_z": float(np.quantile(z, 0.99)),
        "fraction_nonzero": float(np.mean(np.any(z > 0, axis=1))),
    }


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    config = BTSPCAMConfig.from_mapping(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    projection_seed = RNGProvenance.from_seed(config.seed).projection_seed
    encoder = RandomSparseEncoder(
        config.feature_dim, config.code_dim, config.k, projection_seed
    )

    train_codes: dict[int, np.ndarray] = {}
    train_labels: dict[int, np.ndarray] = {}
    test_codes: dict[int, np.ndarray] = {}
    test_labels: dict[int, np.ndarray] = {}
    memory: dict[int, np.ndarray] = {}
    frozen_ridge: dict[int, np.ndarray] = {}
    for task in TASKS:
        train_h, train_y = read_cache("train", task)
        test_h, test_y = read_cache("test", task)
        train_codes[task] = encoder.encode(train_h, config.encoder_batch_size)
        train_labels[task] = train_y
        test_codes[task] = encoder.encode(test_h, config.encoder_batch_size)
        test_labels[task] = test_y
        assert np.all(train_codes[task].sum(axis=1) == config.k)
        assert np.all(test_codes[task].sum(axis=1) == config.k)
        memory[task], frozen_ridge[task] = read_memory(task, config)
        print(json.dumps({"stage": "loaded", "task": task}), flush=True)

    # Exact same-stream feature origin as the formal ridge update: current
    # task's training examples are encoded using the W_B at that task's end.
    train_a = {
        task: overlap_from_codes(train_codes[task], memory[task]) for task in TASKS
    }
    eval_a: dict[int, np.ndarray] = {}
    eval_y: dict[int, np.ndarray] = {}
    for task in TASKS:
        cumulative_s = np.concatenate([test_codes[t] for t in range(1, task + 1)])
        eval_a[task] = overlap_from_codes(cumulative_s, memory[task])
        eval_y[task] = np.concatenate([test_labels[t] for t in range(1, task + 1)])
        assert eval_a[task].shape == (500 * task, config.memory_slots)
    print(json.dumps({"stage": "overlaps_ready"}), flush=True)

    tau_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for tau in TAUS:
        ridge = StreamingRidge(config.memory_slots, 0, config.gamma)
        for task in TASKS:
            z_train = threshold_overlap(train_a[task], tau)
            ridge.expand_classes(task * 5)
            ridge.update(z_train, train_labels[task])
            ridge.solve()
            z_eval = threshold_overlap(eval_a[task], tau)
            stats = activation_stats(z_eval)
            common = {"checkpoint_task": task, "tau": tau, "method": "threshold_sweep"}
            refit = {
                **common,
                "mode": "refit_ridge",
                **accuracy_row(eval_y[task], ridge.logits(z_eval), task),
                **stats,
            }
            frozen = {
                **common,
                "mode": "frozen_ridge",
                **accuracy_row(eval_y[task], z_eval @ frozen_ridge[task], task),
                **stats,
            }
            tau_rows.extend((refit, frozen))
            if tau == 0.0:
                # This duplicates z_tau=0 by construction, as an explicit
                # named unthresholded reference for downstream comparisons.
                baseline_rows.append({**refit, "tau": "", "method": "unthresholded"})
            print(
                json.dumps(
                    {
                        "stage": "tau_sweep",
                        "task": task,
                        "tau": tau,
                        "refit_cumulative": refit["cumulative_accuracy"],
                        "frozen_cumulative": frozen["cumulative_accuracy"],
                    }
                ),
                flush=True,
            )
    tau_rows.extend(baseline_rows)

    # C1/C2 are deliberately 5-way Task1 probes with unchanged Task1 data.
    # Recomputed mode uses historical Task1 training data only as an offline
    # oracle control. It is not a valid no-replay CIL update.
    task1_train_y = train_labels[1]
    task1_test_y = test_labels[1]
    task1_eval_base = eval_a[1]
    base_ridge = StreamingRidge(config.memory_slots, 5, config.gamma)
    base_ridge.update(train_a[1], task1_train_y)
    base_ridge.solve()
    drift_rows: list[dict] = []
    for task in TASKS:
        a_test = overlap_from_codes(test_codes[1], memory[task])
        if task == 1:
            assert np.array_equal(a_test, task1_eval_base)
            current_ridge = base_ridge
        else:
            a_train = overlap_from_codes(train_codes[1], memory[task])
            current_ridge = StreamingRidge(config.memory_slots, 5, config.gamma)
            current_ridge.update(a_train, task1_train_y)
            current_ridge.solve()
        baseline64 = task1_eval_base.astype(np.float64)
        current64 = a_test.astype(np.float64)
        numerator = np.einsum("ij,ij->i", baseline64, current64)
        denominator = np.linalg.norm(baseline64, axis=1) * np.linalg.norm(
            current64, axis=1
        )
        cosine = np.divide(
            numerator,
            denominator,
            out=np.full(len(task1_test_y), np.nan),
            where=denominator > 0,
        )
        assert np.all(np.isfinite(cosine))
        common = {
            "eval_WB": task,
            "mean_cosine_to_WB1": float(np.mean(cosine)),
            "feature_density": float(np.mean(a_test > 0)),
        }
        frozen_accuracy = float(
            np.mean(base_ridge.logits(a_test).argmax(axis=1) == task1_test_y)
        )
        recomputed_accuracy = float(
            np.mean(current_ridge.logits(a_test).argmax(axis=1) == task1_test_y)
        )
        drift_rows.extend(
            (
                {
                    "train_WB": 1,
                    **common,
                    "ridge_mode": "frozen",
                    "accuracy": frozen_accuracy,
                },
                {
                    "train_WB": task,
                    **common,
                    "ridge_mode": "recomputed",
                    "accuracy": recomputed_accuracy,
                },
            )
        )
        print(
            json.dumps(
                {
                    "stage": "drift",
                    "eval_WB": task,
                    "frozen_accuracy": frozen_accuracy,
                    "recomputed_accuracy": recomputed_accuracy,
                }
            ),
            flush=True,
        )

    tau_columns = [
        "checkpoint_task", "tau", "method", "mode", "current_accuracy",
        "cumulative_accuracy", "old_accuracy", "new_accuracy", "ca1_density",
        "mean_z", "p99_z", "fraction_nonzero",
    ]
    drift_columns = [
        "train_WB", "eval_WB", "ridge_mode", "accuracy",
        "mean_cosine_to_WB1", "feature_density",
    ]
    write_csv(DIAGNOSTICS / "tau_sweep.csv", tau_rows, tau_columns)
    write_csv(DIAGNOSTICS / "wb_drift_isolation.csv", drift_rows, drift_columns)
    print(
        json.dumps(
            {
                "stage": "complete",
                "tau_rows": len(tau_rows),
                "drift_rows": len(drift_rows),
                "runtime_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
