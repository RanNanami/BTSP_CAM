"""Read-only W_B drift oracle and fixed-map controls for CIFAR-100 Task1-5.

The oracle reuses formal W_B checkpoints but replays earlier training features.
It is deliberately not an exemplar-free method. No W_B is trained here.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np
import yaml

from run_repair_ladder_task1_5 import (
    ROOT, load_split, original_wb, overlap, geometry,
)
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder

CONFIG_PATH = ROOT / "experiments" / "configs" / "wb_drift_decomposition_task1_5.yaml"
LADDER_PATH = ROOT / "diagnostics" / "repair_ladder_task1_5.csv"


def config_and_paths() -> tuple[dict, BTSPCAMConfig, Path, Path]:
    diagnostic = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if diagnostic["tasks"] != 5 or diagnostic["ca1"] != "unthresholded_overlap":
        raise ValueError("this diagnostic is restricted to Task1-5 and a = WB s / k")
    if diagnostic["degree_swaps_per_edge"] != 5:
        raise ValueError("degree swap depth must remain the preregistered value")
    config_path = (CONFIG_PATH.parent / diagnostic["base_config"]).resolve()
    formal = BTSPCAMConfig.from_mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    if (formal.seed, formal.k, formal.memory_slots, formal.code_dim) != (1993, 41, 4096, 8192):
        raise ValueError("formal settings changed")
    oracle_path = (CONFIG_PATH.parent / diagnostic["oracle_csv"]).resolve()
    controls_path = (CONFIG_PATH.parent / diagnostic["controls_csv"]).resolve()
    if oracle_path != ROOT / "diagnostics" / "current_wb_oracle_ridge.csv":
        raise ValueError("unexpected oracle output")
    if controls_path != ROOT / "diagnostics" / "fixed_memory_controls.csv":
        raise ValueError("unexpected controls output")
    return diagnostic, formal, oracle_path, controls_path


def original_codes(config: BTSPCAMConfig) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    rng = RNGProvenance.from_seed(config.seed)
    encoder = RandomSparseEncoder(config.feature_dim, config.code_dim, config.k, rng.projection_seed)
    with np.load(ROOT / "results" / "btsp_cam_cifar100" / "checkpoint_task1.npz", allow_pickle=False) as state:
        np.testing.assert_array_equal(encoder.projection, state["projection"])
        np.testing.assert_array_equal(encoder.bias, state["bias"])
    train_s, train_y, test_s, test_y = [], [], [], []
    for task in range(1, 6):
        h, y = load_split("train", task)
        code = encoder.encode(h, config.encoder_batch_size)
        if not np.all(code.sum(axis=1) == config.k):
            raise AssertionError("non-exact sparse train code")
        train_s.append(code)
        train_y.append(y)
        h, y = load_split("test", task)
        code = encoder.encode(h, config.encoder_batch_size)
        if not np.all(code.sum(axis=1) == config.k):
            raise AssertionError("non-exact sparse test code")
        test_s.append(code)
        test_y.append(y)
    return train_s, train_y, test_s, test_y


def ridge_batch(z_parts: list[np.ndarray], labels: list[np.ndarray], gamma: float) -> StreamingRidge:
    ridge = StreamingRidge(z_parts[0].shape[1], int(labels[-1].max()) + 1, gamma)
    ridge.update(np.concatenate(z_parts), np.concatenate(labels))
    ridge.solve()
    return ridge


def ridge_stream_fixed(z_parts: list[np.ndarray], labels: list[np.ndarray], gamma: float) -> StreamingRidge:
    ridge = StreamingRidge(z_parts[0].shape[1], 0, gamma)
    for z, y in zip(z_parts, labels):
        ridge.expand_classes(int(y.max()) + 1)
        ridge.update(z, y)
    ridge.solve()
    return ridge


def accuracy_fields(logits: np.ndarray, test_y: list[np.ndarray]) -> tuple[dict, list[float]]:
    labels = np.concatenate(test_y)
    prediction = logits.argmax(axis=1)
    per_source = []
    position = 0
    for y in test_y:
        stop = position + len(y)
        per_source.append(float(np.mean(prediction[position:stop] == y)))
        position = stop
    task = len(test_y)
    return {
        "cumulative_accuracy": float(np.mean(prediction == labels)),
        "old_accuracy": float(np.mean(prediction[:-len(test_y[-1])] == labels[:-len(test_y[-1])])) if task > 1 else float("nan"),
        "current_accuracy": per_source[-1],
    }, per_source


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no diagnostic rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def oracle_rows(config: BTSPCAMConfig, codes: tuple, ladder: dict) -> list[dict]:
    train_s, train_y, test_s, test_y = codes
    rows = []
    for task in range(1, 6):
        started = time.perf_counter()
        wb = original_wb(task, config.memory_slots, config.code_dim)
        z_train = [overlap(s, wb) for s in train_s[:task]]
        z_test = [overlap(s, wb) for s in test_s[:task]]
        combined_test = np.concatenate(z_test)
        batch = ridge_batch(z_train, train_y[:task], config.gamma)
        streamed = ridge_stream_fixed(z_train, train_y[:task], config.gamma)
        batch_logits = batch.logits(combined_test)
        stream_logits = streamed.logits(combined_test)
        batch_metrics, source_acc = accuracy_fields(batch_logits, test_y[:task])
        stream_metrics, _ = accuracy_fields(stream_logits, test_y[:task])
        weight_diff = float(np.max(np.abs(batch.weights - streamed.weights)))
        logit_diff = float(np.max(np.abs(batch_logits - stream_logits)))
        prediction_equal = bool(np.array_equal(batch_logits.argmax(axis=1), stream_logits.argmax(axis=1)))
        if weight_diff > 1e-5 or logit_diff > 1e-5 or not prediction_equal:
            raise AssertionError(f"fixed-WB batch/streaming ridge mismatch at task {task}: {weight_diff}, {logit_diff}")
        for method, metrics in (("O1_current_wb_batch", batch_metrics),
                                ("O2_current_wb_streaming", stream_metrics)):
            rows.append({
                "task": task, "method": method, "source_task": "all",
                **metrics, "source_task_accuracy": float("nan"),
                "W_out_max_abs_diff": weight_diff if method.startswith("O2") else float("nan"),
                "logit_max_abs_diff": logit_diff if method.startswith("O2") else float("nan"),
                "prediction_equal_to_O1": int(prediction_equal) if method.startswith("O2") else "",
                "source_sample_count": int(sum(map(len, test_y[:task]))),
            })
        for source_task, value in enumerate(source_acc, 1):
            rows.append({
                "task": task, "method": "O1_current_wb_batch", "source_task": source_task,
                "cumulative_accuracy": float("nan"), "old_accuracy": float("nan"),
                "current_accuracy": float("nan"), "source_task_accuracy": value,
                "W_out_max_abs_diff": float("nan"), "logit_max_abs_diff": float("nan"),
                "prediction_equal_to_O1": "", "source_sample_count": len(test_y[source_task - 1]),
            })
        # At task 5, reconstruct the official unthresholded R0 readout from
        # historical WBi features. Only that historical A/B differs from O1.
        if task == 5:
            historical_z = [overlap(train_s[i], original_wb(i + 1, config.memory_slots, config.code_dim))
                            for i in range(task)]
            official = ridge_stream_fixed(historical_z, train_y, config.gamma)
            official_metrics, official_source = accuracy_fields(official.logits(combined_test), test_y)
            if abs(official_metrics["cumulative_accuracy"] - float(ladder[("R0", 5)]["cumulative_accuracy"])) > 1e-12:
                raise AssertionError("reconstructed R0 differs from prior repair ladder")
            rows.append({
                "task": task, "method": "R0_historical_wb_streaming", "source_task": "all",
                **official_metrics, "source_task_accuracy": float("nan"),
                "W_out_max_abs_diff": float("nan"), "logit_max_abs_diff": float("nan"),
                "prediction_equal_to_O1": "", "source_sample_count": int(sum(map(len, test_y))),
            })
            for source_task, value in enumerate(official_source, 1):
                rows.append({
                    "task": task, "method": "R0_historical_wb_streaming", "source_task": source_task,
                    "cumulative_accuracy": float("nan"), "old_accuracy": float("nan"),
                    "current_accuracy": float("nan"), "source_task_accuracy": value,
                    "W_out_max_abs_diff": float("nan"), "logit_max_abs_diff": float("nan"),
                    "prediction_equal_to_O1": "", "source_sample_count": len(test_y[source_task - 1]),
                })
            del historical_z, official
        print(json.dumps({"phase": "oracle", "task": task,
                          "O1": batch_metrics["cumulative_accuracy"],
                          "O2_w_diff": weight_diff, "O2_logit_diff": logit_diff,
                          "seconds": round(time.perf_counter() - started, 1)}), flush=True)
        del z_train, z_test, combined_test, batch, streamed
        gc.collect()
    return rows


def degree_randomize(original: np.ndarray, seed: int, swaps_per_edge: int) -> tuple[np.ndarray, int, int]:
    """Bipartite double-edge swaps; exact row and column degrees preserved."""
    randomized = original.copy()
    rows, cols = np.nonzero(randomized)
    rows = rows.astype(np.int32, copy=False)
    cols = cols.astype(np.int32, copy=False)
    edge_count = len(rows)
    target = swaps_per_edge * edge_count
    rng = np.random.default_rng(seed)
    accepted = attempts = 0
    while accepted < target:
        batch = min(250_000, max(20_000, 2 * (target - accepted)))
        left = rng.integers(0, edge_count, size=batch, dtype=np.int32)
        right = rng.integers(0, edge_count, size=batch, dtype=np.int32)
        for a, b in zip(left, right):
            attempts += 1
            ra, rb = int(rows[a]), int(rows[b])
            ca, cb = int(cols[a]), int(cols[b])
            if ra == rb or ca == cb or randomized[ra, cb] or randomized[rb, ca]:
                continue
            randomized[ra, ca] = randomized[rb, cb] = 0
            randomized[ra, cb] = randomized[rb, ca] = 1
            cols[a], cols[b] = cb, ca
            accepted += 1
            if accepted >= target:
                break
    np.testing.assert_array_equal(randomized.sum(axis=0, dtype=np.int64), original.sum(axis=0, dtype=np.int64))
    np.testing.assert_array_equal(randomized.sum(axis=1, dtype=np.int64), original.sum(axis=1, dtype=np.int64))
    return randomized, accepted, attempts


def controls_rows(config: BTSPCAMConfig, diagnostic: dict, codes: tuple, ladder: dict) -> list[dict]:
    train_s, train_y, test_s, test_y = codes
    wb1 = original_wb(1, config.memory_slots, config.code_dim)
    target_density = float(wb1.mean())
    rows: list[dict] = []
    cases = [("TRAINED_WB1", None, wb1, 0, 0)]
    for seed in diagnostic["density_matched_seeds"]:
        rng = np.random.default_rng(int(seed))
        matrix = (rng.random(wb1.shape, dtype=np.float32) < target_density).astype(np.uint8)
        cases.append(("RANDOM_DENSITY_MATCHED", int(seed), matrix, 0, 0))
    for seed in diagnostic["degree_matched_seeds"]:
        started = time.perf_counter()
        matrix, accepted, attempts = degree_randomize(wb1, int(seed), diagnostic["degree_swaps_per_edge"])
        cases.append(("DEGREE_MATCHED_RANDOM", int(seed), matrix, accepted, attempts))
        print(json.dumps({"phase": "degree_swap", "seed": seed, "accepted": accepted,
                          "attempts": attempts, "seconds": round(time.perf_counter()-started, 1)}), flush=True)
    for control, seed, weights, accepted, attempts in cases:
        started = time.perf_counter()
        ridge = StreamingRidge(config.memory_slots, 0, config.gamma)
        exact_row = bool(np.array_equal(weights.sum(axis=1), wb1.sum(axis=1)))
        exact_col = bool(np.array_equal(weights.sum(axis=0), wb1.sum(axis=0)))
        actual_density = float(weights.mean())
        for task in range(1, 6):
            train_a = overlap(train_s[task - 1], weights)
            ridge.expand_classes(int(train_y[task - 1].max()) + 1)
            ridge.update(train_a, train_y[task - 1])
            ridge.solve()
            test_a = overlap(np.concatenate(test_s[:task]), weights)
            labels = np.concatenate(test_y[:task])
            logits = ridge.logits(test_a)
            prediction = logits.argmax(axis=1)
            current = len(test_y[task - 1])
            current_logits = logits[-current:]
            current_y = test_y[task - 1]
            five_way = current_logits[:, (task - 1) * 5 : task * 5].argmax(axis=1) + (task - 1) * 5
            same, diff, margin = geometry(test_a[-current:], current_y)
            if task == 5:
                seen_same, seen_diff, seen_margin = geometry(test_a, labels)
            else:
                seen_same = seen_diff = seen_margin = float("nan")
            rows.append({
                "control": control, "seed": seed if seed is not None else "original_WB1",
                "task": task, "current_5way_accuracy": float(np.mean(five_way == current_y)),
                "current_accuracy": float(np.mean(prediction[-current:] == current_y)),
                "old_accuracy": float(np.mean(prediction[:-current] == labels[:-current])) if task > 1 else float("nan"),
                "cumulative_accuracy": float(np.mean(prediction == labels)),
                "same_cos": same, "diff_cos": diff, "margin": margin,
                "seen_same_cos": seen_same, "seen_diff_cos": seen_diff, "seen_margin": seen_margin,
                "target_WB_density": target_density, "actual_WB_density": actual_density,
                "row_degree_exact": int(exact_row), "column_degree_exact": int(exact_col),
                "swaps_per_edge": diagnostic["degree_swaps_per_edge"] if control == "DEGREE_MATCHED_RANDOM" else 0,
                "swaps_accepted": accepted, "swaps_attempted": attempts,
            })
            if control == "TRAINED_WB1":
                prior = ladder[("R1", task)]
                if abs(rows[-1]["cumulative_accuracy"] - float(prior["cumulative_accuracy"])) > 1e-12:
                    raise AssertionError(f"trained-WB1 control differs from prior R1 at task {task}")
                if abs(rows[-1]["current_5way_accuracy"] - float(prior["current_5way_accuracy"])) > 1e-12:
                    raise AssertionError(f"trained-WB1 5-way differs from prior R1 at task {task}")
        print(json.dumps({"phase": "fixed_control", "control": control, "seed": seed,
                          "task5_cumulative": rows[-1]["cumulative_accuracy"],
                          "task5_5way": rows[-1]["current_5way_accuracy"],
                          "seconds": round(time.perf_counter()-started, 1)}), flush=True)
        del ridge
        gc.collect()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("oracle", "controls"), required=True)
    args = parser.parse_args()
    diagnostic, config, oracle_path, controls_path = config_and_paths()
    with LADDER_PATH.open(newline="", encoding="utf-8") as stream:
        ladder = {(r["branch"], int(r["task"])): r for r in csv.DictReader(stream)}
    if len(ladder) != 25:
        raise ValueError("repair ladder baseline missing")
    codes = original_codes(config)
    if args.phase == "oracle":
        rows = oracle_rows(config, codes, ladder)
        path = oracle_path
    else:
        rows = controls_rows(config, diagnostic, codes, ladder)
        path = controls_path
    write_rows(path, rows)
    print(json.dumps({"phase": args.phase, "output": str(path), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
