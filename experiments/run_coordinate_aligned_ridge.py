"""Validate coordinate-aligned C/D against current-WB replay oracle, Task1-5."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_repair_ladder_task1_5 import ROOT, load_split, original_wb, overlap
from run_wb_drift_decomposition import ridge_batch, accuracy_fields
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder

CONFIG_PATH = ROOT / "experiments" / "configs" / "coordinate_aligned_ridge_task1_5.yaml"
ORACLE_CSV = ROOT / "diagnostics" / "current_wb_oracle_ridge.csv"
LADDER_CSV = ROOT / "diagnostics" / "repair_ladder_task1_5.csv"


def main() -> None:
    experiment = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if experiment["tasks"] != 5 or experiment["variant"] != "coordinate_aligned_ridge":
        raise ValueError("this run is restricted to the five-task experimental variant")
    if experiment["readout"] != "unthresholded_overlap":
        raise ValueError("thresholded CA1 is not part of this experiment")
    formal_path = (CONFIG_PATH.parent / experiment["base_config"]).resolve()
    config = BTSPCAMConfig.from_mapping(yaml.safe_load(formal_path.read_text(encoding="utf-8")))
    if (config.seed, config.code_dim, config.k, config.memory_slots) != (1993, 8192, 41, 4096):
        raise ValueError("shared formal parameters differ from expected values")
    output_path = (CONFIG_PATH.parent / experiment["output_csv"]).resolve()
    if output_path != ROOT / "diagnostics" / "coordinate_aligned_ridge.csv":
        raise ValueError("unexpected diagnostic output")
    encoder = RandomSparseEncoder(config.feature_dim, config.code_dim, config.k,
                                  RNGProvenance.from_seed(config.seed).projection_seed)
    with np.load(ROOT / "results" / "btsp_cam_cifar100" / "checkpoint_task1.npz", allow_pickle=False) as state:
        np.testing.assert_array_equal(encoder.projection, state["projection"])
        np.testing.assert_array_equal(encoder.bias, state["bias"])
    variant = CoordinateAlignedRidge(config.code_dim, config.memory_slots, config.k, config.gamma)
    with ORACLE_CSV.open(newline="", encoding="utf-8") as stream:
        oracle_reference = {int(r["task"]): r for r in csv.DictReader(stream)
                            if r["method"] == "O1_current_wb_batch" and r["source_task"] == "all"}
    with LADDER_CSV.open(newline="", encoding="utf-8") as stream:
        ladder = {(r["branch"], int(r["task"])): r for r in csv.DictReader(stream)}
    if len(oracle_reference) != 5 or len(ladder) != 25:
        raise ValueError("baseline diagnostics are incomplete")

    rows = []
    for task in range(1, 6):
        task_start = time.perf_counter()
        # The variant consumes only this task's sparse codes and labels.
        current_h, current_y = load_split("train", task)
        current_s = encoder.encode(current_h, config.encoder_batch_size)
        del current_h
        variant.update(current_s, current_y)
        del current_s
        wb = original_wb(task, config.memory_slots, config.code_dim)
        a_aligned, b_aligned = variant.aligned_statistics(wb)
        aligned_ridge, aligned_solve_seconds = variant.solve(a_aligned, b_aligned)

        # Validation-only oracle: reads old cached features here, outside the
        # variant. None of these features/codes/CA1 rows enter C/D state.
        replay_z = []
        replay_y = []
        test_z = []
        test_y = []
        for source_task in range(1, task + 1):
            h, y = load_split("train", source_task)
            s = encoder.encode(h, config.encoder_batch_size)
            replay_z.append(overlap(s, wb))
            replay_y.append(y)
            del h, s
            h, y = load_split("test", source_task)
            s = encoder.encode(h, config.encoder_batch_size)
            test_z.append(overlap(s, wb))
            test_y.append(y)
            del h, s
        oracle_start = time.perf_counter()
        oracle_ridge = ridge_batch(replay_z, replay_y, config.gamma)
        oracle_solve_seconds = time.perf_counter() - oracle_start
        combined_test = np.concatenate(test_z)
        aligned_logits = aligned_ridge.logits(combined_test)
        oracle_logits = oracle_ridge.logits(combined_test)
        aligned_acc, _ = accuracy_fields(aligned_logits, test_y)
        oracle_acc, _ = accuracy_fields(oracle_logits, test_y)
        if abs(oracle_acc["cumulative_accuracy"] - float(oracle_reference[task]["cumulative_accuracy"])) > 1e-12:
            raise AssertionError(f"fresh replay oracle differs from archived O1 at task {task}")

        differences = {
            "A_max_abs_diff": float(np.max(np.abs(a_aligned - oracle_ridge.a))),
            "B_max_abs_diff": float(np.max(np.abs(b_aligned - oracle_ridge.b))),
            "ridge_weight_max_abs_diff": float(np.max(np.abs(aligned_ridge.weights - oracle_ridge.weights))),
            "logit_max_abs_diff": float(np.max(np.abs(aligned_logits - oracle_logits))),
            "accuracy_abs_diff": abs(aligned_acc["cumulative_accuracy"] - oracle_acc["cumulative_accuracy"]),
            "prediction_equal": int(np.array_equal(aligned_logits.argmax(axis=1), oracle_logits.argmax(axis=1))),
        }
        if (differences["A_max_abs_diff"] > 1e-7 or differences["B_max_abs_diff"] > 1e-8
                or differences["ridge_weight_max_abs_diff"] > 1e-6
                or differences["logit_max_abs_diff"] > 1e-6
                or differences["prediction_equal"] != 1):
            raise AssertionError(f"coordinate closure failed at task {task}: {differences}")

        rows.append({
            "task": task,
            "R0_official_evolving_cumulative_accuracy": float(ladder[("R0", task)]["cumulative_accuracy"]),
            "coordinate_aligned_cumulative_accuracy": aligned_acc["cumulative_accuracy"],
            "coordinate_aligned_old_accuracy": aligned_acc["old_accuracy"],
            "coordinate_aligned_current_accuracy": aligned_acc["current_accuracy"],
            "current_WB_replay_oracle_cumulative_accuracy": oracle_acc["cumulative_accuracy"],
            "R1_frozen_WB1_cumulative_accuracy": float(ladder[("R1", task)]["cumulative_accuracy"]),
            **differences, **variant.storage(),
            **variant.last_update, **variant.last_alignment,
            "aligned_ridge_solve_seconds": aligned_solve_seconds,
            "oracle_ridge_solve_seconds": oracle_solve_seconds,
            "task_total_seconds": time.perf_counter() - task_start,
            "C_storage_format": "CSR_float64",
            "variant_state": "C_D_only_no_historical_examples",
        })
        print(json.dumps({
            "task": task,
            "R0": rows[-1]["R0_official_evolving_cumulative_accuracy"],
            "aligned": aligned_acc["cumulative_accuracy"],
            "oracle": oracle_acc["cumulative_accuracy"],
            "A_diff": differences["A_max_abs_diff"],
            "C_nnz": variant.c.nnz,
            "C_update_seconds": variant.last_update["update_seconds"],
            "alignment_seconds": variant.last_alignment["alignment_seconds"],
            "task_seconds": round(time.perf_counter() - task_start, 1),
        }), flush=True)
        del a_aligned, b_aligned, aligned_ridge, oracle_ridge, replay_z, test_z, combined_test
        gc.collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output_path), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
