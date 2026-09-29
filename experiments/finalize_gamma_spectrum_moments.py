"""Fill exact all-seen train L2/active moments in gamma spectrum CSV.

This is a read-only diagnostic replay of cached features.  It does not feed
any historical example into the coordinate-aligned C/D update or ridge fit.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder
from run_repair_ladder_task1_5 import load_split
from run_unresolved_parameter_screen import overlap
from run_gamma_spectral_audit import wb_checkpoint

PATH = ROOT / "diagnostics/gamma_spectrum.csv"
CFG = ROOT / "configs/btsp_cam_cifar100.yaml"


def main() -> None:
    with PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 15:
        raise ValueError(f"expected 15 snapshot rows, found {len(rows)}")
    config = BTSPCAMConfig.from_mapping(yaml.safe_load(CFG.read_text(encoding="utf-8")))
    for k in [41, 128, 160]:
        selected = sorted((r for r in rows if int(r["k"]) == k), key=lambda r: int(r["task"]))
        if len(selected) != 5:
            raise ValueError(f"five snapshots missing for k={k}")
        encoder = RandomSparseEncoder(config.feature_dim, config.code_dim, k,
                                      RNGProvenance.from_seed(config.seed).projection_seed)
        historical_codes: list[sparse.csr_matrix] = []
        target_by_task = {int(row["task"]): row for row in selected}
        for task in range(1, 21):
            h, _ = load_split("train", task)
            historical_codes.append(sparse.csr_matrix(encoder.encode(h, config.encoder_batch_size)))
            del h
            if task not in target_by_task:
                continue
            wb = wb_checkpoint(k, task)
            wt = np.ascontiguousarray(wb.T, dtype=np.float32)
            norm_sum = 0.0
            active_sum = 0.0
            active_count = 0
            sample_count = 0
            for codes in historical_codes:
                for start in range(0, codes.shape[0], 250):
                    z = overlap(codes[start:start + 250], wt, k)
                    norm_sum += float(np.linalg.norm(z, axis=1).sum())
                    active_sum += float(z.sum())
                    active_count += int(np.count_nonzero(z))
                    sample_count += len(z)
            row = target_by_task[task]
            if sample_count != int(row["N"]):
                raise AssertionError("sample count mismatch")
            row["Z_mean_l2_per_train_sample"] = norm_sum / sample_count
            row["Z_mean_active_magnitude_all_train"] = active_sum / active_count
            row["Z_density_all_train"] = active_count / (sample_count * wb.shape[0])
            print(f"k={k} task={task:02d} train_mean_l2={norm_sum/sample_count:.6f}", flush=True)
    fields = list(rows[0])
    for field in ["Z_mean_active_magnitude_all_train", "Z_density_all_train"]:
        if field not in fields:
            fields.append(field)
    temp = PATH.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(PATH)


if __name__ == "__main__":
    main()
