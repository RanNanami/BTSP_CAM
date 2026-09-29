"""Read-only numerical/timing check for two exact W C W^T multiplication orders."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder
from coordinate_aligned_ridge import CoordinateAlignedRidge
from run_coordinate_aligned_full20 import formal_wb
from run_repair_ladder_task1_5 import load_split


def dense_right_product(aligned: CoordinateAlignedRidge, wb: np.ndarray):
    started = time.perf_counter()
    w_sparse = sparse.csr_matrix(wb, dtype=np.float64)
    wt_dense = np.ascontiguousarray(wb.T, dtype=np.float64)
    middle = aligned.c @ wt_dense
    a = (w_sparse @ middle) / (aligned.k ** 2)
    b = np.asarray(w_sparse @ aligned.d, dtype=np.float64) / aligned.k
    return a, b, time.perf_counter() - started


def dense_blas_product(aligned: CoordinateAlignedRidge, wb: np.ndarray):
    started = time.perf_counter()
    c = torch.from_numpy(aligned.c.toarray())
    w = torch.from_numpy(np.asarray(wb, dtype=np.float64))
    a = ((w @ c) @ w.T).numpy() / (aligned.k ** 2)
    return a, time.perf_counter() - started


def main():
    config = BTSPCAMConfig.from_mapping(yaml.safe_load(
        (ROOT / "configs" / "btsp_cam_cifar100.yaml").read_text(encoding="utf-8")))
    encoder = RandomSparseEncoder(config.feature_dim, config.code_dim, config.k,
                                  RNGProvenance.from_seed(config.seed).projection_seed)
    aligned = CoordinateAlignedRidge(config.code_dim, config.memory_slots, config.k, config.gamma)
    for task in range(1, 4):
        h, y = load_split("train", task)
        aligned.update(encoder.encode(h, config.encoder_batch_size), y)
        wb = formal_wb(task, config.memory_slots, config.code_dim)
        old_a, old_b = aligned.aligned_statistics(wb)
        old_time = aligned.last_alignment["alignment_seconds"]
        new_a, new_b, new_time = dense_right_product(aligned, wb)
        blas_a, blas_time = dense_blas_product(aligned, wb)
        print(dict(task=task, old_seconds=old_time, new_seconds=new_time,
                   blas_seconds=blas_time,
                   A_max_abs_diff=float(np.max(np.abs(old_a - new_a))),
                   A_blas_max_abs_diff=float(np.max(np.abs(old_a - blas_a))),
                   B_max_abs_diff=float(np.max(np.abs(old_b - new_b)))), flush=True)


if __name__ == "__main__":
    main()
