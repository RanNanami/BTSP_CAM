"""Read-only per-class eligibility snapshots from unchanged formal k=41 checkpoints."""

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

SNAPSHOTS = (1, 5, 10, 20)
OUTPUT = ROOT / "diagnostics" / "lambda_formal_k41_trace_snapshot.csv"


def main() -> None:
    settings = yaml.safe_load((ROOT / "configs" / "btsp_cam_cifar100.yaml").read_text(encoding="utf-8"))
    config = BTSPCAMConfig.from_mapping(settings)
    if (config.k, config.trace_decay, config.seed) != (41, 0.01, 1993):
        raise AssertionError("formal baseline changed")
    encoder = RandomSparseEncoder(config.feature_dim, config.code_dim, config.k,
                                  RNGProvenance.from_seed(config.seed).projection_seed)
    last_seen = np.full(config.num_classes, -1, dtype=np.int64)
    rows = []
    sample_index = 0
    for task in range(1, max(SNAPSHOTS) + 1):
        _, train_y = load_split("train", task)
        for y in train_y:
            last_seen[int(y)] = sample_index
            sample_index += 1
        if task not in SNAPSHOTS:
            continue
        with np.load(ROOT / "results" / "btsp_cam_cifar100" /
                     f"checkpoint_task{task}.npz", allow_pickle=False) as state:
            trace = state["eligibility_values"][:task * 5].copy()
            np.testing.assert_array_equal(encoder.projection, state["projection"])
            np.testing.assert_array_equal(encoder.bias, state["bias"])
        norms = np.linalg.norm(trace, axis=1)
        own_means = np.zeros(task * 5)
        rival_means = np.zeros(task * 5)
        for source in range(1, task + 1):
            h, labels = load_split("test", source)
            codes = sparse.csr_matrix(encoder.encode(h, config.encoder_batch_size))
            dots = np.asarray(codes @ trace.T)
            denominator = np.sqrt(config.k) * norms
            cosine = np.divide(dots, denominator[None, :],
                               out=np.zeros_like(dots), where=denominator[None, :] > 0)
            own = cosine[np.arange(len(labels)), labels]
            cosine[np.arange(len(labels)), labels] = -np.inf
            rival = np.maximum(cosine.max(axis=1), 0.0)
            for label in np.unique(labels):
                mask = labels == label
                own_means[int(label)] = float(own[mask].mean())
                rival_means[int(label)] = float(rival[mask].mean())
        for label in range(task * 5):
            rows.append({
                "checkpoint_task": task, "class": label,
                "source_task": label // 5 + 1,
                "age_since_last_train_sample": sample_index - 1 - int(last_seen[label]),
                "trace_l2_norm": float(norms[label]),
                "trace_density_eps1e_12": float(np.mean(trace[label] > 1e-12)),
                "own_class_test_cosine_mean": float(own_means[label]),
                "max_rival_test_cosine_mean": float(rival_means[label]),
                "own_minus_rival_cosine": float(own_means[label] - rival_means[label]),
                "status": "READ_ONLY_FORMAL_K41_CHECKPOINT",
            })
        print(f"formal trace snapshot task {task}: {task * 5} classes", flush=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
