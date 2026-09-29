"""Recompute only task-end trace descriptors from the actual vectors.

The trained W_B trajectories, C/D readout, predictions, write logs, and all
non-trace CSV fields remain untouched. This is a read-only eligibility replay
followed by a mechanical correction to experimental diagnostic CSV fields.
"""

from __future__ import annotations

from dataclasses import replace
import gc
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder
from run_repair_ladder_task1_5 import load_split
from run_lambda_state_dependence import CONFIG, OUT10, OUT20, StableFastTraces
from run_lambda_state_dependence import read_rows, trace_summary, write_rows


def main() -> None:
    cfg_yaml = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base_path = (CONFIG.parent / cfg_yaml["base_config"]).resolve()
    base = BTSPCAMConfig.from_mapping(yaml.safe_load(base_path.read_text(encoding="utf-8")))
    ten_rows = read_rows(OUT10)
    full_rows = read_rows(OUT20)
    if len(ten_rows) != 70 or len(full_rows) != 80:
        raise AssertionError("lambda diagnostic CSVs are incomplete")
    ten = {(int(r["k"]), r["family"], int(r["task"])): r for r in ten_rows}
    full = {(int(r["k"]), r["family"], int(r["task"])): r for r in full_rows}
    if len(ten) != 70 or len(full) != 80:
        raise AssertionError("duplicate lambda diagnostic keys")
    key_fields = ("cumulative_accuracy", "old_accuracy", "current_accuracy",
                  "current_5way_accuracy", "gross_flips_task", "WB_density")
    protected = {("ten", key): tuple(row[f] for f in key_fields) for key, row in ten.items()}
    protected.update({("full", key): tuple(row[f] for f in key_fields) for key, row in full.items()})
    for k in (128, 160):
        config = replace(base, k=k, tau=0.0)
        encoder = RandomSparseEncoder(config.feature_dim, config.code_dim, config.k,
                                      RNGProvenance.from_seed(config.seed).projection_seed)
        names = list(cfg_yaml["families"]) if k == 128 else cfg_yaml["full20_families"]
        traces = {}
        last_seen = {}
        for name in names:
            decay = cfg_yaml["families"][name]
            traces[name] = StableFastTraces(config.num_classes, config.code_dim,
                                             config.eta, float(decay["current"]),
                                             float(decay["other"]))
            last_seen[name] = np.full(config.num_classes, -1, dtype=np.int64)
        sample_index = 0
        for task in range(1, 21):
            active = [name for name in names if task <= 10 or
                      (k, name, task) in full]
            if not active:
                continue
            h, labels = load_split("train", task)
            codes = encoder.encode(h, config.encoder_batch_size)
            del h
            for name in active:
                tr = traces[name]
                seen = last_seen[name]
                for position, (code, label) in enumerate(zip(codes, labels)):
                    tr.update(code, int(label))
                    seen[int(label)] = sample_index + position
                summary = trace_summary(tr, seen, sample_index + len(labels) - 1, task * 5)
                for mapping in (ten, full):
                    key = (k, name, task)
                    if key in mapping:
                        mapping[key].update(summary)
                        mapping[key]["trace_norm_source"] = "direct_vector_float64"
            sample_index += len(labels)
            print(f"trace descriptors k={k} task={task} active={len(active)}", flush=True)
            del codes, labels
            gc.collect()
    if any(tuple(row[f] for f in key_fields) != protected[("ten", key)]
           for key, row in ten.items()):
        raise AssertionError("non-trace Task10 fields changed")
    if any(tuple(row[f] for f in key_fields) != protected[("full", key)]
           for key, row in full.items()):
        raise AssertionError("non-trace full20 fields changed")
    if any(row.get("trace_norm_source") != "direct_vector_float64"
           for row in [*ten_rows, *full_rows]):
        raise AssertionError("a trace descriptor row was not recomputed")
    write_rows(OUT10, ten_rows)
    write_rows(OUT20, full_rows)


if __name__ == "__main__":
    main()
