"""INFRASTRUCTURE_ONLY: tiny synthetic writer/readout/resume/output smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from run_btsp_cam_imagenet_r import advance_task, evaluate, save_csv, save_json


def tiny_model():
    cfg = BTSPCAMConfig(feature_dim=4, code_dim=32, k=4, memory_slots=64,
                        num_classes=200, base_plateau_probability=1., beta=1.,
                        flip_probability=.5, eta=1., trace_decay=.01,
                        tau=.125, gamma=.1, seed=1993)
    return BTSPCAM(cfg)


def synthetic_task(task: int):
    rng = np.random.default_rng(100 + task)
    codes = np.zeros((40, 32), dtype=np.uint8)
    for cls in range(20):
        first = rng.choice(32, size=4, replace=False)
        rest = rng.choice(np.setdiff1d(np.arange(32), first), size=2, replace=False)
        codes[2 * cls, first] = 1
        codes[2 * cls + 1, np.concatenate((first[:2], rest))] = 1
    return codes, np.repeat(np.arange((task - 1) * 20, task * 20), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/imagenet_r_infrastructure_smoke")
    args = parser.parse_args()
    output = Path(args.output_dir)
    data = {task: synthetic_task(task) for task in (1, 2)}
    model = tiny_model()
    replay1, _ = advance_task(model, 1, *data[1], lambda source: data[source])
    checkpoint = output / "task_01.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    resumed = BTSPCAM.load(checkpoint)
    replay_continuous, _ = advance_task(model, 2, *data[2], lambda source: data[source])
    replay_resumed, writer = advance_task(resumed, 2, *data[2], lambda source: data[source])
    if writer["gross_flips"] <= 0:
        raise AssertionError("synthetic smoke did not exercise writer updates")
    if not (np.array_equal(model.memory.weights, resumed.memory.weights)
            and np.array_equal(model.ridge.a, resumed.ridge.a)
            and np.allclose(replay_continuous.weights, replay_resumed.weights)):
        raise AssertionError("writer/readout resume mismatch")
    test_sources = [data[1], data[2]]
    rows = []
    for mode, ridge in (("EXPERIMENTAL_CURRENT_WB_REPLAY_REFIT", replay_resumed),
                        ("LITERAL_STREAMING_EQ8", resumed.ridge)):
        rows.append(dict(readout=mode, task_id=2,
                         **evaluate(ridge, resumed.memory.weights,
                                    test_sources, 2, 4, .125)))
    save_csv(output / "task_metrics.csv", rows)
    save_json(output / "summary.json",
              dict(status="INFRASTRUCTURE_ONLY_NOT_SCIENTIFIC_RESULTS",
                   checkpoint_reload=True, resume_reproduced=True,
                   writer=writer, readouts=[row["readout"] for row in rows]))
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if len((output / "task_metrics.csv").read_text(encoding="utf-8").splitlines()) != 3:
        raise AssertionError("CSV output missing one branch")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
