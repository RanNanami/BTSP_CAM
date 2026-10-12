"""Infrastructure-only runner tests; all data/models here are synthetic."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import pilot_class_order
from experiments.run_btsp_cam_imagenet_r import (
    CONTROL, MAIN, FIXED, advance_task, comparison_rows, config_from_file, evaluate,
    feature_and_sparse, fit_replay, load_codes,
)
from test_imagenet_r_protocol import fake_imagenet_r
from btsp_cam.imagenet_r import ImageNetRProtocol


def toy_model():
    cfg = BTSPCAMConfig(feature_dim=4, code_dim=32, k=4, memory_slots=8,
                        num_classes=200, base_plateau_probability=.5, beta=1.,
                        flip_probability=.5, eta=1., trace_decay=.01,
                        tau=.125, gamma=.1, seed=1993)
    return BTSPCAM(cfg, class_order=pilot_class_order(200, 1993))


def toy_codes(seed):
    rng = np.random.default_rng(seed)
    codes = np.zeros((20, 32), dtype=np.uint8)
    for row in codes:
        row[rng.choice(32, size=4, replace=False)] = 1
    return codes


def test_frozen_config_and_readout_identity():
    raw, config = config_from_file(Path("configs/btsp_cam_imagenet_r_k96.yaml"))
    assert (config.k, config.tau) == (96, 4 / 96)
    assert raw["readouts"] == dict(main=MAIN, control=CONTROL,
                                   replay_boundary="NOT_STRICT_EXEMPLAR_FREE_NOT_CLAIMED_AS_PAPER_ALGORITHM")
    assert FIXED["q"] == 4
    paired = comparison_rows([
        dict(readout=MAIN, task_id=1, seen_classes=20,
             cumulative_accuracy=.6, wb_density=.1, gross_flips=7),
        dict(readout=CONTROL, task_id=1, seen_classes=20,
             cumulative_accuracy=.4)])
    assert paired[0]["replay_minus_literal_accuracy"] == pytest.approx(.2)


def test_shared_writer_literal_no_replay_and_resume(tmp_path):
    codes = {1: toy_codes(1), 2: toy_codes(2)}
    labels = {1: np.arange(20), 2: np.arange(20, 40)}
    calls = []

    def loader(source):
        calls.append(source)
        return codes[source], labels[source]

    baseline = toy_model()
    replay1, stats1 = advance_task(baseline, 1, codes[1], labels[1], loader)
    assert calls == [1]
    literal_a = baseline.ridge.a.copy()
    assert not np.shares_memory(replay1.a, baseline.ridge.a)
    checkpoint = tmp_path / "task_01.npz"
    baseline.save(checkpoint)
    restored = BTSPCAM.load(checkpoint)
    assert restored.current_task_index == 1
    replay2, stats2 = advance_task(baseline, 2, codes[2], labels[2], loader)
    replay2_restored, stats2_restored = advance_task(restored, 2, codes[2], labels[2], loader)
    assert stats1["train_samples"] == 20 and stats2 == stats2_restored
    assert calls == [1, 1, 2, 1, 2]
    assert np.array_equal(baseline.memory.weights, restored.memory.weights)
    assert np.array_equal(baseline.ridge.a, restored.ridge.a)
    assert np.allclose(replay2.weights, replay2_restored.weights)
    assert np.all(baseline.ridge.a >= literal_a)
    # Literal A is the sum of frozen-at-task-i features, not replay's A.
    assert baseline.ridge.num_classes == replay2.num_classes == 40
    metrics = evaluate(replay2, baseline.memory.weights,
                       [(codes[1], labels[1]), (codes[2], labels[2])], 2, 4, .125)
    assert 0 <= metrics["cumulative_accuracy"] <= 1
    with pytest.raises(ValueError, match="future"):
        evaluate(replay2, baseline.memory.weights,
                 [(codes[1], labels[1]), (codes[2], labels[2]),
                  (codes[2], labels[2])], 2, 4, .125)


def test_mock_feature_cache_extraction_and_replay_boundary(fake_imagenet_r, tmp_path):
    protocol = ImageNetRProtocol(fake_imagenet_r)
    raw = yaml.safe_load(Path("configs/btsp_cam_imagenet_r_k96.yaml").read_text())
    raw["runtime"]["batch_size"] = 4

    class FakeBackbone:
        model_name = "INFRASTRUCTURE_ONLY_FAKE"
        timm_version = "mock"
        pretrained_cfg = {"source": "mock"}

        def extract_loader(self, loader):
            batches = list(loader)
            indices = np.concatenate([b[0].numpy() for b in batches])
            labels = np.concatenate([b[2].numpy() for b in batches])
            features = np.tile(indices[:, None].astype(np.float32), (1, 768))
            return features, labels, indices

    class FakeEncoder:
        def encode(self, features, batch_size):
            codes = np.zeros((len(features), 8192), dtype=np.uint8)
            codes[:, :96] = 1
            return codes

    paths = [tmp_path / part for part in ("feature", "replay", "test")]
    first = feature_and_sparse(protocol, "train", 1, FakeBackbone(), FakeEncoder(),
                               raw, tmp_path, *paths)
    second = feature_and_sparse(protocol, "train", 1, FakeBackbone(), FakeEncoder(),
                                raw, tmp_path, *paths)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2]["split"] == "train"
    assert paths[1].joinpath("train_task01.npz").exists()
    assert not paths[2].joinpath("test_task01.npz").exists()
    with pytest.raises(ValueError, match="provenance"):
        load_codes(paths[1] / "train_task01.npz", {**first[2], "split": "test"})
