"""Dataset-free checks for the isolated CUB transfer runner."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.protocol import pilot_class_order
from btsp_cam.cifar100 import build_pilot_cifar224_transform
from run_btsp_cam_cub200 import (CONFIG, ca1, evaluate, fit_replay, load_codes,
                                load_config, pilot_class_tasks, save_codes)
from utils.data import CUB
from utils.data_manager import DataManager


def test_cub_pilot_class_order_and_20_tasks(monkeypatch):
    original = np.repeat(np.arange(200), 2)
    def fake_data(self):
        self.train_data = np.array([f"train_{i}.jpg" for i in range(len(original))])
        self.test_data = np.array([f"test_{i}.jpg" for i in range(len(original))])
        self.train_targets = original.copy()
        self.test_targets = original.copy()
    monkeypatch.setattr(CUB, "download_data", fake_data)
    manager = DataManager("cub", True, 1993, 10, 10, {"dataset": "cub"})
    order, tasks = pilot_class_tasks()
    np.testing.assert_array_equal(order, pilot_class_order(200, 1993))
    assert manager.nb_tasks == 20
    assert all(manager.get_task_size(t) == 10 for t in range(20))
    np.testing.assert_array_equal(manager._class_order, order)
    np.testing.assert_array_equal(manager._train_targets[:2], [int(np.where(order == 0)[0][0])] * 2)
    assert len(manager.get_dataset(np.arange(0, 10), "train", "train")) == 20
    assert len(manager.get_dataset(np.arange(0, 20), "test", "test")) == 40
    assert len(tasks) == 20


def test_sparse_cache_exact_96_and_metadata(tmp_path):
    code = np.zeros((2, 8192), dtype=np.uint8)
    code[0, :96] = 1
    code[1, 100:196] = 1
    meta = dict(dataset_sha256="fixture", class_order=list(range(200)), k=96,
                sample_count=2, projection_seed=1993)
    path = tmp_path / "codes.npz"
    save_codes(path, code, np.array([0, 1]), meta)
    recovered, labels = load_codes(path, meta)
    np.testing.assert_array_equal(recovered.toarray(), code)
    np.testing.assert_array_equal(labels, [0, 1])
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_codes(path, {**meta, "k": 95})
    bad = code.copy()
    bad[0, 0] = 0
    with pytest.raises(AssertionError, match="exact-96"):
        save_codes(tmp_path / "bad.npz", bad, np.array([0, 1]), meta)


def test_replay_only_calls_seen_training_and_evaluation_counts():
    code = np.zeros((2, 8192), dtype=np.uint8)
    code[:, :96] = 1
    s = sparse.csr_matrix(code, dtype=np.float32)
    wb = np.zeros((4, 8192), dtype=np.uint8)
    wb[0, :96] = 1
    sources = {1:(s, np.array([0, 1])), 2:(s, np.array([10, 11]))}
    called = []
    def train_only(source):
        called.append(source)
        return sources[source]
    ridge, wt = fit_replay(wb, 2, .1, train_only)
    assert called == [1, 2]
    assert ridge.last_solve_status == "ok"
    _, z = ca1(s, wt)
    assert np.any(z > 0) and np.all(np.isfinite(z))
    metrics, geometry = evaluate(ridge, wt, 2, [sources[1], sources[2]])
    assert metrics["test_samples_cumulative"] == 4
    assert metrics["seen_classes"] == 20
    assert len(geometry) == 2
    assert np.all(ridge.logits(z).argmax(axis=1) < 20)


def test_cub_frozen_parameters_and_tamper_rejection(tmp_path):
    raw, cfg = load_config()
    assert (cfg.num_classes, cfg.code_dim, cfg.memory_slots, cfg.k, cfg.tau,
            cfg.base_plateau_probability, cfg.beta, cfg.trace_decay, cfg.eta,
            cfg.flip_probability, cfg.gamma) == (200, 8192, 4096, 96, 4 / 96,
                                                 .005, 1., .01, 1., .5, .1)
    altered = dict(raw)
    altered["parameters"] = dict(raw["parameters"], k=97)
    altered["source_frozen_lock"] = str((ROOT / "experiments/configs/ablation_baseline_frozen.yaml").resolve())
    altered["source_formal_config"] = str((ROOT / "configs/btsp_cam_cifar100.yaml").resolve())
    path = tmp_path / "altered.yaml"
    path.write_text(yaml.safe_dump(altered), encoding="utf-8")
    with pytest.raises(AssertionError, match="frozen CIFAR"):
        load_config(path)


def test_cifar_regression_config_and_transform_unchanged():
    formal = yaml.safe_load((ROOT / "configs/btsp_cam_cifar100.yaml").read_text(encoding="utf-8"))
    cfg = BTSPCAMConfig.from_mapping(formal)
    assert (cfg.num_classes, cfg.k, cfg.tau, cfg.gamma) == (100, 41, .5, .1)
    assert "RandomResizedCrop" in repr(build_pilot_cifar224_transform(True))
    assert "CenterCrop" in repr(build_pilot_cifar224_transform(False))
