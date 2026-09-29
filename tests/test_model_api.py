import unittest

import numpy as np

from btsp_cam.model import BTSPCAM


class _FakeTorchModel:
    training = False

    @staticmethod
    def parameters():
        return []


class _FakeBackbone:
    output_dim = 4
    model_name = "fake_frozen_backbone"
    timm_version = "test"
    pretrained_cfg = {"identifier": "test"}
    model = _FakeTorchModel()

    def __init__(self):
        self.features = np.array(
            [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0], [0.1, 0.9, 0, 0]],
            dtype=float,
        )
        self.labels = np.array([0, 0, 1, 1])

    def extract_loader(self, loader):
        del loader
        return self.features.copy(), self.labels.copy(), np.arange(4)


class ModelAPITests(unittest.TestCase):
    def test_loader_train_evaluate_api(self):
        model = BTSPCAM(
            feature_dim=4,
            code_dim=16,
            k=3,
            memory_slots=8,
            num_classes=2,
            base_plateau_probability=1.0,
            beta=0.5,
            flip_probability=0.5,
            eta=1.0,
            trace_decay=0.1,
            tau=0.1,
            gamma=0.1,
            seed=7,
            backbone=_FakeBackbone(),
        )
        training = model.learn_task(object(), task_id=1)
        evaluation = model.evaluate(object())
        self.assertEqual(training["sample_count"], 4)
        self.assertEqual(evaluation["sample_count"], 4)
        self.assertEqual(model.current_task_index, 1)
        self.assertEqual(model.last_task_slot_plateau_counts.shape, (8,))
        self.assertEqual(
            int(model.last_task_slot_plateau_counts.sum()),
            int(sum(entry["plateau_count"] for entry in training["logs"])),
        )
        self.assertTrue(0.0 <= evaluation["accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
