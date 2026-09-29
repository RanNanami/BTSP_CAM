import tempfile
import unittest
from pathlib import Path

import numpy as np

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM


def make_config() -> BTSPCAMConfig:
    values = {
        "feature_dim": 4,
        "code_dim": 32,
        "k": 5,
        "memory_slots": 12,
        "num_classes": 4,
        "base_plateau_probability": 1.0,
        "beta": 0.5,
        "flip_probability": 0.6,
        "eta": 1.0,
        "trace_decay": 0.2,
        "tau": 0.1,
        "gamma": 0.1,
        "seed": 19,
    }
    return BTSPCAMConfig.from_mapping(
        {
            "parameters": {
                name: {"value": value, "source": "test", "status": "LOCAL_CHOICE"}
                for name, value in values.items()
            }
        }
    )


class CheckpointResumeTests(unittest.TestCase):
    def test_save_load_predictions_and_resume_are_exact(self):
        first_x = np.array(
            [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0], [0.1, 0.9, 0, 0]],
            dtype=float,
        )
        first_y = np.array([0, 0, 1, 1])
        second_x = np.array(
            [[0, 0, 1, 0], [0, 0.1, 0.9, 0], [0, 0, 0, 1], [0, 0, 0.1, 0.9]],
            dtype=float,
        )
        second_y = np.array([2, 2, 3, 3])
        model = BTSPCAM(make_config(), class_order=np.array([2, 0, 3, 1]))
        model.learn_task(first_x, first_y, task_id=1)
        logits_before = model.logits(first_x, model.seen_classes)
        predictions_before = model.predict(first_x, model.seen_classes)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = model.save(Path(directory) / "task1.npz")
            restored = BTSPCAM.load(checkpoint)

            np.testing.assert_array_equal(
                restored.predict(first_x, restored.seen_classes), predictions_before
            )
            np.testing.assert_array_equal(
                restored.logits(first_x, restored.seen_classes), logits_before
            )
            np.testing.assert_array_equal(restored.encoder.projection, model.encoder.projection)
            np.testing.assert_array_equal(restored.encoder.bias, model.encoder.bias)
            np.testing.assert_array_equal(restored.memory.weights, model.memory.weights)
            np.testing.assert_array_equal(restored.traces.values, model.traces.values)
            np.testing.assert_array_equal(restored.ridge.a, model.ridge.a)
            np.testing.assert_array_equal(restored.ridge.b, model.ridge.b)
            np.testing.assert_array_equal(restored.ridge.weights, model.ridge.weights)
            np.testing.assert_array_equal(restored.class_order, model.class_order)
            self.assertEqual(restored.current_task_index, 1)
            self.assertEqual(restored.config.provenance["k"].status, "LOCAL_CHOICE")

            model.learn_task(second_x, second_y, task_id=2)
            restored.learn_task(second_x, second_y, task_id=2)
            np.testing.assert_array_equal(restored.memory.weights, model.memory.weights)
            np.testing.assert_array_equal(restored.traces.values, model.traces.values)
            np.testing.assert_array_equal(restored.ridge.a, model.ridge.a)
            np.testing.assert_array_equal(restored.ridge.b, model.ridge.b)
            np.testing.assert_array_equal(restored.ridge.weights, model.ridge.weights)
            np.testing.assert_array_equal(restored.seen_classes, np.arange(4))
            self.assertEqual(restored.current_task_index, 2)


if __name__ == "__main__":
    unittest.main()
