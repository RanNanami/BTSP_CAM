import unittest

import numpy as np

from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import equal_increment_tasks


class CILProtocolTests(unittest.TestCase):
    def test_b0_local_choice_has_equal_tasks_and_fixed_order(self):
        tasks = equal_increment_tasks(100, 5, seed=1993)
        self.assertEqual(len(tasks), 20)
        self.assertTrue(all(len(task) == 5 for task in tasks))
        self.assertEqual(len(np.unique(np.concatenate(tasks))), 100)

    def test_model_keeps_statistics_not_raw_old_samples(self):
        model = BTSPCAM(
            feature_dim=3,
            code_dim=20,
            k=4,
            memory_slots=8,
            num_classes=2,
            base_plateau_probability=1.0,
            beta=1.0,
            flip_probability=0.5,
            eta=0.5,
            trace_decay=0.1,
            tau=0.1,
            gamma=0.1,
            seed=3,
        )
        model.learn_task(np.array([[1.0, 0, 0], [0.8, 0.1, 0]]), np.array([0, 0]))
        self.assertFalse(hasattr(model, "samples"))
        self.assertEqual(model.ridge.a.shape, (8, 8))

    def test_phase_two_uses_final_post_task_memory(self):
        model = BTSPCAM(
            feature_dim=3,
            code_dim=20,
            k=4,
            memory_slots=8,
            num_classes=2,
            base_plateau_probability=1.0,
            beta=0.5,
            flip_probability=1.0,
            eta=1.0,
            trace_decay=0.3,
            tau=0.1,
            gamma=0.1,
            seed=8,
        )
        x = np.array([[1.0, 0, 0], [0.8, 0.2, 0], [0, 1.0, 0]])
        y = np.array([0, 0, 1])
        model.learn_task(x, y)
        final_z = model.ca1(x)
        np.testing.assert_allclose(model.ridge.a, final_z.T @ final_z)

    def test_prediction_has_no_task_id_argument(self):
        self.assertNotIn("task_id", BTSPCAM.predict.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
