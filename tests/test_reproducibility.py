import unittest

import numpy as np

from btsp_cam.model import BTSPCAM


def make_model():
    return BTSPCAM(
        feature_dim=4,
        code_dim=32,
        k=5,
        memory_slots=12,
        num_classes=2,
        base_plateau_probability=1.0,
        beta=0.5,
        flip_probability=0.6,
        eta=0.4,
        trace_decay=0.2,
        tau=0.1,
        gamma=0.1,
        seed=19,
    )


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_same_state(self):
        x = np.array([[1, 0, 0, 0], [0.8, 0.2, 0, 0], [0, 1, 0, 0]], dtype=float)
        y = np.array([0, 0, 1])
        left, right = make_model(), make_model()
        left.learn_task(x, y)
        right.learn_task(x, y)
        np.testing.assert_array_equal(left.memory.weights, right.memory.weights)
        np.testing.assert_allclose(left.ridge.weights, right.ridge.weights)

    def test_plateau_and_synapse_rngs_are_distinct_streams(self):
        model = make_model()
        self.assertIsNot(model.plateau_rng, model.synapse_rng)


if __name__ == "__main__":
    unittest.main()
