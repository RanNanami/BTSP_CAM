import unittest

import numpy as np

from btsp_cam.ridge import StreamingRidge


class RidgeTests(unittest.TestCase):
    def test_shapes_and_incremental_statistics_are_retained(self):
        ridge = StreamingRidge(3, 2, gamma=0.1)
        z1 = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        ridge.update(z1, np.array([0, 0]))
        a1 = ridge.a.copy()
        b1 = ridge.b.copy()
        z2 = np.array([[0.0, 1.0, 0.0]])
        ridge.update(z2, np.array([1]))
        np.testing.assert_allclose(ridge.a, a1 + z2.T @ z2)
        self.assertGreaterEqual(ridge.b[0, 0], b1[0, 0])
        self.assertEqual(ridge.solve().shape, (3, 2))

    def test_class_expansion_pads_and_preserves_old_columns(self):
        ridge = StreamingRidge(2, 1, gamma=0.1)
        ridge.update(np.array([[1.0, 0.0]]), np.array([0]))
        old_b = ridge.b.copy()
        ridge.expand_classes(3)
        self.assertEqual(ridge.b.shape, (2, 3))
        np.testing.assert_array_equal(ridge.b[:, :1], old_b)
        np.testing.assert_array_equal(ridge.b[:, 1:], 0.0)

    def test_condition_number_uses_unfactored_matrix_norm(self):
        ridge = StreamingRidge(3, 2, gamma=0.1)
        features = np.array(
            [[10.0, 1.0, 0.0], [8.0, 0.0, 0.2], [0.0, 0.1, 0.0]]
        )
        ridge.update(features, np.array([0, 0, 1]))
        expected = np.linalg.cond(ridge.a + ridge.gamma * np.eye(3), p=1)

        ridge.solve()

        self.assertAlmostEqual(ridge.condition_number, expected, places=8)


if __name__ == "__main__":
    unittest.main()
