import unittest

import numpy as np

from btsp_cam.binary_memory import BinaryMemory


class BinaryMemoryTests(unittest.TestCase):
    def test_xor_flip_and_binary_invariant(self):
        memory = BinaryMemory(2, 4)
        rng = np.random.default_rng(1)
        memory.update(np.array([1, 0, 1, 0]), np.array([1, 0]), 1.0, rng)
        np.testing.assert_array_equal(memory.weights[0], [1, 0, 1, 0])
        memory.update(np.array([1, 0, 1, 0]), np.array([1, 0]), 1.0, rng)
        np.testing.assert_array_equal(memory.weights[0], [0, 0, 0, 0])
        self.assertTrue(set(np.unique(memory.weights)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
