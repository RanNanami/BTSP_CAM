import unittest

import numpy as np

from btsp_cam.sparse_encoder import RandomSparseEncoder


class SparseEncoderTests(unittest.TestCase):
    def test_exact_sparsity_and_binary_output(self):
        encoder = RandomSparseEncoder(5, 31, 4, seed=7)
        codes = encoder.encode(np.arange(20, dtype=float).reshape(4, 5))
        np.testing.assert_array_equal(codes.sum(axis=1), np.full(4, 4))
        self.assertTrue(set(np.unique(codes)).issubset({0, 1}))

    def test_seed_reproducibility(self):
        x = np.ones((2, 3))
        left = RandomSparseEncoder(3, 20, 3, seed=4).encode(x)
        right = RandomSparseEncoder(3, 20, 3, seed=4).encode(x)
        np.testing.assert_array_equal(left, right)

    def test_top_k_ties_are_deterministic_and_prefer_lower_index(self):
        encoder = RandomSparseEncoder(2, 6, 3, seed=1)
        encoder.projection[:] = 0.0
        encoder.bias[:] = 0.0
        np.testing.assert_array_equal(encoder.encode(np.ones(2)), [1, 1, 1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
