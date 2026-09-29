"""Small exact-algebra check for the isolated coordinate-aligned variant."""

import numpy as np

from coordinate_aligned_ridge import CoordinateAlignedRidge


def test_sparse_statistics_match_direct_ca1_gram() -> None:
    codes = np.array([
        [1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1],
    ], dtype=np.uint8)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    w = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
    ], dtype=np.uint8)
    variant = CoordinateAlignedRidge(code_dim=6, slots=3, k=2, gamma=0.1)
    variant.update(codes[:2], labels[:2])
    variant.update(codes[2:], labels[2:])
    a, b = variant.aligned_statistics(w)
    z = codes @ w.T / 2.0
    y = np.eye(2)[labels]
    np.testing.assert_allclose(a, z.T @ z, rtol=0, atol=0)
    np.testing.assert_allclose(b, z.T @ y, rtol=0, atol=0)
    assert variant.sample_count == 4
    assert not hasattr(variant, "historical_codes")
