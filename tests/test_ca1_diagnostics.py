import unittest

import numpy as np

from experiments.run_ca1_diagnostics_round1 import (
    overlap_from_codes,
    threshold_overlap,
)


class CA1DiagnosticTests(unittest.TestCase):
    def test_overlap_and_threshold_match_definitions(self):
        codes = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
        weights = np.array(
            [[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]], dtype=bool
        )
        overlap = overlap_from_codes(codes, weights)
        np.testing.assert_allclose(
            overlap,
            np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            threshold_overlap(overlap, 0.5),
            np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
