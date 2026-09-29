import unittest

import numpy as np

from btsp_cam.ca1 import ca1_features


class CA1Tests(unittest.TestCase):
    def test_equation_7_and_non_degenerate_output(self):
        codes = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        weights = np.array([[1, 1, 0, 0], [0, 0, 1, 0]])
        z = ca1_features(codes, weights, tau=0.25)
        np.testing.assert_allclose(z, [[0.75, 0.0], [0.0, 0.25]])
        self.assertGreater(np.count_nonzero(z), 0)
        self.assertLess(np.count_nonzero(z), z.size)


if __name__ == "__main__":
    unittest.main()
