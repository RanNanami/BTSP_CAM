import unittest

import numpy as np

from btsp_cam.binary_memory import official_simplebtsp_step


class OfficialBTSPBehaviorTests(unittest.TestCase):
    def test_exact_truth_table_and_inactive_input(self):
        # Rows exercise inactive, active/no-event, active/LTP, and active/LTD.
        weights = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        inactive = official_simplebtsp_step(weights, [0, 0], [1, 0], [0, 1])
        np.testing.assert_array_equal(inactive, weights)

        active = official_simplebtsp_step(weights, [1, 1], [1, 0], [0, 1])
        np.testing.assert_array_equal(active, np.array([[1, 0], [1, 0]], dtype=np.uint8))

    def test_output_is_binary(self):
        result = official_simplebtsp_step(
            np.ones((3, 2), dtype=np.uint8), [1, 1, 1], [0, 0], [1, 0]
        )
        self.assertTrue(set(np.unique(result)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
