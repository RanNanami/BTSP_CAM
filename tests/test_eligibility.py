import unittest

import numpy as np

from btsp_cam.eligibility import ClassEligibilityTraces


class EligibilityTests(unittest.TestCase):
    def test_all_classes_decay_then_current_class_accumulates(self):
        traces = ClassEligibilityTraces(2, 3, eta=0.5, decay=np.array([0.2, 0.4]))
        traces.values[:] = [[1, 1, 1], [2, 2, 2]]
        traces.update(np.array([1, 0, 1]), label=0)
        np.testing.assert_allclose(traces.values[0], [1.3, 0.8, 1.3])
        np.testing.assert_allclose(traces.values[1], [1.2, 1.2, 1.2])

    def test_familiarity_novelty_and_interference(self):
        traces = ClassEligibilityTraces(2, 3, eta=1.0, decay=0.0)
        traces.values[:] = [[1, 0, 0], [1, 0, 0]]
        signals = traces.signals(np.array([1, 0, 0]), label=0, beta=2.0)
        self.assertAlmostEqual(signals.familiarity, 1.0)
        self.assertAlmostEqual(signals.novelty, 0.0)
        self.assertLess(signals.interference, 1.0)

    def test_eta_is_scale_invariant_for_zero_initialized_cosine_traces(self):
        codes = [np.array([1, 0, 1]), np.array([1, 1, 0])]
        left = ClassEligibilityTraces(2, 3, eta=1.0, decay=0.2)
        right = ClassEligibilityTraces(2, 3, eta=7.0, decay=0.2)
        for code in codes:
            left.update(code, 0)
            right.update(code, 0)
        self.assertAlmostEqual(
            left.signals(codes[-1], 0, beta=1.0).familiarity,
            right.signals(codes[-1], 0, beta=1.0).familiarity,
        )


if __name__ == "__main__":
    unittest.main()
