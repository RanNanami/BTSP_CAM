import unittest

from btsp_cam.eligibility import GatingSignals
from btsp_cam.plateau import plateau_probability


class PlateauTests(unittest.TestCase):
    def test_probability_is_bounded(self):
        signals = GatingSignals(0.7, 0.3, 0.8, 0.2)
        probability = plateau_probability(1.0, signals)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_invalid_base_probability_is_rejected_not_silently_clipped(self):
        signals = GatingSignals(0.7, 0.3, 0.8, 0.2)
        with self.assertRaises(ValueError):
            plateau_probability(10.0, signals)

    def test_familiarity_novelty_and_interference_change_probability(self):
        optimum = plateau_probability(1.0, GatingSignals(2 / 3, 1 / 3, 1.0, 0.0))
        too_familiar = plateau_probability(1.0, GatingSignals(0.95, 0.05, 1.0, 0.0))
        too_novel = plateau_probability(1.0, GatingSignals(0.05, 0.95, 1.0, 0.0))
        interfered = plateau_probability(1.0, GatingSignals(2 / 3, 1 / 3, 0.2, 1.0))
        self.assertGreater(optimum, too_familiar)
        self.assertGreater(optimum, too_novel)
        self.assertLess(interfered, optimum)


if __name__ == "__main__":
    unittest.main()
