import unittest

import numpy as np

from btsp_cam.binary_memory import BinaryMemory
from btsp_cam.config import BTSPCAMConfig
from btsp_cam.eligibility import GatingSignals
from btsp_cam.plateau import plateau_probability
from btsp_cam.sparse_encoder import RandomSparseEncoder


def valid_values():
    return {
        "feature_dim": 4,
        "code_dim": 8,
        "k": 2,
        "memory_slots": 4,
        "num_classes": 2,
        "base_plateau_probability": 0.1,
        "beta": 1.0,
        "flip_probability": 0.5,
        "eta": 1.0,
        "trace_decay": 0.1,
        "tau": 0.5,
        "gamma": 0.1,
        "seed": 3,
    }


class NumericalSafetyTests(unittest.TestCase):
    def test_invalid_tau_gamma_and_k_are_rejected(self):
        for name, value in (("tau", 1.1), ("gamma", 0.0), ("k", 9)):
            values = valid_values()
            values[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                BTSPCAMConfig.from_legacy_kwargs(**values)

    def test_nan_is_rejected_before_probability_or_projection(self):
        signals = GatingSignals(np.nan, 0.5, 0.5, 0.0)
        with self.assertRaises(ValueError):
            plateau_probability(0.1, signals)
        encoder = RandomSparseEncoder(4, 8, 2, seed=1)
        with self.assertRaises(ValueError):
            encoder.encode(np.array([[np.nan, 0.0, 0.0, 0.0]]))

    def test_binary_memory_rejects_nonbinary_inputs(self):
        memory = BinaryMemory(4, 8)
        with self.assertRaises(ValueError):
            memory.update(
                np.array([2, 0, 0, 0, 0, 0, 0, 0]),
                np.ones(4),
                0.5,
                np.random.default_rng(1),
            )


if __name__ == "__main__":
    unittest.main()
