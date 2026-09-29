"""Official simpleBTSP reference update and BTSP-CAM Eq. (6) memory."""

from __future__ import annotations

import numpy as np


def official_simplebtsp_step(
    weights_input_by_output: np.ndarray,
    input_pattern: np.ndarray,
    ltp_slots: np.ndarray,
    ltd_slots: np.ndarray,
) -> np.ndarray:
    """Mirror official main_BTSP_feedforward_model.py lines 123-126.

    The caller is responsible for sampling mutually-exclusive LTP/LTD events.
    The official orientation is (input_dim, output_dim).
    """

    w = np.asarray(weights_input_by_output, dtype=np.uint8)
    x = np.asarray(input_pattern, dtype=np.uint8)[:, None]
    ltp = np.asarray(ltp_slots, dtype=np.uint8)[None, :]
    ltd = np.asarray(ltd_slots, dtype=np.uint8)[None, :]
    if w.shape != (x.shape[0], ltp.shape[1]) or ltp.shape != ltd.shape:
        raise ValueError("incompatible official simpleBTSP shapes")
    return (((w + x * ltp) >= 1) & ((w + x * ltd) <= 1)).astype(np.uint8)


class BinaryMemory:
    """Binary matrix WB with paper orientation (memory_slots, code_dim)."""

    def __init__(self, slots: int, code_dim: int):
        self.slots = int(slots)
        self.code_dim = int(code_dim)
        self.weights = np.zeros((self.slots, self.code_dim), dtype=np.uint8)
        self.last_zero_to_one = 0
        self.last_one_to_zero = 0

    def update(
        self,
        code: np.ndarray,
        plateaus: np.ndarray,
        flip_probability: float,
        rng: np.random.Generator,
    ) -> int:
        s = np.asarray(code, dtype=np.uint8)
        q = np.asarray(plateaus, dtype=np.uint8)
        if s.shape != (self.code_dim,) or q.shape != (self.slots,):
            raise ValueError("code or plateau vector has the wrong shape")
        if not np.all((s == 0) | (s == 1)) or not np.all((q == 0) | (q == 1)):
            raise ValueError("code and plateau vectors must be binary")
        if not np.isfinite(flip_probability) or not 0.0 <= flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")
        active = np.flatnonzero(s)
        selected = np.flatnonzero(q)
        if active.size == 0 or selected.size == 0:
            self.last_zero_to_one = 0
            self.last_one_to_zero = 0
            return 0
        flips = (rng.random((selected.size, active.size)) < flip_probability).astype(
            np.uint8
        )
        target = np.ix_(selected, active)
        before = self.weights[target]
        self.last_zero_to_one = int(np.count_nonzero((before == 0) & (flips == 1)))
        self.last_one_to_zero = int(np.count_nonzero((before == 1) & (flips == 1)))
        self.weights[target] ^= flips
        return int(flips.sum())

    @property
    def active_fraction(self) -> float:
        return float(self.weights.mean())
