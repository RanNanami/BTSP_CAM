"""Equation (7) normalized CA1 activation and threshold competition."""

from __future__ import annotations

import numpy as np


def ca1_features(codes: np.ndarray, memory_weights: np.ndarray, tau: float) -> np.ndarray:
    if not np.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be finite and in [0, 1]")
    s = np.asarray(codes)
    single = s.ndim == 1
    if single:
        s = s[None, :]
    w = np.asarray(memory_weights)
    if s.ndim != 2 or w.ndim != 2 or s.shape[1] != w.shape[1]:
        raise ValueError("codes and memory weights have incompatible shapes")
    if not np.all((s == 0) | (s == 1)) or not np.all((w == 0) | (w == 1)):
        raise ValueError("codes and memory weights must be binary")
    active = s.sum(axis=1)
    if np.any(active == 0):
        raise ValueError("CA1 normalization is undefined for an all-zero code")
    # Codes are exact k-WTA. Summing only active memory columns is identical to
    # dense ``s @ W_B.T`` but makes the paper dimensions (8192 x 4096) usable.
    output = np.empty((s.shape[0], w.shape[0]), dtype=np.float64)
    for row_index, row in enumerate(s):
        indices = np.flatnonzero(row)
        activation = w[:, indices].sum(axis=1, dtype=np.float64) / active[row_index]
        output[row_index] = np.maximum(0.0, activation - float(tau))
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("CA1 produced NaN or Inf")
    return output[0] if single else output
