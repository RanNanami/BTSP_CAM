"""Equation (3): frozen Gaussian random features followed by exact k-WTA."""

from __future__ import annotations

import numpy as np


class RandomSparseEncoder:
    def __init__(self, input_dim: int, code_dim: int, k: int, seed: int = 0):
        if not 0 < k <= code_dim:
            raise ValueError("k must satisfy 0 < k <= code_dim")
        self.input_dim = int(input_dim)
        self.code_dim = int(code_dim)
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        # Paper Eq. (3)/(10): both R and b are standard normal and frozen.
        self.projection = self.rng.standard_normal((self.code_dim, self.input_dim))
        self.bias = self.rng.standard_normal(self.code_dim)

    def encode(self, features: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("features have the wrong shape")
        codes = np.zeros((x.shape[0], self.code_dim), dtype=np.uint8)
        if not np.all(np.isfinite(x)):
            raise ValueError("features contain NaN or Inf")
        batch_size = x.shape[0] if batch_size is None else int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            scores = x[start:stop] @ self.projection.T + self.bias
            if not np.all(np.isfinite(scores)):
                raise FloatingPointError("projection produced NaN or Inf")
            # Paper does not define Top-k ties. Stable descending sort is a
            # LOCAL_CHOICE that makes ties deterministic (lower index wins).
            winners = np.argsort(-scores, axis=1, kind="stable")[:, : self.k]
            np.put_along_axis(codes[start:stop], winners, 1, axis=1)
        return codes[0] if single else codes
