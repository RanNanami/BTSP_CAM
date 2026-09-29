"""Class-specific eligibility traces and geometric gating signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("cosine inputs contain NaN or Inf")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denom == 0.0 else float(np.dot(left, right) / denom)


@dataclass(frozen=True)
class GatingSignals:
    familiarity: float
    novelty: float
    interference: float
    max_rival_similarity: float


class ClassEligibilityTraces:
    """Implements Eq. (4), decaying every class trace at each stream step."""

    def __init__(
        self,
        num_classes: int,
        code_dim: int,
        eta: float,
        decay: float | np.ndarray,
    ):
        self.num_classes = int(num_classes)
        self.code_dim = int(code_dim)
        self.eta = float(eta)
        if self.num_classes <= 0 or self.code_dim <= 0:
            raise ValueError("num_classes and code_dim must be positive")
        if not np.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        decay_array = np.broadcast_to(
            np.asarray(decay, dtype=np.float64), (self.num_classes,)
        ).copy()
        if np.any((decay_array < 0.0) | (decay_array > 1.0)):
            raise ValueError("decay values must be in [0, 1]")
        self.decay = decay_array
        self.values = np.zeros((self.num_classes, self.code_dim), dtype=np.float64)

    def update(self, code: np.ndarray, label: int) -> None:
        s = np.asarray(code, dtype=np.float64)
        if s.shape != (self.code_dim,):
            raise ValueError("code has the wrong shape")
        if not 0 <= label < self.num_classes:
            raise ValueError("label is outside the configured class range")
        if not np.all(np.isfinite(s)):
            raise ValueError("code contains NaN or Inf")
        self.values *= (1.0 - self.decay[:, None])
        self.values[label] += self.eta * s

    def signals(self, code: np.ndarray, label: int, beta: float) -> GatingSignals:
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be non-negative")
        s = np.asarray(code, dtype=np.float64)
        familiarity = float(np.clip(cosine_similarity(s, self.values[label]), 0.0, 1.0))
        # cos(s, 0) is explicitly defined as zero by cosine_similarity. Zero
        # unseen traces are included; with nonnegative traces they cannot raise
        # the maximum. An empty rival set therefore has maximum 0 and I=1.
        rival_similarities = [
            cosine_similarity(s, self.values[c])
            for c in range(self.num_classes)
            if c != label
        ]
        max_rival = max(rival_similarities, default=0.0)
        interference = float(np.exp(-float(beta) * max_rival))
        return GatingSignals(
            familiarity=familiarity,
            novelty=1.0 - familiarity,
            interference=interference,
            max_rival_similarity=max_rival,
        )
