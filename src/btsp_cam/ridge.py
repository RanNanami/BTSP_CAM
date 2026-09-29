"""Equation (8) streaming sufficient statistics and closed-form ridge solve."""

from __future__ import annotations

import numpy as np


class StreamingRidge:
    def __init__(self, feature_dim: int, num_classes: int, gamma: float):
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        if not np.isfinite(self.gamma) or self.gamma <= 0.0:
            raise ValueError("gamma must be finite and positive")
        self.a = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float64)
        self.b = np.zeros((self.feature_dim, self.num_classes), dtype=np.float64)
        self.weights = np.zeros_like(self.b)

    def expand_classes(self, new_num_classes: int) -> None:
        """Pad class columns without changing old sufficient statistics."""

        new_num_classes = int(new_num_classes)
        if new_num_classes < self.num_classes:
            raise ValueError("cannot shrink the ridge class dimension")
        if new_num_classes == self.num_classes:
            return
        extra = new_num_classes - self.num_classes
        self.b = np.pad(self.b, ((0, 0), (0, extra)))
        self.weights = np.pad(self.weights, ((0, 0), (0, extra)))
        self.num_classes = new_num_classes

    def update(self, features: np.ndarray, labels: np.ndarray) -> None:
        z = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if z.ndim != 2 or z.shape[1] != self.feature_dim or y.shape != (z.shape[0],):
            raise ValueError("features or labels have the wrong shape")
        if not np.all(np.isfinite(z)):
            raise ValueError("ridge features contain NaN or Inf")
        if np.any((y < 0) | (y >= self.num_classes)):
            raise ValueError("label is outside the configured class range")
        one_hot = np.zeros((z.shape[0], self.num_classes), dtype=np.float64)
        one_hot[np.arange(z.shape[0]), y] = 1.0
        self.a += z.T @ z
        self.b += z.T @ one_hot

    def solve(self) -> np.ndarray:
        from scipy.linalg import cho_factor, cho_solve, get_lapack_funcs

        self._last_cholesky = None
        regularized = self.a + self.gamma * np.eye(self.feature_dim)
        try:
            # ``cho_factor(..., overwrite_a=True)`` may reuse and overwrite
            # ``regularized``.  LAPACK's condition estimator needs the norm of
            # the original matrix, so capture it before factorization.
            regularized_one_norm = float(np.linalg.norm(regularized, 1))
            factor, lower = cho_factor(
                regularized, lower=True, overwrite_a=True, check_finite=True
            )
            self.weights = cho_solve((factor, lower), self.b, check_finite=True)
            # Retained only in memory for scalable spectral diagnostics. It is
            # reproducible from A/gamma and is intentionally not checkpointed.
            self._last_cholesky = (factor, lower)
            # LAPACK's condition estimator reuses the Cholesky factor instead
            # of performing a second O(S^3) factorization at formal S=4096.
            pocon = get_lapack_funcs("pocon", (factor,))
            rcond, info = pocon(factor, regularized_one_norm, uplo="L")
            if info != 0 or rcond <= 0.0:
                raise np.linalg.LinAlgError(f"condition estimator failed with info={info}")
            self.last_condition_number = float(1.0 / rcond)
            self.last_solve_status = "ok"
        except (np.linalg.LinAlgError, ValueError):
            self.last_solve_status = "failed"
            raise
        if not np.all(np.isfinite(self.weights)):
            self.last_solve_status = "failed_nonfinite"
            raise FloatingPointError("ridge solve produced NaN or Inf")
        return self.weights

    def logits(self, features: np.ndarray) -> np.ndarray:
        z = np.asarray(features, dtype=np.float64)
        if not np.all(np.isfinite(z)):
            raise ValueError("logit features contain NaN or Inf")
        logits = z @ self.weights
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("ridge logits contain NaN or Inf")
        return logits

    @property
    def condition_number(self) -> float:
        if hasattr(self, "last_condition_number"):
            return float(self.last_condition_number)
        regularized = self.a + self.gamma * np.eye(self.feature_dim)
        return float(np.linalg.cond(regularized))
