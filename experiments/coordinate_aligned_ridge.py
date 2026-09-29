"""Experimental sparse-code C/D statistics for an evolving binary W_B.

The persistent state contains only C=sum(s s^T), D=sum(s y^T), and counts.
It never keeps individual codes, images, ViT embeddings, or CA1 features.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse

from btsp_cam.ridge import StreamingRidge


class CoordinateAlignedRidge:
    def __init__(self, code_dim: int, slots: int, k: int, gamma: float):
        self.code_dim = int(code_dim)
        self.slots = int(slots)
        self.k = int(k)
        self.gamma = float(gamma)
        if not 0 < self.k <= self.code_dim or self.slots <= 0 or self.gamma <= 0:
            raise ValueError("invalid coordinate-aligned ridge dimensions or gamma")
        self.c = sparse.csr_matrix((self.code_dim, self.code_dim), dtype=np.float64)
        self.d = np.zeros((self.code_dim, 0), dtype=np.float64)
        self.sample_count = 0
        self.last_update: dict[str, int | float] = {}
        self.last_alignment: dict[str, int | float] = {}

    def update(self, codes: np.ndarray, labels: np.ndarray) -> None:
        """Consume only the current stream batch and retain its aggregates."""
        start = time.perf_counter()
        s = np.asarray(codes)
        y = np.asarray(labels, dtype=np.int64)
        if s.ndim != 2 or s.shape[1] != self.code_dim or y.shape != (len(s),):
            raise ValueError("sparse codes or labels have the wrong shape")
        if len(s) == 0 or not np.all((s == 0) | (s == 1)):
            raise ValueError("codes must be nonempty and binary")
        if not np.all(s.sum(axis=1) == self.k):
            raise ValueError("codes must have exactly k active bits")
        if y.min() < 0:
            raise ValueError("labels must be nonnegative")
        old_nnz = self.c.nnz
        s_csr = sparse.csr_matrix(s, dtype=np.float64)
        gram = (s_csr.T @ s_csr).tocsr()
        self.c = (self.c + gram).tocsr()
        self.c.sum_duplicates()
        self.c.sort_indices()
        new_class_count = int(y.max()) + 1
        if new_class_count > self.d.shape[1]:
            self.d = np.pad(self.d, ((0, 0), (0, new_class_count - self.d.shape[1])))
        one_hot = np.zeros((len(s), self.d.shape[1]), dtype=np.float64)
        one_hot[np.arange(len(s)), y] = 1.0
        self.d += s_csr.T @ one_hot
        self.sample_count += len(s)
        self.last_update = {
            "samples": int(len(s)),
            "pair_contributions": int(len(s) * self.k * self.k),
            "task_gram_nnz": int(gram.nnz),
            "c_new_nnz": int(self.c.nnz - old_nnz),
            "update_seconds": float(time.perf_counter() - start),
        }

    def aligned_statistics(self, memory_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A=W C W^T/k² and B=W D/k for the *current* W_B."""
        start = time.perf_counter()
        w = np.asarray(memory_weights)
        if w.shape != (self.slots, self.code_dim) or not np.all((w == 0) | (w == 1)):
            raise ValueError("memory weights have the wrong shape or are not binary")
        w_csr = sparse.csr_matrix(w, dtype=np.float64)
        left = (w_csr @ self.c).tocsr()
        gram = (left @ w_csr.T).tocsr()
        a = gram.toarray()
        a /= self.k * self.k
        b = np.asarray(w_csr @ self.d, dtype=np.float64)
        b /= self.k
        self.last_alignment = {
            "w_nnz": int(w_csr.nnz),
            "wc_nnz": int(left.nnz),
            "aligned_a_nnz": int(gram.nnz),
            "alignment_seconds": float(time.perf_counter() - start),
        }
        return a, b

    def solve(self, a: np.ndarray, b: np.ndarray) -> tuple[StreamingRidge, float]:
        """Use the unchanged formal ridge solver; A/B are transient outputs."""
        start = time.perf_counter()
        ridge = StreamingRidge(self.slots, b.shape[1], self.gamma)
        ridge.a = np.asarray(a, dtype=np.float64)
        ridge.b = np.asarray(b, dtype=np.float64)
        ridge.solve()
        return ridge, float(time.perf_counter() - start)

    def storage(self) -> dict[str, int | float]:
        """Observed CSR size and dense/COO alternatives; binary MiB in report."""
        m = self.code_dim
        nnz = int(self.c.nnz)
        classes = int(self.d.shape[1])
        index_bytes = self.c.indices.dtype.itemsize
        pointer_bytes = self.c.indptr.dtype.itemsize
        c_csr_bytes = int(self.c.data.nbytes + self.c.indices.nbytes + self.c.indptr.nbytes)
        return {
            "C_nnz": nnz,
            "C_density": nnz / (m * m),
            "C_csr_float64_actual_bytes": c_csr_bytes,
            "C_csr_float32_est_bytes": nnz * (4 + index_bytes) + (m + 1) * pointer_bytes,
            "C_coo_float64_est_bytes": nnz * (8 + 2 * index_bytes),
            "C_coo_float32_est_bytes": nnz * (4 + 2 * index_bytes),
            "C_dense_float64_bytes": m * m * 8,
            "C_dense_float32_bytes": m * m * 4,
            "D_float64_actual_bytes": int(self.d.nbytes),
            "D_float32_est_bytes": m * classes * 4,
            "seen_classes": classes,
            "seen_samples": self.sample_count,
        }
