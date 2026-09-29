"""Class-incremental task partitioning without task-ID inference."""

from __future__ import annotations

import numpy as np


def pilot_class_order(num_classes: int, seed: int) -> np.ndarray:
    """Match PILOT DataManager's legacy NumPy permutation without global RNG."""

    # PILOT commit 7a6e904 uses np.random.seed(seed) followed by
    # np.random.permutation(len(order)). RandomState reproduces that legacy
    # MT19937 API exactly while avoiding mutation of NumPy's global RNG.
    return np.random.RandomState(seed).permutation(int(num_classes))


def equal_increment_tasks(
    num_classes: int, increment: int, seed: int
) -> list[np.ndarray]:
    """PILOT B0-IncN: no separate base session, first real task has N classes."""

    if num_classes % increment:
        raise ValueError("num_classes must be divisible by increment")
    order = pilot_class_order(num_classes, seed)
    return [order[i : i + increment] for i in range(0, num_classes, increment)]


def remap_to_incremental_labels(
    original_labels: np.ndarray, class_order: np.ndarray
) -> np.ndarray:
    """Match PILOT's order.index(label) mapping in vectorized form."""

    order = np.asarray(class_order, dtype=np.int64)
    inverse = np.empty(order.size, dtype=np.int64)
    inverse[order] = np.arange(order.size)
    labels = np.asarray(original_labels, dtype=np.int64)
    return inverse[labels]


def select_classes(
    features: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(labels, classes)
    return features[mask], labels[mask]
