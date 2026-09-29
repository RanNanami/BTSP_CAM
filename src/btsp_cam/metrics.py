"""Small metric helpers used by the smoke CIL runner."""

from __future__ import annotations

import numpy as np


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    y = np.asarray(labels)
    pred = np.asarray(predictions)
    if y.shape != pred.shape:
        raise ValueError("labels and predictions must have the same shape")
    return float((y == pred).mean())


def average_incremental_accuracy(stage_accuracies: list[float]) -> float:
    return float(np.mean(stage_accuracies))
