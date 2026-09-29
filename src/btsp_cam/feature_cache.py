"""Validated frozen-ViT feature cache for practical CIFAR-100 runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _encode_metadata(metadata: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(
        json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )


def _decode_metadata(value: np.ndarray) -> dict[str, Any]:
    return json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8"))


def load_feature_cache(
    path: str | Path, expected_metadata: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    source = Path(path)
    if not source.exists():
        return None
    with np.load(source, allow_pickle=False) as cache:
        metadata = _decode_metadata(cache["metadata"])
        if metadata != dict(expected_metadata):
            raise ValueError(f"feature-cache metadata mismatch: {source}")
        features = cache["features"].copy()
        labels = cache["labels"].copy()
        indices = cache["indices"].copy()
    if features.ndim != 2 or labels.shape != (features.shape[0],) or indices.shape != labels.shape:
        raise ValueError(f"malformed feature cache: {source}")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"feature cache contains NaN or Inf: {source}")
    return features, labels, indices


def save_feature_cache(
    path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    metadata: Mapping[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (features.shape[0],) or indices.shape != labels.shape:
        raise ValueError("feature cache arrays have incompatible shapes")
    if not np.all(np.isfinite(features)):
        raise ValueError("refusing to cache NaN or Inf features")
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            metadata=_encode_metadata(metadata),
            features=features,
            labels=labels,
            indices=indices,
        )
    os.replace(temporary, target)
    return target


def extract_or_load_features(
    backbone,
    loader,
    path: str | Path,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    cached = load_feature_cache(path, metadata)
    if cached is not None:
        return (*cached, True)
    features, labels, indices = backbone.extract_loader(loader)
    save_feature_cache(path, features, labels, indices, metadata)
    return features, labels, indices, False
