"""Runnable, checkpointable BTSP-CAM core with paper-ordered task updates."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .binary_memory import BinaryMemory
from .ca1 import ca1_features
from .config import BTSPCAMConfig
from .eligibility import ClassEligibilityTraces
from .metrics import accuracy
from .plateau import plateau_probability, sample_slot_plateaus
from .ridge import StreamingRidge
from .rng import RNGProvenance
from .sparse_encoder import RandomSparseEncoder


def _json_bytes(value: Any) -> np.ndarray:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ).encode("utf-8")
    return np.frombuffer(payload, dtype=np.uint8)


def _from_json_bytes(value: np.ndarray) -> Any:
    return json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8"))


class BTSPCAM:
    """Independent implementation of the paper's BTSP-CAM core.

    ``BTSPCAM(config)`` is the preferred API. The keyword constructor used by
    the first-round tests remains supported for compatibility.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        config: BTSPCAMConfig | Mapping[str, Any] | None = None,
        *,
        backbone=None,
        class_order: np.ndarray | None = None,
        backbone_metadata: Mapping[str, Any] | None = None,
        rng_provenance: RNGProvenance | Mapping[str, int] | None = None,
        projection_seed: int | None = None,
        plateau_seed: int | None = None,
        synapse_seed: int | None = None,
        **legacy_parameters: Any,
    ):
        if config is None:
            config = BTSPCAMConfig.from_legacy_kwargs(**legacy_parameters)
        elif legacy_parameters:
            raise TypeError("do not mix a config object with legacy model parameters")
        elif not isinstance(config, BTSPCAMConfig):
            config = BTSPCAMConfig.from_mapping(config)
        self.config = config

        if rng_provenance is None:
            rng_provenance = RNGProvenance.from_seed(config.seed)
        if isinstance(rng_provenance, RNGProvenance):
            provenance_dict = rng_provenance.as_dict()
        else:
            provenance_dict = {key: int(value) for key, value in rng_provenance.items()}
        self.rng_provenance = provenance_dict
        projection_seed = (
            provenance_dict["projection_seed"] if projection_seed is None else int(projection_seed)
        )
        plateau_seed = (
            provenance_dict["plateau_seed"] if plateau_seed is None else int(plateau_seed)
        )
        synapse_seed = (
            provenance_dict["synapse_seed"] if synapse_seed is None else int(synapse_seed)
        )

        self.encoder = RandomSparseEncoder(
            config.feature_dim, config.code_dim, config.k, projection_seed
        )
        self.traces = ClassEligibilityTraces(
            config.num_classes, config.code_dim, config.eta, config.trace_decay
        )
        self.memory = BinaryMemory(config.memory_slots, config.code_dim)
        self.ridge = StreamingRidge(config.memory_slots, 0, config.gamma)
        self.plateau_rng = np.random.default_rng(plateau_seed)
        self.synapse_rng = np.random.default_rng(synapse_seed)
        self.backbone = None
        self.backbone_metadata = dict(backbone_metadata or {})
        self.current_task_index = 0
        self.seen_classes = np.empty(0, dtype=np.int64)
        self.train_order_state = {"next_task_index": 0}
        self.last_task_summary: dict[str, Any] = {}
        self.task_metrics_history: list[dict[str, Any]] = []

        if class_order is None:
            class_order = np.arange(config.num_classes, dtype=np.int64)
        self.class_order = np.asarray(class_order, dtype=np.int64).copy()
        if (
            self.class_order.shape != (config.num_classes,)
            or np.unique(self.class_order).size != config.num_classes
            or self.class_order.min() != 0
            or self.class_order.max() != config.num_classes - 1
        ):
            raise ValueError("class_order must be a permutation of all configured classes")
        if backbone is not None:
            self.attach_backbone(backbone)

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    def attach_backbone(self, backbone) -> None:
        if int(backbone.output_dim) != self.config.feature_dim:
            raise ValueError("backbone output dimension does not match model config")
        runtime_metadata = {
            "model_name": str(backbone.model_name),
            "timm_version": str(backbone.timm_version),
            "output_dim": int(backbone.output_dim),
            "pretrained_cfg": dict(backbone.pretrained_cfg),
            "eval": not bool(backbone.model.training),
            "all_parameters_frozen": not any(
                parameter.requires_grad for parameter in backbone.model.parameters()
            ),
        }
        if self.backbone_metadata:
            for key in ("model_name", "output_dim"):
                if key in self.backbone_metadata and self.backbone_metadata[key] != runtime_metadata[key]:
                    raise ValueError(f"loaded checkpoint backbone metadata mismatch: {key}")
        self.backbone_metadata = runtime_metadata
        self.backbone = backbone

    def _learn_code(self, code: np.ndarray, label: int) -> dict[str, float]:
        # PAPER_SPECIFICATION_AMBIGUITY: Algorithm 1 explicitly updates the
        # eligibility trace before computing familiarity. Consequently, a new
        # class's first sample can have F=1, N=0, and f_q_eff=0. We preserve the
        # printed order and do not implement an alternative-order ablation.
        self.traces.update(code, int(label))
        signals = self.traces.signals(code, int(label), self.config.beta)
        probability = plateau_probability(
            self.config.base_plateau_probability, signals
        )
        plateaus = sample_slot_plateaus(
            probability, self.memory.slots, self.plateau_rng
        )
        flips = self.memory.update(
            code, plateaus, self.config.flip_probability, self.synapse_rng
        )
        # Ephemeral diagnostics only. The task learner consumes this before the
        # next sample; it does not affect learning or checkpoint state.
        self._last_sample_plateaus = plateaus
        record = asdict(signals)
        record.update(
            plateau_probability=probability,
            plateau_rate=float(plateaus.mean()),
            plateau_count=float(plateaus.sum()),
            flips=float(flips),
            zero_to_one_flips=float(self.memory.last_zero_to_one),
            one_to_zero_flips=float(self.memory.last_one_to_zero),
        )
        return record

    def ca1_from_codes(self, codes: np.ndarray) -> np.ndarray:
        return ca1_features(codes, self.memory.weights, self.config.tau)

    def ca1(self, features: np.ndarray) -> np.ndarray:
        codes = self.encoder.encode(features, self.config.encoder_batch_size)
        return self.ca1_from_codes(codes)

    def learn_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        task_id: int | None = None,
    ) -> list[dict[str, float]]:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.config.feature_dim:
            raise ValueError("features must be a non-empty [N, feature_dim] array")
        if y.shape != (x.shape[0],):
            raise ValueError("labels must have shape [N]")
        if not np.all(np.isfinite(x)):
            raise ValueError("features contain NaN or Inf")
        if np.any((y < 0) | (y >= self.config.num_classes)):
            raise ValueError("labels are outside the configured class range")
        expected_task = self.current_task_index + 1
        if task_id is not None and int(task_id) != expected_task:
            raise ValueError(f"expected task_id={expected_task}, received {task_id}")

        codes = self.encoder.encode(x, self.config.encoder_batch_size)
        # Phase 1: finish every memory update before constructing any Z_t.
        task_slot_plateau_counts = np.zeros(self.config.memory_slots, dtype=np.int64)
        logs = []
        for code, label in zip(codes, y):
            logs.append(self._learn_code(code, int(label)))
            task_slot_plateau_counts += self._last_sample_plateaus
        self.last_task_slot_plateau_counts = task_slot_plateau_counts
        # Phase 2: freeze current W_B, construct current-task Z_t once, and add
        # only its sufficient statistics. No old raw image or feature is kept.
        z = self.ca1_from_codes(codes)
        self.ridge.expand_classes(int(y.max()) + 1)
        self.ridge.update(z, y)
        self.ridge.solve()

        self.current_task_index = expected_task
        self.seen_classes = np.unique(np.concatenate((self.seen_classes, y)))
        self.train_order_state = {"next_task_index": self.current_task_index}
        self.last_task_summary = {
            "task": self.current_task_index,
            "train_samples": int(y.size),
            "seen_classes": self.seen_classes.tolist(),
            "mean_plateau_probability": float(
                np.mean([entry["plateau_probability"] for entry in logs])
            ),
            "plateau_rate": float(np.mean([entry["plateau_rate"] for entry in logs])),
            "memory_active_fraction": self.memory.active_fraction,
            "ca1_nonzero_density": float(np.count_nonzero(z) / z.size),
            "ridge_solve_status": self.ridge.last_solve_status,
            "ridge_condition_number": self.ridge.condition_number,
        }
        return logs

    def _extract_loader(self, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.backbone is None:
            raise RuntimeError("a frozen backbone must be attached for image loaders")
        return self.backbone.extract_loader(loader)

    def learn_task(self, data, labels: np.ndarray | None = None, *, task_id: int | None = None):
        """Learn one task from frozen features or an attached-backbone loader."""

        if labels is not None:
            return self.learn_features(data, labels, task_id=task_id)
        features, loader_labels, indices = self._extract_loader(data)
        logs = self.learn_features(features, loader_labels, task_id=task_id)
        return {
            "logs": logs,
            "sample_count": int(loader_labels.size),
            "unique_source_indices": int(np.unique(indices).size),
            **self.last_task_summary,
        }

    def logits(
        self, features: np.ndarray, seen_classes: np.ndarray | None = None
    ) -> np.ndarray:
        logits = self.ridge.logits(self.ca1(features))
        if seen_classes is not None:
            allowed = np.zeros(self.ridge.num_classes, dtype=bool)
            indices = np.asarray(seen_classes, dtype=np.int64)
            if np.any((indices < 0) | (indices >= self.ridge.num_classes)):
                raise ValueError("seen class is outside the fitted ridge dimension")
            allowed[indices] = True
            logits[:, ~allowed] = -np.inf
        return logits

    def predict(
        self, features: np.ndarray, seen_classes: np.ndarray | None = None
    ) -> np.ndarray:
        if self.ridge.num_classes == 0:
            raise RuntimeError("the classifier has not learned any class")
        return self.logits(features, seen_classes).argmax(axis=1)

    def evaluate(
        self,
        data,
        labels: np.ndarray | None = None,
        *,
        seen_classes: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if labels is None:
            features, labels, indices = self._extract_loader(data)
        else:
            features = np.asarray(data)
            labels = np.asarray(labels, dtype=np.int64)
            indices = np.arange(labels.size, dtype=np.int64)
        classes = self.seen_classes if seen_classes is None else np.asarray(seen_classes)
        predictions = self.predict(features, classes)
        return {
            "accuracy": accuracy(labels, predictions),
            "sample_count": int(labels.size),
            "predictions": predictions,
            "labels": labels,
            "source_indices": indices,
        }

    def save(self, path: str | Path, *, compressed: bool = False) -> Path:
        """Atomically save all learned state needed for exact continuation."""

        target = Path(path)
        if target.suffix.lower() != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not np.all((self.memory.weights == 0) | (self.memory.weights == 1)):
            raise ValueError("refusing to save a non-binary memory")
        for name, value in (
            ("projection", self.encoder.projection),
            ("bias", self.encoder.bias),
            ("eligibility", self.traces.values),
            ("ridge_a", self.ridge.a),
            ("ridge_b", self.ridge.b),
            ("ridge_weights", self.ridge.weights),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"refusing to save non-finite {name}")
        metadata = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "config": self.config.to_dict(),
            "rng_provenance": self.rng_provenance,
            "projection_rng_state": self.encoder.rng.bit_generator.state,
            "plateau_rng_state": self.plateau_rng.bit_generator.state,
            "synapse_rng_state": self.synapse_rng.bit_generator.state,
            "train_order_state": self.train_order_state,
            "backbone_metadata": self.backbone_metadata,
            "current_task_index": self.current_task_index,
            "ridge_num_classes": self.ridge.num_classes,
            "ridge_solve_status": getattr(self.ridge, "last_solve_status", "not_solved"),
            "ridge_condition_number": getattr(
                self.ridge, "last_condition_number", None
            ),
            "last_task_summary": self.last_task_summary,
            "task_metrics_history": self.task_metrics_history,
        }
        arrays = {
            "metadata": _json_bytes(metadata),
            "projection": self.encoder.projection,
            "bias": self.encoder.bias,
            "eligibility_values": self.traces.values,
            "eligibility_decay": self.traces.decay,
            "memory_weights": self.memory.weights,
            "ridge_a": self.ridge.a,
            "ridge_b": self.ridge.b,
            "ridge_weights": self.ridge.weights,
            "seen_classes": self.seen_classes,
            "class_order": self.class_order,
        }
        temporary = target.with_name(target.name + ".tmp")
        writer = np.savez_compressed if compressed else np.savez
        with temporary.open("wb") as handle:
            writer(handle, **arrays)
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path, *, backbone=None) -> "BTSPCAM":
        source = Path(path)
        with np.load(source, allow_pickle=False) as state:
            metadata = _from_json_bytes(state["metadata"])
            if metadata.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
                raise ValueError("unsupported BTSP-CAM checkpoint version")
            config = BTSPCAMConfig.from_mapping(metadata["config"])
            model = cls(
                config,
                class_order=state["class_order"],
                backbone_metadata=metadata["backbone_metadata"],
                rng_provenance=metadata["rng_provenance"],
            )
            model.encoder.projection = state["projection"].copy()
            model.encoder.bias = state["bias"].copy()
            model.traces.values = state["eligibility_values"].copy()
            model.traces.decay = state["eligibility_decay"].copy()
            model.memory.weights = state["memory_weights"].copy()
            model.ridge.a = state["ridge_a"].copy()
            model.ridge.b = state["ridge_b"].copy()
            model.ridge.weights = state["ridge_weights"].copy()
            model.ridge.num_classes = int(metadata["ridge_num_classes"])
            model.seen_classes = state["seen_classes"].copy()

        expected_shapes = {
            "projection": (config.code_dim, config.feature_dim),
            "bias": (config.code_dim,),
            "eligibility": (config.num_classes, config.code_dim),
            "memory": (config.memory_slots, config.code_dim),
            "ridge_a": (config.memory_slots, config.memory_slots),
            "ridge_b": (config.memory_slots, model.ridge.num_classes),
            "ridge_weights": (config.memory_slots, model.ridge.num_classes),
        }
        actual_shapes = {
            "projection": model.encoder.projection.shape,
            "bias": model.encoder.bias.shape,
            "eligibility": model.traces.values.shape,
            "memory": model.memory.weights.shape,
            "ridge_a": model.ridge.a.shape,
            "ridge_b": model.ridge.b.shape,
            "ridge_weights": model.ridge.weights.shape,
        }
        if actual_shapes != expected_shapes:
            raise ValueError(f"checkpoint array shape mismatch: {actual_shapes}")
        if not np.all((model.memory.weights == 0) | (model.memory.weights == 1)):
            raise ValueError("checkpoint memory is not binary")
        for value in (
            model.encoder.projection,
            model.encoder.bias,
            model.traces.values,
            model.ridge.a,
            model.ridge.b,
            model.ridge.weights,
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError("checkpoint contains NaN or Inf")

        model.encoder.rng.bit_generator.state = metadata["projection_rng_state"]
        model.plateau_rng.bit_generator.state = metadata["plateau_rng_state"]
        model.synapse_rng.bit_generator.state = metadata["synapse_rng_state"]
        model.train_order_state = dict(metadata["train_order_state"])
        model.current_task_index = int(metadata["current_task_index"])
        model.last_task_summary = dict(metadata.get("last_task_summary", {}))
        model.task_metrics_history = list(metadata.get("task_metrics_history", []))
        model.ridge.last_solve_status = metadata["ridge_solve_status"]
        if metadata.get("ridge_condition_number") is not None:
            model.ridge.last_condition_number = float(metadata["ridge_condition_number"])
        if backbone is not None:
            model.attach_backbone(backbone)
        return model
