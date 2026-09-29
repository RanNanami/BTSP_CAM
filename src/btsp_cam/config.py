"""Validated model configuration with explicit parameter provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ParameterProvenance:
    value: Any
    source: str
    status: str

    @classmethod
    def from_value(cls, value: Any) -> "ParameterProvenance":
        return cls(value=value, source="legacy constructor", status="UNSPECIFIED")


@dataclass(frozen=True)
class BTSPCAMConfig:
    feature_dim: int
    code_dim: int
    k: int
    memory_slots: int
    num_classes: int
    base_plateau_probability: float
    beta: float
    flip_probability: float
    eta: float
    trace_decay: float
    tau: float
    gamma: float
    seed: int
    encoder_batch_size: int = 128
    provenance: dict[str, ParameterProvenance] = field(default_factory=dict)

    PARAMETER_NAMES = (
        "feature_dim",
        "code_dim",
        "k",
        "memory_slots",
        "num_classes",
        "base_plateau_probability",
        "beta",
        "flip_probability",
        "eta",
        "trace_decay",
        "tau",
        "gamma",
        "seed",
    )

    def __post_init__(self) -> None:
        integer_positive = {
            "feature_dim": self.feature_dim,
            "code_dim": self.code_dim,
            "k": self.k,
            "memory_slots": self.memory_slots,
            "num_classes": self.num_classes,
            "encoder_batch_size": self.encoder_batch_size,
        }
        for name, value in integer_positive.items():
            if int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.k > self.code_dim:
            raise ValueError("k must satisfy 0 < k <= code_dim")
        finite = {
            "base_plateau_probability": self.base_plateau_probability,
            "beta": self.beta,
            "flip_probability": self.flip_probability,
            "eta": self.eta,
            "trace_decay": self.trace_decay,
            "tau": self.tau,
            "gamma": self.gamma,
        }
        for name, value in finite.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.base_plateau_probability <= 1.0:
            raise ValueError("base_plateau_probability must be in [0, 1]")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if not 0.0 <= self.trace_decay <= 1.0:
            raise ValueError("trace_decay must be in [0, 1]")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BTSPCAMConfig":
        section = mapping.get("parameters", mapping)
        values: dict[str, Any] = {}
        provenance: dict[str, ParameterProvenance] = {}
        for name in cls.PARAMETER_NAMES:
            if name not in section:
                raise ValueError(f"missing model parameter: {name}")
            entry = section[name]
            if isinstance(entry, Mapping):
                if "value" not in entry or "source" not in entry or "status" not in entry:
                    raise ValueError(f"{name} must provide value, source, and status")
                values[name] = entry["value"]
                provenance[name] = ParameterProvenance(
                    value=entry["value"],
                    source=str(entry["source"]),
                    status=str(entry["status"]),
                )
            else:
                values[name] = entry
                provenance[name] = ParameterProvenance.from_value(entry)
        values["encoder_batch_size"] = int(mapping.get("encoder_batch_size", 128))
        values["provenance"] = provenance
        return cls(**values)

    @classmethod
    def from_legacy_kwargs(cls, **kwargs: Any) -> "BTSPCAMConfig":
        values = {name: kwargs.pop(name) for name in cls.PARAMETER_NAMES}
        values["encoder_batch_size"] = int(kwargs.pop("encoder_batch_size", 128))
        if kwargs:
            raise TypeError(f"unexpected model arguments: {sorted(kwargs)}")
        values["provenance"] = {
            name: ParameterProvenance.from_value(value) for name, value in values.items()
            if name != "encoder_batch_size"
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = {
            name: {
                "value": getattr(self, name),
                "source": self.provenance.get(
                    name, ParameterProvenance.from_value(getattr(self, name))
                ).source,
                "status": self.provenance.get(
                    name, ParameterProvenance.from_value(getattr(self, name))
                ).status,
            }
            for name in self.PARAMETER_NAMES
        }
        return {"parameters": result, "encoder_batch_size": self.encoder_batch_size}

