"""Named, reproducible RNG provenance for the BTSP-CAM pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RNGProvenance:
    base_seed: int
    class_order_seed: int
    train_order_seed: int
    augmentation_seed: int
    projection_seed: int
    plateau_seed: int
    synapse_seed: int

    @classmethod
    def from_seed(cls, seed: int) -> "RNGProvenance":
        seed = int(seed)
        children = np.random.SeedSequence(seed).spawn(5)
        derived = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
        return cls(
            base_seed=seed,
            # Protocol parity requires the literal PILOT seed here.
            class_order_seed=seed,
            train_order_seed=derived[0],
            augmentation_seed=derived[1],
            projection_seed=derived[2],
            plateau_seed=derived[3],
            synapse_seed=derived[4],
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
