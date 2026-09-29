"""PILOT-compatible CIFAR-100/cifar224 data path."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .protocol import remap_to_incremental_labels


def build_pilot_cifar224_transform(train: bool):
    """Reproduce PILOT commit 7a6e904 utils/data.py for non-CODA methods."""

    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


class IndexedRemappedSubset:
    """Select original CIFAR classes and emit PILOT-remapped labels."""

    def __init__(
        self,
        dataset,
        original_classes: Sequence[int],
        class_order: np.ndarray,
    ):
        self.dataset = dataset
        targets = np.asarray(dataset.targets, dtype=np.int64)
        self.indices = np.flatnonzero(np.isin(targets, np.asarray(original_classes)))
        self.labels = remap_to_incremental_labels(targets[self.indices], class_order)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int):
        source_index = int(self.indices[position])
        image, _ = self.dataset[source_index]
        return source_index, image, int(self.labels[position])


def load_cifar100(root: str, train: bool, download: bool = True):
    from torchvision.datasets import CIFAR100

    return CIFAR100(
        root=root,
        train=train,
        download=download,
        transform=build_pilot_cifar224_transform(train),
    )
