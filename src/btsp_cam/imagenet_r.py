"""PILOT-compatible ImageNet-R path protocol; no model algorithm changes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .protocol import pilot_class_order


CLASSES = 200
INCREMENT = 20
TASKS = 10
SEED = 1993


def pilot_imagenet_r_transform(train: bool):
    """Pinned PILOT utils/data.py build_transform, non-CODA path."""
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(0.75, 4 / 3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


@dataclass(frozen=True)
class Sample:
    relative_path: str
    remapped_label: int


class ImageNetRProtocol:
    """Supplied PILOT train/test ImageFolders with PILOT label remapping."""

    def __init__(self, root: str | Path):
        from torchvision.datasets import ImageFolder

        self.root = Path(root).expanduser().resolve()
        train_root, test_root = self.root / "train", self.root / "test"
        if not train_root.is_dir() or not test_root.is_dir():
            raise FileNotFoundError("ImageNet-R requires supplied train/ and test/ folders")
        train, test = ImageFolder(train_root), ImageFolder(test_root)
        if len(train.classes) != CLASSES or train.classes != test.classes:
            raise ValueError("ImageNet-R needs 200 identical, sorted train/test class folders")
        self.class_names = tuple(train.classes)
        self.order = pilot_class_order(CLASSES, SEED)
        inverse = np.empty(CLASSES, dtype=np.int64)
        inverse[self.order] = np.arange(CLASSES)
        self.samples: dict[str, tuple[Sample, ...]] = {}
        for split, folder in (("train", train), ("test", test)):
            items = tuple(Sample(Path(path).relative_to(self.root).as_posix(),
                                 int(inverse[label])) for path, label in folder.samples)
            labels = np.bincount([item.remapped_label for item in items], minlength=CLASSES)
            if np.any(labels == 0):
                raise ValueError(f"{split} is missing at least one ImageNet-R class")
            if len({item.relative_path for item in items}) != len(items):
                raise ValueError(f"duplicate {split} sample paths")
            self.samples[split] = items
        train_paths = {item.relative_path.removeprefix("train/") for item in self.samples["train"]}
        test_paths = {item.relative_path.removeprefix("test/") for item in self.samples["test"]}
        if train_paths & test_paths:
            raise ValueError("train/test relative path overlap")

    def task_samples(self, split: str, task: int) -> tuple[Sample, ...]:
        if split not in ("train", "test") or not 1 <= task <= TASKS:
            raise ValueError("invalid split or task")
        low, high = (task - 1) * INCREMENT, task * INCREMENT
        # PILOT DataManager.get_dataset concatenates one remapped class at a time.
        result = tuple(item for label in range(low, high)
                       for item in self.samples[split] if item.remapped_label == label)
        if not result:
            raise ValueError("empty task")
        return result

    def dataset(self, split: str, task: int):
        from PIL import Image
        from torch.utils.data import Dataset

        protocol = self
        samples = self.task_samples(split, task)
        transform = pilot_imagenet_r_transform(split == "train")

        class TaskDataset(Dataset):
            def __len__(self):
                return len(samples)

            def __getitem__(self, index):
                sample = samples[index]
                with Image.open(protocol.root / sample.relative_path) as image:
                    tensor = transform(image.convert("RGB"))
                return index, tensor, sample.remapped_label

        return TaskDataset()

    def manifest(self) -> dict:
        def record(item: Sample) -> dict:
            stat = (self.root / item.relative_path).stat()
            return dict(**item.__dict__, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

        return dict(root_layout="PILOT_ImageFolder_supplied_train_test",
                    class_names=list(self.class_names), class_order=self.order.tolist(),
                    splits={key: [record(item) for item in values]
                            for key, values in self.samples.items()})
