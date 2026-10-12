"""Protocol parity tests use synthetic ImageFolders, not ImageNet-R results."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from btsp_cam.imagenet_r import ImageNetRProtocol, pilot_imagenet_r_transform
from btsp_cam.protocol import pilot_class_order


@pytest.fixture
def fake_imagenet_r(tmp_path: Path) -> Path:
    for split, stem in (("train", "tr"), ("test", "te")):
        for class_id in range(200):
            folder = tmp_path / split / f"n{class_id:03d}"
            folder.mkdir(parents=True)
            Image.new("RGB", (8, 8), color=(class_id % 255, 4, 7)).save(folder / f"{stem}.png")
    return tmp_path


def test_class_order_and_tasks_match_pilot(fake_imagenet_r):
    protocol = ImageNetRProtocol(fake_imagenet_r)
    assert np.array_equal(protocol.order, pilot_class_order(200, 1993))
    assert protocol.order.tolist() == np.random.RandomState(1993).permutation(200).tolist()
    train_tasks = [protocol.task_samples("train", task) for task in range(1, 11)]
    assert all(len(task) == 20 for task in train_tasks)
    assert [set(x.remapped_label for x in task) for task in train_tasks] == [
        set(range(i * 20, (i + 1) * 20)) for i in range(10)]
    assert len(set(x.relative_path for task in train_tasks for x in task)) == 200
    assert len(protocol.task_samples("test", 1)) == 20


def test_non_coda_pilot_preprocess(fake_imagenet_r):
    protocol = ImageNetRProtocol(fake_imagenet_r)
    transform = pilot_imagenet_r_transform(False)
    assert [type(x).__name__ for x in transform.transforms] == ["Resize", "CenterCrop", "ToTensor"]
    train = pilot_imagenet_r_transform(True)
    assert [type(x).__name__ for x in train.transforms] == [
        "RandomResizedCrop", "RandomHorizontalFlip", "ToTensor"]
    _, tensor, label = protocol.dataset("test", 1)[0]
    assert tuple(tensor.shape) == (3, 224, 224)
    assert 0 <= label < 20
