"""Split and task-access safeguards for the supplied PILOT folder protocol."""

from pathlib import Path

import pytest

from btsp_cam.imagenet_r import ImageNetRProtocol
from test_imagenet_r_protocol import fake_imagenet_r


def test_no_train_test_ids_or_paths_overlap(fake_imagenet_r: Path):
    protocol = ImageNetRProtocol(fake_imagenet_r)
    train = {x.relative_path for x in protocol.samples["train"]}
    test = {x.relative_path for x in protocol.samples["test"]}
    assert not train & test
    assert all(x.relative_path.startswith("train/") for x in protocol.task_samples("train", 1))
    assert all(x.relative_path.startswith("test/") for x in protocol.task_samples("test", 1))
    assert not set(x.remapped_label for x in protocol.task_samples("train", 1)) & set(
        x.remapped_label for x in protocol.task_samples("train", 2))


def test_reject_relative_path_overlap(fake_imagenet_r: Path):
    source = fake_imagenet_r / "test" / "n000" / "te.png"
    source.rename(source.with_name("tr.png"))
    with pytest.raises(ValueError, match="overlap"):
        ImageNetRProtocol(fake_imagenet_r)


def test_missing_test_class_rejected(fake_imagenet_r: Path):
    (fake_imagenet_r / "test" / "n000" / "te.png").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        ImageNetRProtocol(fake_imagenet_r)
