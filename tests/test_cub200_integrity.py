"""Tiny original-metadata fixtures for CUB split reconstruction safeguards."""
from pathlib import Path
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_cub200_pilot_split import prepare


def fixture(tmp_path: Path) -> Path:
    source = tmp_path / "CUB_200_2011"
    names = ["001.Bird_A", "002.Bird_B"]
    entries = []
    for cid, name in enumerate(names, 1):
        for part in (1, 0):
            image_id = len(entries) + 1
            rel = f"{name}/bird_{image_id}.jpg"
            path = source / "images" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), color=(image_id * 20, 0, 0)).save(path)
            entries.append((image_id, rel, cid, part))
    (source / "classes.txt").write_text("1 001.Bird_A\n2 002.Bird_B\n")
    (source / "images.txt").write_text("".join(f"{i} {p}\n" for i, p, _, _ in entries))
    (source / "image_class_labels.txt").write_text("".join(f"{i} {c}\n" for i, _, c, _ in entries))
    (source / "train_test_split.txt").write_text("".join(f"{i} {s}\n" for i, _, _, s in entries))
    return source


def test_official_split_integrity_and_idempotency(tmp_path):
    source = fixture(tmp_path)
    output, diagnostics = tmp_path / "pilot", tmp_path / "diagnostics"
    first = prepare(source, output, diagnostics, expected_classes=2, expected_images=4)
    assert first["status"] == "PASS"
    assert (first["train_images"], first["test_images"]) == (2, 2)
    assert first["linked"] + first["copied"] == 4
    assert (output / "train/001.Bird_A/bird_1.jpg").is_file()
    assert (output / "test/002.Bird_B/bird_4.jpg").is_file()
    second = prepare(source, output, diagnostics, expected_classes=2, expected_images=4)
    assert second["skipped_existing"] == 4
    assert second["linked"] == second["copied"] == 0
    assert (diagnostics / "cub200_dataset_integrity.json").is_file()
    assert (diagnostics / "cub200_class_counts.csv").is_file()


def test_split_rejects_metadata_conflict_and_extra_image(tmp_path):
    source = fixture(tmp_path)
    output = tmp_path / "pilot"
    diagnostics = tmp_path / "diagnostics"
    split_file = source / "train_test_split.txt"
    split_file.write_text(split_file.read_text() + "1 0\n")
    with pytest.raises(ValueError, match="duplicate id"):
        prepare(source, output, diagnostics, expected_classes=2, expected_images=4)
    split_file.write_text(split_file.read_text().replace("1 0\n", ""))
    prepare(source, output, diagnostics, expected_classes=2, expected_images=4)
    Image.new("RGB", (4, 4)).save(output / "train/001.Bird_A/extra.jpg")
    with pytest.raises(ValueError, match="extra image"):
        prepare(source, output, diagnostics, expected_classes=2, expected_images=4)
