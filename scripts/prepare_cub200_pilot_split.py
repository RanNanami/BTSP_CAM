"""Materialize PILOT's ImageFolder layout from the original CUB-200-2011 metadata.

The official train_test_split.txt is authoritative. Existing images are never
overwritten: identical files are skipped and conflicting/extra files are errors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder


def read_index(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        image_id = int(key)
        if image_id in result:
            raise ValueError(f"duplicate id {image_id} in {path}")
        result[image_id] = value
    return result


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def archive_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(source: Path, output: Path, diagnostics: Path, *, expected_classes: int = 200,
            expected_images: int = 11788) -> dict:
    source, output, diagnostics = source.resolve(), output.resolve(), diagnostics.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and PILOT output must be independent directories")
    names = read_index(source / "classes.txt")
    images = read_index(source / "images.txt")
    labels = {k: int(v) for k, v in read_index(source / "image_class_labels.txt").items()}
    split = {k: int(v) for k, v in read_index(source / "train_test_split.txt").items()}
    ids = set(images)
    if (len(names) != expected_classes or len(ids) != expected_images
            or ids != set(labels) or ids != set(split)
            or set(names) != set(range(1, expected_classes + 1))
            or any(value not in (0, 1) for value in split.values())):
        raise ValueError("CUB metadata count, ID set, class IDs, or split values invalid")
    if len(set(images.values())) != len(images) or len(set(names.values())) != len(names):
        raise ValueError("duplicate image path or class name in metadata")
    source_image_root = source / "images"
    actual_source_images = {p.relative_to(source_image_root).as_posix()
                            for p in source_image_root.rglob("*") if p.is_file()}
    if actual_source_images != set(images.values()):
        raise ValueError(f"raw image directory differs from metadata: missing={len(set(images.values()) - actual_source_images)}, "
                         f"extra={len(actual_source_images - set(images.values()))}")
    expected: dict[str, dict[Path, Path]] = {"train": {}, "test": {}}
    counts = {cid: {"train": 0, "test": 0} for cid in names}
    source_hashes: dict[Path, str] = {}
    for image_id, relative in images.items():
        rel = Path(relative)
        cid = labels[image_id]
        if (cid not in names or rel.is_absolute() or ".." in rel.parts
                or len(rel.parts) != 2 or rel.parts[0] != names[cid]):
            raise ValueError(f"invalid class/path for image id {image_id}: {relative}")
        src = (source / "images" / rel).resolve()
        if source / "images" not in src.parents or not src.is_file():
            raise FileNotFoundError(src)
        with Image.open(src) as image:
            image.verify()
        source_hashes[src] = digest(src)
        part = "train" if split[image_id] == 1 else "test"
        dst = output / part / rel
        if dst in expected[part]:
            raise ValueError(f"duplicate target path: {dst}")
        expected[part][dst] = src
        counts[cid][part] += 1
    if any(not all(counts[cid].values()) for cid in names):
        raise ValueError("class missing from a split")
    # Reject conflicts and extras before creating any links/copies.
    for part in ("train", "test"):
        root = output / part
        if root.exists():
            actual_dirs = {p.name for p in root.iterdir() if p.is_dir()}
            if actual_dirs - set(names.values()):
                raise ValueError(f"extra class directories in {root}")
            actual_files = {p.resolve() for p in root.rglob("*") if p.is_file()}
            if actual_files - set(expected[part]):
                raise ValueError(f"extra image files in {root}")
        for dst, src in expected[part].items():
            if dst.exists() and digest(dst) != source_hashes[src]:
                raise ValueError(f"conflicting existing image: {dst}")
    linked = copied = skipped = 0
    for part in ("train", "test"):
        for dst, src in expected[part].items():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                skipped += 1
                continue
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1
            if digest(dst) != source_hashes[src]:
                raise ValueError(f"copy/link verification failed: {dst}")
    train, test = ImageFolder(output / "train"), ImageFolder(output / "test")
    ordered_names = [names[cid] for cid in sorted(names)]
    if train.classes != ordered_names or test.classes != ordered_names:
        raise ValueError("PILOT ImageFolder class index mismatch")
    if len(train) + len(test) != expected_images:
        raise ValueError("PILOT ImageFolder image count mismatch")
    if set(train.imgs) & set(test.imgs):
        raise ValueError("train/test path overlap")
    # PILOT DataManager uses np.random.seed(1993); np.random.permutation(n).
    order = np.random.RandomState(1993).permutation(expected_classes).tolist()
    remap = {original_index: incremental for incremental, original_index in enumerate(order)}
    source_manifest = hashlib.sha256()
    for image_id in sorted(ids):
        src = source / "images" / images[image_id]
        source_manifest.update(f"{image_id}\0{images[image_id]}\0{split[image_id]}\0{source_hashes[src]}\n".encode())
    report = {
        "status": "PASS", "source": str(source), "output": str(output),
        "split_rule": "official train_test_split.txt: 1=train, 0=test",
        "classes": len(names), "images": len(images), "train_images": len(train),
        "test_images": len(test), "metadata_image_ids_unique": True,
        "metadata_id_sets_equal": True, "no_missing_or_extra_images": True,
        "all_images_opened": True, "train_test_paths_disjoint": True,
        "train_test_class_dirs_identical": True,
        "imagefolder_class_to_idx": train.class_to_idx,
        "pilot_original_to_incremental_label": remap,
        "pilot_class_order_seed": 1993, "pilot_original_class_order": order,
        "source_manifest_sha256": source_manifest.hexdigest(),
        "linked": linked, "copied": copied, "skipped_existing": skipped,
    }
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "cub200_dataset_integrity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (diagnostics / "cub200_class_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class_id", "class_name", "train_count", "test_count", "total_count"])
        writer.writeheader()
        for cid in sorted(names):
            tr, te = counts[cid]["train"], counts[cid]["test"]
            writer.writerow(dict(class_id=cid, class_name=names[cid], train_count=tr,
                                 test_count=te, total_count=tr + te))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="extracted CUB_200_2011 directory")
    parser.add_argument("--archive", type=Path, help="original archive; require published MD5 before preparation")
    parser.add_argument("--output", type=Path, default=Path("data/cub"))
    parser.add_argument("--diagnostics", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    if args.archive:
        actual_md5 = archive_md5(args.archive)
        if actual_md5 != "97eceeb196236b17998738112f37df78":
            raise ValueError(f"original CUB archive MD5 mismatch: {actual_md5}")
    report = prepare(args.source, args.output, args.diagnostics)
    if args.archive:
        report["archive"] = str(args.archive.resolve())
        report["archive_md5"] = actual_md5
        (args.diagnostics / "cub200_dataset_integrity.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
