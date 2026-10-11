"""Read-only, independent CUB official-split and cache provenance audit."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream/LAMDA-PILOT"))
sys.path.insert(0, str(ROOT / "src"))
from utils.data_manager import DataManager
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder

RAW = ROOT / "data/CUB_200_2011"
PILOT = ROOT / "data/cub"
FEATURES = ROOT / "data/feature_cache_cub200"
SPARSE = ROOT / "diagnostics/cub200_sparse_cache"
OUT = ROOT / "diagnostics"


def metadata_table(path: Path) -> dict[int, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"malformed metadata line in {path}: {line}")
        key = int(parts[0])
        if key in result:
            raise ValueError(f"duplicate image/class ID {key}: {path}")
        result[key] = parts[1]
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path: Path) -> np.uint64:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    return np.uint64(int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big"))


def near_pairs(train: list[tuple], test: list[tuple], limit: int = 50) -> list[dict]:
    train_bits = np.array([row[4] for row in train], dtype=np.uint64)
    test_bits = np.array([row[4] for row in test], dtype=np.uint64)
    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)
    candidates: list[tuple[int, int, int]] = []
    for start in range(0, len(train), 64):
        xor = (train_bits[start:start + 64, None] ^ test_bits[None, :]).view(np.uint8)
        distance = popcount[xor].reshape(min(64, len(train) - start), len(test), 8).sum(2)
        flat = distance.ravel()
        n = min(limit, flat.size)
        chosen = np.argpartition(flat, n - 1)[:n]
        for index in chosen:
            local, te = divmod(int(index), len(test))
            candidates.append((int(flat[index]), start + local, te))
        candidates.sort(key=lambda row: (row[0], train[row[1]][0], test[row[2]][0]))
        candidates = candidates[:limit]
    return [dict(train_path=train[i][0], test_path=test[j][0], train_class=train[i][1],
                 test_class=test[j][1], hash_distance=distance,
                 flag="NEEDS_REVIEW" if distance <= 3 else "DIAGNOSTIC_ONLY")
            for distance, i, j in candidates]


def audit_raw() -> tuple[dict, dict[str, dict], dict[int, str]]:
    images = metadata_table(RAW / "images.txt")
    splits = {k: int(v) for k, v in metadata_table(RAW / "train_test_split.txt").items()}
    labels = {k: int(v) for k, v in metadata_table(RAW / "image_class_labels.txt").items()}
    classes = metadata_table(RAW / "classes.txt")
    ids = set(images)
    assert len(ids) == 11788 and ids == set(splits) == set(labels)
    assert len(classes) == 200 and set(classes) == set(range(1, 201))
    assert len(set(images.values())) == 11788 and set(splits.values()) == {0, 1}
    assert sum(splits.values()) == 5994
    inverse = {path: image_id for image_id, path in images.items()}
    folders = {split: ImageFolder(PILOT / split) for split in ("train", "test")}
    assert folders["train"].class_to_idx == folders["test"].class_to_idx
    assert len(folders["train"].classes) == 200
    assert folders["train"].classes == [classes[c] for c in range(1, 201)]
    details: dict[str, dict] = {}
    paths, hashes, file_ids = {}, {}, {}
    hash_rows = defaultdict(list)
    drows = {"train": [], "test": []}
    digest = hashlib.sha256()
    for split, dataset in folders.items():
        paths[split], hashes[split], file_ids[split] = set(), set(), set()
        for path_string, imagefolder_index in dataset.samples:
            path = Path(path_string)
            canonical = path.relative_to(PILOT / split).as_posix()
            assert canonical in inverse
            image_id = inverse[canonical]
            assert splits[image_id] == (1 if split == "train" else 0)
            assert labels[image_id] - 1 == imagefolder_index
            assert canonical not in paths[split]
            paths[split].add(canonical)
            content = sha256(path)
            hashes[split].add(content)
            hash_rows[(split, content)].append((image_id, canonical))
            stat = path.stat()
            if stat.st_ino:
                file_ids[split].add((stat.st_dev, stat.st_ino))
            details[f"{split}:{image_id}"] = dict(split=split, image_id=image_id,
                  canonical_path=canonical, source_path=str(path.resolve()),
                  original_class_id=labels[image_id], imagefolder_class_index=imagefolder_index,
                  file_size=stat.st_size, sha256=content)
            digest.update(f"{split}/{canonical}\0{imagefolder_index}\0{content}\n".encode("utf-8"))
            drows[split].append((canonical, path.parent.name, image_id, imagefolder_index, dhash(path)))
    assert len(paths["train"]) == 5994 and len(paths["test"]) == 5794
    duplicate_hashes = hashes["train"] & hashes["test"]
    pairs = [dict(sha256=content, train=[dict(image_id=i, path=p) for i, p in hash_rows[("train", content)]],
                  test=[dict(image_id=i, path=p) for i, p in hash_rows[("test", content)]])
             for content in sorted(duplicate_hashes)]
    near = near_pairs(drows["train"], drows["test"])
    report = dict(status="PASS" if not (paths["train"] & paths["test"] or duplicate_hashes
                                         or file_ids["train"] & file_ids["test"]) else "FAIL",
                  metadata_image_ids=11788, classes=200, train_count=5994, test_count=5794,
                  metadata_id_sets_identical=True, image_path_sets_exhaustive=True,
                  canonical_path_overlap=len(paths["train"] & paths["test"]),
                  resolved_absolute_path_overlap=len({details[k]["source_path"] for k in details if k.startswith("train:")} &
                                                     {details[k]["source_path"] for k in details if k.startswith("test:")}),
                  exact_duplicate_hash_count=len(duplicate_hashes), exact_duplicate_pairs=pairs,
                  filesystem_identity_supported=bool(file_ids["train"] and file_ids["test"]),
                  filesystem_identity_overlap=len(file_ids["train"] & file_ids["test"]),
                  imagefolder_class_to_idx=folders["train"].class_to_idx,
                  dataset_sha256=digest.hexdigest(), near_duplicate_top50=near,
                  near_duplicate_review_count=sum(row["flag"] == "NEEDS_REVIEW" for row in near))
    (OUT / "cub200_raw_split_leakage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, details, images


def audit_caches(raw: dict, details: dict[str, dict]) -> dict:
    manager = DataManager("cub", True, 1993, 10, 10, {"dataset": "cub"})
    order = np.asarray(manager._class_order)
    assert manager.nb_tasks == 20 and np.array_equal(order, np.random.RandomState(1993).permutation(200))
    per_task = []
    id_sets = {"train": set(), "test": set()}
    provenance_rows = []
    canonical_to_id = {(value["split"], value["canonical_path"]): value["image_id"]
                       for value in details.values()}
    projection_seed = RNGProvenance.from_seed(1993).projection_seed
    encoder = RandomSparseEncoder(768, 8192, 96, projection_seed)
    for split in ("train", "test"):
        for task in range(1, 21):
            selected = np.arange((task - 1) * 10, task * 10)
            dataset = manager.get_dataset(selected, source=split, mode=split)
            feature_path = FEATURES / f"{split}_task{task:02d}.npz"
            sparse_path = SPARSE / f"{split}_task{task:02d}.npz"
            with np.load(feature_path, allow_pickle=False) as cache:
                meta = json.loads(cache["metadata"].tobytes().decode())
                features, labels, indices = cache["features"], cache["labels"], cache["indices"]
            with np.load(sparse_path, allow_pickle=False) as cache:
                sparse_meta = json.loads(cache["metadata"].tobytes().decode())
                packed, sparse_labels = cache["packed"], cache["labels"]
            count = len(dataset)
            assert features.shape == (count, 768) and labels.shape == indices.shape == (count,)
            assert packed.shape == (count, 1024) and sparse_labels.shape == (count,)
            assert np.array_equal(np.sort(indices), np.arange(count))
            assert np.array_equal(labels, np.asarray(dataset.labels)[indices])
            assert np.array_equal(sparse_labels, labels)
            assert np.all(np.unpackbits(packed, axis=1).sum(1) == 96)
            assert np.isfinite(features).all()
            reencoded = np.packbits(encoder.encode(features, 128), axis=1)
            assert np.array_equal(reencoded, packed), f"sparse row/code mismatch: {split} task {task}"
            assert (meta["split"], meta["task"], meta["dataset_sha256"], meta["dataset_sample_count"],
                    meta["class_order"], meta["original_classes"], meta["output_point"]) == (
                    split, task, raw["dataset_sha256"], 5994 if split == "train" else 5794,
                    order.tolist(), order[selected].tolist(), "final_LayerNorm_CLS_768D")
            assert (sparse_meta["split"], sparse_meta["task"], sparse_meta["dataset_sha256"],
                    sparse_meta["sample_count"], sparse_meta["k"], sparse_meta["density"],
                    Path(sparse_meta["source_feature_cache"]).resolve()) == (
                    split, task, raw["dataset_sha256"], count, 96, 96 / 8192, feature_path.resolve())
            assert sparse_meta["projection_seed"] == projection_seed
            for row, index in enumerate(indices):
                image_path = Path(dataset.images[int(index)]).resolve()
                canonical = image_path.relative_to(PILOT / split).as_posix()
                original_id = canonical_to_id[(split, canonical)]
                identity = f"{split}:{original_id}"
                record = details[identity]
                assert record["canonical_path"] == canonical
                assert int(labels[row]) == int(np.where(order == record["imagefolder_class_index"])[0][0])
                assert identity not in id_sets[split]
                id_sets[split].add(identity)
                provenance_rows.append(dict(sample_id=identity, source_path=str(image_path.resolve()),
                                            split=split, task=task, label=int(labels[row]),
                                            feature_row_index=row, sparse_row_index=row,
                                            feature_cache=feature_path.name, sparse_cache=sparse_path.name))
            per_task.append(dict(task=task, split=split, rows=count, feature_shape=list(features.shape),
                                 sparse_packed_shape=list(packed.shape), unique_source_indices=count,
                                 feature_sparse_label_mismatch=0, non_exact_k_rows=0,
                                 cache_metadata_valid=True))
    assert len(id_sets["train"]) == 5994 and len(id_sets["test"]) == 5794
    overlap = id_sets["train"] & id_sets["test"]
    report = dict(status="PASS" if not overlap else "FAIL", train_feature_count=5994,
                  test_feature_count=5794, train_sparse_count=5994, test_sparse_count=5794,
                  train_test_feature_id_overlap=len(overlap), train_test_sparse_id_overlap=len(overlap),
                  exact_k=96, source_index_order_and_labels_verified=True,
                  feature_sparse_row_correspondence_verified=True,
                  sparse_codes_recomputed_exactly_from_feature_rows=True,
                  dataset_sha256=raw["dataset_sha256"], per_task=per_task,
                  provenance_rows_file="cub200_cache_row_provenance.jsonl")
    with (OUT / "cub200_cache_row_provenance.jsonl").open("w", encoding="utf-8") as stream:
        for row in provenance_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "cub200_feature_cache_provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from project root: {ROOT}")
    OUT.mkdir(exist_ok=True)
    raw_report, detail_map, image_map = audit_raw()
    cache_report = audit_caches(raw_report, detail_map)
    print(json.dumps(dict(raw_status=raw_report["status"], hash_overlap=raw_report["exact_duplicate_hash_count"],
                          path_overlap=raw_report["canonical_path_overlap"],
                          near_review=raw_report["near_duplicate_review_count"],
                          cache_status=cache_report["status"],
                          cache_id_overlap=cache_report["train_test_feature_id_overlap"])))
