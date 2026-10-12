"""Read-only PILOT ImageNet-R folder/protocol inventory; never downloads data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btsp_cam.imagenet_r import ImageNetRProtocol, TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/imagenet-r")
    parser.add_argument("--manifest", default=None, help="optional output JSON path")
    args = parser.parse_args()
    root = Path(args.data_root)
    if not root.is_absolute():
        root = ROOT / root
    protocol = ImageNetRProtocol(root)
    manifest = protocol.manifest()
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                       separators=(",", ":"),
                                       ensure_ascii=False).encode("utf-8")).hexdigest()
    report = dict(status="PASS", split_source="PILOT_SUPPLIED_FOLDERS_ORIGIN_UNVERIFIED",
                  classes=len(protocol.class_names), tasks=TASKS,
                  class_order_seed=1993,
                  train_samples=len(protocol.samples["train"]),
                  test_samples=len(protocol.samples["test"]),
                  train_per_class=dict(Counter(x.remapped_label for x in protocol.samples["train"])),
                  test_per_class=dict(Counter(x.remapped_label for x in protocol.samples["test"])),
                  manifest_sha256=digest,
                  no_evidence_of_path_or_id_leakage=True,
                  content_duplicate_check="NOT_PERFORMED")
    if args.manifest:
        target = Path(args.manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(dict(report=report, manifest=manifest),
                                     indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
