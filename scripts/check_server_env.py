"""Read-only server environment readiness report; does not install packages."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def version(name: str):
    try:
        return importlib.import_module(name).__version__
    except (ImportError, AttributeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("IMAGENET_R_ROOT", "data/imagenet-r"))
    parser.add_argument("--output-dir", default="results/imagenet_r_k96")
    parser.add_argument("--cache-dir", default="cache/imagenet_r")
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except ImportError:
        cuda = False
        cuda_version = gpu = None
    try:
        import psutil
        ram_gib = psutil.virtual_memory().total / 2**30
    except ImportError:
        ram_gib = None
    disk = shutil.disk_usage(Path.cwd().anchor)
    dirs = {}
    locations = dict(output_dir=args.output_dir, cache_dir=args.cache_dir,
                     checkpoint_dir=str(Path(args.output_dir) / "checkpoints"),
                     replay_dir="experimental_replay_codes/imagenet_r")
    for name, value in locations.items():
        path = Path(value).expanduser().resolve()
        parent = next((p for p in (path, *path.parents) if p.exists()), None)
        dirs[name] = dict(path=str(path), writable=bool(parent and os.access(parent, os.W_OK)))
    report = dict(python=sys.version.split()[0], platform=platform.platform(),
                  torch=version("torch"), torchvision=version("torchvision"),
                  timm=version("timm"), numpy=version("numpy"), scipy=version("scipy"),
                  cuda_available=cuda, cuda_version=cuda_version, gpu_name=gpu,
                  disk_free_gib=disk.free / 2**30, ram_total_gib=ram_gib,
                  data_root=str(root), data_root_exists=root.is_dir(), dirs=dirs)
    if root.is_dir():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        try:
            from btsp_cam.imagenet_r import ImageNetRProtocol
            protocol = ImageNetRProtocol(root)
            report["class_count"] = len(protocol.class_names)
            report["train_image_count"] = len(protocol.samples["train"])
            report["test_image_count"] = len(protocol.samples["test"])
        except Exception as error:
            report["dataset_error"] = str(error)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
