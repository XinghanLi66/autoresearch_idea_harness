#!/usr/bin/env python3
"""Download and prepare few-shot classification datasets for easy-few-shot-learning.

Creates:
    <data_root>/mini_imagenet/images/<class>/<image>.JPEG
    <data_root>/cifar_fs/images/<class>/<image>.png
    <data_root>/cifar_fs/{train,val,test}.json
    <data_root>/CUB/images/<class>/<image>.jpg
"""
import argparse
import json
import os
import shutil
import tarfile
import pickle
from pathlib import Path

import numpy as np


def prepare_mini_imagenet(data_root: Path):
    """Prepare miniImageNet from learn2learn cached pickle files.

    Re-uses the learn2learn data already downloaded during package build.
    Converts from pkl cache to class-organized image directories.
    """
    out_dir = data_root / "mini_imagenet" / "images"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[mini_imagenet] Already exists at {out_dir}, skipping")
        return

    # Find learn2learn data
    l2l_root = Path(__file__).resolve().parent.parent.parent / "external_packages" / "l2l_data"
    if not l2l_root.exists():
        l2l_root = data_root.parent / "external_packages" / "l2l_data"

    cache_files = {
        "train": l2l_root / "mini-imagenet-cache-train.pkl",
        "validation": l2l_root / "mini-imagenet-cache-validation.pkl",
        "test": l2l_root / "mini-imagenet-cache-test.pkl",
    }

    for split, path in cache_files.items():
        if not path.exists():
            print(f"[mini_imagenet] Cache file missing: {path}")
            print("[mini_imagenet] Please run: mlsbench build learn2learn first")
            return

    print("[mini_imagenet] Converting learn2learn cache to image directories...")
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    for split, cache_path in cache_files.items():
        print(f"  Processing {split}...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        # l2l cache format: {"image_data": tensor, "class_dict": {class_name: [indices]}}
        image_data = data["image_data"]  # numpy array [N, 84, 84, 3]
        class_dict = data["class_dict"]   # {class_name: [idx, ...]}

        for class_name, indices in class_dict.items():
            class_dir = out_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i, idx in enumerate(indices):
                img_array = image_data[idx]
                img = Image.fromarray(img_array.astype(np.uint8))
                img_path = class_dir / f"{class_name}_{i:05d}.JPEG"
                if not img_path.exists():
                    img.save(img_path)

    print(f"[mini_imagenet] Done. Images at {out_dir}")


def prepare_cifar_fs(data_root: Path):
    """Prepare CIFAR-FS from learn2learn's downloaded CIFAR-100.

    Uses Bertinetto et al. (2019) standard 64/16/20 split.
    """
    out_img_dir = data_root / "cifar_fs" / "images"
    out_spec_dir = data_root / "cifar_fs"

    if (out_img_dir.exists() and any(out_img_dir.iterdir()) and
        (out_spec_dir / "train.json").exists()):
        print(f"[cifar_fs] Already exists at {out_spec_dir}, skipping")
        return

    # Find learn2learn CIFAR-FS data
    l2l_root = Path(__file__).resolve().parent.parent.parent / "external_packages" / "l2l_data"
    if not l2l_root.exists():
        l2l_root = data_root.parent / "external_packages" / "l2l_data"

    cifarfs_dir = l2l_root / "cifarfs"
    cifar100_data = cifarfs_dir / "cifar100" / "data"
    splits_dir = cifarfs_dir / "cifar100" / "splits" / "bertinetto"

    if not cifar100_data.exists() or not splits_dir.exists():
        print(f"[cifar_fs] CIFAR-FS data missing at {cifar100_data} or {splits_dir}")
        print("[cifar_fs] Please run: mlsbench build learn2learn first")
        return

    print("[cifar_fs] Converting learn2learn CIFAR-FS to image directories...")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_spec_dir.mkdir(parents=True, exist_ok=True)

    # Read Bertinetto splits
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}.txt"
        with open(split_file) as f:
            classes = [line.strip() for line in f if line.strip()]
        splits[split_name] = classes

    # Copy images for all classes used in splits
    all_classes = set()
    for class_list in splits.values():
        all_classes.update(class_list)

    for class_name in sorted(all_classes):
        src_class = cifar100_data / class_name
        if not src_class.exists():
            print(f"  Warning: class directory not found: {src_class}")
            continue
        dst_class = out_img_dir / class_name
        if not dst_class.exists():
            shutil.copytree(src_class, dst_class)

    # Create JSON specs
    for split_name, class_list in splits.items():
        class_roots = [f"./data/cifar_fs/images/{cn}" for cn in sorted(class_list)]
        spec = {
            "class_names": sorted(class_list),
            "class_roots": class_roots,
        }
        spec_path = out_spec_dir / f"{split_name}.json"
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"  Wrote {spec_path} ({len(class_list)} classes)")

    print(f"[cifar_fs] Done. Images at {out_img_dir}")


def prepare_cub(data_root: Path):
    """Prepare CUB-200-2011 for few-shot learning.

    Downloads from official source and organizes into class directories.
    """
    out_dir = data_root / "CUB" / "images"
    spec_dir = data_root / "CUB"

    if (out_dir.exists() and any(out_dir.iterdir()) and
        (spec_dir / "train.json").exists()):
        print(f"[CUB] Already exists at {out_dir}, skipping")
        return

    print("[CUB] Downloading CUB-200-2011...")
    import urllib.request

    tar_path = data_root / "CUB_200_2011.tgz"
    if not tar_path.exists():
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, tar_path)

    print("  Extracting...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(data_root)

    src_images = data_root / "CUB_200_2011" / "images"
    if not src_images.exists():
        print(f"[CUB] Extraction failed: {src_images} not found")
        return

    # Copy images
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted(src_images.iterdir()):
        if class_dir.is_dir():
            dst = out_dir / class_dir.name
            if not dst.exists():
                shutil.copytree(class_dir, dst)

    # Create train/val/test splits
    # Standard CUB few-shot split: 100/50/50
    all_classes = sorted([d.name for d in out_dir.iterdir() if d.is_dir()])
    n_total = len(all_classes)
    print(f"  Found {n_total} classes")

    # Use alphabetical order split (standard in many papers)
    train_classes = all_classes[:100]
    val_classes = all_classes[100:150]
    test_classes = all_classes[150:200]

    for split_name, class_list in [("train", train_classes), ("val", val_classes), ("test", test_classes)]:
        class_roots = [f"./data/CUB/images/{cn}" for cn in class_list]
        spec = {
            "class_names": class_list,
            "class_roots": class_roots,
        }
        spec_path = spec_dir / f"{split_name}.json"
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"  Wrote {spec_path} ({len(class_list)} classes)")

    # Cleanup
    if tar_path.exists():
        tar_path.unlink()
    cub_extracted = data_root / "CUB_200_2011"
    if cub_extracted.exists():
        shutil.rmtree(cub_extracted)

    print(f"[CUB] Done. Images at {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare few-shot classification datasets")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory for dataset storage")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "mini_imagenet", "cifar_fs", "cub"],
                        help="Which dataset to prepare")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "mini_imagenet"):
        prepare_mini_imagenet(data_root)

    if args.dataset in ("all", "cifar_fs"):
        prepare_cifar_fs(data_root)

    if args.dataset in ("all", "cub"):
        prepare_cub(data_root)


if __name__ == "__main__":
    main()
