#!/usr/bin/env python3
"""Prepare COCO val2014 images and pre-compute inception stats for FID.

Downloads COCO val2014 images (~6GB), then computes Inception-v3 activation
statistics (mu, sigma) and saves them as an .npz file for pytorch-fid.

Output: {data_root}/coco_val2014/
    val2014/              # ~40K JPEG images
    inception_stats.npz   # pre-computed (mu, sigma) for FID

Usage:
    python vendor/data_scripts/CFGpp-main/prepare_coco.py --data-root vendor/data
    # or via mlsbench:
    mlsbench data CFGpp-main
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


COCO_VAL_URL = "http://images.cocodataset.org/zips/val2014.zip"


def run_cmd(cmd, cwd=None):
    """Run a shell command, raising on failure."""
    print(f"  $ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def prepare_coco_images(coco_dir: Path):
    """Download and extract COCO val2014 images."""
    img_dir = coco_dir / "val2014"

    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 40000:
        print("[coco_val2014] Images already exist, skipping download")
        return img_dir

    coco_dir.mkdir(parents=True, exist_ok=True)
    zip_path = coco_dir / "val2014.zip"

    if not zip_path.exists():
        print("[coco_val2014] Downloading COCO val2014 (~6GB)...", flush=True)
        run_cmd(f"wget -q {COCO_VAL_URL} -O {zip_path}")
    else:
        print("[coco_val2014] Zip already exists, skipping download")

    print("[coco_val2014] Extracting...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(coco_dir))

    # Clean up zip to save space
    zip_path.unlink(missing_ok=True)
    print("[coco_val2014] Extraction done")

    assert img_dir.exists(), "COCO val2014 extraction failed"
    return img_dir


def prepare_inception_stats(img_dir: Path, npz_path: Path, batch_size: int = 50):
    """Compute Inception-v3 activation statistics and save as npz.

    COCO images have variable sizes, so we manually resize to 299x299
    (Inception input) instead of relying on pytorch_fid's default loader.
    """
    if npz_path.exists():
        print("[coco_val2014] Inception stats already exist, skipping")
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from pytorch_fid.inception import InceptionV3
    from PIL import Image
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[coco_val2014] Computing inception stats on {device}...", flush=True)

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).eval()

    # Custom dataset that resizes all images to 299x299
    class ResizedImageDataset(Dataset):
        def __init__(self, img_dir):
            self.files = sorted([
                f for f in img_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img)

    dataset = ResizedImageDataset(img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[coco_val2014] Processing {len(dataset)} images...", flush=True)

    all_acts = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            pred = model(batch)[0]  # (B, dims, 1, 1)
            pred = pred.squeeze(-1).squeeze(-1).cpu().numpy()
            all_acts.append(pred)

    all_acts = np.concatenate(all_acts, axis=0)
    mu = np.mean(all_acts, axis=0)
    sigma = np.cov(all_acts, rowvar=False)

    np.savez(str(npz_path), mu=mu, sigma=sigma)
    print(f"[coco_val2014] Saved inception stats to {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare COCO val2014 images and inception stats for FID"
    )
    parser.add_argument("--data-root", type=str, required=True, help="Root data directory")
    args = parser.parse_args()

    coco_dir = Path(args.data_root) / "coco_val2014"
    img_dir = prepare_coco_images(coco_dir)

    npz_path = coco_dir / "inception_stats.npz"
    prepare_inception_stats(img_dir, npz_path)

    print("\n[coco_val2014] Done")
    print(f"  Images:  {img_dir}")
    print(f"  Stats:   {npz_path}")


if __name__ == "__main__":
    main()
