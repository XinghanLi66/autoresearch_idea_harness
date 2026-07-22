#!/usr/bin/env python3
"""Download VGG16 and LPIPS weights for perceptual loss in VAE training.

Output:
  {data_root}/diffusers_data/hub/checkpoints/vgg16-397923af.pth
  {data_root}/diffusers_data/hub/checkpoints/lpips_vgg-*.pth (via lpips)

The host directory is mounted as TORCH_HOME=/data/pretrained in the container,
so torchvision and lpips find cached weights at the standard paths.

Usage:
    python vendor/data_scripts/diffusers-main/prepare_data.py --data-root vendor/data
"""

import argparse
import os
import subprocess
from pathlib import Path

VGG16_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"


def main(data_root: str):
    ckpt_dir = Path(data_root) / "diffusers_data" / "hub" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # VGG16 weights (for torchvision.models.vgg16(pretrained=True))
    vgg_path = ckpt_dir / "vgg16-397923af.pth"
    if vgg_path.exists():
        print(f"VGG16 weights already at {vgg_path}")
    else:
        print("Downloading VGG16 weights...")
        subprocess.check_call([
            "wget", "-q", "--show-progress", "-O", str(vgg_path), VGG16_URL,
        ])
        print(f"Done: {vgg_path} ({vgg_path.stat().st_size / 1e6:.1f} MB)")

    # LPIPS VGG weights (for lpips.LPIPS(net='vgg'))
    torch_home = Path(data_root) / "diffusers_data"
    env = os.environ.copy()
    env["TORCH_HOME"] = str(torch_home)
    try:
        subprocess.check_call([
            "python", "-c",
            "import lpips; lpips.LPIPS(net='vgg'); print('LPIPS VGG cached OK')",
        ], env=env)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: could not pre-cache LPIPS weights (lpips not installed on host)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    args = p.parse_args()
    main(args.data_root)
