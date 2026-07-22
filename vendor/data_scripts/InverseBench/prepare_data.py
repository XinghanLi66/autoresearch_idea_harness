"""Prepare data for InverseBench package.

Downloads pretrained diffusion models and test datasets for inverse problem benchmarks.
Run via: mlsbench data InverseBench

Creates:
  <data_root>/inversebench/checkpoints/{inv-scatter-5m.pt, blackhole-50k.pt}
  <data_root>/inversebench/inv-scatter-test/
  <data_root>/inversebench/blackhole/{test/, measure/}
  <data_root>/inversebench/cache/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


CHECKPOINTS = {
    "inv-scatter-5m.pt": "https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/inv-scatter-5m.pt",
    "blackhole-50k.pt": "https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/blackhole-50k.pt",
}

TEST_DATA = {
    "inv-scatter-test": "https://sdsc.osn.xsede.org/ini230004-bucket01/zg89b-mpv16/inv-scatter-test.zip",
    "blackhole": "https://sdsc.osn.xsede.org/ini230004-bucket01/zg89b-mpv16/blackhole.zip",
}

# Torch Hub cached weights used by runtime evaluators (e.g. piq.LPIPS for ffhq256
# inpainting). Compute nodes have NO network access, so these must be pre-fetched
# at build time. We write them under <data_root>/inversebench/cache/torch/hub/
# which the container mounts at /workspace/InverseBench/cache/torch, and set
# TORCH_HOME to that path via pkg_configs/InverseBench/config.json env.
TORCH_HUB_CHECKPOINTS = {
    "lpips_weights.pt": "https://github.com/photosynthesis-team/photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt",
    # piq.LPIPS loads a VGG16 backbone via torchvision, which fetches this file
    # through torch.hub. Pre-download it so offline compute nodes can construct
    # the LPIPS metric without network access.
    "vgg16-397923af.pth": "https://download.pytorch.org/models/vgg16-397923af.pth",
}


def download(url: str, dest: str) -> None:
    print(f"  Downloading {url} -> {dest}")
    subprocess.run(["wget", "-q", "-O", dest, url], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.data_root) / "inversebench"

    # Checkpoints
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name, url in CHECKPOINTS.items():
        dest = ckpt_dir / name
        if dest.exists():
            print(f"  [SKIP] {name} already exists")
        else:
            download(url, str(dest))

    # Test data
    for name, url in TEST_DATA.items():
        dest_dir = root / name
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(f"  [SKIP] {name} already exists")
        else:
            zip_path = f"/tmp/{name}.zip"
            download(url, zip_path)
            subprocess.run(["unzip", "-oq", zip_path, "-d", str(root)], check=True)
            os.remove(zip_path)
            print(f"  Extracted {name}")

    # Cache dir
    (root / "cache").mkdir(parents=True, exist_ok=True)

    # Torch Hub checkpoints (for runtime evaluators with no network access).
    # TORCH_HOME=/workspace/InverseBench/cache/torch in the container, so torch.hub
    # resolves weights at /workspace/InverseBench/cache/torch/hub/checkpoints/.
    hub_dir = root / "cache" / "torch" / "hub" / "checkpoints"
    hub_dir.mkdir(parents=True, exist_ok=True)
    for name, url in TORCH_HUB_CHECKPOINTS.items():
        dest = hub_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [SKIP] torch hub {name} already exists")
        else:
            download(url, str(dest))

    # Verify
    checks = [
        ckpt_dir / "inv-scatter-5m.pt",
        ckpt_dir / "blackhole-50k.pt",
        root / "inv-scatter-test",
        root / "blackhole" / "test",
        root / "blackhole" / "measure",
        hub_dir / "lpips_weights.pt",
        hub_dir / "vgg16-397923af.pth",
    ]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"ERROR: Missing: {missing}", file=sys.stderr)
        sys.exit(1)
    print("All InverseBench data verified.")


if __name__ == "__main__":
    main()
