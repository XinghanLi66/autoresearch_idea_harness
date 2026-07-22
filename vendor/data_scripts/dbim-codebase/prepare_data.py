#!/usr/bin/env python3
"""Prepare all data for dbim-codebase tasks (sampler + scheduler).

Downloads and organises:
  1. Datasets: edges2handbags, DIODE-256, ImageNet
  2. Model checkpoints (adapted for current PyTorch)
  3. FID reference statistics
  4. Inception TorchScript model for FID evaluator

Output layout (maps 1:1 to /workspace/dbim-codebase/assets inside container):
  {data_root}/dbim_data/
  ├── datasets/
  │   ├── edges2handbags/{train,val}
  │   ├── DIODE-256/train/
  │   ├── ImageNet/{train,val}
  │   ├── val_faster_imagefolder_10k_fn.txt
  │   └── val_faster_imagefolder_10k_label.txt
  ├── ckpts/
  │   ├── e2h_ema_0.9999_420000_adapted.pt
  │   ├── diode_ema_0.9999_440000_adapted.pt
  │   └── imagenet256_inpaint_ema_0.9999_400000.pt
  ├── stats/
  │   ├── edges2handbags_ref_64_data.npz
  │   └── diode_ref_256_data.npz
  └── inception-2015-12-05.pt

Usage:
    python vendor/data_scripts/dbim-codebase/prepare_data.py --data-root vendor/data
    python vendor/data_scripts/dbim-codebase/prepare_data.py --data-root vendor/data --skip-imagenet
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd=None):
    """Run a shell command, raising on failure."""
    print(f"  $ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


# ---------------------------------------------------------------------------
# 1. edges2handbags
# ---------------------------------------------------------------------------
def prepare_edges2handbags(datasets_dir: Path):
    """Download and extract edges2handbags dataset."""
    out_dir = datasets_dir / "edges2handbags"
    if out_dir.exists() and (out_dir / "train").exists() and (out_dir / "val").exists():
        print("[edges2handbags] Already exists, skipping")
        return

    print("[edges2handbags] Downloading...", flush=True)
    tarball = datasets_dir / "edges2handbags.tar.gz"
    run_cmd(
        "wget -q http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2handbags.tar.gz"
        f" -O {tarball}"
    )

    print("[edges2handbags] Extracting...", flush=True)
    run_cmd(f"tar -xzf {tarball} -C {datasets_dir}")
    tarball.unlink(missing_ok=True)

    assert (out_dir / "train").exists(), "edges2handbags extraction failed"
    print("[edges2handbags] Done")


# ---------------------------------------------------------------------------
# 2. DIODE
# ---------------------------------------------------------------------------
def prepare_diode(datasets_dir: Path):
    """Download DIODE raw data and preprocess to DIODE-256."""
    out_dir = datasets_dir / "DIODE-256" / "train"
    if out_dir.exists() and len(list(out_dir.iterdir())) > 100:
        print("[DIODE] DIODE-256 already exists, skipping")
        return

    raw_dir = datasets_dir / "DIODE"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        ("http://diode-dataset.s3.amazonaws.com/train.tar.gz", "train.tar.gz"),
        ("http://diode-dataset.s3.amazonaws.com/train_normals.tar.gz", "train_normals.tar.gz"),
        ("https://diode-1254389886.cos.ap-hongkong.myqcloud.com/data_list.zip", "data_list.zip"),
    ]

    for url, fname in files_to_download:
        dest = raw_dir / fname
        if fname == "data_list.zip":
            extracted_marker = raw_dir / "data_list"
        elif fname == "train.tar.gz":
            extracted_marker = raw_dir / "train"
        else:
            extracted_marker = None

        if extracted_marker and extracted_marker.exists():
            print(f"[DIODE] {fname}: already extracted, skipping download")
            continue

        if not dest.exists():
            print(f"[DIODE] Downloading {fname}...", flush=True)
            run_cmd(f"wget -q {url} -O {dest}")

        print(f"[DIODE] Extracting {fname}...", flush=True)
        if fname.endswith(".tar.gz"):
            run_cmd(f"tar -xzf {dest} -C {raw_dir}")
        elif fname.endswith(".zip"):
            run_cmd(f"unzip -qo {dest} -d {raw_dir}")
        dest.unlink(missing_ok=True)

    print("[DIODE] Preprocessing to DIODE-256...", flush=True)
    _preprocess_diode(raw_dir, datasets_dir)
    print("[DIODE] Done")


def _preprocess_diode(raw_dir: Path, datasets_dir: Path):
    """Replicate preprocess_depth.py from DiffusionBridge repo."""
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from PIL import Image

    img_size = 256
    split = "train"
    data_csv = raw_dir / "data_list" / "train_outdoor.csv"
    target_dir = datasets_dir / f"DIODE-{img_size}" / split
    target_dir.mkdir(parents=True, exist_ok=True)

    all_files = []
    with open(data_csv, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            if row[-1] == "Unavailable":
                continue
            all_files.append(row[0].split("/")[-1])

    print(f"  Processing {len(all_files)} images...", flush=True)

    def plot_depth_map(dm, validity_mask, name):
        validity_mask = validity_mask > 0
        MIN_DEPTH = 0.5
        MAX_DEPTH = min(300, np.percentile(dm, 99))
        dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
        dm = np.log(dm, where=validity_mask)
        dm = np.ma.masked_where(~validity_mask, dm)
        cmap = plt.cm.get_cmap("jet")
        cmap.set_bad(color="black")
        norm = plt.Normalize(vmin=0, vmax=np.log(MAX_DEPTH + 1.01))
        image = cmap(norm(dm))
        plt.imsave(name, np.clip(image, 0.0, 1.0))

    def plot_normal_map(normal_map, name):
        normal_viz = normal_map[:, :, :]
        normal_viz = normal_viz + np.equal(
            np.sum(normal_viz, 2, keepdims=True), 0.0
        ).astype(np.float32) * np.min(normal_viz)
        normal_viz = (normal_viz - np.min(normal_viz)) / 2.0
        plt.imsave(name, np.clip(normal_viz, 0.0, 1.0))

    for i, file in enumerate(all_files):
        out_path = target_dir / file
        if out_path.exists():
            continue

        scene_id, scan_id = file.split("_")[0], file.split("_")[1]
        base_path = raw_dir / split / "outdoor" / f"scene_{scene_id}" / f"scan_{scan_id}"

        pil_image = (
            Image.open(base_path / file)
            .convert("RGB")
            .resize((img_size, img_size), Image.BICUBIC)
        )

        depth = np.load(str(base_path / (file[:-4] + "_depth.npy"))).squeeze().astype(np.float32)
        depth_mask = np.load(str(base_path / (file[:-4] + "_depth_mask.npy"))).astype(np.float32)
        normal = np.load(str(base_path / (file[:-4] + "_normal.npy"))).astype(np.float32)

        image_depth = cv2.resize(depth, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)
        image_depth_mask = cv2.resize(depth_mask, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)
        normal = cv2.resize(normal, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)

        pil_image.save(str(out_path))
        plot_depth_map(image_depth, image_depth_mask, str(target_dir / (file[:-4] + "_depth.png")))
        plot_normal_map(normal, str(target_dir / (file[:-4] + "_normal.png")))

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(all_files)}", flush=True)

    print(f"  Preprocessed {len(all_files)} images to {target_dir}")


# ---------------------------------------------------------------------------
# 3. ImageNet
# ---------------------------------------------------------------------------
def prepare_imagenet(datasets_dir: Path):
    """Download and extract ImageNet ILSVRC2012 for inpainting task."""
    out_dir = datasets_dir / "ImageNet"
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    if (
        train_dir.exists()
        and val_dir.exists()
        and len(list(train_dir.iterdir())) >= 1000
        and len(list(val_dir.iterdir())) >= 1000
    ):
        print("[ImageNet] Already exists, skipping")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

        train_tar = datasets_dir / "ILSVRC2012_img_train.tar"
        val_tar = datasets_dir / "ILSVRC2012_img_val.tar"

        if not train_tar.exists() and not (train_dir.exists() and len(list(train_dir.iterdir())) >= 1000):
            print("[ImageNet] Downloading train set (~138 GB)...", flush=True)
            run_cmd(
                f"wget --no-check-certificate -q"
                f" https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
                f" -O {train_tar}"
            )

        if not val_tar.exists() and not (val_dir.exists() and len(list(val_dir.iterdir())) >= 1000):
            print("[ImageNet] Downloading val set (~6.3 GB)...", flush=True)
            run_cmd(
                f"wget --no-check-certificate -q"
                f" https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
                f" -O {val_tar}"
            )

        if not train_dir.exists() or len(list(train_dir.iterdir())) < 1000:
            print("[ImageNet] Extracting train set...", flush=True)
            train_dir.mkdir(parents=True, exist_ok=True)
            run_cmd(f"tar -xf {train_tar} -C {train_dir}")
            run_cmd(
                'find . -name "*.tar" | while read NAME; do'
                ' mkdir -p "${NAME%.tar}";'
                ' tar -xf "${NAME}" -C "${NAME%.tar}";'
                ' rm -f "${NAME}";'
                " done",
                cwd=train_dir,
            )
            train_tar.unlink(missing_ok=True)

        if not val_dir.exists() or len(list(val_dir.iterdir())) < 1000:
            print("[ImageNet] Extracting val set...", flush=True)
            val_dir.mkdir(parents=True, exist_ok=True)
            run_cmd(f"tar -xf {val_tar} -C {val_dir}")
            run_cmd(
                "wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash",
                cwd=val_dir,
            )
            val_tar.unlink(missing_ok=True)

        print("[ImageNet] Extraction done")

    _download_val10k_files(datasets_dir)
    print("[ImageNet] Done")


def _download_val10k_files(datasets_dir: Path):
    """Download val_faster_imagefolder_10k_{fn,label}.txt from DiffusionBridge repo."""
    base_url = (
        "https://raw.githubusercontent.com/thu-ml/DiffusionBridge"
        "/92522733cc602686df77f07a1824bb89f89cda1a/assets/datasets"
    )
    for fname in [
        "val_faster_imagefolder_10k_fn.txt",
        "val_faster_imagefolder_10k_label.txt",
    ]:
        dest = datasets_dir / fname
        if dest.exists():
            print(f"[ImageNet] {fname} already exists, skipping")
            continue
        print(f"[ImageNet] Downloading {fname}...", flush=True)
        run_cmd(f"wget -q {base_url}/{fname} -O {dest}")


# ---------------------------------------------------------------------------
# 4. Checkpoints
# ---------------------------------------------------------------------------
def prepare_checkpoints(ckpts_dir: Path):
    """Download and adapt model checkpoints."""
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    HF_BASE = "https://huggingface.co/alexzhou907/DDBM/resolve/main"

    # --- edges2handbags ---
    raw_e2h = ckpts_dir / "e2h_ema_0.9999_420000.pt"
    adapted_e2h = ckpts_dir / "e2h_ema_0.9999_420000_adapted.pt"
    if not adapted_e2h.exists():
        if not raw_e2h.exists():
            print("[ckpts] Downloading e2h checkpoint...", flush=True)
            run_cmd(f"wget -q {HF_BASE}/e2h_ema_0.9999_420000.pt -O {raw_e2h}")
        print("[ckpts] Adapting e2h checkpoint...", flush=True)
        _adapt_e2h(raw_e2h, adapted_e2h)
    else:
        print("[ckpts] e2h adapted checkpoint already exists, skipping")

    # --- DIODE ---
    raw_diode = ckpts_dir / "diode_ema_0.9999_440000.pt"
    adapted_diode = ckpts_dir / "diode_ema_0.9999_440000_adapted.pt"
    if not adapted_diode.exists():
        if not raw_diode.exists():
            print("[ckpts] Downloading diode checkpoint...", flush=True)
            run_cmd(f"wget -q {HF_BASE}/diode_ema_0.9999_440000.pt -O {raw_diode}")
        print("[ckpts] Adapting diode checkpoint...", flush=True)
        _adapt_diode(raw_diode, adapted_diode)
    else:
        print("[ckpts] diode adapted checkpoint already exists, skipping")

    # --- ImageNet inpainting ---
    imagenet_ckpt = ckpts_dir / "imagenet256_inpaint_ema_0.9999_400000.pt"
    if not imagenet_ckpt.exists():
        print("[ckpts] Downloading ImageNet inpainting checkpoint (Google Drive)...", flush=True)
        gdrive_id = "1WozJyVOAFukj0nUYLS-ZUp1-QHuGNfox"
        run_cmd(f"gdown {gdrive_id} -O {imagenet_ckpt}")
    else:
        print("[ckpts] ImageNet checkpoint already exists, skipping")


def _adapt_e2h(raw_path: Path, adapted_path: Path):
    """Squeeze attention weights for e2h checkpoint (removes flash_attn dim)."""
    import torch
    sd = torch.load(raw_path, map_location="cpu")
    modules = []
    for i in range(5, 16):
        if i in (8, 12):
            continue
        modules.append(f"input_blocks.{i}.1.qkv.weight")
        modules.append(f"input_blocks.{i}.1.proj_out.weight")
    modules.append("middle_block.1.qkv.weight")
    modules.append("middle_block.1.proj_out.weight")
    for i in range(12):
        modules.append(f"output_blocks.{i}.1.qkv.weight")
        modules.append(f"output_blocks.{i}.1.proj_out.weight")
    for name in modules:
        sd[name] = sd[name].squeeze(-1)
    torch.save(sd, adapted_path)
    print(f"  Saved adapted checkpoint to {adapted_path}")


def _adapt_diode(raw_path: Path, adapted_path: Path):
    """Squeeze attention weights for diode checkpoint."""
    import torch
    sd = torch.load(raw_path, map_location="cpu")
    modules = []
    for i in range(10, 18):
        if i in (12, 15):
            continue
        modules.append(f"input_blocks.{i}.1.qkv.weight")
        modules.append(f"input_blocks.{i}.1.proj_out.weight")
    modules.append("middle_block.1.qkv.weight")
    modules.append("middle_block.1.proj_out.weight")
    for i in range(9):
        modules.append(f"output_blocks.{i}.1.qkv.weight")
        modules.append(f"output_blocks.{i}.1.proj_out.weight")
    for name in modules:
        sd[name] = sd[name].squeeze(-1)
    torch.save(sd, adapted_path)
    print(f"  Saved adapted checkpoint to {adapted_path}")


# ---------------------------------------------------------------------------
# 5. FID reference statistics
# ---------------------------------------------------------------------------
def prepare_stats(stats_dir: Path):
    """Download FID reference statistics."""
    stats_dir.mkdir(parents=True, exist_ok=True)

    HF_BASE = "https://huggingface.co/alexzhou907/DDBM/resolve/main"
    files = [
        (f"{HF_BASE}/edges2handbags_ref_64_data.npz",
         stats_dir / "edges2handbags_ref_64_data.npz"),
        (f"{HF_BASE}/diode_ref_256_data.npz",
         stats_dir / "diode_ref_256_data.npz"),
    ]
    for url, dest in files:
        if dest.exists():
            print(f"[stats] {dest.name} already exists, skipping")
        else:
            print(f"[stats] Downloading {dest.name}...", flush=True)
            run_cmd(f"wget -q {url} -O {dest}")


# ---------------------------------------------------------------------------
# 6. Inception model (for FID evaluator)
# ---------------------------------------------------------------------------
def prepare_inception(base_dir: Path):
    """Download Inception V3 TorchScript model for FID evaluation.

    The pre_edit points feature_extractor.py to look for inception at
    /workspace/dbim-codebase/assets/ which maps to base_dir.
    """
    inception_path = base_dir / "inception-2015-12-05.pt"
    if inception_path.exists():
        print("[inception] inception-2015-12-05.pt already exists, skipping")
    else:
        print("[inception] Downloading Inception V3 TorchScript...", flush=True)
        run_cmd(
            "wget -q https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/"
            f"pretrained/metrics/inception-2015-12-05.pt -O {inception_path}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare all data for dbim-codebase tasks"
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root data directory (e.g. vendor/data)")
    parser.add_argument("--skip-imagenet", action="store_true",
                        help="Skip ImageNet download (~144 GB, needs image-net.org)")
    parser.add_argument("--skip-diode", action="store_true",
                        help="Skip DIODE download (~20 GB)")
    args = parser.parse_args()

    # Base dir maps to /workspace/dbim-codebase/assets via data_bind
    base_dir = Path(args.data_root) / "dbim_data"
    base_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: edges2handbags dataset ===")
    prepare_edges2handbags(datasets_dir)

    if not args.skip_diode:
        print("\n=== Step 2: DIODE dataset ===")
        prepare_diode(datasets_dir)
    else:
        print("\n=== Step 2: DIODE (skipped) ===")

    if not args.skip_imagenet:
        print("\n=== Step 3: ImageNet dataset ===")
        prepare_imagenet(datasets_dir)
    else:
        print("\n=== Step 3: ImageNet (skipped) ===")

    print("\n=== Step 4: Checkpoints ===")
    prepare_checkpoints(base_dir / "ckpts")

    print("\n=== Step 5: FID reference statistics ===")
    prepare_stats(base_dir / "stats")

    print("\n=== Step 6: Inception model ===")
    prepare_inception(base_dir)

    print("\n=== All done ===")
    print(f"Output: {base_dir}/")
    print("Data bind: {data_root}/dbim_data:/workspace/dbim-codebase/assets")


if __name__ == "__main__":
    main()
