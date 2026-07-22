"""Prepare data for HypSeek package.

Downloads UniMol pretrained weights, ESM2 model, and virtual screening training/validation data.
Run via: mlsbench data HypSeek

Creates:
  <data_root>/HypSeek/pretrain/mol_pre_no_h_220816.pt
  <data_root>/HypSeek/pretrain/pocket_pre_220816.pt
  <data_root>/HypSeek/hf_cache/  (ESM2 model cache)
  <data_root>/HypSeek/vs_data/train_lig_all_blend.lmdb
  <data_root>/HypSeek/vs_data/train_prot_all_blend.lmdb
  <data_root>/HypSeek/vs_data/train_label_seq/  (training labels)
  <data_root>/HypSeek/vs_data/valid_label_seq.json
  <data_root>/HypSeek/vs_data/valid_lig.lmdb
  <data_root>/HypSeek/vs_data/valid_prot.lmdb
  <data_root>/HypSeek/vs_data/uniport80.clstr
  <data_root>/HypSeek/vs_data/uniport40.clstr
  <data_root>/HypSeek/vs_data/pocket_name2idx_train_blend.json
  <data_root>/HypSeek/vs_data/mol_smi2idx_train_blend.json
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PRETRAINED_WEIGHTS = {
    "mol_pre_no_h_220816.pt": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt",
    "pocket_pre_220816.pt": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt",
}

VS_DATA_FILES = {
    "train_lig_all_blend.lmdb": "https://ndownloader.figshare.com/files/50992728",
    "train_prot_all_blend.lmdb": "https://ndownloader.figshare.com/files/50992194",
    "valid_label_seq.json": "https://ndownloader.figshare.com/files/50992173",
    "valid_lig.lmdb": "https://ndownloader.figshare.com/files/50992176",
    "valid_prot.lmdb": "https://ndownloader.figshare.com/files/50992182",
    "uniport80.clstr": "https://ndownloader.figshare.com/files/54239555",
    "uniport40.clstr": "https://ndownloader.figshare.com/files/54239558",
    "pocket_name2idx_train_blend.json": "https://ndownloader.figshare.com/files/50992185",
    "mol_smi2idx_train_blend.json": "https://ndownloader.figshare.com/files/50992191",
}

# train_label is a zip that needs extraction
TRAIN_LABEL = {
    "url": "https://ndownloader.figshare.com/files/50992179",
    "archive": "train_label.zip",
}


def download(url: str, dest: str) -> None:
    print(f"  Downloading {url} -> {dest}")
    subprocess.run(["wget", "-q", "-O", dest, url], check=True)


def cache_esm2(hf_cache_dir: str) -> None:
    """Download and cache ESM2 model for offline use."""
    print("  Caching ESM2 model (facebook/esm2_t12_35M_UR50D)...")
    subprocess.run(
        [
            sys.executable, "-c",
            f"import os; os.environ['HF_HOME'] = '{hf_cache_dir}'; "
            "from transformers import AutoTokenizer, AutoModelForMaskedLM; "
            "AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D'); "
            "AutoModelForMaskedLM.from_pretrained('facebook/esm2_t12_35M_UR50D'); "
            "print('ESM2 cached')"
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.data_root) / "HypSeek"
    root.mkdir(parents=True, exist_ok=True)

    # Download pretrained weights
    pretrain_dir = root / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    for name, url in PRETRAINED_WEIGHTS.items():
        dest = pretrain_dir / name
        if dest.exists():
            print(f"  [SKIP] pretrain/{name} already exists")
        else:
            download(url, str(dest))
            print(f"  Downloaded pretrain/{name}")

    # Cache ESM2 model
    hf_cache_dir = root / "hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    # Check if already cached by looking for model files
    hub_dir = hf_cache_dir / "hub"
    if hub_dir.exists() and any(hub_dir.glob("models--facebook--esm2*")):
        print("  [SKIP] ESM2 model already cached")
    else:
        cache_esm2(str(hf_cache_dir))

    # Download VS training/validation data
    vs_dir = root / "vs_data"
    vs_dir.mkdir(parents=True, exist_ok=True)

    for name, url in VS_DATA_FILES.items():
        dest = vs_dir / name
        if dest.exists():
            print(f"  [SKIP] vs_data/{name} already exists")
        else:
            download(url, str(dest))
            print(f"  Downloaded vs_data/{name}")

    # Download and extract train_label zip
    train_label_dir = vs_dir / "train_label_seq"
    if train_label_dir.exists() and any(train_label_dir.iterdir()):
        print("  [SKIP] vs_data/train_label_seq already exists")
    else:
        zip_path = f"/tmp/{TRAIN_LABEL['archive']}"
        download(TRAIN_LABEL["url"], zip_path)
        subprocess.run(["unzip", "-oq", zip_path, "-d", str(vs_dir)], check=True)
        os.remove(zip_path)
        print("  Extracted train_label_seq")

    # Create placeholder files for FEP data
    for name in ["fep_repeat_ligands_can.json", "fep_assays.json"]:
        dest = vs_dir / name
        if not dest.exists():
            dest.write_text("[]")
            print(f"  Created placeholder {name}")

    # Create cache dir
    (vs_dir / "cache").mkdir(parents=True, exist_ok=True)

    # Verify
    checks = [
        pretrain_dir / "mol_pre_no_h_220816.pt",
        pretrain_dir / "pocket_pre_220816.pt",
        vs_dir / "train_lig_all_blend.lmdb",
        vs_dir / "train_prot_all_blend.lmdb",
        vs_dir / "valid_label_seq.json",
        vs_dir / "valid_lig.lmdb",
        vs_dir / "valid_prot.lmdb",
    ]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"ERROR: Missing: {missing}", file=sys.stderr)
        sys.exit(1)
    print("All HypSeek data verified.")


if __name__ == "__main__":
    main()
