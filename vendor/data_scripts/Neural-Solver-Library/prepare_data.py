#!/usr/bin/env python3
"""Prepare data for Neural-Solver-Library tasks (PDE benchmarks).

Downloads FNO, PDEBench, AirfRANS, and ShapeNet-Car datasets.

Output: {data_root}/data_staging/{fno,PDEBench,AirfRANS,AirCraft,PDE_data}/

Usage:
    python vendor/data_scripts/Neural-Solver-Library/prepare_data.py --data-root vendor/data
"""

import argparse
import os
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file if it doesn't already exist."""
    if dest.exists():
        print(f"  {desc or dest.name}: already exists")
        return
    print(f"  Downloading {desc or dest.name}...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Done: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def download_gdrive(file_id: str, dest: Path, desc: str = ""):
    """Download a file from Google Drive."""
    if dest.exists():
        print(f"  {desc or dest.name}: already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name} from GDrive...", flush=True)
    # Use gdown if available, otherwise urllib
    try:
        import gdown
        gdown.download(id=file_id, output=str(dest), quiet=False)
    except ImportError:
        # Fallback: direct URL with confirmation bypass
        page = urllib.request.urlopen(
            f"https://drive.google.com/uc?export=download&id={file_id}"
        ).read().decode()
        uuid_match = re.search(r'uuid.*?value="([^"]+)', page)
        if uuid_match:
            uuid = uuid_match.group(1)
            url = (f"https://drive.usercontent.google.com/download?"
                   f"id={file_id}&export=download&confirm=t&uuid={uuid}")
            urllib.request.urlretrieve(url, str(dest))
        else:
            raise RuntimeError(f"Cannot extract download UUID for GDrive file {file_id}")
    print(f"  Done: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def prepare_fno(staging_dir: Path):
    """Download FNO benchmark datasets."""
    fno_dir = staging_dir / "fno"
    fno_dir.mkdir(parents=True, exist_ok=True)

    # FNO airfoil/pipe/darcy data from GDrive
    datasets = {
        "airfoil": "1D_Pipe_A_Nu1.000_T0_N2000_D32",
        "darcy": "Darcy_421",
        "pipe": "naca",
    }

    print("\n--- FNO datasets ---")
    # These are typically hosted on Google Drive or custom URLs
    # The exact file IDs may need to be updated
    for name, folder in datasets.items():
        target = fno_dir / name
        if target.exists():
            print(f"  {name}: already exists")
        else:
            print(f"  {name}: needs manual download or GDrive ID")
            print(f"    Expected at: {target}")


def prepare_pdebench(staging_dir: Path):
    """Download PDEBench datasets from DARUS."""
    pde_dir = staging_dir / "PDEBench"

    # DARUS dataset file IDs (University of Stuttgart)
    downloads = [
        ("2D/SWE/2D_rdb_NA_NA.h5", "https://darus.uni-stuttgart.de/api/access/datafile/133021"),
        ("2D/DiffReact/2D_diff-react_NA_NA.h5", "https://darus.uni-stuttgart.de/api/access/datafile/133017"),
        ("1D/Burgers/1D_Burgers_Sols_Nu0.001.hdf5", "https://darus.uni-stuttgart.de/api/access/datafile/268190"),
    ]

    print("\n--- PDEBench datasets ---")
    for rel_path, url in downloads:
        dest = pde_dir / rel_path
        download_file(url, dest, rel_path)


def prepare_airfrans(staging_dir: Path):
    """Download AirfRANS dataset."""
    af_dir = staging_dir / "AirfRANS"

    print("\n--- AirfRANS dataset ---")
    if af_dir.exists() and any(af_dir.iterdir()):
        print("  AirfRANS: already exists")
        return

    zip_path = staging_dir / "AirfRANS.zip"
    download_file(
        "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip",
        zip_path,
        "AirfRANS",
    )

    print("  Extracting AirfRANS...")
    af_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(af_dir))
    zip_path.unlink()
    print("  Done")


def prepare_aircraft(staging_dir: Path):
    """Download ShapeNet-Car (AirCraft) dataset from Google Drive."""
    ac_dir = staging_dir / "AirCraft"

    print("\n--- AirCraft (ShapeNet-Car) dataset ---")
    if ac_dir.exists() and any(ac_dir.iterdir()):
        print("  AirCraft: already exists")
        return

    # Google Drive file ID for AirCraft dataset
    file_id = "1UDGgtOM8UYBFbDe_t2FP7Ij9N5SA3w-g"
    zip_path = staging_dir / "AirCraft.zip"
    download_gdrive(file_id, zip_path, "AirCraft")

    print("  Extracting AirCraft...")
    ac_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(ac_dir))
    zip_path.unlink()
    print("  Done")


def main():
    parser = argparse.ArgumentParser(description="Prepare Neural-Solver-Library data")
    parser.add_argument("--data-root", required=True, help="Root data directory")
    args = parser.parse_args()

    staging_dir = Path(args.data_root) / "data_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    print("=== Preparing Neural-Solver-Library data ===")

    prepare_fno(staging_dir)
    prepare_pdebench(staging_dir)
    prepare_airfrans(staging_dir)
    prepare_aircraft(staging_dir)

    print(f"\n=== Data preparation complete ===")
    print(f"Output: {staging_dir}/")


if __name__ == "__main__":
    main()
