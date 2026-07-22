#!/bin/bash
# Prepare VICON datasets for the pde-foundation-icl task.
# Run this script on a login node with network access before building the container.
#
# Usage:
#   bash vendor/pkg_configs/VICON/prepare_data.sh /path/to/data_root
#
# This will create:
#   <data_root>/vicon_data/2D_Train_Rand/{train,valid,test}/*.h5
#   <data_root>/vicon_data/2D_Train_Turb/{train,valid,test}/*.h5
#   <data_root>/vicon_data/NavierStokes-2D-conditoned/*.h5
#
# PDEBench DaRUS dataset (doi:10.18419/darus-2986):
#   - 2D_CFD_Rand: ID 164687 (M0.1, Eta0.01, 128 res, 10K samples, high viscosity)
#   - 2D_CFD_Turb: ID 164686 (M1.0, Eta1e-08, 512 res, 1K samples, low viscosity)
# PDEArena: NavierStokes-2D-conditoned from HuggingFace (pdearena)

set -e

mkdir -p "${1:?Usage: $0 <data_root>}"
DATA_ROOT="$(cd "$1" && pwd)"
VICON_DATA="${DATA_ROOT}/vicon_data"
RAW_DIR="${VICON_DATA}/raw"

echo "=== Preparing VICON data in ${VICON_DATA} ==="
mkdir -p "${RAW_DIR}" "${VICON_DATA}/2D_Train_Rand" "${VICON_DATA}/2D_Train_Turb"

RAND_MIN_BYTES=$((40 * 1024 * 1024 * 1024))
TURB_MIN_BYTES=$((70 * 1024 * 1024 * 1024))

# 1. Download PDEBench 2D CFD Rand (high viscosity, 128 res, single file with 10K samples)
echo "=== Downloading PDEBench 2D CFD Rand (ID 164687) ==="
RAND_FILE="${RAW_DIR}/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
if [ -f "${RAND_FILE}" ] && [ "$(stat -c%s "${RAND_FILE}")" -ge "${RAND_MIN_BYTES}" ]; then
    echo "  Skipping (exists: $(du -h "${RAND_FILE}" | cut -f1))"
else
    echo "  Downloading/resuming (~55 GB)..."
    wget -q -c -O "${RAND_FILE}" "https://darus.uni-stuttgart.de/api/access/datafile/164687"
    echo "  Done: $(stat -c%s "${RAND_FILE}") bytes"
fi

# 2. Download PDEBench 2D CFD Turb (low viscosity, 512 res, single file with 1K samples)
echo "=== Downloading PDEBench 2D CFD Turb (ID 164686) ==="
TURB_FILE="${RAW_DIR}/2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"
if [ -f "${TURB_FILE}" ] && [ "$(stat -c%s "${TURB_FILE}")" -ge "${TURB_MIN_BYTES}" ]; then
    echo "  Skipping (exists: $(du -h "${TURB_FILE}" | cut -f1))"
else
    echo "  Downloading/resuming (~88 GB)..."
    wget -q -c -O "${TURB_FILE}" "https://darus.uni-stuttgart.de/api/access/datafile/164686"
    echo "  Done: $(stat -c%s "${TURB_FILE}") bytes"
fi

# 3. Download PDEArena NS2D from HuggingFace
echo "=== Downloading PDEArena NavierStokes-2D ==="
if [ -d "${VICON_DATA}/NavierStokes-2D-conditoned" ] && [ "$(ls ${VICON_DATA}/NavierStokes-2D-conditoned/*.h5 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Skipping (already exists)"
else
    (
        cd "${VICON_DATA}"
        git lfs install
        GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D-conditoned
    )
fi

# Remove known-bad upstream shard shipped as a 96-byte LFS pointer.
rm -f "${VICON_DATA}/NavierStokes-2D-conditoned/NavierStokes2D_train_496019_0.38774_32.h5"

# The HuggingFace clone is only used as a one-time data fetch. Keep the working tree
# but drop git metadata so Docker builds don't spend time/context on LFS history.
if [ -d "${VICON_DATA}/NavierStokes-2D-conditoned/.git" ]; then
    echo "=== Removing PDEArena git metadata ==="
    rm -rf "${VICON_DATA}/NavierStokes-2D-conditoned/.git"
fi

# Older prepare_data runs could leave an accidental nested vendor tree under vicon_data.
# It is not used by VICON and only inflates Docker build context.
if [ -d "${VICON_DATA}/vendor" ]; then
    echo "=== Removing stale nested vendor directory ==="
    rm -rf "${VICON_DATA}/vendor"
fi

# 4. Convert PDEBench Rand data (128 res, chunk into 500-sample files)
echo "=== Converting PDEBench high-viscosity Rand data ==="
python3 -c "
import h5py, numpy as np, os

def convert_highvis(raw_file, save_folder, split_size=500):
    os.makedirs(save_folder, exist_ok=True)
    save_file_idx = 0
    with h5py.File(raw_file, 'r') as f:
        total_size = f['density'].shape[0]
        n_chunks = total_size // split_size
        print(f'  {total_size} samples -> {n_chunks} chunks of {split_size}', flush=True)
        for ci in range(n_chunks):
            data_dict = {}
            start = ci * split_size
            for key in f.keys():
                if key == 't-coordinate' or key.endswith('coordinate'):
                    data_dict[key] = np.array(f[key])
                else:
                    data_dict[key] = np.array(f[key][start:start+split_size])
            traj_path = os.path.join(save_folder, f'{save_file_idx}.h5')
            save_file_idx += 1
            with h5py.File(traj_path, 'w') as g:
                for key, val in data_dict.items():
                    g.create_dataset(key, data=val)
            print(f'  Chunk {ci}/{n_chunks}', flush=True)
    print(f'Done: {save_file_idx} chunks', flush=True)

convert_highvis('${RAND_FILE}', '${VICON_DATA}/2D_Train_Rand/')
"

# 5. Convert PDEBench Turb data (512 res -> downsample 4x to 128, chunk into 50-sample files)
echo "=== Converting PDEBench low-viscosity Turb data (512->128 downsample) ==="
python3 -c "
import h5py, numpy as np, os

def convert_lowvis(raw_file, save_folder, split_size=50, step=4):
    os.makedirs(save_folder, exist_ok=True)
    save_file_idx = 0
    with h5py.File(raw_file, 'r') as f:
        total_size = f['density'].shape[0]
        n_chunks = total_size // split_size
        print(f'  {total_size} samples -> {n_chunks} chunks of {split_size} (downsample {step}x)', flush=True)
        for ci in range(n_chunks):
            data_dict = {}
            start = ci * split_size
            for key in f.keys():
                if key == 't-coordinate':
                    data_dict[key] = np.array(f[key])
                elif key.endswith('coordinate'):
                    coord = np.array(f[key])
                    data_dict[key] = coord.reshape(-1, step).mean(axis=1)
                else:
                    chunk = np.array(f[key][start:start+split_size])
                    N, T, H, W = chunk.shape
                    chunk = chunk.reshape(N, T, H//step, step, W//step, step).mean(axis=(3,5))
                    data_dict[key] = chunk
            traj_path = os.path.join(save_folder, f'{save_file_idx}.h5')
            save_file_idx += 1
            with h5py.File(traj_path, 'w') as g:
                for key, val in data_dict.items():
                    g.create_dataset(key, data=val)
            print(f'  Chunk {ci}/{n_chunks}', flush=True)
    print(f'Done: {save_file_idx} chunks', flush=True)

convert_lowvis('${TURB_FILE}', '${VICON_DATA}/2D_Train_Turb/')
"

# 6. Split both datasets into train/valid/test (80/10/10)
echo "=== Splitting datasets ==="
python3 -c "
import numpy as np, os, glob, shutil

def split_dataset(path, split_percent=[0.8, 0.1, 0.1], seed=42):
    modes = ['train', 'valid', 'test']
    filenames = sorted(glob.glob(os.path.join(path, '*.h5')))
    print(f'  {path}: {len(filenames)} files', flush=True)
    np.random.seed(seed)
    np.random.shuffle(filenames)
    train_size = int(len(filenames) * split_percent[0])
    valid_size = int(len(filenames) * split_percent[1])
    splits = [filenames[:train_size], filenames[train_size:train_size+valid_size], filenames[train_size+valid_size:]]
    for m in modes:
        os.makedirs(os.path.join(path, 'tmp_' + m), exist_ok=True)
    for im, mode_files in enumerate(splits):
        for i, f in enumerate(mode_files):
            os.rename(f, os.path.join(path, 'tmp_' + modes[im], f'{i}.h5'))
    for m in modes:
        target = os.path.join(path, m)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.rename(os.path.join(path, 'tmp_' + m), target)
    print(f'  Split: {[len(s) for s in splits]}', flush=True)

split_dataset('${VICON_DATA}/2D_Train_Rand/')
split_dataset('${VICON_DATA}/2D_Train_Turb/')
"

# 7. Clean up raw files
echo "=== Cleaning up raw files ==="
rm -rf "${RAW_DIR}"

# 8. Verify
echo "=== Verifying ==="
python3 -c "
import os
checks = [
    '${VICON_DATA}/2D_Train_Rand/train',
    '${VICON_DATA}/2D_Train_Rand/valid',
    '${VICON_DATA}/2D_Train_Rand/test',
    '${VICON_DATA}/2D_Train_Turb/train',
    '${VICON_DATA}/2D_Train_Turb/valid',
    '${VICON_DATA}/2D_Train_Turb/test',
    '${VICON_DATA}/NavierStokes-2D-conditoned',
]
missing = [f for f in checks if not os.path.exists(f)]
assert not missing, f'Missing: {missing}'
for c in checks:
    n = len([f for f in os.listdir(c) if f.endswith('.h5')])
    print(f'  {c}: {n} h5 files')
print('All datasets verified!')
"

echo "=== Data preparation complete ==="
echo "Now build the container: mlsbench build VICON"
