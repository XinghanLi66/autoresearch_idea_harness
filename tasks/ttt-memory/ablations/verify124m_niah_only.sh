#!/bin/bash
# Re-run NIAH eval on an existing trained checkpoint (no training).
#
# Usage: sbatch verify124m_niah_only.sh <baseline_name>
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120g
#SBATCH --partition=ailab
#SBATCH --account=chij
#SBATCH --qos=ailab
#SBATCH --time=2:00:00
#SBATCH --job-name=mls-ttt-niah-only
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/MLS-Bench/logs/ttt-memory/verify124m/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lyubh22@gmail.com

set -e
BASELINE="${1:?usage: $0 <baseline_name>}"
ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench
SAVE_DIR="${ROOT}/.saves/ttt-memory/verify124m_${BASELINE}/seed_42"
CKPT="${SAVE_DIR}/ckpt_gpt-124m.pt"
SRC="${SAVE_DIR}/model_source_gpt-124m.py"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: missing checkpoint ${CKPT}"
    exit 1
fi

echo "=== NIAH-only re-eval: ${BASELINE} ==="
echo "ckpt: ${CKPT}"
echo "src:  ${SRC}"

CTX_LENS="${CTX_LENS:-1024,1536,2048,2560,3072,4096,6144,8192}"
apptainer exec --nv --writable-tmpfs --no-home \
    --env TIKTOKEN_CACHE_DIR=/workspace/_task/tiktoken_cache \
    --env TRITON_CACHE_DIR=/tmp/triton_cache \
    --bind "${ROOT}/tasks/ttt-memory:/workspace/_task" \
    --bind "${SAVE_DIR}:/workspace/_ckpt" \
    --pwd /workspace/_task \
    "${ROOT}/vendor/images/nanoGPT.sif" \
    python /workspace/_task/niah_eval.py \
        --checkpoint /workspace/_ckpt/ckpt_gpt-124m.pt \
        --source /workspace/_ckpt/model_source_gpt-124m.py \
        --ctx-lens "${CTX_LENS}" \
        --n-samples 40 \
        --seed 42 \
        --train-ctx 2048

echo "done at $(date)"
