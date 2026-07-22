#!/bin/bash
# Verification run: train one baseline at 124M / 2.5B tokens / **2 GPUs DDP** /
# train_ctx=2048, then NIAH-eval at 1.25x..8x extrapolation.
#
# Token budget is preserved: trainer divides GRAD_ACCUM by ddp_world_size
# (custom_template.py line 561), so per-iter tokens stay at 16*8*2048=262144,
# and 9537 iters still trains exactly 2.5B tokens (Chinchilla-optimal for 124M).
#
# Usage: sbatch verify124m_run_2gpu.sh <baseline_name>
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=240g
#SBATCH --partition=ailab
#SBATCH --account=chij
#SBATCH --qos=ailab
#SBATCH --time=14:00:00
#SBATCH --job-name=mls-ttt-verify124m-2g
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/MLS-Bench/logs/ttt-memory/verify124m/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lyubh22@gmail.com

set -e
BASELINE="${1:?usage: $0 <baseline_name>}"
ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench
LOG_DIR="${ROOT}/logs/ttt-memory/verify124m"
SAVE_DIR="${ROOT}/.saves/ttt-memory/verify124m_${BASELINE}/seed_42"
WS_ROOT="${ROOT}/.verify124m_ws/${BASELINE}"
mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

echo ""
echo "========================================================================"
echo ">>> verify124m (2 GPU DDP): baseline=${BASELINE}"
echo "    workspace: ${WS_ROOT}"
echo "    save:      ${SAVE_DIR}"
echo "    started at $(date)"
echo "========================================================================"

# 1. Set up isolated workspace (copy nanoGPT, render template, apply edit)
echo ""
echo ">>> [1/3] setting up workspace"
python3 "${ROOT}/tasks/ttt-memory/ablations/verify124m_setup.py" "${BASELINE}" "${WS_ROOT}"

# 2. Train (2 GPU DDP, 124M, train_ctx=2048, 2.5B tokens)
echo ""
echo ">>> [2/3] training (this is the long step)"
T_START=$(date +%s)
apptainer exec --nv --writable-tmpfs --no-home \
    --env DATA_DIR=/data/climbmix \
    --env EVAL_DIR=/data/eval \
    --env 'PYTHONPATH=/workspace/nanoGPT:${PYTHONPATH}' \
    --env TRITON_CACHE_DIR=/tmp/triton_cache \
    --env TIKTOKEN_CACHE_DIR=/workspace/_task/tiktoken_cache \
    --env SEED=42 \
    --env ENV=gpt-124m \
    --env OUTPUT_DIR="${SAVE_DIR}" \
    --bind "${WS_ROOT}/nanoGPT:/workspace/nanoGPT" \
    --bind "${ROOT}/tasks/ttt-memory:/workspace/_task" \
    --bind "/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data/climbmix:/data/climbmix" \
    --bind "/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data/eval:/data/eval" \
    --bind "${SAVE_DIR}:${SAVE_DIR}" \
    --bind /dev/shm \
    --pwd /workspace/nanoGPT \
    "${ROOT}/vendor/images/nanoGPT.sif" \
    bash /workspace/_task/scripts/gpt_124m.sh
T_TRAIN=$(( $(date +%s) - T_START ))
echo ">>> training took ${T_TRAIN}s"

CKPT="${SAVE_DIR}/ckpt_gpt-124m.pt"
SRC="${SAVE_DIR}/model_source_gpt-124m.py"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: training did not produce ${CKPT}"
    exit 1
fi

# 3. NIAH eval (1 GPU is enough for eval)
echo ""
echo ">>> [3/3] NIAH eval"
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

echo ""
echo "========================================================================"
echo ">>> verify124m done: baseline=${BASELINE}"
echo "    finished at $(date)"
echo "========================================================================"
