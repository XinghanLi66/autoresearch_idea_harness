#!/bin/bash
# Verification run: train at block_size=4096 (2x normal train_ctx) on 2 GPU,
# then NIAH-eval with NTK kicking in past 4096.
#
# Token budget held at Chinchilla-optimal 2.5B:
#   tokens/iter = N_GPU(2) * (per_rank_GA = GA / world_size = 16/2 = 8)
#                 * BS(8) * BLOCK(4096) = 524288
#   MAX_ITERS = 2.5e9 / 524288 ≈ 4768 (so total ≈ 2.5B tokens)
#
# BATCH_SIZE=8 (not 16) is required: at block=4096 the outer transformer
# backward needs activations for the entire sequence (TBPTT only detaches
# the fast-weight recurrence). BS=16 OOM's at ~108GB on H200 (140GB).
#
# Per-iter time ~doubles vs 2k ctx because each sample has 64 chunks instead
# of 32, but we halve MAX_ITERS, so total walltime is similar (~6-7h on 2 GPU).
#
# Usage: sbatch verify124m_run_2gpu_4kctx.sh <baseline_name>
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
#SBATCH --job-name=mls-ttt-verify124m-2g-4k
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
echo ">>> verify124m (2 GPU, train_ctx=4096): baseline=${BASELINE}"
echo "    started at $(date)"
echo "========================================================================"

echo ""
echo ">>> [1/3] setting up workspace"
python3 "${ROOT}/tasks/ttt-memory/ablations/verify124m_setup.py" "${BASELINE}" "${WS_ROOT}"

echo ""
echo ">>> [2/3] training (2 GPU, block=4096, max_iters=4768, ~2.5B tokens)"
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
    --env BLOCK_SIZE=4096 \
    --env BATCH_SIZE=8 \
    --env GRAD_ACCUM=16 \
    --env MAX_ITERS="${MAX_ITERS_OVERRIDE:-4768}" \
    --env EVAL_INTERVAL=500 \
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
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: training did not produce ${CKPT}"
    exit 1
fi

echo ""
echo ">>> [3/3] NIAH eval (NTK base scales for T > 4096)"
CTX_LENS="${CTX_LENS:-2048,3072,4096,5120,6144,8192,12288,16384}"
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
        --train-ctx 4096

echo ""
echo "========================================================================"
echo ">>> verify124m done: baseline=${BASELINE}"
echo "    finished at $(date)"
echo "========================================================================"
