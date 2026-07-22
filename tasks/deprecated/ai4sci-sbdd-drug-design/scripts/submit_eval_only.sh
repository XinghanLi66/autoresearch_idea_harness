#!/bin/bash
# submit_eval_only.sh — Submit eval-only SLURM jobs for SBDD linker/frag baselines
# Usage: bash submit_eval_only.sh <baseline> <task>
#   baseline: targetdiff | diffbp | pocket2mol
#   task:     linker | frag
#
# Example:
#   bash tasks/ai4sci-sbdd-drug-design/scripts/submit_eval_only.sh targetdiff linker

set -euo pipefail

BASELINE="${1:?Usage: $0 <baseline> <task>}"
TASK="${2:?Usage: $0 <baseline> <task>}"

# Validate inputs
case "$BASELINE" in
    targetdiff|diffbp|pocket2mol) ;;
    *) echo "ERROR: baseline must be targetdiff, diffbp, or pocket2mol"; exit 1 ;;
esac
case "$TASK" in
    linker|frag) ;;
    *) echo "ERROR: task must be linker or frag"; exit 1 ;;
esac

PROJECT_ROOT="/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench"
SAVE_PATH="/scratch/gpfs/CHIJ/st3812/models/saves"
SEED=42
TAG="seed${SEED}"

# --- Find workspace for this baseline ---
# Map baseline to workspace directory (from 20260406 submission)
declare -A WORKSPACE_MAP=(
    [targetdiff]="${PROJECT_ROOT}/vendor/workspace/20260406_075130_862287_934182/ai4sci-sbdd-drug-design/targetdiff/CBGBench"
    [diffbp]="${PROJECT_ROOT}/vendor/workspace/20260406_075131_667685_935364/ai4sci-sbdd-drug-design/diffbp/CBGBench"
    [pocket2mol]="${PROJECT_ROOT}/vendor/workspace/20260406_075132_657622_936615/ai4sci-sbdd-drug-design/pocket2mol/CBGBench"
)

WORKSPACE="${WORKSPACE_MAP[$BASELINE]}"
if [ ! -d "$WORKSPACE" ]; then
    echo "ERROR: Workspace not found: $WORKSPACE"
    echo "Searching for alternative workspace..."
    WORKSPACE=$(ls -d "${PROJECT_ROOT}"/vendor/workspace/*/ai4sci-sbdd-drug-design/"${BASELINE}"/CBGBench 2>/dev/null | tail -1)
    if [ -z "$WORKSPACE" ] || [ ! -d "$WORKSPACE" ]; then
        echo "ERROR: No workspace found for baseline ${BASELINE}"
        exit 1
    fi
    echo "Found alternative: $WORKSPACE"
fi

# Verify custom_sbdd.py has baseline edits (not just the template)
CUSTOM_SBDD="${WORKSPACE}/repo/models/custom_sbdd.py"
if [ ! -f "$CUSTOM_SBDD" ]; then
    echo "ERROR: custom_sbdd.py not found in workspace"
    exit 1
fi
LINE_COUNT=$(wc -l < "$CUSTOM_SBDD")
if [ "$LINE_COUNT" -lt 200 ]; then
    echo "WARNING: custom_sbdd.py only has ${LINE_COUNT} lines - may not have baseline edit applied"
fi
echo "Workspace: $WORKSPACE (custom_sbdd.py: ${LINE_COUNT} lines)"

# --- Checkpoint ---
CKPT_DIR="${SAVE_PATH}/ai4sci-sbdd-drug-design/${BASELINE}/seed_${SEED}/train_logs/${TASK}/custom/${TAG}/checkpoints"
BEST_CKPT=$(ls "${CKPT_DIR}"/*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Checkpoint: $BEST_CKPT"

# --- Paths ---
CONTAINER="${PROJECT_ROOT}/vendor/images/CBGBench.sif"
TASK_DIR="${PROJECT_ROOT}/tasks/ai4sci-sbdd-drug-design"
DATA_DIR="${PROJECT_ROOT}/vendor/data/CBGBench"
OUTPUT_DIR="${SAVE_PATH}/ai4sci-sbdd-drug-design/${BASELINE}/seed_${SEED}"

# Create log directory
LOG_DIR="${PROJECT_ROOT}/logs/ai4sci-sbdd-drug-design/${BASELINE}/eval_only"
mkdir -p "$LOG_DIR"

# --- Determine test config version ---
# linker and frag configs are identical except for `version:` field
VERSION="${TASK}"

# --- Generate SLURM script ---
SLURM_SCRIPT="${LOG_DIR}/${TASK}_s${SEED}.slurm"

cat > "${SLURM_SCRIPT}" << 'SLURM_OUTER'
#!/bin/bash
#SBATCH --job-name=JOBNAME_PLACEHOLDER
#SBATCH --partition=gpu-ee
#SBATCH --account=chij
#SBATCH --qos=della-gpuee
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --output=OUTPUT_PLACEHOLDER
#SBATCH --error=OUTPUT_PLACEHOLDER

echo "=== SBDD Eval-Only: BASELINE_PH / TASK_PH ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Start: $(date)"

apptainer exec --nv --writable-tmpfs \
    --bind WORKSPACE_PH:/workspace/CBGBench \
    --bind TASKDIR_PH:/workspace/_task \
    --bind DATADIR_PH:/data/crossdocked \
    --pwd /workspace/CBGBench \
    CONTAINER_PH \
    bash -c '
set -e

PY=python3
TASK=TASK_PH
SEED=SEED_PH
TAG=seed${SEED}
WORK=OUTPUTDIR_PH

export PYTHONPATH=/workspace/CBGBench:${PYTHONPATH:-}
export DATA_DIR=/data/crossdocked

echo "=== Setting up environment ==="

# Data symlinks
mkdir -p data
ln -sf /data/crossdocked/crossdocked_test data/crossdocked_test

# Python symlink for evaluate_chem.sh
ln -sf /usr/bin/python3 /usr/local/bin/python 2>/dev/null || true

# Install eval deps from pre-downloaded wheels
WHEELS=/data/crossdocked/wheels
if [ -d "$WHEELS" ]; then
    pip3 install --no-deps "$WHEELS"/meeko-0.5.0*.whl 2>/dev/null || true
    pip3 install --no-deps "$WHEELS"/gemmi-*.whl "$WHEELS"/AutoDockTools_py3-*.zip 2>/dev/null || true
fi

# Use existing checkpoint (no training)
BEST_CKPT=CKPT_PH
echo "Using existing checkpoint: ${BEST_CKPT}"

if [ ! -f "${BEST_CKPT}" ]; then
    echo "ERROR: Checkpoint file not found: ${BEST_CKPT}"
    exit 1
fi

# Create test config (MUST be named custom.yml for sample.py method dir)
TEST_CONFIG=${WORK}/custom.yml
cat > ${TEST_CONFIG} << YMLEOF
model:
  type: custom
  checkpoint: ${BEST_CKPT}

data:
  test:
    name: pl_decomp
    version: VERSION_PH
    raw_path: /data/crossdocked/crossdocked_v1.1_rmsd1.0_pocket10
    processed_dir: /data/crossdocked/pl_decomp/
    split_path: /data/crossdocked/split_by_name_10m.pt
    transform:
      - type: choose_ctx_gen
        sampling: fix_zero
      - type: featurize_protein_fa
      - type: remove_ligand_gen
        mode: add_aromatic
      - type: assign_gensize
        distribution: prior_distcond
      - type: assign_genatomtype
        distribution: uniform
        mode: add_aromatic
      - type: center_pos
        center_flag: ligand
        mask_flag: ctx_flag
      - type: assign_genpos
        distribution: gaussian
      - type: merge
        keys:
          - protein
          - ligand
        excluded_subkeys:
          - gen_bond_index
          - gen_bond_type
          - bond_index
          - bond_type
          - ctx_bond_index
          - ctx_bond_type
          - gen_index
          - ctx_index
          - cross_bond_index
          - cross_bond_type
  follow_batch:
    - protein_element
    - ligand_element

sampling:
  seed: ${SEED}
  num_samples: 20
  translate: true

reconstruct:
  basic_mode: false

mode: add_aromatic
YMLEOF

echo "=== Test config written to ${TEST_CONFIG} ==="
cat ${TEST_CONFIG}

# Sample
echo "=== Running sample.py ==="
$PY sample.py --config ${TEST_CONFIG} --out_root ${WORK}/results/${TASK}/ --tag ${TAG} --seed ${SEED}

# Evaluate chemistry
rm -rf ./results 2>/dev/null || true
ln -sf ${WORK}/results ./results

# Clean old eval results
find ${WORK}/results/${TASK}/custom/${TAG}/ -name "chem_eval_results.pt" -delete 2>/dev/null || true
find ${WORK}/results/${TASK}/custom/${TAG}/ -name "chem_reference_results.pt" -delete 2>/dev/null || true
find ${WORK}/results/${TASK}/custom/${TAG}/ -name "molecule_properties.csv" -delete 2>/dev/null || true
find ${WORK}/results/${TASK}/custom/${TAG}/ -type d -name "docking_results" -exec rm -rf {} + 2>/dev/null || true

echo "=== Running evaluate_chem.sh ==="
cd evaluate_scripts
bash evaluate_chem.sh --method custom --tasks ${TASK} --tag ${TAG}
cd ..

# Extract and print metrics
echo "=== Extracting metrics ==="
$PY -c "
import os, torch, numpy as np

result_root = '"'"'${WORK}/results/${TASK}/custom/${TAG}'"'"'
all_qed, all_sa = [], []
validity_list = []
score_metrics, dock_metrics = [], []

for subdir, dirs, files in os.walk(result_root):
    chem_file = os.path.join(subdir, '"'"'chem_eval_results.pt'"'"')
    if os.path.isfile(chem_file):
        try:
            result = torch.load(chem_file)
            for res in result:
                all_qed.append(res['"'"'chem_results'"'"']['"'"'qed'"'"'])
                all_sa.append(res['"'"'chem_results'"'"']['"'"'sa'"'"'])
            validity_list.append(len(result) / 20)
            vina_s = [r['"'"'vina'"'"']['"'"'score_only'"'"']['"'"'affinity'"'"'] for r in result
                      if r.get('"'"'vina'"'"',{}).get('"'"'score_only'"'"') is not None]
            if vina_s:
                score_metrics.append(np.mean(vina_s))
            vina_d = [r['"'"'vina'"'"']['"'"'dock'"'"']['"'"'affinity'"'"'] for r in result
                      if r.get('"'"'vina'"'"',{}).get('"'"'dock'"'"') is not None]
            if vina_d:
                dock_metrics.append(np.mean(vina_d))
        except Exception as e:
            print(f'"'"'Warning: {e}'"'"')

if all_qed:
    print(f'"'"'TEST_METRICS qed={np.mean(all_qed):.6f}'"'"', flush=True)
    print(f'"'"'TEST_METRICS sa={np.mean(all_sa):.6f}'"'"', flush=True)
    print(f'"'"'TEST_METRICS validity={np.mean(validity_list):.6f}'"'"', flush=True)
if score_metrics:
    print(f'"'"'TEST_METRICS vina_score={np.mean(score_metrics):.6f}'"'"', flush=True)
if dock_metrics:
    print(f'"'"'TEST_METRICS vina_dock={np.mean(dock_metrics):.6f}'"'"', flush=True)
"

echo "=== Done: $(date) ==="
'

echo "Job finished: $(date)"
SLURM_OUTER

# --- Replace placeholders ---
sed -i "s|JOBNAME_PLACEHOLDER|mls-sbdd-eval-${BASELINE}-${TASK}|g" "${SLURM_SCRIPT}"
sed -i "s|OUTPUT_PLACEHOLDER|${LOG_DIR}/${TASK}_s${SEED}.out|g" "${SLURM_SCRIPT}"
sed -i "s|WORKSPACE_PH|${WORKSPACE}|g" "${SLURM_SCRIPT}"
sed -i "s|TASKDIR_PH|${TASK_DIR}|g" "${SLURM_SCRIPT}"
sed -i "s|DATADIR_PH|${DATA_DIR}|g" "${SLURM_SCRIPT}"
sed -i "s|CONTAINER_PH|${CONTAINER}|g" "${SLURM_SCRIPT}"
sed -i "s|TASK_PH|${TASK}|g" "${SLURM_SCRIPT}"
sed -i "s|SEED_PH|${SEED}|g" "${SLURM_SCRIPT}"
sed -i "s|OUTPUTDIR_PH|${OUTPUT_DIR}|g" "${SLURM_SCRIPT}"
sed -i "s|VERSION_PH|${VERSION}|g" "${SLURM_SCRIPT}"
sed -i "s|CKPT_PH|${BEST_CKPT}|g" "${SLURM_SCRIPT}"
sed -i "s|BASELINE_PH|${BASELINE}|g" "${SLURM_SCRIPT}"

echo ""
echo "=== SLURM script: ${SLURM_SCRIPT} ==="
echo "Submitting..."
sbatch "${SLURM_SCRIPT}"
echo "Log will be at: ${LOG_DIR}/${TASK}_s${SEED}.out"
