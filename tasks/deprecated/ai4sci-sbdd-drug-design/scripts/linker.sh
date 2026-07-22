#!/bin/bash

# Use python3 explicitly (container may not have python symlink)
PY=python3

TASK=linker
CONFIG=/workspace/_task/configs/linker_custom.yml
TAG=seed${SEED:-42}

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR:-/workspace/output}
WORK=${OUTPUT_DIR}

# Symlink data for evaluate scripts (expects ../data/crossdocked_test relative to evaluate_scripts/)
mkdir -p data
ln -sf /data/crossdocked/crossdocked_test data/crossdocked_test

# Create python symlink for sub-scripts (evaluate_chem.sh uses 'python')
ln -sf /usr/bin/python3 /usr/local/bin/python 2>/dev/null || true

# Install missing evaluation deps from pre-downloaded wheels (no network on compute nodes)
# meeko 0.7.1 needs gemmi; AutoDockTools needed by docking_vina.py; meeko<0.6 for obutils compat
WHEELS=/data/crossdocked/wheels
if [ -d "$WHEELS" ]; then
    pip3 install --no-deps "$WHEELS"/meeko-0.5.0*.whl 2>/dev/null || true
    pip3 install --no-deps "$WHEELS"/gemmi-*.whl "$WHEELS"/AutoDockTools_py3-*.zip 2>/dev/null || true
fi

# Train (limit to 20h to leave time for sampling+evaluation)
LOGDIR=${OUTPUT_DIR}/train_logs/${TASK}/custom
timeout --signal=INT 72000 $PY train.py --config ${CONFIG} --logdir ${LOGDIR} --tag ${TAG} --num_workers 4 || true
echo "Training phase complete"

# Find best checkpoint
CKPT_DIR=${LOGDIR}/${TAG}/checkpoints
echo "Looking for checkpoints in: ${CKPT_DIR}"
ls -la ${CKPT_DIR}/ 2>/dev/null || echo "Checkpoint dir does not exist"
BEST_CKPT=$(ls ${CKPT_DIR}/*.pt 2>/dev/null | sort -t/ -k1 -V | tail -1)
if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: No checkpoint found"
    exit 1
fi
echo "Using checkpoint: ${BEST_CKPT}"

# Create test config (name MUST be custom.yml so sample.py saves under "custom/" method dir)
TEST_CONFIG=${WORK}/custom.yml
cat > ${TEST_CONFIG} << EOF
model:
  type: custom
  checkpoint: ${BEST_CKPT}

data:
  test:
    name: pl_decomp
    version: linker
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
  seed: ${SEED:-42}
  num_samples: 20
  translate: true

reconstruct:
  basic_mode: false

mode: add_aromatic
EOF

# Sample
$PY sample.py --config ${TEST_CONFIG} --out_root ${WORK}/results/${TASK}/ --tag ${TAG} --seed ${SEED:-42}

# Evaluate chemistry
# Must rm first: CBGBench repo has an existing results/ directory that ln -sf cannot replace
rm -rf ./results 2>/dev/null || true
ln -sf ${WORK}/results ./results

# Clean old evaluation results to ensure fresh metrics from this run's model
find ${WORK}/results/${TASK}/custom/${TAG}/ -name "chem_eval_results.pt" -delete 2>/dev/null || true
find ${WORK}/results/${TASK}/custom/${TAG}/ -name "chem_reference_results.pt" -delete 2>/dev/null || true
find ${WORK}/results/${TASK}/custom/${TAG}/ -name "molecule_properties.csv" -delete 2>/dev/null || true
find ${WORK}/results/${TASK}/custom/${TAG}/ -type d -name "docking_results" -exec rm -rf {} + 2>/dev/null || true

cd evaluate_scripts
bash evaluate_chem.sh --method custom --tasks ${TASK} --tag ${TAG}
cd ..

# Extract and print metrics
$PY -c "
import os, torch, numpy as np

result_root = '${WORK}/results/${TASK}/custom/${TAG}'
all_qed, all_sa = [], []
validity_list = []
score_metrics, dock_metrics = [], []

for subdir, dirs, files in os.walk(result_root):
    chem_file = os.path.join(subdir, 'chem_eval_results.pt')
    if os.path.isfile(chem_file):
        try:
            result = torch.load(chem_file)
            for res in result:
                all_qed.append(res['chem_results']['qed'])
                all_sa.append(res['chem_results']['sa'])
            validity_list.append(len(result) / 20)
            # Vina score_only (skip None entries)
            vina_s = [r['vina']['score_only']['affinity'] for r in result
                      if r.get('vina',{}).get('score_only') is not None]
            if vina_s:
                score_metrics.append(np.mean(vina_s))
            # Vina dock (skip None entries)
            vina_d = [r['vina']['dock']['affinity'] for r in result
                      if r.get('vina',{}).get('dock') is not None]
            if vina_d:
                dock_metrics.append(np.mean(vina_d))
        except Exception as e:
            print(f'Warning: {e}')

if all_qed:
    print(f'TEST_METRICS qed={np.mean(all_qed):.6f}', flush=True)
    print(f'TEST_METRICS sa={np.mean(all_sa):.6f}', flush=True)
    print(f'TEST_METRICS validity={np.mean(validity_list):.6f}', flush=True)
if score_metrics:
    print(f'TEST_METRICS vina_score={np.mean(score_metrics):.6f}', flush=True)
if dock_metrics:
    print(f'TEST_METRICS vina_dock={np.mean(dock_metrics):.6f}', flush=True)
"
