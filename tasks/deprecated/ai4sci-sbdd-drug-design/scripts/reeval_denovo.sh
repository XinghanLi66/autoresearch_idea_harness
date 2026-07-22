#!/bin/bash
# Re-evaluate existing sampling results for denovo task.
# Usage: Set BASELINE (targetdiff/diffbp/pocket2mol) and SEED env vars.
# Expects sampling results already in OUTPUT_DIR/results/denovo/custom/seed${SEED}

PY=python3
BASELINE=${BASELINE:-targetdiff}
TAG=seed${SEED:-42}
TASK=denovo
WORK=${OUTPUT_DIR}

echo "Re-evaluating ${BASELINE} denovo results (tag=${TAG})"
echo "Result root: ${WORK}/results/${TASK}/custom/${TAG}"

# Verify sampling results exist
RESULT_PATH="${WORK}/results/${TASK}/custom/${TAG}"
if [ ! -d "$RESULT_PATH" ]; then
    echo "ERROR: No sampling results at ${RESULT_PATH}"
    exit 1
fi
N_POCKETS=$(ls -d "${RESULT_PATH}"/*/ 2>/dev/null | wc -l)
echo "Found ${N_POCKETS} pocket directories"

# Setup data symlink for evaluate scripts
mkdir -p data
ln -sf /data/crossdocked/crossdocked_test data/crossdocked_test

# Create python symlink
ln -sf /usr/bin/python3 /usr/local/bin/python 2>/dev/null || true

# Fix: remove existing results dir before symlinking
rm -rf ./results 2>/dev/null || true
ln -sf ${WORK}/results ./results

# Run evaluate_chem
cd evaluate_scripts
bash evaluate_chem.sh --method custom --tasks ${TASK} --tag ${TAG}
cd ..

# Extract and print metrics
$PY -c "
import os, torch, numpy as np, glob

result_root = '${RESULT_PATH}'
all_qed, all_sa, all_logp, all_lipinski = [], [], [], []
validity_list = []
score_metrics, dock_metrics = [], []

for subdir, dirs, files in os.walk(result_root):
    chem_file = os.path.join(subdir, 'chem_eval_results.pt')
    ref_file = os.path.join(subdir, 'chem_reference_results.pt')
    if os.path.isfile(chem_file) and os.path.isfile(ref_file):
        try:
            result = torch.load(chem_file)
            ref = torch.load(ref_file)
            for res in result:
                all_qed.append(res['chem_results']['qed'])
                all_sa.append(res['chem_results']['sa'])
                all_logp.append(res['chem_results']['logp'])
                all_lipinski.append(res['chem_results']['lipinski'])
            validity_list.append(len(result) / 20)

            # Vina score_only
            vina_s = np.array([r['vina']['score_only']['affinity'] for r in result])
            ref_vina = ref['vina']['score_only']['affinity']
            if ref_vina < 0:
                score_metrics.append(np.mean(vina_s))

            # Vina dock
            vina_d = np.array([r['vina']['dock']['affinity'] for r in result])
            ref_vina_d = ref['vina']['dock']['affinity']
            if ref_vina_d < 0:
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

print(f'Processed {len(validity_list)} pockets total')
"
