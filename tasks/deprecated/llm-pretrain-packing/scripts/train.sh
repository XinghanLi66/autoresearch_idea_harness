#!/bin/bash
# Train a depth-4 nanochat model (~10M params) for packing strategy evaluation.
# Uses small config: depth=4, 500 iterations, frequent eval.
set -e

cd nanochat

python -m scripts.base_train \
    --depth=4 \
    --aspect-ratio=64 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=16384 \
    --num-iterations=500 \
    --eval-every=100 \
    --eval-tokens=1048576 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --warmup-steps=20 \
    --warmdown-ratio=0.65 \
    --window-pattern=L \
    2>&1 | tee /tmp/train_output.log

# Extract final validation BPB from the training output
python -c "
import re, sys

log = open('/tmp/train_output.log').read()

# Parse all 'Validation bpb: X.XXXX' lines for training feedback
bpb_lines = re.findall(r'Step (\d+).*Validation bpb: ([\d.]+)', log)
for step, bpb in bpb_lines:
    print(f'TRAIN_METRICS: step={step}, val_bpb={bpb}', flush=True)

# Final metrics: use the last validation BPB
if bpb_lines:
    final_step, final_bpb = bpb_lines[-1]
    print(f'TEST_METRICS: val_bpb={final_bpb}', flush=True)
else:
    # Fallback: look for Minimum validation bpb
    m = re.search(r'Minimum validation bpb: ([\d.]+)', log)
    if m:
        print(f'TEST_METRICS: val_bpb={m.group(1)}', flush=True)
    else:
        print('ERROR: No validation BPB found in output', file=sys.stderr)
        sys.exit(1)
"
