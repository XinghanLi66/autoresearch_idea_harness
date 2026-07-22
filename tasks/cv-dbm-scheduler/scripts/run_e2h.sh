export eta=0.0
export ds=e2h
export num_samples=10000
export doob_scale=1.0
export sampler=dbim
export nfe=5

export sample_dir=${OUTPUT_DIR:-output}/$ds-$nfe-$sampler-$eta-seed${SEED:-42}
rm -rf "$sample_dir"
mkdir -p "$sample_dir"
bash scripts/sample.sh $ds $nfe $sampler $eta
bash scripts/evaluate.sh $ds $nfe $sampler $eta
# evaluations/evaluator.py writes fid.json next to the sample NPZ, i.e. under
# workdir/<ckpt>/sample_*/split=train/<sampler>/steps=*/fid.json — NOT into
# $sample_dir. Glob it there and surface "FID: <num>" for the task parser.
FID_JSON=$(ls -t workdir/e2h_ema_*/sample_*/split=train/*/steps=*/fid.json 2>/dev/null | head -1)
if [ -n "$FID_JSON" ]; then
    echo "FID: $(python3 -c "import json; print(json.load(open('$FID_JSON'))['fid'])")"
fi

# Clean up sample NPZ files once FID has been computed — each agent iteration
# would otherwise keep a ~60 MB (10k × 64x64x3 uint8) NPZ on Vepfs.
find workdir/ -name "samples_*.npz" -delete 2>/dev/null || true
rm -rf "$sample_dir"
