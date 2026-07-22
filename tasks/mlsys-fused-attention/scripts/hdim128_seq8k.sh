#!/bin/bash
# FA3 paper config: headdim=128, nheads=16, seqlen=8192, batch=2, causal
# Total tokens = 16384, hidden_dim = 2048
# Reference: FA3 ~602 TFLOPs/s, FA2 ~333 TFLOPs/s (H100 FP16 fwd causal)

cd /workspace

python flash-attention/custom_triton_bench.py \
    --batch 2 \
    --seqlen 8192 \
    --nheads 16 \
    --headdim 128 \
    --causal \
    --dtype float16 \
    --output-dir ${OUTPUT_DIR} \
    --seed ${SEED:-42}
