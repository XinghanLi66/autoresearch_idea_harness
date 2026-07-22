#!/bin/bash
# FA3 paper config: headdim=64, nheads=32, seqlen=4096, batch=4, causal
# Total tokens = 16384, hidden_dim = 2048
# Reference: FA3 ~420 TFLOPs/s, FA2 ~284 TFLOPs/s (H100 FP16 fwd causal)

cd /workspace

python flash-attention/custom_triton_bench.py \
    --batch 4 \
    --seqlen 4096 \
    --nheads 32 \
    --headdim 64 \
    --causal \
    --dtype float16 \
    --output-dir ${OUTPUT_DIR} \
    --seed ${SEED:-42}
