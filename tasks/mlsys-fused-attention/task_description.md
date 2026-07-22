# Fused Attention Kernel Design for H100 GPUs

## Research Question

Design an efficient fused self-attention forward pass kernel using OpenAI Triton that maximizes throughput (TFLOPs/s) on H100 GPUs while maintaining numerical correctness.

## Background

Self-attention is the computational bottleneck of Transformer models. The standard implementation materializes the full N×N attention score matrix, requiring O(N²) memory and O(N²d) FLOPs. Flash Attention (Dao et al., 2022) introduced a tiled, IO-aware algorithm using online softmax that avoids materializing the full matrix, reducing HBM accesses from O(Nd + N²) to O(N²d²/M) where M is SRAM size.

Subsequent versions improved throughput through better parallelism strategies and hardware-specific optimizations:
- **FlashAttention-2** (Dao, 2024): Reduced non-matmul FLOPs, parallelized over sequence length, better warp-level work partitioning. Reaches ~200 TFLOPs/s on A100.
- **FlashAttention-3** (Shah et al., 2024): Exploits H100 Hopper features — warp specialization (producer/consumer warps overlapping TMA loads with GMMA compute), GEMM-softmax interleaving. Reaches ~740 TFLOPs/s on H100 (75% utilization).

## Task

Modify the `custom_attention_forward` function and the associated Triton kernel `_custom_attn_fwd` to implement an efficient fused attention forward pass. You may:

- Redesign the tiling strategy (block sizes, tile shapes)
- Optimize the online softmax computation (e.g., use exp2 instead of exp, delay rescaling)
- Improve memory access patterns (coalescing, prefetching)
- Split the causal/non-causal iteration into separate passes to avoid per-block masking overhead
- Use Triton auto-tuning (`@triton.autotune`) to search over configurations
- Define multiple helper kernels if needed

## Interface

```python
def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    """
    Args:
        q, k, v: (batch, nheads, seqlen, headdim), contiguous, FP16
        causal: if True, apply causal mask (key_pos <= query_pos)
        sm_scale: softmax scale factor (default: 1/sqrt(headdim))
    Returns:
        output: (batch, nheads, seqlen, headdim), same dtype as input
    """
```

Correctness constraint: max absolute difference from reference (PyTorch SDPA) must be < 1e-2.

## Evaluation

Benchmarked on three configurations aligned with the FA3 paper (total tokens = 16384):

| Config | Batch | SeqLen | Heads | HeadDim | FA3 Paper (TFLOPs/s) | FA2 Paper (TFLOPs/s) |
|--------|-------|--------|-------|---------|----------------------|----------------------|
| hdim64_seq4k | 4 | 4096 | 32 | 64 | ~420 | ~284 |
| hdim128_seq8k | 2 | 8192 | 16 | 128 | ~602 | ~333 |
| hdim256_seq16k | 1 | 16384 | 8 | 256 | ~642 | ~298 |

All configurations use FP16, causal masking, on H100 80GB SXM5.

**Metrics** (per configuration):
- `tflops`: Achieved TFLOPs/s (higher is better) — primary metric
- `latency_ms`: Kernel latency in milliseconds (lower is better)
- `correct`: Binary (1 if max_diff < 1e-2, else 0) — hard constraint

FLOP formula (FA2/FA3 convention): `4 * batch * seqlen² * nheads * headdim / 2` (causal).

## Hints

- The default template provides a basic flash attention kernel (~200-300 TFLOPs/s). Key optimization opportunities:
  1. **Two-pass causal**: Split the K/V loop into non-causal blocks (no mask check) and causal boundary blocks, reducing branch overhead
  2. **Block size tuning**: Different (BLOCK_M, BLOCK_N) for different headdims — larger blocks amortize loop overhead but increase register pressure
  3. **Triton autotuning**: Use `@triton.autotune` with `configs=[...]` to search block sizes at compile time
  4. **Reduced rescaling**: In the online softmax, the rescaling `acc *= alpha` can be deferred or batched to reduce non-matmul operations
  5. **Memory coalescing**: Ensure K/V loads are coalesced along the headdim dimension
- The Triton tutorial fused attention (triton_flash_v2 baseline) demonstrates the two-pass approach
- FA3 achieves its speedup through Hopper-specific CUDA features (warp specialization, TMA, GMMA) that are not accessible from Triton — closing the gap requires algorithmic cleverness in the Triton DSL
- Available imports: `torch`, `triton`, `triton.language as tl`, `math`, `torch.nn.functional as F`
