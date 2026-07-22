# LLM Pretraining: Sparse Attention Mechanism

## Research Question
Design a *sparse* causal self-attention mechanism for GPT-2 Medium pretraining
that attends to **at most 25% of the (Q,K) pairs** of full causal attention,
yet matches (or approaches) the validation loss of the dense reference. The
research goal is to identify which sparsity pattern is most useful when the
compute budget is fixed and the model has to learn from scratch at 345M scale.

## What You Can Modify
The `SparseSelfAttention` class (lines 34-72 in `custom_pretrain.py`). You
implement the entire `(q, k, v) -> output` computation. Allowed tools:
- `torch.nn.attention.flex_attention` (FlexAttention with `score_mod` /
  `mask_mod`) — works on A100 (sm80+).
- `torch.nn.functional.scaled_dot_product_attention` with explicit attn masks.
- Hand-written PyTorch ops (gather/scatter/segment, block-sparse matmul).
- **Triton kernels are out of scope** — do not write `@triton.jit` code.

You may also adjust the `CONFIG_OVERRIDES` dict (lines 269-271) to override
`learning_rate`, `weight_decay`, `warmup_iters`, `min_lr`, or `grad_clip`.

## Sparsity Budget (HARD CONSTRAINT)
The FIXED region (lines 74-90) defines `_BUDGET_DENSITY = 0.25` and a runtime
check `_assert_density_budget(self.attn)` invoked from `Block.forward` after
every attention call.

You **MUST** set `self.reported_density` (a float in `[0, 1]`) on every forward
pass to honestly reflect the fraction of (Q,K) pairs your pattern attends to,
relative to full causal attention `T*(T+1)/2`. If your reported density exceeds
`0.25` the run aborts with `RuntimeError`.

The `is_dense_oracle` flag is reserved for the `dense` reference baseline —
**agents must not set it to `True`**.

Examples of how to compute reported density honestly:
- Sliding window of size W on length T: `reported_density ≈ W / (T/2)` capped
  at 1, or more simply `min(W*T, T*(T+1)/2) / (T*(T+1)/2)`.
- Block-sparse with K active blocks per query block, block size B, T=N*B:
  `reported_density ≈ K / N`.

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better), perplexity
  (WikiText-2, LAMBADA), and downstream accuracy (ARC-Easy, HellaSwag, PIQA,
  WinoGrande).
- **Model**: GPT-2 Medium (24L / 16H / 1024D, ~355M params).
- **Dataset**: ClimbMix (curated, GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla).
- **Training**: 13535 iterations, BSZ=64, GA=8, 2-GPU DDP.
- **Hardware**: A100 (sm80+) GPU.

## Hints
- FlexAttention with a `mask_mod` (returning a `BlockMask` from
  `create_block_mask`) is the cleanest way to express block-sparse, sliding,
  dilated, or sink+window patterns without writing custom kernels.
- A standard sliding window of W=256 on T=1024 gives density ≈ 0.50 — too
  high. W=128 gives density ≈ 0.25.
- Mixing local + a few global tokens (BigBird, StreamingLLM, NSA-block)
  preserves long-range information without inflating density much.
- Set `self.use_pos_emb = False` if your attention provides its own positional
  signal (RoPE, ALiBi). Otherwise leave it `True`.
