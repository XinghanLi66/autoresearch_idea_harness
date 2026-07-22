# LLM Pretraining: Linear/Subquadratic Attention Mechanism

## Research Question
Design a novel linear or subquadratic attention mechanism for GPT-2 language model pretraining that achieves competitive validation loss while replacing standard softmax attention. The mechanism should scale better than O(n^2) in sequence length.

## What You Can Modify
Two editable regions in `custom_pretrain.py`:

1. **`CausalSelfAttention` class** (lines 33-70): The attention mechanism itself, including:
   - The attention computation (replace softmax attention with linear/subquadratic alternatives)
   - Feature maps, gating mechanisms, decay factors
   - Query/Key/Value projections and transformations
   - Internal state management (recurrent states, convolutions, etc.)

2. **`Block` class** (lines 88-100): The transformer block structure, including:
   - How attention and MLP sublayers are composed
   - Normalization placement (pre-norm, post-norm)
   - Residual connection patterns

**Note**: The `flash-linear-attention` (FLA) library is pre-installed and provides 27+ optimized linear attention implementations with Triton kernels. You can import from `fla.layers` (e.g., `GatedLinearAttention`, `DeltaNet`, `MultiScaleRetention`, `LinearAttention`, `HGRN2`, `Mamba2`, etc.) or implement your own mechanism from scratch.

**Note**: If your attention mechanism does not use learned absolute position embeddings, set `self.use_pos_emb = False` in `__init__` — the model will then skip adding position embeddings in the forward pass.

**Note**: `torch.compile` is disabled for this task since FLA's Triton kernels are not compatible with it.

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better), plus perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 13535 iterations, BSZ=32, GA=16, 2-GPU DDP
- **Hardware**: H200 GPU

