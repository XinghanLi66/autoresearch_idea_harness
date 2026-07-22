# LLM Pretraining: Embedding Strategy Optimization

## Research Question
Design an improved embedding strategy for GPT-2 language model pretraining. Your modifications should reduce validation loss compared to standard token + position embeddings with weight tying.

## What You Can Modify
The `TokenEmbedding` class (lines 116-140) in `custom_pretrain.py`:
- Token embedding representation (default: learned token + position embeddings)
- Weight tying strategy (default: input embedding = output lm_head weight)
- Additional embedding sources (e.g., n-gram embeddings, hash-based embeddings)
- Value embeddings that inject into transformer layers via `get_value_embed(layer_idx)`

**Interface**: Your `TokenEmbedding` class must implement:
- `forward(idx) -> x`: Takes token indices `(B, T)`, returns embeddings `(B, T, n_embd)`
- `get_lm_head_weight() -> weight`: Returns the weight tensor for the output projection
- `get_num_pos_params() -> int`: Returns count of position parameters (excluded from reported param count)
- `get_value_embed(layer_idx) -> Optional[Tensor]`: (Optional) Returns per-layer value embedding residual `(B, T, n_embd)` or `None`

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better), plus perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 12030 iterations, BSZ=96, GA=6, 2-GPU DDP
- **Hardware**: H200 GPU

