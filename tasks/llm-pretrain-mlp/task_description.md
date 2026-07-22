# LLM Pretraining: Feed-Forward Network Optimization

## Research Question
Design an improved feed-forward network (MLP) for GPT-2 language model pretraining. Your modifications should reduce validation loss compared to the standard GELU MLP.

## What You Can Modify
The `MLP` class (lines 73-86 in `custom_pretrain.py`), including:
- Activation function (default: GELU)
- Network architecture (default: two linear layers with 4x expansion)
- Gating mechanisms
- Hidden dimension sizing

**Constraint**: The MLP must accept input of shape `(B, T, n_embd)` and return output of the same shape.

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better), plus perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 12030 iterations, BSZ=96, GA=6, 2-GPU DDP
- **Hardware**: H200 GPU

