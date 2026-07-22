# LLM Pretraining: Normalization & Block Architecture Optimization

## Research Question
Design improved normalization and/or transformer block architecture for GPT-2 language model pretraining. Your modifications should reduce validation loss compared to the standard LayerNorm with Pre-LN block structure.

## What You Can Modify
Two regions in `custom_pretrain.py`:
1. **LayerNorm class** (lines 23-31): The normalization implementation
2. **Block class** (lines 89-100): How attention and MLP are composed with residual connections

You can modify:
- The normalization algorithm (default: LayerNorm with bias)
- Where normalization is applied (Pre-LN, Post-LN, or other placements)
- The residual connection structure
- How attention and MLP sublayers are combined (sequential vs parallel)

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better), plus perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 12030 iterations, BSZ=96, GA=6, 2-GPU DDP
- **Hardware**: H200 GPU

