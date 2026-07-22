# LLM Pretraining: Optimizer & Learning Rate Schedule Optimization

## Research Question
Design an improved optimizer and/or learning rate schedule for GPT-2 language model pretraining. Your modifications should reduce validation loss compared to the standard AdamW with cosine annealing schedule.

## What You Can Modify
Two regions in `custom_pretrain.py`:
1. **configure_optimizers method** (lines 172-189): Optimizer creation and parameter grouping
2. **get_lr function** (lines 192-201): Learning rate schedule

You can modify:
- The optimization algorithm (default: AdamW with fused implementation)
- Parameter grouping strategy (default: weight decay for 2D params, no decay for 1D)
- Learning rate schedule shape (default: cosine with linear warmup)
- Any optimizer hyperparameters

**Note**: The training loop calls `get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)` — keep this signature compatible. The optimizer returned by `configure_optimizers` must support `.zero_grad()`, `.step()`, and `.param_groups`.

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better), plus perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 12030 iterations, BSZ=96, GA=6, 2-GPU DDP
- **Hardware**: H200 GPU

