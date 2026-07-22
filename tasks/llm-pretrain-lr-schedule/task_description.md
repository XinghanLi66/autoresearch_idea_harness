# LLM Pretraining: Learning Rate Schedule Optimization

## Research Question
Design an improved learning rate schedule for GPT-2 language model pretraining. Your modifications should reduce validation loss compared to the standard cosine annealing schedule with linear warmup.

## What You Can Modify
The `get_lr` function (lines 192-201) in `custom_pretrain.py`:
- Schedule shape (default: cosine decay with linear warmup)
- Warmup strategy and duration
- Decay behavior (shape, rate, final LR)
- Multi-phase scheduling (e.g., warmup-stable-decay)

**Note**: The function signature `get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)` must be preserved. The training loop calls this function at every iteration to set the learning rate.

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better), plus perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 12030 iterations, BSZ=96, GA=6, 2-GPU DDP
- **Hardware**: H200 GPU

