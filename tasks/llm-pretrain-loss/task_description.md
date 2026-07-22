# LLM Pretraining: Loss Function Optimization

## Research Question
Design an improved loss function for GPT-2 language model pretraining. Your modifications should reduce validation loss compared to standard cross-entropy.

## What You Can Modify
The `compute_loss` function (lines 189-191) in `custom_pretrain.py`:
- Loss function formulation (default: standard cross-entropy)
- Logit processing (e.g., softcapping, temperature scaling)
- Regularization terms (e.g., z-loss, entropy penalties)
- Label distribution modifications (e.g., label smoothing)

**Note**: The function signature `compute_loss(logits, targets)` must be preserved. `logits` has shape `(B, T, V)` and `targets` has shape `(B, T)`. The function is called inside the model's forward pass during training.

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better), plus perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 13535 iterations, BSZ=64, GA=8, 2-GPU DDP
- **Hardware**: H200 GPU

