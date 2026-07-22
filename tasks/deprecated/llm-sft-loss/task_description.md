# SFT Loss Function Design

## Research Question

Design a novel loss function for supervised fine-tuning (SFT) of large language models that improves downstream task performance compared to standard cross-entropy.

## Background

Standard SFT uses cross-entropy loss to train models on instruction-response pairs. However, cross-entropy treats all tokens equally and does not account for varying token difficulty or the distribution of model confidence across the vocabulary. Recent work has explored alternative loss functions that can improve SFT quality:

- **Focal Loss**: Down-weights easy (high-confidence) tokens so the model focuses on harder tokens, controlled by a focusing parameter gamma.
- **Label Smoothing**: Regularizes by distributing a small probability mass across all vocabulary tokens, preventing overconfident predictions.
- **GEM (Generalized Entropy Minimization)**: Uses an auxiliary softmax distribution at a different temperature to create a contrastive weighting scheme that emphasizes hard examples.
- **DFT (Distribution-aware Fine-Tuning)**: Reweights per-token losses by the model's own prediction confidence.
- **Entropy Regularization**: Adds a term encouraging higher entropy in the output distribution, which can improve exploration and generalization.

## Task

Implement `custom_loss_func` in `custom_sft_loss.py`. Your loss function receives:
- `outputs`: model output dict (use `outputs.get("logits")` for logits)
- `labels`: target token IDs (NOT pre-shifted; apply causal shift yourself)
- `num_items_in_batch`: optional normalization scalar for gradient accumulation

The model is fine-tuned on MetaMathQA (20K samples) with Qwen3-1.7B, then evaluated on hellaswag, arc_challenge, and piqa using lm-evaluation-harness. Your goal is to maximize downstream evaluation accuracy.

