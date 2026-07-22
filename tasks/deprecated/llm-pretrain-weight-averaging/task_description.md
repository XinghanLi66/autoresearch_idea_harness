# LLM Pretraining: Weight Averaging Strategy

## Research Question
Design a weight averaging strategy that produces a better final model during GPT-2 language model pretraining. Your modifications should reduce the validation loss of the averaged model compared to the final checkpoint (no averaging) and standard approaches like SWA or fixed-decay EMA.

## What You Can Modify
The `WeightAverager` class (lines 205-252) in `custom_pretrain.py`:
- How averaged weights are initialized and stored
- The update rule applied after each optimizer step (EMA decay schedule, selective averaging, etc.)
- When averaging starts or stops during training
- How averaged weights are applied to the model for evaluation

**Note**: The class interface must be preserved:
- `__init__(self, model, max_iters)` — called once before training starts (before torch.compile and DDP wrapping), receives the raw model and total iteration count
- `update(self, model, step)` — called after every optimizer step with the raw (unwrapped) model and current iteration number (0-indexed)
- `apply(self, model)` — called before final evaluation to load averaged weights into the model; must save original weights internally
- `restore(self, model)` — called after evaluation to restore original training weights

The model architecture, optimizer (AdamW), learning rate schedule (cosine), data pipeline, and training loop are all fixed.

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better) of the averaged model
- **Model sizes**: GPT-2 124M (12L/12H/768D) and GPT-2 1.5B (48L/25H/1600D, 4-GPU DDP)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer)
- **Training**: 5000 iterations, batch_size=12, block_size=1024, grad_accum=5

