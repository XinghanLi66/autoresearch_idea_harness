# JEPA Self-Supervised Learning: Anti-Collapse Regularization

## Research Question
Design an improved anti-collapse regularization loss for Joint Embedding Predictive Architecture (JEPA) self-supervised image representation learning. Your regularizer should prevent representation collapse (where all inputs map to the same output) while encouraging the model to learn useful, discriminative features.

## What You Can Modify
The editable region in `custom_regularizer.py` is lines 33-58: the `CustomRegularizer` class plus the `CONFIG_OVERRIDES` dictionary. The class receives two projected embedding tensors from different augmented views of the same images and must return a loss dictionary.

Interface:
- **Input**: `z1: [B, D]` and `z2: [B, D]` -- projected embeddings from two augmented views
- **Output**: `dict` with at least a `"loss"` key containing a scalar tensor

You may add any parameters to `__init__`, define helper methods, and use any PyTorch operations. The imports at the top of the file (torch, torch.nn, torch.nn.functional, etc.) are available.

## Evaluation
- **Metric**: `val_acc` -- linear probe classification accuracy on CIFAR-10 (higher is better)
- **Benchmarks**: Three backbone architectures (ResNet-18, ResNet-34, ResNet-50) test regularizer generalization across model scales
- **Projector**: features_dim -> 2048 -> 2048 MLP
- **Training**: 100 epochs, batch size 256, LARS optimizer (lr=0.3), warmup cosine schedule
- **Dataset**: CIFAR-10 (50k train / 10k val)
