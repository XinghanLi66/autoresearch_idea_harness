# DL Regularization Strategy Design

## Research Question
Design a novel regularization strategy for deep convolutional neural networks that improves generalization (test accuracy) across different architectures and datasets.

## Background
Regularization is essential for preventing overfitting and improving generalization in deep neural networks. Beyond standard weight decay (L2 penalty), many regularization techniques have been proposed:

- **DropBlock-inspired spatial co-activation penalty** (Ghiasi et al., 2018): Penalizes local spatial co-activation in feature maps, discouraging reliance on contiguous regions — captures the core insight of DropBlock as a loss-based regularizer
- **Confidence penalty** (Pereyra et al., 2017): Penalizes low-entropy output distributions to prevent overconfidence
- **Orthogonal regularization** (Brock et al., 2017): Encourages weight matrices to be orthogonal, preserving gradient flow

However, these methods typically apply a fixed penalty throughout training and do not adapt to training dynamics, model architecture, or the relationship between different layer types. There is room to design regularization strategies that are more adaptive, architecture-aware, or that combine multiple complementary penalties.

## What You Can Modify
The `compute_regularization(model, inputs, outputs, targets, config)` function (lines 155-183) in `custom_reg.py`. This function is called every training step and returns a scalar loss that is added to the cross-entropy loss.

You can use:
- **model**: the full `nn.Module` — iterate over `model.named_parameters()` or `model.named_modules()` for weight-based penalties
- **inputs**: `[B, 3, 32, 32]` — the input batch (for input-dependent regularization)
- **outputs**: `[B, num_classes]` — the model logits (for output-based penalties like confidence/entropy)
- **targets**: `[B]` — integer class labels
- **config**: dict with `num_classes` (int), `epoch` (int, 0-indexed), `total_epochs` (int)

Design ideas:
- Weight-based: L1/L2 norms, orthogonality, spectral norms, weight correlation
- Output-based: entropy, confidence penalty, label smoothing effect, logit penalties
- Activation-based: sparsity, diversity (requires forward hooks)
- Epoch-dependent: warm-up schedules, annealing, curriculum regularization
- Architecture-aware: different penalties for conv vs linear, depth-dependent scaling

Note: Standard L2 weight decay (5e-4) is already applied via the optimizer. Your regularization term is *additional*.

## Evaluation
- **Metric**: Best test accuracy (%, higher is better)
- **Architectures & datasets**:
  - ResNet-56 on CIFAR-100 (deep residual, 100 classes)
  - VGG-16-BN on CIFAR-100 (deep non-residual with BatchNorm, 100 classes)
  - MobileNetV2 on FashionMNIST (lightweight inverted-residual, 10 classes) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip

