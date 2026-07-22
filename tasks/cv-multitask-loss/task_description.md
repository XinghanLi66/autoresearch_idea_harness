# CV Multi-Task Loss Combination Strategy Design

## Research Question
Design a novel multi-task loss combination strategy for jointly training fine-grained (100-class) and coarse (20-superclass) classification on CIFAR-100 that maximizes fine-class test accuracy.

## Background
CIFAR-100 contains 100 fine classes organized into 20 coarse superclasses. Training a model with two classification heads (fine + coarse) provides a natural multi-task learning setup where the coarse task acts as an auxiliary signal. The key challenge is how to combine the two losses effectively.

Classic approaches include:
- **Equal weighting**: Simply sum the losses (baseline default)
- **Uncertainty weighting** (Kendall et al., 2018): Learn task-specific uncertainty as log-variance parameters
- **Dynamic Weight Average** (Liu et al., 2019): Weight tasks by their relative loss change rate
- **PCGrad** (Yu et al., NeurIPS 2020): Project conflicting task gradients onto each other's normal plane to reduce gradient interference

The coarse labels encode semantic hierarchy. The task is to balance the auxiliary coarse signal against the primary fine-class objective across different architectures and training stages.

## What You Can Modify
The `MultiTaskLoss` class (lines 195-216) in `custom_mtl.py`. This class receives individual task losses and must combine them into a single scalar loss.

You can modify:
- The `__init__` method: add learnable parameters (log-variances, weights, etc.)
- The `forward` method: implement any combination strategy
- Use `epoch` and `total_epochs` for curriculum/scheduling approaches
- Add any auxiliary state (e.g., loss history buffers)

The `forward` method receives:
- `fine_loss`: scalar tensor, cross-entropy for 100-class fine prediction
- `coarse_loss`: scalar tensor, cross-entropy for 20-class coarse prediction
- `epoch`: int, current epoch (0-indexed)
- `total_epochs`: int, total number of training epochs

Note: The `MultiTaskLoss` parameters are included in the optimizer, so learnable parameters will be trained.

## Evaluation
- **Metric**: Best fine-class test accuracy (%, higher is better)
- **Architectures** (all on CIFAR-100 with fine+coarse heads):
  - ResNet-20 (shallow residual network)
  - ResNet-56 (deeper residual network)
  - VGG-16-BN (deep non-residual with BatchNorm) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip
