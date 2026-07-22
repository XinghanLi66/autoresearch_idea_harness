# CV Global Pooling / Feature Aggregation Design

## Research Question
Design a novel global pooling or feature aggregation strategy for image classification that improves test accuracy across different CNN architectures and datasets.

## Background
Global pooling is the final spatial aggregation step in modern CNNs, reducing feature maps from [B, C, H, W] to [B, C] before the classifier head. The standard approach is Global Average Pooling (GAP), which computes the spatial mean per channel. While simple and effective, GAP discards spatial structure and treats all positions equally. Alternative strategies include:

- **Global Max Pooling (GMP)**: Selects the strongest activation per channel, capturing the most salient features but ignoring distribution information.
- **Generalized Mean (GeM) Pooling** (Radenovic et al., 2018): Learnable power-mean that interpolates between average and max pooling.
- **Average + Max**: Element-wise combination of GAP and GMP, capturing both mean-field and peak statistics.

There is room to design pooling strategies that better capture the spatial statistics of feature maps, adapt to different architectures, or learn task-specific aggregation patterns.

## What You Can Modify
The `CustomPool` class (lines 31-48) in `custom_pool.py`. This class receives a 4D tensor [B, C, H, W] and must return a 2D tensor [B, C].

You can modify:
- The aggregation function (mean, max, learned weights, attention, higher-order statistics)
- Whether to use learnable parameters
- How spatial information is summarized (single-point, multi-scale, distribution-based)
- Channel-wise or spatial-wise weighting mechanisms
- Any combination of the above

Constraints:
- Input: [B, C, H, W] tensor (C varies by architecture: 64 for ResNet-56, 512 for VGG-16-BN, 1280 for MobileNetV2)
- Output: [B, C] tensor (must match input channel dimension exactly)
- Must work with variable spatial sizes (8×8 for ResNet on CIFAR, 1×1 for VGG after max-pools, 1×1 for MobileNetV2)
- No access to training data or labels within the pooling layer

## Evaluation
- **Metric**: Best test accuracy (%, higher is better)
- **Architectures & datasets**:
  - ResNet-56 on CIFAR-100 (deep residual, 100 classes; final feature map 8×8, C=64)
  - VGG-16-BN on CIFAR-100 (deep non-residual with BatchNorm, 100 classes; final feature map 1×1 after max-pools, C=512)
  - MobileNetV2 on FashionMNIST (lightweight inverted-residual, 10 classes) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip

