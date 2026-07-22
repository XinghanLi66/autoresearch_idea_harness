# DL Normalization Layer Design

## Research Question
Design a novel normalization layer for deep convolutional neural networks that improves training stability and final test accuracy across different architectures and datasets.

## Background
Normalization layers are critical components in modern deep networks, controlling internal covariate shift and enabling stable training at higher learning rates. Classic methods include:

- **BatchNorm** (Ioffe & Szegedy, 2015): Normalizes across the batch dimension per channel. The de facto standard, but depends on batch statistics and behaves differently at train/test time.
- **GroupNorm** (Wu & He, 2018): Divides channels into groups and normalizes within each group. Batch-size independent.
- **InstanceNorm** (Ulyanov et al., 2016): Normalizes each channel independently per instance. Common in style transfer.
- **LayerNorm** (Ba et al., 2016): Normalizes across all channels for each sample. Standard in transformers but less common in CNNs.

However, each method has limitations: BatchNorm degrades with small batches, GroupNorm requires choosing the number of groups, InstanceNorm discards inter-channel information, and LayerNorm may not suit spatial feature maps well. There is room to design normalization strategies that combine strengths of multiple approaches or introduce novel normalization statistics.

## What You Can Modify
The `CustomNorm` class (lines 31-45) in `custom_norm.py`. This class must be a drop-in replacement for `nn.BatchNorm2d`:

- Constructor takes `num_features` (number of channels C)
- Input shape: `[B, C, H, W]`
- Output shape: `[B, C, H, W]`

You can modify:
- The normalization statistics (mean/variance computation: over batch, channel, spatial, or combinations)
- Learnable affine parameters (scale and shift)
- The normalization grouping strategy
- Combining multiple normalization approaches
- Adaptive or input-dependent normalization
- Any other normalization design that maintains the interface

## Evaluation
- **Metric**: Best test accuracy (%, higher is better)
- **Architectures & datasets**:
  - ResNet-56 on CIFAR-100 (deep residual, 100 classes)
  - MobileNetV2 on FashionMNIST (lightweight inverted-residual, 10 classes)
  - ResNet-110 on CIFAR-100 (very deep residual, 100 classes) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip

