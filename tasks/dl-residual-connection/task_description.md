# DL Residual Connection Block Design

## Research Question
Design a novel residual/skip connection block for deep convolutional neural networks that improves test accuracy across different depths and datasets.

## Background
Residual connections (He et al., 2016) enabled training of very deep networks by providing identity shortcut paths. The basic residual block adds the input to the output of two stacked 3x3 convolutions. Several improvements have been proposed:

- **Pre-activation ResBlock** (He et al., 2016 v2): BN-ReLU-Conv order instead of Conv-BN-ReLU, enabling cleaner gradient flow
- **Gated Residual** (ReZero / learnable scaling): A learnable scalar gate scales the residual branch before addition, allowing the network to learn optimal residual contribution per block
- **Stochastic Depth** (Huang et al., ECCV 2016): Randomly drops entire residual blocks during training with linearly decaying survival probability, acting as an implicit ensemble regularizer that is especially effective for very deep networks
- **ResNeXt** (Xie et al., 2017): Grouped convolutions for multi-branch aggregation
- **Res2Net** (Gao et al., 2019): Multi-scale feature extraction within a single residual block

There is room for novel block designs that better balance gradient flow, feature reuse, and computational efficiency, particularly for varying network depths.

## What You Can Modify
The `CustomBlock` class (lines 31-61) in `custom_residual.py`. This is the residual block used by the ResNet backbone.

You can modify:
- The internal convolution structure (number, kernel sizes, grouping)
- The activation function placement and type
- The normalization layer placement and type
- The shortcut/skip connection design
- Channel attention or spatial attention mechanisms
- The `expansion` class attribute (1 for basic, 4 for bottleneck, etc.)
- Any additional modules within the block

Constraints:
- The block must accept `(in_planes, planes, stride)` constructor arguments
- The block must have an `expansion` class attribute
- `forward(x)` must return a tensor with channels = `planes * expansion`
- The shortcut must handle dimension mismatches (stride != 1 or channel mismatch)

## Evaluation
- **Metric**: Best test accuracy (%, higher is better)
- **Architectures & datasets**:
  - ResNet-20 ([3,3,3]) on CIFAR-10 (shallow, 10 classes)
  - ResNet-56 ([9,9,9]) on CIFAR-100 (deep, 100 classes)
  - ResNet-110 ([18,18,18]) on CIFAR-100 (very deep, 100 classes — tests gradient flow) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip

