# DL Activation Function Design

## Research Question
Design a novel activation function for deep convolutional neural networks that improves test accuracy across different architectures (ResNet, VGG) and datasets (CIFAR-10, CIFAR-100).

## Background
Activation functions introduce nonlinearity into neural networks and critically affect training dynamics and generalization. Classic choices include:

- **ReLU** (2010): max(0, x) — simple, sparse, but zero gradient for negative inputs ("dying ReLU")
- **GELU** (2016): x * Phi(x) — smooth approximation weighting by Gaussian CDF
- **Swish/SiLU** (2017): x * sigmoid(x) — self-gated, smooth, non-monotonic
- **Mish** (2019): x * tanh(softplus(x)) — self-regularized, smooth

These functions differ in smoothness, gating behavior, and negative-domain behavior, and may interact differently with modern network components such as residual connections and batch normalization.

## What You Can Modify
The `CustomActivation` class (lines 31-48) in `custom_activation.py`. This is an `nn.Module` used as a drop-in replacement for ReLU throughout the network.

You can modify:
- The forward computation (any element-wise or channel-wise operation)
- Learnable parameters (registered in `__init__`)
- The shape of the activation curve (monotonic, non-monotonic, bounded, etc.)
- Negative-domain behavior (zero, linear, bounded, learnable)
- Any stateless or stateful activation logic

The activation is used in:
- ResNet: BasicBlock (2x per block) + initial conv
- VGG: after every Conv-BN pair + in the classifier head

## Evaluation
- **Metric**: Best test accuracy (%, higher is better)
- **Architectures & datasets**:
  - ResNet-20 on CIFAR-10 (shallow residual, 10 classes)
  - VGG-16-BN on CIFAR-100 (deep non-residual with BatchNorm, 100 classes)
  - MobileNetV2 on FashionMNIST (lightweight inverted-residual with ReLU6 baseline, 10 classes) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip
