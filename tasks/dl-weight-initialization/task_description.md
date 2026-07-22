# DL Weight Initialization Strategy Design

## Research Question
Design a novel weight initialization strategy for deep convolutional neural networks that improves convergence speed and final test accuracy across different architectures and datasets.

## Background
Weight initialization is fundamental to training deep neural networks. Poor initialization leads to vanishing/exploding gradients, slow convergence, or suboptimal generalization. Classic methods include:

- **Kaiming/He** (2015): Accounts for ReLU nonlinearity, N(0, sqrt(2/fan_out))
- **Orthogonal** (2014): Preserves gradient norms via orthogonal matrices
- **Fixup** (2019): Scales the last conv in each residual block by L^(-0.5) where L is the number of blocks, controlling variance accumulation across depth; zero-initializes the last BN per block so residual branches start near identity

However, these methods each address only one aspect of initialization. There is room to design strategies that jointly account for residual connections, batch normalization's re-scaling effect, depth-dependent scaling, and the interaction between different layer types (convolution vs classifier).

## What You Can Modify
The `initialize_weights(model, config)` function (lines 147-180) in `custom_init.py`. This function receives the fully constructed model and a config dict, and must initialize all parameters.

You can modify:
- How `nn.Conv2d` weights are initialized (distribution, fan-in/fan-out, gain)
- How `nn.BatchNorm2d` parameters (weight/bias) are initialized
- How `nn.Linear` weights and biases are initialized
- Per-layer or depth-dependent scaling strategies
- Special handling for residual shortcut projections vs main-path convolutions
- Any data-independent initialization logic (no training data access)

The `config` dict provides: `arch` (str), `num_classes` (int), `depth` (int = number of Conv2d + Linear layers). You can also iterate over `model.named_modules()` or `model.named_parameters()`.

## Evaluation
- **Metric**: Best test accuracy (%, higher is better)
- **Architectures & datasets**:
  - ResNet-56 on CIFAR-100 (deep residual, 100 classes)
  - VGG-16-BN on CIFAR-100 (deep non-residual with BatchNorm, 100 classes)
  - MobileNetV2 on FashionMNIST (lightweight inverted-residual, 10 classes) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip

