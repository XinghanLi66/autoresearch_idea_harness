# DL Learning Rate Schedule Design

## Research Question
Design a novel learning rate schedule for training deep convolutional neural networks that improves convergence speed and final test accuracy across different architectures and datasets.

## Background
Learning rate scheduling is critical for training deep neural networks effectively. A fixed learning rate often leads to suboptimal results — too high causes instability, too low results in slow convergence. Classic schedules include:

- **Step decay** (He et al., 2016): Divide LR by 10 at fixed milestones (e.g., 50% and 75% of training)
- **Cosine annealing** (Loshchilov & Hutter, 2017): Smooth decay following a cosine curve from base_lr to 0
- **Warmup + cosine** (Goyal et al., 2017): Linear warmup phase followed by cosine decay, stabilizes large-batch training

However, these schedules are designed without considering architecture-specific properties (depth, residual connections, batch normalization) or dataset characteristics. There is room to design schedules that adapt to the training context.

## What You Can Modify
The `get_lr(epoch, total_epochs, base_lr, config)` function (lines 155-178) in `custom_schedule.py`. This function is called once per epoch to compute the learning rate.

You can modify:
- The shape of the LR decay curve (cosine, polynomial, exponential, linear, piecewise, ...)
- Whether and how long to warm up
- The minimum/final learning rate
- Architecture-aware scheduling (config provides 'arch' and 'dataset')
- Any epoch-dependent logic (cyclic restarts, sharp transitions, plateau regions, ...)

The `config` dict provides: `arch` (str: 'resnet20', 'resnet56', 'mobilenetv2'), `dataset` (str: 'cifar10', 'cifar100', 'fmnist').

## Evaluation
- **Metric**: Best test accuracy (%, higher is better)
- **Architectures & datasets**:
  - ResNet-20 on CIFAR-10 (shallow residual, 10 classes)
  - ResNet-56 on CIFAR-100 (deeper residual, 100 classes)
  - MobileNetV2 on FashionMNIST (lightweight inverted-residual, 10 classes) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), 200 epochs, NO built-in scheduler
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip
- **Weight init**: Kaiming normal (fixed, not editable)

