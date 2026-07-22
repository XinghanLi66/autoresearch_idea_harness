# CV Data Augmentation Strategy Design

## Research Question
Design a novel data augmentation strategy for image classification that improves test accuracy across different architectures and datasets.

## Background
Data augmentation is a key regularization technique for training deep neural networks on limited data. By applying transformations to training images, augmentation increases effective dataset diversity and reduces overfitting. Classic and modern methods include:

- **Standard** (baseline): RandomCrop with padding + RandomHorizontalFlip — the minimal CIFAR augmentation
- **Cutout** (DeVries & Taylor, 2017): Randomly masks square regions, forcing the network to use broader spatial context
- **RandAugment** (Cubuk et al., 2020): Applies N randomly selected operations at uniform magnitude M, avoiding expensive search
- **TrivialAugmentWide** (Mueller & Hutter, 2021): Single random operation with random magnitude per image, zero hyperparameters

These methods make different choices about geometric, photometric, and masking transforms, and may behave differently across datasets and model families.

## What You Can Modify
The `build_train_transform(config)` function (lines 165-194) in `custom_augment.py`. This function receives a config dict and must return a `transforms.Compose` pipeline that includes `ToTensor()` and `Normalize()`.

You can modify:
- Which geometric transforms to apply (crop, flip, rotation, affine, perspective)
- Which photometric transforms to apply (color jitter, equalize, posterize, solarize)
- Erasing/masking strategies (cutout, random erasing)
- Automated augmentation policies (AutoAugment, RandAugment, TrivialAugment)
- Custom transform classes (defined inside the function)
- The ordering and composition of transforms
- Any transform that operates on PIL images (before ToTensor) or tensors (after ToTensor)

The `config` dict provides: `img_size` (32), `mean` (tuple), `std` (tuple), `dataset` ('cifar10' or 'cifar100'). You may use dataset-specific augmentation if desired.

**Important**: The pipeline MUST include `transforms.ToTensor()` and `transforms.Normalize(config['mean'], config['std'])` to produce properly normalized tensors for the model.

## Evaluation
- **Metric**: Best test accuracy (%, higher is better)
- **Architectures & datasets**:
  - ResNet-20 on CIFAR-10 (shallow residual, 10 classes)
  - ResNet-56 on CIFAR-100 (deeper residual, 100 classes)
  - MobileNetV2 on FashionMNIST (lightweight inverted-residual, 10 classes) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Weight init**: Standard Kaiming normal (fixed)
