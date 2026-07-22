# Sparse Adversarial Attack (L0 Constraint)

## Objective
Implement a stronger sparse attack in `bench/custom_attack.py`.
Your method should maximize attack success rate (ASR) under a strict `L0` perturbation budget:

- Threat model: full model access for custom attack implementation.
- Norm constraint: number of modified spatial pixels is bounded.
- Budget: `L0(x_adv, x) <= pixels`, where `pixels = 10`.

## Editable Interface
You must implement:

`run_attack(model, images, labels, pixels, device, n_classes) -> adv_images`

Inputs:
- `images`: tensor of shape `(N, C, H, W)`, values in `[0, 1]`.
- `labels`: tensor of shape `(N,)`.
- `pixels`: maximum number of modified spatial pixels per sample.
- `n_classes`: 10 for CIFAR-10, 100 for CIFAR-100.

Output:
- `adv_images`: same shape as `images`, also in `[0, 1]`.

## Evaluation Protocol
Each evaluation script:
1. Loads one pretrained model.
2. Collects up to 1000 samples that are initially classified correctly.
3. Runs your `run_attack`.
4. Checks `L0` validity (`<= pixels` modified spatial pixels).
5. Reports:
   - `clean_acc`
   - `robust_acc`
   - `asr = 1 - robust_acc`

Important:
- ASR denominator is the number of initially correct samples.
- Invalid adversarial outputs (shape mismatch or violated budget) are treated as failure.

## Scenarios
Six scenarios are evaluated in parallel:

- ResNet20 on CIFAR-10
- VGG11-BN on CIFAR-10
- MobileNetV2 on CIFAR-10
- ResNet20 on CIFAR-100
- VGG11-BN on CIFAR-100
- MobileNetV2 on CIFAR-100

## Baselines
- `onepixel`: one-pixel differential evolution based sparse baseline.
- `sparsefool`: gradient-based sparse perturbation baseline.
- `jsma`: Jacobian saliency map based targeted sparse baseline.
- `pixle`: pixel rearrangement based sparse baseline.

Your goal is to improve ASR while respecting the L0 budget.
