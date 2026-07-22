# White-Box Evasion Attack under Linf Constraint

## Objective
Implement a stronger white-box `L_inf` attack in `bench/custom_attack.py`.
Your method should maximize attack success rate (ASR) under a strict perturbation budget:

- Threat model: white-box (full model access, including gradients).
- Norm constraint: `||x_adv - x||_inf <= eps`.
- Budget: `eps = 2/255` (small-budget regime used to differentiate attack quality on
  undefended models; RobustBench uses 8/255 for *defended* models, which saturates ASR
  to ~1.0 on undefended ones, leaving no headroom for agents).

## Editable Interface
You must implement:

`run_attack(model, images, labels, eps, device, n_classes) -> adv_images`

Inputs:
- `images`: tensor of shape `(N, C, H, W)`, values in `[0, 1]`.
- `labels`: tensor of shape `(N,)`.
- `n_classes`: 10 for CIFAR-10, 100 for CIFAR-100.

Output:
- `adv_images`: same shape as `images`, also in `[0, 1]`.

## Evaluation Protocol
Each evaluation script:
1. Loads one pretrained model.
2. Collects up to 1000 samples that are initially classified correctly.
3. Runs your `run_attack`.
4. Checks `L_inf` validity.
5. Reports:
   - `clean_acc`
   - `robust_acc`
   - `asr = 1 - robust_acc`

Important:
- ASR denominator is the number of initially correct samples.
- Invalid adversarial outputs (shape mismatch or violated norm) are treated as failure.

## Scenarios
Six scenarios are evaluated in parallel:

- ResNet20 on CIFAR-10
- VGG11-BN on CIFAR-10
- MobileNetV2 on CIFAR-10
- ResNet20 on CIFAR-100
- VGG11-BN on CIFAR-100
- MobileNetV2 on CIFAR-100

## Baselines
- `fgsm`: one-step FGSM baseline (simplest first-order attack).
- `pgd`: iterative PGD baseline (strong first-order baseline).
- `mifgsm`: momentum iterative FGSM.
- `autoattack`: `torchattacks.AutoAttack(version="standard")` as a strong upper baseline.

Your goal is to improve ASR while respecting the Linf budget.

## Note on per-architecture natural robustness

At `eps=2/255`, ASR differs substantially across architectures on undefended models:
- ResNet20 / MobileNetV2: PGD-40 and AutoAttack both hit ~99% ASR — near-saturated.
- VGG11-BN: PGD-40 and AutoAttack plateau near 72% ASR — leaves meaningful headroom.

This is an architectural property, not an evaluation bug: VGG11-BN's wider-but-shallower
activations at low-resolution feature maps absorb small Linf perturbations more robustly
than bottlenecked ResNet / depthwise-separable MobileNetV2. Agents targeting ASR
improvements over AutoAttack should focus on VGG11-BN envs where there is genuine margin;
ResNet20 and MobileNetV2 are effectively capped near 100%.
