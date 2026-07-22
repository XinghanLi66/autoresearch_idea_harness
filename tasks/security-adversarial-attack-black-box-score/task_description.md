# Score-Based Query Black-Box Attack under Linf Constraint

## Research Question
Can you design a stronger score-based query black-box attack that improves attack success rate (ASR) under a fixed query budget and `L_inf` perturbation constraint?

## Objective
Implement a better black-box attack in `bench/custom_attack.py`:

- Threat model: query black-box (no gradient access).
- Constraint: `||x_adv - x||_inf <= eps`.
- Budget: `n_queries` is a per-sample query budget.
- Primary metric: maximize `ASR` under fixed budget.
- Tie-break: for similar ASR, lower `avg_queries` is better.

## Editable Interface
You must implement:

`run_attack(model, images, labels, eps, n_queries, device, n_classes) -> adv_images`

Inputs:
- `model`: black-box wrapper that returns logits only.
- `images`: tensor of shape `(N, C, H, W)`, in `[0, 1]`.
- `labels`: tensor of shape `(N,)`.
- `n_classes`: 10 for CIFAR-10, 100 for CIFAR-100.

Output:
- `adv_images`: tensor with same shape as `images`, values in `[0, 1]`.

## Trusted Evaluation Logic
The evaluation logic in `bench/run_eval.py` is trusted and not editable.

- It tracks all model queries through a wrapper.
- If a batch exceeds query budget (`batch_size * n_queries`), the entire batch is marked as attack failure.
- `L_inf` and `[0, 1]` validity are checked per sample; only invalid samples are marked as attack failure.

Wrapper behavior and evaluation logic are fixed. Improvements should be confined to the attack algorithm in `custom_attack.py`.

## Query Semantics
- One call to `model(x)` consumes `x.shape[0]` queries.
- Repeated calls on the same sample still consume additional queries.
- Different batch partitioning should be treated as equivalent total budget usage.

## Evaluation Scenarios (6)
- ResNet20 on CIFAR-10
- VGG11-BN on CIFAR-10
- MobileNetV2 on CIFAR-10
- ResNet20 on CIFAR-100
- VGG11-BN on CIFAR-100
- MobileNetV2 on CIFAR-100

Reported metrics line format:

`ATTACK_METRICS asr=... clean_acc=... robust_acc=... avg_queries=...`
