# Offline-to-Online RL: Preventing Catastrophic Forgetting in Fine-Tuning


## Objective
Design and implement an offline-to-online RL algorithm that pretrains from an offline dataset (1M steps), then fine-tunes with online interaction (1M steps) without catastrophic forgetting or Q-value collapse. Your code goes in `custom_finetune.py`. Three reference implementations (AWAC, SPOT, Cal-QL) are provided as read-only.

## Background
The critical challenge is the offline-to-online transition: naive fine-tuning often causes Q-value collapse (conservative estimates become overoptimistic) and catastrophic forgetting. The Adroit `cloned-v1` datasets mix expert and noisy demonstrations, making this transition particularly challenging.

## Constraints
- **Network dimensions are fixed at 256.** All MLP hidden layers must use 256 units. A `_mlp()` factory function is provided in the FIXED section for convenience. You may define custom network classes but hidden widths must remain 256.
- **Total parameter count is enforced.** The training loop checks that total trainable parameters do not exceed 1.2x the largest baseline architecture. Focus on algorithmic innovations (loss functions, regularization, training procedures), not network capacity.
- Do NOT simply copy a reference implementation with minor changes

## Evaluation
Trained and evaluated on Pen, Door, Hammer using Adroit `cloned-v1` datasets. Additional held-out environments (not shown during intermediate testing) are used to assess generalization. Metric: D4RL normalized score (0 = random, 100 = expert), evaluated throughout both phases.
