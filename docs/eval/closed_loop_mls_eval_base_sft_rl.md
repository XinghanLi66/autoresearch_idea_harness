# Closed-loop MLS eval — 32B base vs SFT vs RL (worker-implemented, 10 tasks)

Worker = Claude Code implements each proposal, trains ResNet-20/CIFAR-10 (or task model) 200 ep, returns signed metric. Higher = better (except where noted). Pass = metric ≥ threshold.

| task | thresh | base | SFT | RL | best |
|---|---|---|---|---|---|
| cv_classification_loss | 73.48 | 72.77 | 72.68 | 71.99 | base |
| cv_data_augmentation | 93.76 | 92.29 | 92.91 | 91.92 | sft |
| cv_multitask_loss | 68.79 | 68.67 | 25.89 | 68.51 | base |
| cv_pooling_aggregation | 72.19 | 72.64✅ | 71.75 | 72.27✅ | base |
| cv_sample_weighting | 74.44 | 72.65 | 72.16 | 72.74 | rl |
| dl_activation_function | 93.35 | 92.8 | 91.95 | 92.55 | base |
| dl_lr_schedule | 93.19 | 92.65 | 92.8 | 93.08 | rl |
| dl_regularization | 73.0 | 72.19 | 73.07✅ | 72.2 | sft |
| dl_residual_connection | 92.96 | 93.12✅ | 92.28 | 92.83 | base |
| dl_weight_initialization | 73.26 | 72.98 | 72.14 | 72.42 | base |

- **base**: 2/10 passed · completed 10/10 · mean metric 80.28 · median 72.88
- **sft**: 1/10 passed · completed 10/10 · mean metric 75.76 · median 72.88
- **rl**: 1/10 passed · completed 10/10 · mean metric 80.05 · median 72.58

Recorded V2.5 baseline (prior eval, 16k-ctx): base 1/10, old proposal-SFT 0/10.
