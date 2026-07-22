# LLM Offline RL: Preference Optimization for Math Reasoning

## Objective
Design a custom preference loss for offline preference optimization of a math
LLM. Implement your loss in the `compute_preference_loss` method of
`trainer.py` (selected via `pref_loss=custom`, registered in
`finetuning_args.py`).

## Background
DPO and its variants (Hinge, IPO, KTO-pair, ORPO, SimPO) directly optimize a
preference loss on chosen/rejected response pairs without a separate reward
model. Each variant trades off stability, length bias, calibration, and
reference dependence differently. This task asks you to design a single
preference loss that improves math reasoning over the standard variants.

## Setup (Step-DPO recipe — Lai et al. 2024, arXiv:2406.18629)
- **Base model**: `Qwen2.5-Math-1.5B-Instruct` (math-specialized SFT, 1.5B
  params). Trained with the `qwen` chat template.
- **Preference data**: `xinlai/Math-Step-DPO-10K` (~10K math problems with
  step-level chosen/rejected solutions). We use the full responses as
  response-level chosen/rejected pairs, so all DPO variants apply directly.
- **Training**: full-parameter, 4× GPU, ZeRO-2, β=0.1 (or variant-specific),
  4 epochs, lr=5e-7, cosine schedule.

## Evaluation (judge-free)
Three math reasoning benchmarks — graded by MathRuler's sympy + mathd
checker, no LLM judge involved:

1. **GSM8K** — grade-school math (1.32K problems). Metric: `gsm8k_accuracy`.
2. **MATH-500** — 500-problem subset of MATH competition. Metric: `math500_accuracy`.
3. **AIME 2024** — 30 American Invitational Math Exam problems. Metric: `aime2024_accuracy`.

A single vLLM engine is loaded once per evaluation pass and runs greedy
decoding (temperature=0) for all three benchmarks.

## Baselines
| Name  | `pref_loss` | Reference            |
|-------|-------------|----------------------|
| dpo   | sigmoid     | Rafailov et al. 2023 |
| simpo | simpo       | Meng et al. 2024     |
| ipo   | ipo         | Azar et al. 2023     |
| orpo  | orpo        | Hong et al. 2024     |

Your `pref_loss=custom` implementation should beat at least one of these on
the average of the three math benchmarks.
