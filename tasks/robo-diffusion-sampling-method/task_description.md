# Robo-Diffusion: Sampling Algorithm Design

## Objective
Design a single efficient diffusion sampler for a fixed DQL-style diffusion
policy, maximizing D4RL MuJoCo return at low inference NFE.

This task is deliberately about inference-time sampler choice, not policy
learning, guidance, or trajectory planning. The trained actor/critic, dataset,
environment list, seeds, and evaluation loop are fixed.

## What You Can Modify
- `solver` in `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`
- `sampling_steps` in the same YAML file

## What Is Fixed
- The pipeline code, model architecture, critic, and training objective
- `diffusion_steps`, training budgets, checkpoint selection, and EMA use
- D4RL environment names, seeds, vectorized evaluation, and hidden split

The score's NFE term is read from the same `sampling_steps` field passed to
CleanDiffuser's sampler. Custom pipeline-code samplers are intentionally out of
scope here because they would decouple true NFE from the reported score column.

## Evaluation
Evaluated on three D4RL MuJoCo environments:
1. **hopper-medium-v2**
2. **walker2d-medium-v2**
3. **halfcheetah-medium-v2**

Metrics: `normalized_score` (D4RL return) and `sampling_steps` (NFE per inference call).

`halfcheetah-medium-v2` is hidden during intermediate agent tests.

### Score formula
The per-env score multiplies a quality term by an NFE penalty:

```
score(env) = sigmoid(normalized_score) * penalty_upper(sampling_steps, target=10)
penalty_upper(x, target=10) = exp(-0.015 * (x - 10))   for x > 10
                              1.0                       for x <= 10
```

NFE penalty cheat-sheet:

| sampling_steps | penalty | example                |
|---------------:|--------:|------------------------|
| 10             | 1.000   | DPM-Solver++ baseline  |
| 20             | 0.861   | DDIM baseline          |
| 50             | 0.549   |                        |
| 100            | 0.259   | DDPM baseline          |

Task score is the geometric mean of the three env scores. **Submitting at lower
NFE is strictly preferred when quality is comparable.**

## Baselines

### ddpm
DDPM sampling with 100 steps - Standard but slow

### ddim
DDIM sampling with 20 steps - Faster deterministic sampling

### dpm_solver
DPM-Solver++ with 10 steps - Fastest high-quality sampling
