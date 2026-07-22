# Flow Matching Training Objective

## Background

Flow matching trains a neural network to predict a velocity field that transports
samples from noise to data. Given a clean image x_0 and noise x_1 ~ N(0,I),
the noisy sample at time t is:

    x_t = (1 - t) * x_0 + t * x_1

The instantaneous velocity is v = x_1 - x_0 (independent of t).

**MeanFlow** extends this by training the network to predict the *mean velocity*
u(x_t, t, t_next) — the average velocity over the interval [t_next, t]. This
enables high-quality generation in very few steps (1-2 NFE).

The mean velocity satisfies:
    u(x_t, t, t_next) = v(x_t, t) - (t - t_next) * dv/dt

which can be computed via a Jacobian-vector product (JVP).

## Research Question

Can we improve upon MeanFlow's training objective to achieve better FID?

## Task

You are given `custom_train.py`, a self-contained training script that trains a
small DiT on CIFAR-10 (32x32) and evaluates FID.

The editable region contains two functions:

1. `sample_traj_params(batch_size, cur_step, max_steps, device)` — controls the
   training objective via `ratio_fm` (fraction of flow matching samples) and
   `alpha` (discrete training weight).

2. `compute_mean_velocity_target(net, x_t, t, t_next, dt, velocity, device)` —
   computes the mean velocity training target.

Your goal is to implement a training objective that achieves **lower FID** than
the MeanFlow baseline.

## Evaluation

- Dataset: CIFAR-10 (32x32)
- Model: SmallDiT (512 hidden, 8 layers, ~40M params)
- Training: 10000 steps, batch size 128
- Metric: FID (lower is better), computed with clean-fid against CIFAR-10 train set
- Inference: 10-step Euler sampler
