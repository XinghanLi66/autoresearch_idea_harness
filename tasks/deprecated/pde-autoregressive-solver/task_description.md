# Autoregressive PDE Solving: Custom Neural Operator Design

## Objective
Design and implement a custom neural operator for autoregressive time-dependent PDE solving. Your code goes in the `Model` class in `models/Custom.py`. Three reference implementations (FNO, Transformer, Transolver) are provided as read-only context.

## Model Interface
Your model receives `args` at initialization and must implement:
```python
forward(self, x, fx, T=None, geo=None) -> output
```
- `x`: spatial coordinates, shape `(B, N, space_dim)`
- `fx`: input function values (previous timesteps), shape `(B, N, fun_dim)` where `fun_dim=10`
- `T`: time embedding (unused for autoregressive, always `None`)
- `geo`: geometry info (unused, always `None`)
- output: predicted next timestep, shape `(B, N, out_dim)` where `out_dim=1`

The model is called autoregressively: at each step, the oldest input channel is dropped, and the new prediction is appended as the newest channel in `fx`.

Key `args` attributes: `n_hidden`, `n_layers`, `n_heads`, `space_dim`, `fun_dim`, `out_dim`, `act`, `mlp_ratio`, `dropout`, `geotype` (`structured_2D` or `structured_1D`), `shapelist`, `unified_pos`, `ref`, `slice_num`, `modes`.

## Evaluation
Trained and evaluated on four autoregressive PDE benchmarks (relative L2 error, lower is better):
- **NS** (2D Navier-Stokes vorticity, structured_2D, 64x64, viscosity=1e-5, 10-step rollout)
- **DiffSorp** (1D diffusion-sorption, structured_1D, PDEBench, 10-step rollout)
- **Advection** (1D advection equation, structured_1D, PDEBench, beta=1.0, 10-step rollout)
- **Burgers** (1D Burgers equation, structured_1D, PDEBench, viscosity=1e-3, 10-step rollout)

Your model must handle both `structured_1D` and `structured_2D` geometry types via `args.geotype`.
