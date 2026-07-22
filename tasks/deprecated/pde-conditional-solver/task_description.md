# Conditional Dynamic PDE Solving: Custom Neural Operator Design

## Objective
Design and implement a custom neural operator for time-conditional dynamic PDE solving. Unlike autoregressive models, this model receives a **timestep embedding** `T` and directly predicts the solution at that timestep. Your code goes in the `Model` class in `models/Custom.py`.

## Model Interface
Your model receives `args` at initialization and must implement:
```python
forward(self, x, fx, T=None, geo=None) -> output
```
- `x`: spatial coordinates, shape `(B, N, space_dim)` where `space_dim=2`
- `fx`: input function values (initial conditions), shape `(B, N, fun_dim)` where `fun_dim` varies by dataset (1 for Plasticity/SWE, 4 for CFD)
- `T`: **timestep conditioning**, shape `(B, 1)` — a scalar timestep for the prediction target
- `geo`: geometry info (unused, always `None`)
- output: predicted solution at timestep T, shape `(B, N, out_dim)` where `out_dim` varies by dataset (4 for Plasticity/CFD, 1 for SWE)

The model is called for each of `T_out=20` timesteps. Unlike autoregressive tasks, the model always receives the same initial conditions `fx` and a different `T` for each call.

Key `args` attributes: `n_hidden`, `n_layers`, `n_heads`, `space_dim`, `fun_dim`, `out_dim`, `act`, `mlp_ratio`, `dropout`, `geotype` (`structured_2D`), `shapelist` (`[101, 31]`), `unified_pos`, `ref`, `slice_num`, `modes`, `time_input` (True).

## Evaluation
Trained and evaluated on three conditional PDE benchmarks (relative L2 error over all timesteps, lower is better):
- **Plasticity** (2D elasto-plastic deformation, structured_2D, 101x31, 20 output timesteps, fun_dim=1, out_dim=4)
- **SWE** (2D shallow water equations, structured_2D, 128x128 downsampled to 64x64, 20 output timesteps, fun_dim=1, out_dim=1)
- **CFD** (2D compressible Navier-Stokes, structured_2D, 128x128 downsampled to 64x64, Mach 0.1, 20 output timesteps, fun_dim=4, out_dim=4)

Training epochs: Plasticity (50), SWE (30), CFD (30), all with OneCycleLR scheduler.
