# Steady-State PDE Solving: Custom Neural Operator Design

## Objective
Design and implement a custom neural operator for steady-state PDE solving on structured 2D meshes. Your code goes in the `Model` class in `models/Custom.py`. Three reference implementations (FNO, Transformer, Transolver) are provided as read-only context.

## Model Interface
Your model receives `args` at initialization and must implement:
```python
forward(self, x, fx, T=None, geo=None) -> output
```
- `x`: spatial coordinates, shape `(B, N, space_dim)` where `space_dim=2`
- `fx`: input function values, shape `(B, N, fun_dim)` — can be `None` if `fun_dim=0`
- `T`: time embedding (unused for steady tasks, always `None`)
- `geo`: geometry info (unused, always `None`)
- output: predicted solution, shape `(B, N, out_dim)`

Key `args` attributes: `n_hidden`, `n_layers`, `n_heads`, `space_dim`, `fun_dim`, `out_dim`, `act`, `mlp_ratio`, `dropout`, `geotype` (always `structured_2D`), `shapelist` (grid dimensions `[H, W]`), `unified_pos`, `ref`, `slice_num`, `modes`.

## Evaluation
Trained and evaluated on three structured 2D PDE benchmarks (relative L2 error, lower is better):
- **Airfoil** (NACA airflow, fun_dim=2, out_dim=1, 221x51 grid)
- **Darcy** (Darcy flow, fun_dim=1, out_dim=1, 85x85 grid after downsampling)
- **Pipe** (pipe flow, fun_dim=2, out_dim=1, 129x129 grid)

Training epochs vary per dataset: Airfoil (100), Darcy (300), Pipe (50), all with OneCycleLR scheduler.
