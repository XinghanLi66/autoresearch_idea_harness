# Graph Message Passing for Spatio-Temporal Traffic Forecasting

> **⚠️ DEPRECATED (2026-04-22)**: Baselines cannot currently be reproduced on this
> infrastructure. BasicTS loads METR-LA/PEMS datasets fully into GPU memory (~15-20 GB
> per run), which exceeds the free GPU memory available on the gpublaze cluster when
> shared with other workloads. Task is kept for historical reference; do not run
> `mlsbench agent graph-temporal` or `mlsbench baseline graph-temporal` until this
> constraint is addressed (e.g., by reducing batch/lookback in `edits/run_template.py`
> or running on a dedicated GPU).

## Research Question
Design a novel graph message passing mechanism for spatial aggregation in spatio-temporal traffic forecasting networks.

## Background
Traffic forecasting on sensor networks requires modeling both temporal dynamics and spatial dependencies between sensors. While temporal modeling (via convolutions or RNNs) is relatively well-understood, the spatial component — how information is passed between graph nodes — remains an active area of research.

Classical approaches include:
- **Spectral methods**: Chebyshev polynomial approximation of graph convolutions (STGCN)
- **Diffusion methods**: Random walk-based diffusion on directed graphs (DCRNN, Graph WaveNet)
- **Attention methods**: Spatial attention mechanisms (ASTGCN, STAEformer)
- **Adaptive methods**: Learned graph structures combined with multi-hop propagation (MTGNN)

The task is to design a spatial aggregation layer for complex, distance-dependent, and potentially asymmetric relationships between traffic sensors.

## Task
Modify the `SpatialLayer` class in `custom_graph_model.py`. This class defines the graph message passing component used within each spatio-temporal block. The temporal backbone (gated dilated causal convolutions) and training pipeline are fixed.

Your `SpatialLayer` receives:
- `x`: Node features `[B, N, D]` — B=batch, N=nodes (sensors), D=features
- `adj`: Normalized adjacency matrix `[N, N]` — symmetric-normalized, weighted by sensor distance

And must return spatially aggregated node features `[B, N, D']`.

## Interface
```python
class SpatialLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        ...
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], adj: [N, N] -> output: [B, N, D']
        ...
```

The class is instantiated with `SpatialLayer(hidden_dim, hidden_dim, dropout)` where `hidden_dim=32` by default. You may add parameters, intermediate layers, or learnable components as needed.

## Evaluation
Trained and evaluated on three traffic datasets:
- **METR-LA** (207 sensors, traffic speed, Los Angeles highway network)
- **PEMS-BAY** (325 sensors, traffic speed, San Francisco Bay Area)
- **PEMS04** (307 sensors, traffic flow, California district 4)

All use `input_len=12`, `output_len=12` (5-minute intervals, 1 hour history -> 1 hour prediction).
Metrics: MAE, RMSE, MAPE (lower is better). Data is Z-score normalized; metrics computed after inverse transform. For MAPE, METR-LA / PEMS-BAY mask entries with value `== 0` (standard speed-sensor missing-value sentinel); PEMS04 (flow counts) masks entries below `1e-5` to avoid near-zero denominators inflating MAPE.

**Important — fair comparison note.** All baselines in this task share a single fixed backbone (gated dilated causal temporal convolutions + mean pooling over time + per-node fully-connected readout), and only the `SpatialLayer` is swapped out for each method (DCRNN diffusion, STGCN Chebyshev, GWNet adaptive, MTGNN mix-hop, ASTGCN/STAEformer spatial attention). This isolates the research question (spatial aggregation) but means the numbers here are **not directly comparable to those reported in the original method papers**, which use method-specific temporal blocks, readout heads, and training recipes. Compare within this task's leaderboard, not against paper-reported numbers.
