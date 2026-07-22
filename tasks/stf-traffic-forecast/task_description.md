# Spatial-Temporal Traffic Forecasting: Custom Model Design

## Objective
Design and implement a custom deep learning model for spatial-temporal traffic forecasting. Your code goes in `custom_model.py` (both the `Custom` model class and `CustomConfig` config class). Three reference implementations (STID, DLinear, StemGNN) are provided as read-only.

## Background
Spatial-temporal forecasting predicts future values across a network of spatial nodes (e.g., traffic sensors), leveraging both temporal patterns and spatial correlations between nodes. Unlike standard time series forecasting, STF models must capture inter-node dependencies (e.g., traffic at nearby sensors is correlated). Key design choices include:
- **Spatial modeling**: learnable node embeddings, graph convolutions, spatial attention
- **Temporal modeling**: RNNs, temporal convolutions, Transformers
- **Spatial-temporal fusion**: how to combine spatial and temporal information

## Model Interface
```python
def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
    """
    inputs: [batch_size, input_len, num_features]
        - input_len=12 (1 hour of 5-minute intervals)
        - num_features = number of spatial nodes (sensors)
    inputs_timestamps: [batch_size, input_len, 2]
        - channel 0: normalized time-of-day (0 to 1)
        - channel 1: normalized day-of-week (0 to 1)
    Returns: [batch_size, output_len, num_features]
        - output_len=12 (predict next 1 hour)
    """
```

## Evaluation
Trained and evaluated on three traffic datasets:
- **METR-LA** (207 sensors, traffic speed, Los Angeles highway)
- **PEMS-BAY** (325 sensors, traffic speed, San Francisco Bay Area)
- **PEMS04** (307 sensors, traffic flow, California district 4)

All use `input_len=12`, `output_len=12`. Metrics: MAE, RMSE, MAPE (lower is better). Data is Z-score normalized; metrics are computed after inverse transform. Missing values (0.0) are masked during loss computation.

## Available Modules
You may import and use components from `basicts.modules`:
- `basicts.modules.mlps`: MLP layers (MLPLayer, ResMLPLayer)
- `basicts.modules.norm`: Normalization (RevIN, LayerNorm)
- `basicts.modules.embed`: Sequence embeddings
- `basicts.modules.transformer`: Transformer components (Encoder, MultiHeadAttention)
- `basicts.modules.activations`: Activation functions

## Optimizer Hyperparameter Override
The training harness uses Adam with `lr=2e-3`, `weight_decay=1e-4`, and a `MultiStepLR(milestones=[1, 50, 80], gamma=0.5)` schedule for 100 epochs (batch_size=64). If your method needs a different learning rate or weight decay, edit the `CONFIG_OVERRIDES` dict at the bottom of `custom_model.py`:

```python
# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 5e-4, 'weight_decay': 1e-3}
```

Only `lr` and `weight_decay` are forwarded to the optimizer; all other training settings (epochs, batch size, scheduler, gradient clipping) are fixed by the harness for fair comparison.
