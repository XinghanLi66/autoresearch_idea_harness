# Time Series Imputation via Dual Diffusion on Trend and Seasonal Components

## Core Idea
Reframe missing-value imputation as a denoising diffusion process, but adapt it to time-series structure: (1) replace gaussian noise with a binary mask (zero out missing entries) as the forward process, and (2) run two parallel diffusions on the decomposed trend and seasonal components rather than one flat diffusion, reconstructing masked values with the observed values as prior.

## Design

**Forward Process (Noise Model)**:
- Corrupt the input with a binary mask: set missing values to zero, leave observed values untouched. This directly models the missingness mechanism instead of forcing gaussian noise onto a temporally-correlated signal.

**Signal Decomposition**:
- Decompose the time series into trend (slow-moving global structure) and seasonal (repeating periodic patterns) components before diffusion. Each component gets its own reverse process, because their noise dynamics differ.

**Dual Diffusion**:
- **Trend diffusion**: Standard reverse step (denoising network conditioned on current noisy sample + observed values).
- **Seasonal diffusion**: Reverse step aware of the seasonal period — either a learned module or a fixed seasonal prior (simpler, less prone to overfitting but risks the seasonal prior dominating the trend). The choice between fixed and learned is to be settled on the masked metric.

**Reconstruction**:
- Compose the cleaned trend and seasonal components back, then impute only the originally-missing positions using the observed values as prior. The observed values are never touched — they act as the clean anchor.

**Training Objective**:
- Minimize MSE/MAE on the masked regions only, keeping the same loss and optimizer as the reference models for a fair comparison.

## Non-Trivial Crux
The seasonal-diffusion branch must be justified by the masked-region metric: if a single trend-only diffusion is competitive, prefer it for parsimony; only keep two components if the seasonal-aware reverse step clearly improves error. The fixed-seasonal-prior route is tempting for simplicity but risks the seasonal signal overpowering the trend — this dominance should be checked, not assumed.