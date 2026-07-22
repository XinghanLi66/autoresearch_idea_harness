# Temporal JEPA Prediction Loss Optimization

## Research Question
Design a better prediction cost function for multi-step temporal Joint Embedding Predictive Architecture (JEPA). The prediction loss measures discrepancy between predicted and target representations in the latent space, directly influencing how well the predictor learns to model temporal dynamics.

## What You Can Modify
The `CustomPredictionLoss` class in `custom_prediction_loss.py`. You may modify the `__init__` and `forward` methods, add helper methods, and import additional modules.

## Interface
```python
class CustomPredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        """
        Args:
            state: [B, C, T, H, W] - target encoded representations from the encoder
            predicted: [B, C, T, H, W] - predicted representations from the predictor

        Returns:
            Scalar loss tensor (lower means predicted is closer to state)
        """
```

The loss is called during JEPA's `unroll()` method as `predcost(state, predicted_states)`, where both tensors share the same shape. The returned scalar is added to the regularization loss and backpropagated.

## Evaluation
Mean detection Average Precision (AP) across prediction timesteps on Moving MNIST. Higher is better. The model is trained for 50 epochs with the Adam optimizer (lr=1e-3), and the final mean detection AP is reported.

The prediction loss is evaluated across three model sizes to test generalization:
- **small**: henc=16, dstc=8, hpre=16
- **base**: henc=32, dstc=16, hpre=32
- **large**: henc=64, dstc=32, hpre=64

## Background
- The current baseline uses simple MSE loss (`F.mse_loss(state, predicted)`), treating all representation dimensions equally
- The JEPA architecture includes a VCLoss regularizer (Variance-Covariance) that encourages representation diversity
- The encoder produces spatial feature maps (not just vectors), so spatial structure matters
- The predictor operates autoregressively over time steps

