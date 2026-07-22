Let me reason from the training dynamics of modern CNNs toward a specific activation design, rather than starting from a known formula. I'll focus on the three regimes of activation behavior and their interactions with residual connections.

### Step 1: Regime Analysis
Consider the three regimes of an activation f(x):
1. **Positive regime**: x > 0
2. **Negative regime**: x < 0
3. **Transition regime**: x ≈ 0

For a residual connection y = F(x) + x, the effective residual is r = F(x) - x. In the positive regime, if f(x) grows faster than x, the residual can dominate the output, leading to exploding activations. If f(x) grows slower, the residual is small and training is slow. In the negative regime, the residual must be large enough to overcome the bias of the negative input.

### Step 2: Interaction with Residual Connections
Residual connections are designed to ease gradient flow and allow deeper networks. But they also impose constraints on the activation's slope:
- A large positive slope amplifies the residual, potentially causing explosion.
- A small positive slope weakens the residual, slowing training.
- In the negative regime, the residual needs to be strong enough to lift the activation out of zero.

### Step 3: Smooth vs. Non-Smooth Transitions
Smooth activations (GELU, Swish) have continuous first derivatives, which helps gradient propagation. However, a very smooth transition can blur the boundary between positive and negative regimes, making the model less decisive. A non-smooth transition can create a sharp decision boundary but risks introducing discontinuities that hurt optimization.

### Step 4: Channel-Wise Adaptation
Modern attention mechanisms show that per-channel adaptivity can help. An activation that adapts its shape per channel could capture diverse feature behaviors better than a fixed global shape.

### Step 5: Bounded vs. Unbounded Growth
An unbounded activation (ReLU, GELU) can grow without limit, risking saturation and numerical issues. A bounded activation caps the maximum value, but must still support sufficient range for expressive power.

### Step 6: Negative-Domain Behavior
Zeroing negative inputs (ReLU) is simple but kills gradient and information. Linearizing the negative domain allows some signal to pass through, but the slope must be chosen carefully — too steep and it dominates the positive regime; too shallow and it provides little signal.

### Core Idea:
Design a channel-wise adaptive activation that combines a bounded positive regime with a learnable negative-domain slope and a smooth, non-monotonic transition near zero. Specifically: f(x) = min(max(x, a * x), b) where a and b are per-channel learnable parameters, and the transition near zero is smoothed via a softplus-like mechanism.

### Non-trivial crux:
The key is balancing the negative-domain slope a and the upper bound b so that the negative regime contributes useful signal while the positive regime remains stable under residual connections — this requires careful initialization and possibly a constraint on a to prevent dominance of the negative slope over the positive regime.