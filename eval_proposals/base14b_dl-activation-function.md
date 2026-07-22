Core idea: Introduce a learnable "negative slope modulator" in a self-gated activation function, allowing the model to dynamically adjust the negative-domain behavior based on layer-specific statistics, combining the benefits of smoothness (Swish) with adaptive sparsity (Leaky ReLU).

Non-trivial crux: Designing a parameterization that avoids gradient instability while enabling the activation to learn optimal negative-domain responses without overfitting to specific layer characteristics.

**Mechanism**:  
The activation function is defined as $ f(x) = x \cdot \sigma(\gamma x) + \beta \cdot x \cdot \sigma(-\gamma x) $, where $ \sigma $ is the sigmoid, $ \gamma $ is a learnable scaling parameter, and $ \beta $ is a learnable negative slope coefficient. This creates a smooth, self-gated function that:  
1. Maintains Swish-like behavior for positive inputs ($ \gamma x $ scales the sigmoid's sensitivity).  
2. Introduces a learnable negative response ($ \beta $) that adapts during training, avoiding "dying" while preserving gradient flow.  
3. Uses a shared $ \gamma $ parameter to ensure consistency across layers, preventing excessive parameter proliferation.  

The negative slope $ \beta $ is initialized to a small value (e.g., 0.1) to encourage sparsity but remains trainable, allowing the network to optimize it per-layer during training. This design balances adaptability with stability, leveraging BN's normalization to maintain reasonable input ranges for the sigmoid gates. The parameterization is differentiable and compatible with residual connections, as the function remains bounded and smooth.