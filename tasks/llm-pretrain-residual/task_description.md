# LLM Pretraining: Residual Connection Strategy

## Research Question
Improve the residual connection strategy in a GPT-style language model. The current architecture uses standard Pre-LN residual connections (`x + sublayer(x)`) in each transformer block. Your task is to redesign how information flows through the residual stream across layers to achieve lower validation loss.

## Background

### Standard Residual Connections
The default GPT architecture uses simple additive residual connections in each block:
```python
x = x + self.attn(self.ln_1(x))   # attention sublayer
x = x + self.mlp(self.ln_2(x))    # MLP sublayer
```
While effective, this fixed accumulation pattern may not be optimal for deep networks. The residual stream is the primary information highway through the model, and its design critically affects gradient flow, feature reuse, and training dynamics.

### Research Directions
Several recent works have proposed improvements to residual connections:

1. **Per-layer residual scaling**: Learnable scalars that modulate the residual stream at each layer (inspired by modded-nanogpt, ReZero, SkipInit).
2. **Initial embedding blending**: Blending the initial token embedding back at each layer to preserve token identity (x0 residual connections).
3. **Hyper-Connections**: Maintaining m parallel residual streams with learned transition matrices for richer information flow across layers (Zhu et al., 2025).
4. **Attention Residuals**: Using softmax attention over all previous layer outputs to dynamically select which representations to combine (Kimi Team, 2026).

## What You Can Modify

### Block Class (lines 88-99)
The `Block` class defines per-block residual behavior. You can change how attention and MLP outputs are combined with the residual stream within each block.

### Residual Stream Parameters (lines 128-130)
Add custom parameters to `GPT.__init__` for your residual connection strategy (e.g., per-layer scalars, transition matrices, query vectors).

### Block Loop in GPT.forward (lines 162-164)
The main loop that iterates through transformer blocks. You can modify how blocks are called and how their outputs are accumulated (e.g., multi-stream processing, attention over layer outputs).

### Optimizer Configuration (lines 175-192)
The `configure_optimizers` method. If you add new parameters, you may want to assign them to appropriate optimizer groups with custom learning rates and weight decay.

### Training Hyperparameters (line 251)
The `CONFIG_OVERRIDES` dictionary for adjusting learning rate, weight decay, etc.

**Note**: The `CausalSelfAttention`, `MLP`, `LayerNorm`, and `GPTConfig` classes are fixed. The `Block.forward` signature must accept `x` and return a tensor of the same shape. The `GPT.forward` must accept `(idx, targets=None)` and return `(logits, loss)`.

## Evaluation
- **Primary metric**: Validation loss (`val_loss`, lower is better)
- **Secondary metrics**: Perplexity on WikiText-2 and LAMBADA, plus downstream task accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 13535 iterations, BSZ=32, GA=16, 2-GPU DDP
- **Hardware**: H200 GPU
