# LLM Pretraining: Residual Stream Design

## Research Question
Improve the residual stream design in a GPT-style language model. The current architecture uses three interacting mechanisms for residual connections: per-layer residual scaling (`resid_lambdas`), initial embedding blending (`x0_lambdas`), and value embedding gating. Your task is to redesign how information flows through the residual stream to achieve lower validation bits-per-byte (val_bpb).

## Background

### Residual Lambdas
Each transformer layer scales its residual stream by a learnable scalar `resid_lambdas[i]` (initialized to 1.0). This allows the model to learn layer-dependent residual scaling, potentially helping with gradient flow in deeper networks. Inspired by modded-nanogpt.

### x0 Lambdas (Initial Embedding Blending)
At each layer, a fraction of the initial normalized embedding `x0` is blended back into the residual stream via `x0_lambdas[i]` (initialized to 0.1). This creates a "skip connection to the input" that helps preserve token identity information throughout the network depth.

### Value Embeddings (ResFormer-style)
Alternating layers have learnable value embeddings that are mixed into the attention values via an input-dependent gate. The `has_ve()` function determines which layers get value embeddings. The gate uses the first 12 channels of the input to produce per-head gating scalars via sigmoid, scaled by 3x.

### How They Interact
In the forward pass, before each block:
```python
x = resid_lambdas[i] * x + x0_lambdas[i] * x0
```
And inside attention, value embeddings are gated and added to V:
```python
gate = 3 * sigmoid(ve_gate(x[:, :, :12]))
v = v + gate * ve
```

## What You Can Modify
The residual stream mechanism spans several locations in `gpt.py`:

1. **`has_ve()` function** (line 53-55): Controls which layers get value embeddings
2. **`CausalSelfAttention.__init__`** (lines 79-80): Value embedding gate initialization
3. **`CausalSelfAttention.forward`** (lines 91-95): Value embedding mixing in attention
4. **`GPT.__init__`** (lines 176-185): Residual parameter definitions (resid_lambdas, x0_lambdas, value_embeds)
5. **`GPT.init_weights`** (lines 227-238): Initialization of residual parameters and gates
6. **`GPT.estimate_flops`** (lines 314-318): FLOPs estimation for residual params
7. **`GPT.num_scaling_params`** (lines 340-356): Parameter counting for scaling laws
8. **`GPT.setup_optimizer`** (lines 362-382): Optimizer groups for residual parameters
9. **`GPT.forward`** (lines 413-417): The residual mixing computation in the forward pass

**Note**: You must keep the model interface intact. The `Block.forward(x, ve, cos_sin, window_size, kv_cache)` signature must be preserved. The `setup_optimizer` must return a valid optimizer with all model parameters included. The `estimate_flops` and `num_scaling_params` methods must remain correct.

## Evaluation
- **Metric**: Validation bits-per-byte (`val_bpb`, lower is better)
- **Model**: depth=4 (4 layers, ~256 dim), ~500 training steps
- **Dataset**: ClimbMix (tokenized, BPE)
- **Training**: nanochat base_train.py with default Muon+AdamW optimizer

