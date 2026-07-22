# LLM Pretraining: Native Low-Bit Linear (BitLinear)

## Research Question
Design a low-bit linear layer for GPT-2 pretraining that uses native low-precision weights (binary/ternary) during both training and inference, instead of standard float weights. The goal is to minimize validation loss while constraining weights to a small discrete set.

## Background
Standard neural networks store and compute with full-precision (FP32/BF16) weights. Post-training quantization (PTQ) and quantization-aware training (QAT) attempt to compress these weights after or during training, but the model fundamentally trains with float weights. Native low-bit training takes a different approach: weights are inherently discrete (e.g., {-1, +1} or {-1, 0, +1}) during every forward pass, with float latent weights maintained only for gradient accumulation.

This paradigm was introduced by BitNet (Wang et al., 2023), which binarized weights to {-1, +1} using the sign function, and extended by BitNet b1.58 (Ma et al., 2024), which used ternary {-1, 0, +1} weights via absmean quantization. The key insight is that these models can match or approach full-precision performance at a fraction of the effective parameter cost.

**Key differences from QAT** (the `llm-pretrain-quantization` task):
- QAT: float weights -> fake quantize during training -> real quantize at deployment (weights are float during training)
- BitLinear: float latent weights -> discrete quantize in every forward pass (weights are always discrete during computation)

**Key differences from mixed-precision** (the `llm-pretrain-precision` task):
- Mixed precision: changes the float format (FP32 -> BF16/FP8) but values are still continuous
- BitLinear: weights are restricted to a small discrete set (1-2 bits)

## What You Can Modify
The BitLinear module (lines 38-115) in `custom_pretrain.py`:
- `weight_quant(weight)` -- quantizes float latent weights to discrete values, returns (quantized_weight, scale)
- `activation_quant(x)` -- optional activation quantization, returns (quantized_x, scale)
- `BitLinear` class -- linear layer that uses the above functions

**Notes on the interface**:
- `BitLinear.__init__(self, in_features, out_features, bias=True)` must keep `self.weight` as a Parameter
- `BitLinear.forward(self, x) -> output` where x has shape `(..., in_features)` and output has shape `(..., out_features)`
- The quantization is applied in every forward pass (both training and eval) -- there is no separate train/eval path
- `weight_quant` should return `(quantized_weight, scale)` where `quantized_weight * scale` approximates the original weight
- `activation_quant` should return `(quantized_x, scale)` similarly
- All model linear layers (attention, MLP, lm_head) use `BitLinear`
- You may add helper classes (autograd Functions, learned parameters) alongside the existing functions
- Must be compatible with `torch.compile` (no `@torch.compiler.disable`)

## Evaluation
- **Primary metric**: `val_loss` -- validation loss (cross-entropy, lower is better)
- **Additional metrics**: Perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 13535 iterations, BSZ=64, GA=8, 2-GPU DDP
- **Hardware**: H200 GPU
