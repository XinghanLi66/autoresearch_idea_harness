# LLM Pretraining: Quantization-Aware Training (QAT)

## Research Question
Design a quantization-aware training (QAT) scheme for GPT-2 pretraining that minimizes validation loss degradation when model weights are quantized to INT4 (4-bit integer) precision after training.

## Background
Post-training quantization (PTQ) — converting trained float weights to int4 — offers aggressive model compression but causes significant accuracy degradation because the model never "saw" quantization noise during training. With only 16 discrete levels (vs 256 for int8), INT4 quantization is far more destructive, making QAT essential. Quantization-Aware Training (QAT) inserts simulated (fake) quantization into the forward pass during training, so the model learns weight distributions that are robust to quantization. The key challenge is designing the fake-quantization operator to be compatible with gradient-based optimization while faithfully simulating the true INT4 quantization effect.

Per-channel quantization (one scale factor per output channel of each weight matrix) is used for evaluation, as it significantly reduces quantization error compared to per-tensor quantization at INT4 precision.

This differs from mixed-precision training (FP8/BF16 casting): mixed precision changes the compute format, while QAT specifically trains for robustness to integer quantization at deployment.

## What You Can Modify
The quantization module (lines 34-116) in `custom_pretrain.py`:
- `fake_quantize_weight(weight, num_bits)` — simulated quantization applied to weights during training
- `fake_quantize_activation(x, num_bits)` — optional simulated quantization for activations
- `quantize_dequantize_weight(weight, num_bits)` — real quantize-then-dequantize roundtrip for evaluation
- `QATLinear` class — linear layer that applies the above functions

**Notes on the interface**:
- `QATLinear.__init__(self, in_features, out_features, bias=True)` must keep `self.weight` as a Parameter
- `QATLinear.forward(self, x) -> output` where x has shape `(..., in_features)` and output has shape `(..., out_features)`
- During training (`self.training=True`): apply fake quantization so gradients flow through
- During eval (`self.training=False`): apply real quantize-dequantize roundtrip to measure post-quantization quality
- All model linear layers (attention, MLP, lm_head) use `QATLinear`
- You may add helper classes (autograd Functions, learned parameters) alongside the existing functions

## Evaluation
- **Primary metric**: `quant_val_loss` -- validation loss after quantize-dequantize roundtrip (lower is better)
- **Secondary metric**: `quant_degradation` -- difference between quantized and float val_loss (lower is better)
- **Additional metrics**: Perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 13535 iterations, BSZ=64, GA=8, 2-GPU DDP
- **Hardware**: H200 GPU

