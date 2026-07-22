# LLM Pretraining: Mixed Precision & Quantization Strategy

## Research Question
Design an improved precision strategy for linear layers in GPT-2 pretraining. Your modifications should improve training throughput or model quality by using lower-precision arithmetic (FP8, dynamic quantization) while maintaining numerical stability.

## What You Can Modify
The `CustomLinear` class (lines 34-52) in `custom_pretrain.py`:
- Forward pass precision (default: standard FP32/BF16 F.linear)
- Weight/activation quantization (e.g., FP8 E4M3 for forward, E5M2 for backward)
- Scale factor computation (static, dynamic, or EMA-based)
- Custom autograd Functions for precision-aware backward pass
- You may add helper classes (e.g., autograd Functions) alongside `CustomLinear`

**Note**: The `CustomLinear` class must maintain the interface:
- `__init__(self, in_features, out_features, bias=True)` with `self.weight` parameter
- `forward(self, x) -> output` where x has shape `(..., in_features)` and output has shape `(..., out_features)`
- All model linear layers (attention, MLP, lm_head) use `CustomLinear`

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better)
- **Model sizes**: GPT-2 124M (12L/12H/768D) and GPT-2 1.5B (48L/25H/1600D, 4-GPU DDP)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer)
- **Training**: 5000 iterations, batch_size=12, block_size=1024, grad_accum=5
- **Hardware**: H100 GPU with FP8 tensor core support

