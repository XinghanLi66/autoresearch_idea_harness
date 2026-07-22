# Gradient Compression for Communication-Efficient Distributed Training

## Research Question
Design a gradient compression operator that reduces communication cost in distributed training while maintaining convergence quality (test accuracy).

## Background
In distributed data-parallel training, gradient communication is often the bottleneck. Workers compute local gradients, which must be aggregated (e.g., via all-reduce) before the optimizer step. Gradient compression reduces the volume of data communicated by applying lossy compression to gradients before transmission.

Three main families of compression exist:
- **Sparsification**: Keep only a subset of gradient elements (e.g., TopK selects the largest magnitudes)
- **Quantization**: Reduce the precision of gradient values (e.g., QSGD uses stochastic rounding to discrete levels)
- **Low-rank approximation**: Approximate gradient matrices with low-rank factors (e.g., PowerSGD)

A key challenge is that naive compression introduces bias or variance that degrades convergence. Error feedback (accumulating compression residuals for the next iteration) is a widely-used technique to correct this.

## Task
Modify the `Compressor` class in `custom_compressor.py`. Your compressor must implement:
- `__init__(self, compress_ratio)`: Initialize with a target compression ratio (0.01 = 100x compression)
- `compress(self, tensor, name)`: Compress a gradient tensor, returning `(compressed_tensors, ctx)`
- `decompress(self, compressed_tensors, ctx)`: Reconstruct the gradient

The compressor may maintain internal state (e.g., error feedback residuals) across calls. The `name` parameter identifies parameters for per-parameter state tracking.

## Interface
```python
class Compressor:
    def __init__(self, compress_ratio=0.01): ...
    def compress(self, tensor, name) -> (list[Tensor], ctx): ...
    def decompress(self, compressed_tensors, ctx) -> Tensor: ...
```
- `compress_ratio`: Fraction of gradient elements/information to retain (0.01 = keep 1%)
- `compressed_tensors`: List of tensors that would be communicated over the network
- `ctx`: Local context (not communicated) needed for decompression
- The decompressed tensor must have the same shape as the original input

## Evaluation
Trained and evaluated on three settings with 100x compression (compress_ratio=0.01):
- **ResNet-20 / CIFAR-10** (0.27M params): Small model, standard benchmark
- **VGG-11-BN / CIFAR-100** (9.8M params): Larger model, harder 100-class problem
- **ResNet-56 / CIFAR-10** (0.85M params): Deeper model, tests scalability

Metric: **best test accuracy** (higher is better). All settings use SGD with momentum, cosine LR schedule, and 200 training epochs.

