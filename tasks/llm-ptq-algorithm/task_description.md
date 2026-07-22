# LLM Post-Training Quantization (PTQ) Algorithm

## Research Question
Design a post-training quantization algorithm that minimizes accuracy degradation when quantizing a pretrained Mistral-7B-v0.1 model (7.24B parameters) to low-bit integer precision, without any retraining or fine-tuning.

## Background
Post-training quantization (PTQ) compresses neural network weights from floating-point to low-bit integer representations after training is complete. Unlike quantization-aware training (QAT), which modifies the training procedure, PTQ works on already-trained models and requires no gradient updates to the original weights. This makes PTQ especially attractive for large language models where retraining is prohibitively expensive.

The challenge is severe at low bit-widths: INT4 has only 16 discrete levels (vs 256 for INT8), and INT3 has only 8 levels, so naive rounding causes significant accuracy loss. This effect is amplified in larger models like Mistral-7B (7.24B params), where weight distributions are complex and quantization errors accumulate across 32 transformer layers. State-of-the-art methods use various strategies to minimize this degradation:

- **RTN (Round-To-Nearest)**: Simply round each weight to its nearest quantized value. Fast but high degradation.
- **SmoothQuant** (Xiao et al., 2023): Migrate quantization difficulty from activations to weights by applying per-channel scaling, making weight distributions more uniform before quantization.
- **GPTQ** (Frantar et al., 2023): Use calibration data to compute a Hessian approximation, then quantize column-by-column while optimally compensating for error.
- **AWQ** (Lin et al., MLSys 2024): Identify salient weight channels via activation magnitudes and protect them with per-channel scaling during quantization, without requiring Hessian computation.

The quantization uses symmetric group quantization: weights are partitioned into groups of consecutive columns (group size 64 or 128), and one scale factor is computed per group per output row.

## What You Can Modify
The `LayerQuantizer` class and helper functions in `custom_ptq.py` (editable region):
- `quantize_tensor()` / `dequantize_tensor()`: Basic quantization primitives
- `find_scale_zero()`: Scale/zero-point computation (per-channel or per-group)
- `LayerQuantizer.__init__()`: Set hyperparameters; receives `num_bits` and `group_size` from the evaluation script
- `LayerQuantizer.add_batch(inp)`: Collect statistics from calibration data (128 sequences)
- `LayerQuantizer.quantize()`: Apply quantization to the layer's weight matrix

You can implement any approach:
- **Error compensation**: Redistribute quantization error across weights using second-order information
- **Weight transformation**: Transform weight distributions before quantization (scaling, rotation, smoothing)
- **Mixed strategies**: Combine multiple techniques (e.g., smoothing + Hessian-based error compensation)
- **Outlier handling**: Special treatment for weight outliers that dominate quantization error
- **Adaptive grouping**: Different strategies for different group sizes or bit-widths

## Architecture
The task loads real Mistral-7B-v0.1 weights (downloaded from HuggingFace) and quantizes them using your algorithm. No training is done -- the task is purely about the quantization algorithm quality.

Mistral-7B-v0.1 specs: 32 layers, 32 attention heads, 8 KV heads (GQA), 4096 hidden dimension, 14336 intermediate size, ~7.24B parameters.

The script (`custom_ptq.py`):
1. Loads Mistral-7B-v0.1 weights from `/data/mistral-7b-v01` (pre-downloaded HuggingFace snapshot)
2. Evaluates the FP16 (unquantized) model as baseline
3. Runs your `LayerQuantizer.add_batch()` on calibration data layer by layer
4. Quantizes each linear layer using your `LayerQuantizer.quantize()`
5. Evaluates the quantized model and reports perplexity degradation

## Interface
```python
class LayerQuantizer:
    def __init__(self, layer, num_bits=4, group_size=-1):
        # layer: nn.Linear to quantize
        # num_bits: target bit width (4 or 3, set by evaluation)
        # group_size: columns per group (-1 = per-channel, 128 or 64)
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        # ... initialize calibration buffers

    def add_batch(self, inp):
        # inp: layer input tensor, shape (batch*seq_len, in_features)
        # Collect activation stats, Hessians, etc.
        pass

    def quantize(self):
        # Returns: quantized-dequantized weight tensor (same shape as original)
        # Must respect self.num_bits and self.group_size
        return W_dq

    def free(self):
        # Release calibration buffers
        pass
```

**Constraints**:
- You must NOT retrain or fine-tune the model (no gradient updates to original weights)
- All linear layers in each transformer block are quantized (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Embeddings, LayerNorm, and LM head are NOT quantized
- The returned weight must have the same shape and dtype as the original
- `copy`, `math`, `torch`, `torch.nn`, `F`, `np`, `os`, `time` are available
- Your algorithm must work for both INT4 and INT3, and for different group sizes

## Evaluation
The algorithm is evaluated across multiple quantization settings to test generalizability:

- **ptq-7b-int4**: INT4 (4-bit) quantization with group size 128 -- standard PTQ setting
- **ptq-7b-int3**: INT3 (3-bit) quantization with group size 128 -- harder setting with only 8 levels
- **ptq-7b-int4-g64**: INT4 with group size 64 -- finer granularity setting

**Primary metric**: `wikitext2_ppl` -- WikiText-2 perplexity after quantization (lower is better)
**Secondary metric**: `degradation` -- perplexity increase over FP16 baseline (lower is better)
**Model**: Mistral-7B-v0.1 (32 layers, GQA, ~7.24B params)
**Weights**: Real Mistral-7B-v0.1 from HuggingFace (no pretraining needed)
**Calibration**: 128 sequences from WikiText-2 training set, 2048 tokens each
