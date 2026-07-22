# Masked Diffusion LM: Demasking Strategy

## Research Question

Design a better demasking (decoding) strategy for masked diffusion language
models. The strategy must generalize across **different decoding regimes**:

- **Block-based semi-autoregressive decoding** for downstream task accuracy
  (LLaDA on MATH/HumanEval, following the KLASS protocol)
- **Fully-parallel decoding** for open-ended text generation (Dream on
  prefix-conditioned C4 continuation, measured by perplexity / diversity)

## Background

Masked diffusion LMs (LLaDA, Dream) generate by starting from a fully masked
generation region and iteratively unmasking over `steps` denoising iterations.
A **demasking strategy** decides at each step:

1. **Schedule**: how many tokens to unmask
2. **Position selection**: which masked positions to unmask
3. **Token assignment**: what token id to place

Decoding can be **semi-autoregressive** (when `block_length < gen_length`,
process one block at a time) or **fully parallel** (`block_length ==
gen_length`, all positions decoded together).

## What You Can Modify

Edit the `DemaskDecoder` class in `LLaDA/custom_demask_eval.py`
(lines 59-151).

### Interface

```python
class DemaskDecoder:
    def __init__(self, mask_id, temperature=0.0,
                 conf_threshold=0.9, kl_threshold=0.01, history_length=2):
        ...

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        # Returns (x_output [1, prompt_len + gen_length], used_steps)
```

`get_num_transfer_tokens(mask, steps)` is available outside the editable
region — returns the uniform schedule (`mask.sum() // steps` per step).

### Constraints

- `gen_length % block_length == 0`. When equal, decoding is fully parallel.
- Process blocks sequentially (no early-decoding into later blocks).
- Always return `[1, prompt_len + gen_length]`.
- `used_steps` counts model forward passes (lower = more efficient).

## Evaluation

### Benchmarks

| Label | Task | Model | gen_len | steps | block_len | Metrics |
|-------|------|-------|---------|-------|-----------|---------|
| `llada-math` | MATH-500 | LLaDA-8B-Instruct | 256 | 256 | 64 | accuracy + avg_steps |
| `llada-humaneval` | HumanEval (164) | LLaDA-8B-Instruct | 256 | 256 | 64 | accuracy + avg_steps |
| `dream-text` | C4 prefix-continuation (256 samples, 32-tok prefix → 224-tok continuation) | Dream-v0-Instruct-7B | 224 | 256 | 224 | gen_ppl + MAUVE + entropy + rep2 + avg_steps |

### Metrics

| Metric | Direction | Where | Description |
|--------|-----------|-------|-------------|
| `accuracy` | ↑ | math/humaneval | exact-match (MATH) or pass@1 (HumanEval) |
| `gen_ppl` | ↓ | text | Conditional perplexity via GPT-2-Large |
| `mauve` | ↑ | text | Distributional similarity to C4 reference text |
| `entropy` | ↑ | text | Bigram entropy (lexical diversity) |
| `rep2` | ↓ | text | Repeated bigram ratio |
| `avg_steps` | ↓ | all | Actual model forward passes used |

### Protocol references

- **MATH/HumanEval**: KLASS (Kim et al., NeurIPS 2025; arXiv 2511.05664).
  We use KLASS's exact `data/math_test.json`, prompts, and `utils.py` for
  answer extraction (`extract_math_answer`, `compare_answers`).
- **Text generation**: prefix-conditioned C4 continuation, similar to MDLM /
  ReMDM evaluation but with conditioning on a 32-token prefix.

### Baselines (from KLASS algorithms)

- `confidence_greedy` — LLaDA's `low_confidence` remasking: top-k by max prob.
- `topk_margin` — Dream's `topk_margin`: top-k by (top1 prob − top2 prob).
- `klass` — SOTA: KL-adaptive stability + confidence thresholds.

## Reference Performance

LLaDA paper (EVAL.md, gen_length=256/steps=256/block_length=256):
**MATH = 30.3%, HumanEval = 32.9%** on LLaDA-8B-Base.

KLASS paper on LLaDA-8B-Instruct, MATH (with block_length=64):
**~33.8%** (KLASS), reducing steps by 40-70%.
