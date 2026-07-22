# Test-Time Memory Module Design (Titans / Nested Learning)

## Research Question

Design a better **test-time memory module**. The module is a small chunk-wise
sub-layer whose *fast weights* update online during each forward pass from a
surprise signal, so the network can consolidate context-dependent associations
beyond what standard attention can store within a fixed context window. The
research question is about the memory module's **inner learning dynamics**:
**what surprise signal, online update rule, forget-gate parameterization, and
storage structure give the best length-generalization and quality-per-parameter
trade-off?**

A 345M-parameter GPT trained on ClimbMix serves only as the **evaluation
substrate** — it is held fixed across baselines so that the only varying
component is the memory module. The test-time learning framing is faithful to
Titans (Behrouz et al., 2025): the slow params (projections, forget-gate
weights, optionally a learned warm start) are co-trained with the substrate by
the outer AdamW; the fast weights themselves are reinitialized per sequence and
only ever updated by the inner surprise step.

The five reference baselines span the design space:

* **`titans_default`** = Titans (Behrouz et al., 2025, *Learning to Memorize
  at Test Time*, arXiv:2501.00663) Eq. 13–14: depth=2 MLP memory with MSE
  surprise loss, momentum SGD update, and three data-dependent gates
  (αₜ forget gate, ηₜ momentum decay, θₜ surprise weight) per Titans §3.2's
  chunk-wise simplification.
* **`linear_memory`** = Titans §3.1 / Table 5 depth ablation: L_M = 1
  (linear regression memory), otherwise identical to `titans_default` —
  clean depth-only ablation.
* **`deep_momentum`** = Nested Learning (Behrouz et al., 2025,
  arXiv:2512.24695) §4 Eq. 51 (Deep Momentum Gradient Descent):
  `m = α m − η · φ(∇L); W += m`. No weight decay on W; a learnable
  feature map φ is applied to the gradient before it enters the momentum.
* **`fast_weights_hebbian`** = Ba, Hinton et al. (NeurIPS 2016,
  *Using Fast Weights to Attend to the Recent Past*, arXiv:1610.06258)
  Eq. 1+5: pure Hebbian outer-product update
  `A(t) = λ A(t-1) + η h(t) h(t)^T` with paper hyperparameters λ=0.95,
  η=0.5, and an iterative LayerNorm-stabilized inner-loop read. No
  surprise gradient — direct associative storage.
* **`vanilla`** = attention-only control: `TitansMemoryLayer.forward` returns
  zeros, so the model uses the same RoPE causal attention substrate with no
  memory residual.

The slow-parameter counts are intentionally disclosed rather than artificially
balanced: Titans/linear use `{k,v,q,out}` plus three gate projections,
`deep_momentum` replaces one gate with a learned `φ` map, and Hebbian ties
`k_proj`/`v_proj` with constant λ/η. These differences are intrinsic to the
published update rules and are covered by the task-level parameter budget.

## What You Can Modify

Three editable regions in `nanoGPT/custom_pretrain.py`:

All memory baselines attach a `TitansMemoryLayer` to every transformer block,
matching the Titans block-level memory composition. The memory update uses
256-token chunks and closed-form fast-weight gradients for the provided linear
and two-layer SiLU memory networks; this is algebraically equivalent to the
inner MSE surprise gradient while avoiding one `autograd.grad` call per chunk
per memory layer.

1. **TTT-memory region** (lines 35–274, between read-only `# BEGIN/END TTT-MEMORY EDITABLE REGION` markers — do NOT delete or replace the marker lines), containing:
   - `init_memory_state(memory, dim)` — allocate the memory MLP's "fast" state,
     momentum buffers, learning-rate / decay hyperparameters, optional replay FIFOs.
   - `compute_keys_values(memory, x)` — decide *what* the memory stores. Given the
     per-chunk hidden input `x` (B, C_chunk, dim), return `(keys, values)`.
   - `compute_surprise(memory, fast_W, keys, values)` — decide *how surprising* the
     current chunk is. Returns the scalar loss that drives the online update.
     Common choices: L2 reconstruction, cosine distance, InfoNCE-style contrastive.
   - `apply_update(memory, fast_W, fast_S, grads, gates=None)` — decide *how* the gradients
     update the fast memory. SGD+momentum, Adam moments, cascaded (Nested) momentum,
     surprise-gated steps, EMA blending, context-dependent decay, etc.
   - `TitansMemoryLayer(nn.Module)` — glues the four functions into a chunk-wise
     online-update loop and a final read path.
   - `CausalSelfAttention(nn.Module)` — the attention block that wraps
     `TitansMemoryLayer` (MAC / MAG / MAL composition is up to the baseline).

2. **Block region** (lines 335–346) — how `CausalSelfAttention` composes with the
   MLP inside a residual block. Most baselines leave it unchanged; some may want
   to route the memory residual separately from the attention residual.

3. **CONFIG_OVERRIDES region** (lines 531–533) — per-method overrides for
   `learning_rate / weight_decay / warmup_iters / min_lr / grad_clip`.

## Intended Task Boundary

* Research question is the **memory module's inner learning dynamics**, *not* the
  outer GPT architecture. The evaluator enforces the top-level region boundary
  with an AST validator: the six required definitions must be present, and
  additional top-level helper functions/classes are allowed. Other top-level
  statements are still disallowed inside the TTT-memory span.
* **What gets co-trained vs. per-sequence reset.** The Titans framing is
  test-time learning: each forward pass starts the memory's *fast weights* from
  a stored prior and updates them online via the surprise loop. The "memory
  module is co-trained with the GPT backbone" means the **slow params**
  (`{k,v,q,out}_proj`, the data-dependent forget-gate `alpha_proj`, the
  warm-start tensors stored on the memory layer) are trained by the outer
  AdamW; the **fast weights** are reinitialized per sequence and only ever
  updated by the inner surprise step. By default `M1_init/M2_init` are
  registered as buffers (random per-layer prior). If you want a *learned*
  warm start, you may declare them as `nn.Parameter` instead — they will be
  picked up by `configure_optimizers` automatically. Any state that must
  receive outer gradient must be a `nn.Parameter`.
* `configure_optimizers` is fixed and trains everything in `named_parameters()`
  with `requires_grad=True`. During training, fast-weight tensors are **not**
  detached between chunks: the outer loss can backpropagate through the
  differentiable `apply_update` chain, so `k_proj`, `v_proj`, gates, and other
  slow update parameters receive gradients from future reads. Provided
  baselines leave chunk checkpointing disabled (`use_checkpoint = False`) to
  keep the closed-form update path straightforward. Evaluation may detach fast
  state between chunks because no outer gradient is needed.
* Length generalization is evaluated with NIAH passkey retrieval at 1.25× / 1.5× /
  2× the training context (1280 / 1536 / 2048), with NTK-aware RoPE base scaling
  applied at eval time so positions past the training length stay roughly
  in-distribution. The passkey is a six-letter uppercase sequence. The baseline
  `CausalSelfAttention` uses RoPE
  (`use_pos_emb=False`); the memory module is what discriminates baselines once
  context exceeds the 1024 training seq_len.

## Evaluation

Evaluation follows the same setup as other `llm-pretrain-*` tasks: training at
345M scale (24L/16H/1024D) with Chinchilla-optimal 7.1B tokens.

* **Score**: geometric mean of three components:
  - training quality: `val_loss`, `wikitext2_ppl`, and `lambada_ppl`
  - downstream 0-shot: `arc_easy`, `hellaswag`, `piqa`, and `winogrande`
    with equal weight. `piqa` and `winogrande` are hidden in the leaderboard
    but contribute to the score.
  - length generalization: `niah_pass_1536`, `niah_pass_2k`, and
    `niah_rank_2k`. Rank is scored after `log1p(rank)` compression so one
    very large raw rank cannot dominate the component.
* **Visible benchmark regimes**:
  - `gpt-345m` — 345M pretraining on ClimbMix + WT2/LAMBADA PPL
  - `lm-eval-345m` — 0-shot downstream evaluation (ARC-Easy, HellaSwag) on the
    345M checkpoint
  - `niah-eval-345m` — standalone NIAH passkey retrieval with NTK-aware RoPE at
    contexts 1280 / 1536 / 2048
* **Training data**: ClimbMix tokenized training split (~58GB)
* **Held-out eval data**: WikiText-2, LAMBADA (packaged `eval` dependency)
* **NIAH**: synthesized at eval time with the GPT-2 BPE tokenizer; 40 samples per
  context length, six-letter uppercase passkeys, teacher-forced accuracy at the
  answer positions
* **Training schedule**: 345M uses ~7.1B tokens (13535 steps, 2-GPU DDP, LR=3e-4),
  same as `llm-pretrain-attention` / `llm-pretrain-linear-attention`
* **Metric instrumentation**: nanoGPT metric printing is task-local in
  `custom_pretrain.py`; there is intentionally no package-level nanoGPT
  `pre_edit.py` for this task.
