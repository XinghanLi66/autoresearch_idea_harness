"""GPT-2 Medium Test-Time Training Evaluation Script.

Loads GPT-2 Medium weights from HuggingFace (pre-downloaded to /data/gpt2-medium),
converts them to nanoGPT format, then applies a test-time training (TTT) adaptation
strategy before final evaluation. Only the TTTAdapter class is editable.

No pretraining is done -- this script uses the real GPT-2 Medium weights directly.
"""

import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# Model Components (nanoGPT architecture)
# ============================================================================

# -- Normalization ------------------------------------------------------------
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# -- Self-Attention -----------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self.use_pos_emb = True

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

# -- Feed-Forward Network ----------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# -- Transformer Block -------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ============================================================================
# GPT Model
# ============================================================================

from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 original vocab size
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = True  # GPT-2 uses bias

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        use_pos = getattr(self.transformer.h[0].attn, 'use_pos_emb', True)
        if use_pos:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

# ============================================================================
# HuggingFace GPT-2 -> nanoGPT Weight Conversion
# ============================================================================

def load_hf_gpt2_weights(model, hf_model_dir):
    """Load HuggingFace GPT-2 weights into a nanoGPT model.

    HF GPT-2 and nanoGPT use different naming conventions:
      HF: transformer.h.{i}.attn.c_attn.weight  (Conv1D, transposed)
      nanoGPT: transformer.h.{i}.attn.c_attn.weight  (nn.Linear)

    HF GPT-2 uses Conv1D which stores weights as (in, out) whereas
    nn.Linear stores as (out, in), so we need to transpose attention
    and MLP weight matrices.
    """
    from safetensors import safe_open
    import json
    import glob

    # Load HF state dict from safetensors or pytorch_model.bin
    hf_sd = {}
    safetensor_files = sorted(glob.glob(os.path.join(hf_model_dir, '*.safetensors')))
    if safetensor_files:
        for sf in safetensor_files:
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    hf_sd[key] = f.get_tensor(key)
    else:
        bin_path = os.path.join(hf_model_dir, 'pytorch_model.bin')
        if os.path.exists(bin_path):
            hf_sd = torch.load(bin_path, map_location='cpu', weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights found in {hf_model_dir}")

    # Build mapping from HF keys to nanoGPT keys
    # HF Conv1D weights need transposing; nanoGPT uses nn.Linear
    nano_sd = model.state_dict()
    converted = {}

    # Keys that need transposing (Conv1D -> Linear: (in, out) -> (out, in))
    transpose_keys = {
        'attn.c_attn.weight', 'attn.c_proj.weight',
        'mlp.c_fc.weight', 'mlp.c_proj.weight',
    }

    for hf_key, hf_val in hf_sd.items():
        # Map HF key names to nanoGPT key names
        nano_key = hf_key

        # HF: h.{i}.ln_1 -> nanoGPT: h.{i}.ln_1 (same)
        # HF: h.{i}.attn.c_attn -> nanoGPT: h.{i}.attn.c_attn (same)
        # HF: h.{i}.mlp.c_fc -> nanoGPT: h.{i}.mlp.c_fc (same)
        # HF: ln_f -> nanoGPT: ln_f (same)
        # HF: wte, wpe -> nanoGPT: wte, wpe (same)

        # Check if this key needs transposing (Conv1D weight)
        needs_transpose = any(nano_key.endswith(tk) for tk in transpose_keys)
        if needs_transpose:
            hf_val = hf_val.t()

        if nano_key in nano_sd:
            if nano_sd[nano_key].shape == hf_val.shape:
                converted[nano_key] = hf_val
            else:
                print(f"  Shape mismatch for {nano_key}: "
                      f"nanoGPT {nano_sd[nano_key].shape} vs HF {hf_val.shape}, skipping")
        else:
            # lm_head.weight is tied to wte.weight, skip it
            if nano_key == 'lm_head.weight':
                pass
            else:
                print(f"  Key {nano_key} not found in nanoGPT model, skipping")

    # Check coverage
    missing = set(nano_sd.keys()) - set(converted.keys())
    # lm_head.weight is tied to wte, so it's expected to be "missing"
    missing.discard('lm_head.weight')
    if missing:
        print(f"  Warning: {len(missing)} keys not loaded from HF: {missing}")

    model.load_state_dict(converted, strict=False)
    print(f"Loaded {len(converted)} parameter tensors from HuggingFace GPT-2 Medium")
    return model

# ============================================================================
# Data Loading
# ============================================================================

def get_eval_chunks(data_dir, block_size, device, split='val', max_chunks=200):
    """Load evaluation data as sequential non-overlapping chunks."""
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    chunks = []
    for start in range(0, len(data) - block_size, block_size):
        x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
        chunks.append((x, y))
        if len(chunks) >= max_chunks:
            break
    return chunks

# ============================================================================
# Test-Time Training (TTT) Adapter
# ============================================================================
# EDITABLE REGION START (lines 276-346)
# The agent modifies the TTTAdapter class to implement test-time training
# strategies that improve language modeling performance at inference time.

class TTTAdapter:
    """Test-time training adapter for GPT models.

    This adapter takes a pretrained GPT model and adapts it at inference time
    using the evaluation context. The goal is to reduce validation loss by
    specializing the model to the local data distribution at test time.

    The adapter has two main methods:
    - setup(model, config): Initialize any TTT-specific parameters or state.
      Called once after the model is loaded. `config` is the GPTConfig.
    - adapt_and_evaluate(model, eval_chunks, ctx): Run TTT adaptation and
      return the final evaluation loss. `eval_chunks` is a list of (x, y)
      pairs from the validation set. `ctx` is the autocast context manager.

    Constraints:
    - You may create a copy of the model for adaptation (use copy.deepcopy)
    - You may add lightweight parameters (LoRA, adapters, memory modules)
    - Total additional parameter count should be < 5% of the base model
    - The adaptation must complete within a reasonable time budget
    - You must return a valid loss value (float) from adapt_and_evaluate
    - Do NOT modify the base model's weights directly -- always work on a copy

    Available imports: copy, math, torch, torch.nn, torch.nn.functional (as F),
                       numpy (as np), os, time
    """

    def __init__(self):
        """Initialize the TTT adapter (called before checkpoint loading)."""
        self.ttt_lr = 1e-4
        self.ttt_steps = 0  # No adaptation by default
        self.ttt_optimizer_cls = torch.optim.AdamW

    def setup(self, model, config):
        """Set up TTT-specific state after loading the pretrained model.

        Args:
            model: The pretrained GPT model (nn.Module).
            config: The GPTConfig used to build the model.
        """
        self.config = config

    def adapt_and_evaluate(self, model, eval_chunks, ctx):
        """Adapt the model at test time and compute final evaluation loss.

        This method should:
        1. Create a copy of the model (or selected parameters)
        2. Use part of eval_chunks as adaptation context
        3. Run TTT optimization on the context
        4. Evaluate on the remaining chunks (or all chunks)
        5. Return the average evaluation loss

        Args:
            model: The pretrained GPT model (nn.Module, in eval mode).
            eval_chunks: List of (x, y) tensor pairs, each of shape
                         (1, block_size) from the validation set.
            ctx: torch.amp.autocast context manager for mixed precision.

        Returns:
            float: The average cross-entropy loss on the evaluation data.
        """
        # Default: no adaptation, just evaluate the pretrained model
        model.eval()
        total_loss = 0.0
        n_chunks = 0
        with torch.no_grad():
            for x, y in eval_chunks:
                with ctx:
                    _, loss = model(x, y)
                total_loss += loss.item()
                n_chunks += 1
        return total_loss / max(n_chunks, 1)

# EDITABLE REGION END (lines 276-346)
# ============================================================================

# ============================================================================
# Evaluation Script (loads HF GPT-2 Medium, runs TTT, evaluates)
# ============================================================================

if __name__ == '__main__':
    # -- Configuration from environment --
    output_dir = os.environ.get('OUTPUT_DIR', 'out')
    seed = int(os.environ.get('SEED', 1337))
    data_dir = os.environ.get('DATA_DIR', '/data/climbmix')
    gpt2_medium_dir = os.environ.get('GPT2_MEDIUM_DIR', '/data/gpt2-medium')
    eval_iters = 200

    device = 'cuda'
    device_type = 'cuda'
    dtype = 'bfloat16'
    block_size = 1024
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # -- Build nanoGPT model with GPT-2 Medium config --
    gptconf = GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        dropout=0.0,
        bias=True,
    )
    model = GPT(gptconf)

    # -- Load HuggingFace GPT-2 Medium weights --
    print(f"Loading GPT-2 Medium weights from: {gpt2_medium_dir}")
    model = load_hf_gpt2_weights(model, gpt2_medium_dir)
    model.to(device)
    model.eval()

    # -- Compute baseline val loss (before TTT) --
    eval_chunks = get_eval_chunks(data_dir, block_size, device, split='val', max_chunks=eval_iters)
    print(f"Loaded {len(eval_chunks)} eval chunks")

    base_val_loss = 0.0
    with torch.no_grad():
        for x, y in eval_chunks:
            with ctx:
                _, loss = model(x, y)
            base_val_loss += loss.item()
    base_val_loss /= max(len(eval_chunks), 1)
    print(f"Pre-TTT val loss: {base_val_loss:.4f}")

    # -- Initialize TTT Adapter --
    ttt_adapter = TTTAdapter()
    ttt_adapter.setup(model, gptconf)

    # -- Run TTT Adaptation and Evaluation --
    t_ttt_start = time.time()
    ttt_val_loss = ttt_adapter.adapt_and_evaluate(model, eval_chunks, ctx)
    t_ttt_end = time.time()
    print(f"TTT adaptation took {t_ttt_end - t_ttt_start:.2f}s")

    improvement = base_val_loss - ttt_val_loss
    print(f"Final: base_val_loss {base_val_loss:.4f}, ttt_val_loss {ttt_val_loss:.4f}")
    print(f"TTT improvement: {improvement:.4f} ({improvement/base_val_loss*100:.2f}%)")

    # -- PPL on Benchmark Datasets (with TTT adaptation) --
    eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
    eval_datasets = ['wikitext2', 'lambada']
    ppl_results = {}
    for ds_name in eval_datasets:
        ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
        if not os.path.exists(ds_path):
            print(f"Eval dataset not found: {ds_path}")
            continue
        data = np.memmap(ds_path, dtype=np.uint16, mode='r')
        ds_chunks = []
        for start in range(0, len(data) - block_size, block_size):
            x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
            y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
            ds_chunks.append((x, y))
            if len(ds_chunks) >= 500:
                break

        # Evaluate with TTT on each benchmark dataset
        ds_loss = ttt_adapter.adapt_and_evaluate(model, ds_chunks, ctx)
        ppl = math.exp(ds_loss)
        ppl_results[ds_name] = ppl
        print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={ds_loss:.4f})")

    ppl_str = ', '.join(f'{k}_ppl={v:.2f}' for k, v in ppl_results.items())
    print(f"TEST_METRICS: val_loss={ttt_val_loss:.4f}, {ppl_str}", flush=True)
