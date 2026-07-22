"""Custom GPT-2 pretraining for ttt-memory (Titans / Nested Learning).

Based on Andrej Karpathy's nanoGPT, with a wide editable region for
test-time memory module design. The memory module receives the per-chunk
hidden state, computes a surprise loss, updates its own fast weights
online (not via the outer AdamW), and produces a residual read path.
"""

import ast
import inspect
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# BEGIN TTT-MEMORY EDITABLE REGION
def init_memory_state(memory, dim):
    """Initialize multi-head fast-weight state for chunk-wise TTT memory.

    The benchmark substrate is GPT-345M (dim=1024, 16 attention heads). We
    intentionally use the same 16-way factorization as TTT-Linear/TTT-MLP's
    350M config: head_dim = 1024 / 16 = 64. This keeps fast weights isolated
    per sequence and per head as (B, H, 64, 64), matching the published TTT
    convention and lucidrains' Titans implementation that flattens B*H for
    per-sample memory gradients.
    """
    memory.memory_dim = dim
    memory.depth = 2
    memory.chunk_size = 256
    memory.learning_rate = 1.0
    memory.momentum_coef = 0.9
    memory.decay = 0.99
    memory.head_dim = 64
    assert dim % memory.head_dim == 0
    memory.num_memory_heads = dim // memory.head_dim
    assert dim == memory.num_memory_heads * memory.head_dim
    std = 1.0 / math.sqrt(memory.head_dim)
    W1 = torch.empty(memory.num_memory_heads, memory.head_dim, memory.head_dim)
    W2 = torch.empty(memory.num_memory_heads, memory.head_dim, memory.head_dim)
    torch.nn.init.normal_(W1, mean=0.0, std=std)
    torch.nn.init.normal_(W2, mean=0.0, std=std)
    memory.register_buffer('M1_init', W1)
    memory.register_buffer('M2_init', W2)


def _linear(layer, x):
    return F.linear(x, layer.weight, layer.bias)


def _silu_grad(x):
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def compute_keys_values(memory, x):
    """Project hidden state into per-head keys and values: (B, T, H, D)."""
    B, T, _ = x.shape
    H, D = memory.num_memory_heads, memory.head_dim
    k = _linear(memory.k_proj, x).view(B, T, H, D)
    v = _linear(memory.v_proj, x).view(B, T, H, D)
    return k, v


def compute_surprise(memory, fast_W, keys, values):
    """Per-sequence/per-head MSE surprise for the online fast-weight update."""
    pred = memory._apply_fast_mlp(fast_W, keys)
    return (pred - values).pow(2).sum(dim=-1).mean(dim=1).sum()


def compute_fast_gradients(memory, fast_W, keys, values):
    """Closed-form gradient of the chunk MSE surprise w.r.t. fast weights."""
    scale = 2.0 / max(keys.shape[1], 1)
    h1_pre = torch.einsum('bthi,bhoi->btho', keys, fast_W[0])
    if len(fast_W) == 1:
        err = h1_pre - values
        return [scale * torch.einsum('btho,bthi->bhoi', err, keys)]
    h1 = F.silu(h1_pre)
    pred = torch.einsum('bthi,bhoi->btho', h1, fast_W[1])
    err = pred - values
    grad_W2 = scale * torch.einsum('btho,bthi->bhoi', err, h1)
    grad_h1 = scale * torch.einsum('btho,bhoi->bthi', err, fast_W[1])
    grad_pre = grad_h1 * _silu_grad(h1_pre)
    grad_W1 = torch.einsum('btho,bthi->bhoi', grad_pre, keys)
    return [grad_W1, grad_W2]


def apply_update(memory, fast_W, fast_S, grads, gates=None):
    """Fast-weight update rule (SGD + momentum + decay)."""
    lr = memory.learning_rate
    mu = memory.momentum_coef
    alpha = memory.decay
    new_W, new_S = [], []
    for W, S, g in zip(fast_W, fast_S, grads):
        S_new = mu * S - lr * g
        W_new = alpha * W + S_new
        new_W.append(W_new)
        new_S.append(S_new)
    return new_W, new_S


class TitansMemoryLayer(nn.Module):
    """Chunk-wise online memory with multi-head test-time fast weights.

    Fast tensors have shape (B, H, D, D). The read path uses chunk-start fast
    weights, then the same chunk produces a differentiable per-head surprise
    update for future chunks. Every transformer block owns a memory layer, and
    closed-form update gradients keep slow projections co-trained without
    per-chunk autograd.grad calls.
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.use_checkpoint = False
        init_memory_state(self, self.dim)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_memory_heads, self.head_dim)

    def _merge_heads(self, x):
        B, T, H, D = x.shape
        return x.contiguous().view(B, T, H * D)

    def _apply_fast_mlp(self, fast_W, x):
        h = torch.einsum('bthi,bhoi->btho', x, fast_W[0])
        if len(fast_W) >= 2:
            h = F.silu(h)
            h = torch.einsum('bthi,bhoi->btho', h, fast_W[1])
        return h

    def _init_fast(self, batch_size, dtype):
        W = [self.M1_init.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()]
        if self.depth >= 2:
            W.append(self.M2_init.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous())
        if self.training:
            for w in W:
                w.requires_grad_(True)
        S = [torch.zeros_like(w) for w in W]
        return W, S

    def _chunk_step(self, chunk, *state):
        depth = self.depth
        fast_W = list(state[:depth])
        fast_S = list(state[depth:])
        chunk_f = chunk

        q = self._split_heads(_linear(self.q_proj, chunk_f))
        out = self._apply_fast_mlp(fast_W, q)
        keys, values = compute_keys_values(self, chunk_f)
        grads = compute_fast_gradients(self, fast_W, keys, values)
        fast_W, fast_S = apply_update(self, fast_W, fast_S, grads, None)
        if not self.training:
            fast_W = [w.detach() for w in fast_W]
            fast_S = [s_.detach() for s_ in fast_S]
        return (out.to(chunk.dtype), *fast_W, *fast_S)

    @torch.compiler.disable
    def forward(self, x):
        B, T, C = x.shape
        fast_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        fast_W, fast_S = self._init_fast(B, fast_dtype)
        outs = []
        cs = self.chunk_size
        if self.training and self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
        for s in range(0, T, cs):
            e = min(s + cs, T)
            chunk = x[:, s:e]
            state = tuple(fast_W + fast_S)
            if self.training and self.use_checkpoint:
                result = checkpoint(self._chunk_step, chunk, *state, use_reentrant=False)
            else:
                result = self._chunk_step(chunk, *state)
            outs.append(result[0])
            state = result[1:]
            fast_W = list(state[:self.depth])
            fast_S = list(state[self.depth:])
        y = self._merge_heads(torch.cat(outs, dim=1))
        return self.out_proj(y)

class CausalSelfAttention(nn.Module):
    """Standard causal softmax attention (RoPE) + parallel Titans memory residual."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.memory = TitansMemoryLayer(config)
        self.use_pos_emb = False
        self._rope_base = 10000.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def _rope_cache(self, T, device, dtype):
        hd = self.head_dim
        inv = 1.0 / (self._rope_base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
        pos = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype).view(1, 1, T, hd), emb.sin().to(dtype).view(1, 1, T, hd)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x, cos, sin):
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = self._rope_cache(T, x.device, q.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        if self.flash:
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.head_dim)
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            att = att.masked_fill(~mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            a = torch.einsum('bhqk,bhkd->bhqd', att, v)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        a = self.resid_dropout(self.c_proj(a))
        m = self.memory(x)
        return a + m
















# END TTT-MEMORY EDITABLE REGION


def _validate_ttt_memory_editable_region():
    def _fail(message):
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            raise RuntimeError(message)
        raise SystemExit(message)

    required_names = {
        "init_memory_state",
        "compute_keys_values",
        "compute_surprise",
        "apply_update",
        "TitansMemoryLayer",
        "CausalSelfAttention",
    }
    with open(__file__, 'r') as _f:
        source = _f.read()
    start = source.index("# BEGIN TTT-MEMORY EDITABLE REGION") + len("# BEGIN TTT-MEMORY EDITABLE REGION")
    end = source.index("# END TTT-MEMORY EDITABLE REGION")
    snippet = source[start:end]
    parsed = ast.parse(snippet)
    seen = set()
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            seen.add(node.name)
        else:
            _fail("TTT-memory region may only contain top-level function/class definitions")
    missing = required_names - seen
    if missing:
        _fail(f"TTT-memory region is missing required definitions: {sorted(missing)}")
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


_validate_ttt_memory_editable_region()


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


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


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
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


def build_niah_sample(ctx_len, key, enc, filler_toks):
    needle = enc.encode(f" The pass key is {key}. Remember it. {key} is the pass key.")
    question = enc.encode(" The pass key is")
    answer = enc.encode(f" {key}")
    reserve = len(needle) + len(question) + len(answer) + 4
    budget = ctx_len - reserve
    if budget < 100:
        raise ValueError(f"ctx_len={ctx_len} too small for NIAH template")
    pre_len = budget // 2
    post_len = budget - pre_len
    def _pad(n):
        out = []
        while len(out) < n:
            out.extend(filler_toks)
        return out[:n]
    tokens = _pad(pre_len) + needle + _pad(post_len) + question + answer
    tokens = tokens[:ctx_len]
    prompt_len = len(tokens) - len(answer)
    return tokens, prompt_len, answer


@torch.no_grad()
def run_niah(raw_model, ctx_len, n_samples, device, seed=42):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    filler_toks = enc.encode(filler)
    rng = np.random.default_rng(seed)
    raw_model.eval()
    correct = 0
    done = 0
    for _ in range(n_samples):
        key = " ".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=6))
        try:
            tokens, prompt_len, answer = build_niah_sample(ctx_len, key, enc, filler_toks)
        except ValueError as exc:
            print(f"  niah @ {ctx_len}: skipping ({exc})")
            continue
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        y = x.clone()
        try:
            logits, _ = raw_model(x, y)
        except Exception as exc:
            print(f"  niah @ {ctx_len}: forward failed ({exc})")
            return 0.0
        preds = logits.argmax(dim=-1).squeeze(0)
        hit = all(preds[prompt_len - 1 + i].item() == answer[i] for i in range(len(answer)))
        if hit:
            correct += 1
        done += 1
    return correct / max(done, 1)


if __name__ == '__main__':
    output_dir = os.environ.get('OUTPUT_DIR', 'out')
    seed = int(os.environ.get('SEED', 1337))
    data_dir = os.environ.get('DATA_DIR', '/data/climbmix')

    n_layer = int(os.environ.get('N_LAYER', 12))
    n_head = int(os.environ.get('N_HEAD', 12))
    n_embd = int(os.environ.get('N_EMBD', 768))

    max_iters = int(os.environ.get('MAX_ITERS', 5000))
    eval_interval = int(os.environ.get('EVAL_INTERVAL', 500))
    eval_iters = 200
    log_interval = 10
    batch_size = int(os.environ.get('BATCH_SIZE', 12))
    block_size = 2048                                # training seq slice (2048 control)
    model_block_size = int(os.environ.get('MODEL_BLOCK_SIZE', 8192))  # model capacity (for NIAH@8K)
    gradient_accumulation_steps = int(os.environ.get('GRAD_ACCUM', 5))
    learning_rate = float(os.environ.get('LEARNING_RATE', 6e-4))
    min_lr = learning_rate / 10
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    warmup_iters = int(max_iters * 0.04)
    lr_decay_iters = max_iters
    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
    CONFIG_OVERRIDES = {}

    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': learning_rate = _v; min_lr = learning_rate / 10
        elif _k == 'weight_decay': weight_decay = _v
        elif _k == 'warmup_iters': warmup_iters = _v
        elif _k == 'min_lr': min_lr = _v
        elif _k == 'grad_clip': grad_clip = _v

    # TitansMemoryLayer stays outside torch.compile because its chunk-wise
    # recurrent state is small and dynamic; compiling the remaining fragments
    # has not paid back its cold-start overhead in this task.
    compile_model = False
    dtype = 'bfloat16'

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        device = 'cuda'
        seed_offset = 0

    device_type = 'cuda'
    torch.manual_seed(seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if master_process:
        os.makedirs(output_dir, exist_ok=True)

    tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
    if ddp:
        tokens_per_iter *= int(os.environ.get('WORLD_SIZE', 1))
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    if master_process:
        print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      block_size=model_block_size, bias=False, vocab_size=50304, dropout=0.0)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    if compile_model:
        if master_process:
            print("compiling the model...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        raw = model.module if ddp else model
        raw.eval()
        for split, data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(data, batch_size, block_size, device)
                with ctx:
                    logits, loss = raw(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        raw.train()
        return out

    t0 = time.time()
    best_val_loss = 1e9

    for iter_num in range(max_iters + 1):
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            train_loss = losses['train'].item()
            val_loss = losses['val'].item()
            print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            print(f"TRAIN_METRICS: step={iter_num}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                X, Y = get_batch(train_data, batch_size, block_size, device)
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and iter_num > 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")

    del optimizer, scaler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    if master_process:
        raw = model.module if ddp else model
        raw.eval()

        # ── Save checkpoint + model source FIRST, before any fragile eval step.
        # Post-training evals (PPL / NIAH) are wrapped in try/except below so
        # a single crash can't wipe 16h of training progress.
        import shutil
        env_label = os.environ.get('ENV', 'model')
        save_model = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
        ckpt_data = {'model_state_dict': save_model.state_dict(), 'model_args': model_args}
        ckpt_path = os.path.join(output_dir, f'ckpt_{env_label}.pt')
        torch.save(ckpt_data, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}", flush=True)
        src_path = os.path.join(output_dir, f'model_source_{env_label}.py')
        shutil.copy2(os.path.abspath(__file__), src_path)
        print(f"Model source saved to {src_path}", flush=True)

        try:
            losses = estimate_loss()
            val_loss = losses['val'].item()
            train_loss = losses['train'].item()
            print(f"Final: train loss {train_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}")
        except Exception as exc:
            print(f"estimate_loss failed: {exc}", flush=True)
            val_loss = float('nan')

        eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
        eval_datasets = ['wikitext2', 'lambada']
        ppl_results = {}
        for ds_name in eval_datasets:
            try:
                ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
                if not os.path.exists(ds_path):
                    print(f"Eval dataset not found: {ds_path}")
                    continue
                data = np.memmap(ds_path, dtype=np.uint16, mode='r')
                n_tokens = len(data)
                total_loss = 0.0
                n_chunks = 0
                with torch.no_grad():
                    for start in range(0, n_tokens - block_size, block_size):
                        x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
                        y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
                        with ctx:
                            _, loss = raw(x, y)
                        total_loss += loss.item()
                        n_chunks += 1
                avg_loss = total_loss / max(n_chunks, 1)
                ppl = math.exp(avg_loss)
                ppl_results[ds_name] = ppl
                print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={avg_loss:.4f}, {n_chunks} chunks)")
            except Exception as exc:
                print(f"PPL {ds_name} eval failed: {exc}", flush=True)

        # NIAH is now run by the standalone scripts/niah_eval.sh test_cmd
        # (group 3) so the eval can apply NTK-aware RoPE scaling and target
        # the discrimination range (1.25-2x training length) without changing
        # training-time behavior. Skip in-training NIAH to avoid wasted time.

        ppl_str = ', '.join(f'{k}_ppl={v:.2f}' for k, v in ppl_results.items())
        tail = ppl_str
        print(f"TEST_METRICS: val_loss={val_loss:.4f}, {tail}", flush=True)

    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
