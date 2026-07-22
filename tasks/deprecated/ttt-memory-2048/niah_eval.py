"""Standalone NIAH passkey eval for ttt-memory checkpoints.

Loads the saved model_source + checkpoint, monkey-patches every
CausalSelfAttention's `_rope_cache` to use NTK-aware base scaling for
T > train_ctx (so positions beyond training length stay roughly
in-distribution), and runs synthetic passkey retrieval at a few context
lengths chosen to live just past the vanilla-RoPE extrapolation cliff.

Outputs:
  TEST_METRICS niah_pass_1280=<acc> niah_pass_1536=<acc> niah_pass_2k=<acc>
                niah_rank_1536=<mean_rank> niah_rank_2k=<mean_rank>

Mean rank is the rank of the first answer token from a six-letter uppercase
passkey sequence, averaged over samples — a finer-grained metric than binary
accuracy at the discrimination range where the model is *almost* getting it.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from types import MethodType

import numpy as np
import torch

# Cap CPU threads. Default torch.set_num_threads picks up nproc which on
# beefy nodes can be 192+; with TitansMemoryLayer's many small inner-grad
# ops this thrashes. SLURM allocates an explicit core count via
# OMP_NUM_THREADS / SLURM_CPUS_PER_TASK; honor that, else cap at 16.
_n_threads = int(os.environ.get('OMP_NUM_THREADS') or
                 os.environ.get('SLURM_CPUS_PER_TASK') or 16)
torch.set_num_threads(min(_n_threads, 16))

TRAIN_CTX = 2048  # training seq_len; NTK kicks in only for T > this
ALPHABET = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def _load_module(src_path: str):
    spec = importlib.util.spec_from_file_location("niah_model_src", src_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["niah_model_src"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_ntk_rope_cache(self, T, device, dtype):
    """NTK-aware RoPE: scale base by s^(d/(d-2)) when T > TRAIN_CTX."""
    hd = self.head_dim
    if T > TRAIN_CTX:
        s = T / TRAIN_CTX
        base = self._rope_base * (s ** (hd / (hd - 2)))
    else:
        base = self._rope_base
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    pos = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype).view(1, 1, T, hd), emb.sin().to(dtype).view(1, 1, T, hd)


def _patch_ntk(model):
    patched = 0
    for block in model.transformer.h:
        attn = block.attn
        if hasattr(attn, "_rope_cache"):
            attn._rope_cache = MethodType(_make_ntk_rope_cache, attn)
            patched += 1
    print(f"  patched {patched} attention layers with NTK-aware RoPE (s = T/{TRAIN_CTX})", flush=True)


def _build_sample(ctx_len, key, enc, filler_toks):
    needle = enc.encode(f" The pass key is {key}. Remember it. {key} is the pass key.")
    question = enc.encode(" The pass key is")
    answer = enc.encode(f" {key}")
    reserve = len(needle) + len(question) + len(answer) + 4
    budget = ctx_len - reserve
    if budget < 100:
        raise ValueError(f"ctx_len={ctx_len} too small for needle template")
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


def _sample_key(rng):
    return " ".join(rng.choice(ALPHABET, size=6))


@torch.no_grad()
def evaluate(model, ctx_len, n_samples, device, seed=42, enc=None, filler_toks=None):
    rng = np.random.default_rng(seed)
    correct = 0
    rank0_sum = 0.0
    n = 0
    for s_idx in range(n_samples):
        key = _sample_key(rng)
        try:
            tokens, prompt_len, answer = _build_sample(ctx_len, key, enc, filler_toks)
        except ValueError as exc:
            print(f"  niah @ {ctx_len}: skip ({exc})", flush=True)
            continue
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        y = x.clone()
        try:
            logits, _ = model(x, y)
        except Exception as exc:
            print(f"  niah @ {ctx_len}: forward failed ({type(exc).__name__}: {exc})", flush=True)
            return 0.0, float('nan')
        hit = True
        for i in range(len(answer)):
            pos = prompt_len - 1 + i
            l = logits[0, pos].float()
            target = answer[i]
            if l.argmax().item() != target:
                hit = False
            if i == 0:
                rank0_sum += (l > l[target]).sum().item() + 1
        if hit:
            correct += 1
        n += 1
        if (s_idx + 1) % 5 == 0:
            print(f"    [ctx={ctx_len}] {s_idx + 1}/{n_samples} done, running acc={correct}/{n}", flush=True)
    if n == 0:
        return 0.0, float('nan')
    return correct / n, rank0_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--ctx-lens", type=str, default="1280,1536,2048")
    p.add_argument("--n-samples", type=int, default=int(os.environ.get("NIAH_N_SAMPLES", 40)))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"=== Standalone NIAH eval ===", flush=True)
    print(f"checkpoint: {args.checkpoint}")
    print(f"source:     {args.source}")
    print(f"ctx_lens:   {args.ctx_lens}")
    print(f"n_samples:  {args.n_samples}")
    print(f"device:     {args.device}", flush=True)

    print(f"\nLoading source...", flush=True)
    mod = _load_module(args.source)

    print(f"Loading checkpoint...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ma = ckpt["model_args"]
    print(f"  model_args: {ma}")
    gptconf = mod.GPTConfig(**ma)
    model = mod.GPT(gptconf)
    sd = ckpt["model_state_dict"]
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    model = model.to(args.device)
    # bf16 helps both CPU and CUDA paths; CPU bf16 matmul is ~2x fp32 on x86.
    model = model.to(torch.bfloat16)
    print(f"  loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M)", flush=True)

    _patch_ntk(model)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    filler_toks = enc.encode(
        "The grass is green. The sky is blue. The sun is yellow. Here we go. ")
    print(f"  tiktoken ready (filler={len(filler_toks)} toks)", flush=True)

    metrics = {}
    for ctx_str in args.ctx_lens.split(","):
        ctx_len = int(ctx_str.strip())
        if ctx_len > ma["block_size"]:
            print(f"\nctx={ctx_len}: skip (> model block_size {ma['block_size']})", flush=True)
            continue
        print(f"\n=== ctx_len={ctx_len} ===", flush=True)
        t = time.time()
        acc, mean_rank = evaluate(model, ctx_len, args.n_samples, args.device, seed=args.seed,
                                   enc=enc, filler_toks=filler_toks)
        elapsed = time.time() - t
        print(f"  acc={acc:.4f} ({int(round(acc * args.n_samples))}/{args.n_samples}), "
              f"mean rank(answer[0])={mean_rank:.1f}, {elapsed:.1f}s", flush=True)
        # Tag metric by ctx (1280 stays "1280", 2048 -> "2k", 1536 stays "1536", etc).
        tag = f"{ctx_len // 1024}k" if ctx_len % 1024 == 0 else str(ctx_len)
        metrics[f"niah_pass_{tag}"] = acc
        metrics[f"niah_rank_{tag}"] = mean_rank

    metric_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"\nTEST_METRICS: {metric_str}", flush=True)


if __name__ == "__main__":
    main()
