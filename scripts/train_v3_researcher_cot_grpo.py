#!/usr/bin/env python3
"""GRPO-style RL on the researcher-CoT policy with the composite reward (QS pod, single GPU).

Policy = full-SFT 32B checkpoint + trainable LoRA (base frozen). Reference for the KL penalty is
the same model with adapters disabled (zero extra memory). Per prompt we sample K completions,
score each with the composite reward, and use group-relative advantages (GRPO):

  R = w_fp * P_fingerprint(target researcher | masked cot)     # slim LogReg head on MiniLM emb
    + w_fmt * format_score(cot, researcher)                    # **Mocking:** + Core idea + crux anchors
    + w_cr  * creativity_aux(cot)                              # slim Ridge head, normalized to [0,1]

Loss = -mean_tokens(adv * logp(completion)) + kl_coef * mean_tokens(logp - logp_ref).

Designed for a GB200 (~189GB) single GPU: bf16 weights ~65GB + LoRA + grad-ckpt activations.
Writes events.jsonl (per-step rewards/KL), periodic adapter checkpoints, and a final summary.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def _patch_transformers_modeling_layers() -> None:
    if importlib.util.find_spec("transformers.modeling_layers") is not None:
        return
    module = types.ModuleType("transformers.modeling_layers")
    module.__spec__ = importlib.machinery.ModuleSpec("transformers.modeling_layers", loader=None)

    class GradientCheckpointingLayer:  # minimal stub, mirrors full-SFT trainer shim
        pass

    module.GradientCheckpointingLayer = GradientCheckpointingLayer
    sys.modules["transformers.modeling_layers"] = module


def _patch_broken_apex_amp() -> None:
    try:
        import apex  # type: ignore
    except Exception:
        apex = types.ModuleType("apex")
        apex.__spec__ = importlib.machinery.ModuleSpec("apex", loader=None)
        sys.modules["apex"] = apex
    if hasattr(sys.modules.get("apex"), "amp"):
        return
    amp = types.ModuleType("apex.amp")
    amp.__spec__ = importlib.machinery.ModuleSpec("apex.amp", loader=None)
    sys.modules["apex"].amp = amp
    sys.modules["apex.amp"] = amp


_patch_transformers_modeling_layers()
_patch_broken_apex_amp()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CJK = re.compile(r"[一-鿿]")
MOCK_RE = re.compile(r"\*\*Mocking:\*\*\s*(.+)")


# ---------------- composite reward ----------------

def _name_tokens(name: str) -> list[str]:
    toks = [t for t in re.split(r"[\s()·,]+", name) if len(t) >= 2 and not CJK.search(t)]
    cjk = "".join(CJK.findall(name))
    if cjk:
        toks.append(cjk)
    return toks


def format_score(text: str, researcher: str) -> float:
    has_mock = bool(MOCK_RE.search(text))
    has_core = "**Core idea:**" in text
    has_crux = "**Non-trivial crux:**" in text
    name_ok = False
    m = MOCK_RE.search(text)
    if m:
        name_ok = any(t in m.group(1) for t in _name_tokens(researcher))
    return 0.3 * has_mock + 0.2 * name_ok + 0.25 * has_core + 0.25 * has_crux


def mask_name(text: str, researcher: str) -> str:
    text = re.sub(r"^\*\*Mocking:\*\*.*\n?", "", text)
    for t in re.split(r"[\s()·,]+", researcher):
        if len(t) >= 2 and not CJK.search(t):
            text = re.sub(re.escape(t), "[NAME]", text, flags=re.I)
    cjk = "".join(CJK.findall(researcher))
    if cjk:
        text = text.replace(cjk, "[NAME]")
    return text


class CompositeReward:
    def __init__(self, heads_dir: Path, w_fp: float, w_fmt: float, w_cr: float) -> None:
        from sentence_transformers import SentenceTransformer

        meta = json.loads((heads_dir / "reward_heads_meta.json").read_text())
        npz = np.load(heads_dir / "reward_heads.npz")
        self.fp_coef, self.fp_b = npz["fp_coef"], npz["fp_intercept"]
        self.cr_coef, self.cr_b = npz["cr_coef"], npz["cr_intercept"]
        self.classes = {c: i for i, c in enumerate(meta["fingerprint_classes"])}
        self.embedder = SentenceTransformer(meta["embedder"], device="cpu")
        self.w_fp, self.w_fmt, self.w_cr = w_fp, w_fmt, w_cr

    def score(self, cots: list[str], researchers: list[str]) -> list[dict]:
        masked = [mask_name(c, r) for c, r in zip(cots, researchers)]
        Z = self.embedder.encode(masked, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        logits = Z @ self.fp_coef.T + self.fp_b  # (n, 125)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        cr_raw = Z @ self.cr_coef + self.cr_b[0]  # judge scale ~1-5
        out = []
        for i, (cot, res) in enumerate(zip(cots, researchers)):
            idx = self.classes.get(res)
            fp = float(probs[i, idx]) if idx is not None else 0.0
            fmt = format_score(cot, res)
            cr = float(np.clip((cr_raw[i] - 3.0) / 2.0, 0.0, 1.0))
            out.append({
                "fingerprint": fp, "format": fmt, "creativity": cr,
                "reward": self.w_fp * fp + self.w_fmt * fmt + self.w_cr * cr,
            })
        return out


# ---------------- data ----------------

def load_prompts(path: Path, limit: int | None, seed: int = 42) -> list[dict]:
    rows = [json.loads(l) for l in path.open()]
    prompts = []
    for r in rows:
        msgs = r["messages"][:2]  # researcher-conditioned system + user setup
        prompts.append({"messages": msgs, "researcher": r["researcher"]})
    rng = np.random.RandomState(seed)
    rng.shuffle(prompts)
    return prompts[:limit] if limit else prompts


# ---------------- logprob helpers ----------------

def completion_logprobs(model, input_ids, attention_mask, completion_mask):
    """Token logprobs of the labeled completion positions. Returns (sum over tokens per seq, count)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = completion_mask[:, 1:].to(logits.dtype)
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)
    return tok_logp * mask, mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-model", required=True, help="full-SFT 32B checkpoint dir (or base for smoke)")
    ap.add_argument("--train-jsonl", required=True, help="anchored dataset; prompts = system+user")
    ap.add_argument("--reward-heads-dir", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4, help="K samples per prompt")
    ap.add_argument("--max-new-tokens", type=int, default=1400)
    ap.add_argument("--gen-temperature", type=float, default=0.9)
    ap.add_argument("--gen-top-p", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--w-fp", type=float, default=0.4)
    ap.add_argument("--w-fmt", type=float, default=0.3)
    ap.add_argument("--w-cr", type=float, default=0.3)
    ap.add_argument("--prompt-limit", type=int, default=512)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--micro-batch", type=int, default=2, help="sequences per grad-forward micro-batch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary-name", default="grpo_summary.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = (args.output_dir / "events.jsonl").open("a")

    def log_event(**kw):
        kw["time"] = time.time()
        events.write(json.dumps(kw, ensure_ascii=False) + "\n")
        events.flush()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.policy_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.policy_model, torch_dtype=torch.bfloat16, device_map=device)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear",
                      lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    log_event(event="model_loaded", load_s=round(time.time() - t0, 1),
              policy=args.policy_model)

    reward = CompositeReward(Path(args.reward_heads_dir), args.w_fp, args.w_fmt, args.w_cr)
    prompts = load_prompts(Path(args.train_jsonl), args.prompt_limit, args.seed)
    log_event(event="setup", n_prompts=len(prompts), args=vars(args) | {"output_dir": str(args.output_dir)})

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    rng = np.random.RandomState(args.seed)
    running = []

    for step in range(1, args.max_steps + 1):
        step_t0 = time.time()
        batch = [prompts[i] for i in rng.choice(len(prompts), size=args.prompts_per_step, replace=False)]

        # ---- rollout: K samples per prompt (batched generation, adapters ON) ----
        prompt_texts, researchers = [], []
        for p in batch:
            text = tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
            prompt_texts += [text] * args.group_size
            researchers += [p["researcher"]] * args.group_size
        enc = tok(prompt_texts, return_tensors="pt", padding=True).to(device)
        model.eval()
        model.config.use_cache = True
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.gen_temperature, top_p=args.gen_top_p,
                                 pad_token_id=tok.pad_token_id)
        model.config.use_cache = False
        prompt_len = enc["input_ids"].shape[1]
        completions = [tok.decode(g[prompt_len:], skip_special_tokens=True).strip() for g in gen]

        # ---- rewards + group-relative advantages ----
        scores = reward.score(completions, researchers)
        rewards = np.array([s["reward"] for s in scores], dtype=np.float32)
        adv = np.zeros_like(rewards)
        for gi in range(args.prompts_per_step):
            grp = rewards[gi * args.group_size:(gi + 1) * args.group_size]
            adv[gi * args.group_size:(gi + 1) * args.group_size] = (grp - grp.mean()) / (grp.std() + 1e-4)

        # ---- policy update (micro-batched; ref = adapters disabled) ----
        model.train()
        attn = (gen != tok.pad_token_id).long()
        comp_mask = torch.zeros_like(gen)
        comp_mask[:, prompt_len:] = (gen[:, prompt_len:] != tok.pad_token_id).long()
        adv_t = torch.tensor(adv, device=device)

        opt.zero_grad(set_to_none=True)
        total_loss = total_kl = total_tok = 0.0
        n_seq = gen.shape[0]
        for mb in range(0, n_seq, args.micro_batch):
            sl = slice(mb, mb + args.micro_batch)
            ids, am, cm = gen[sl], attn[sl], comp_mask[sl]
            with torch.no_grad(), model.disable_adapter():
                ref_logp, _ = completion_logprobs(model, ids, am, cm)
            pol_logp, mask = completion_logprobs(model, ids, am, cm)
            tok_count = mask.sum().clamp(min=1.0)
            pg = -(adv_t[sl].unsqueeze(1) * pol_logp).sum() / tok_count
            kl = ((pol_logp - ref_logp).sum() / tok_count)
            loss = (pg + args.kl_coef * kl) * (ids.shape[0] / n_seq)
            loss.backward()
            total_loss += float(loss.detach())
            total_kl += float(kl.detach()) * ids.shape[0] / n_seq
            total_tok += float(tok_count)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        comp_means = {k: float(np.mean([s[k] for s in scores])) for k in ("fingerprint", "format", "creativity", "reward")}
        running.append(comp_means["reward"])
        log_event(event="step", step=step, loss=round(total_loss, 4), kl=round(total_kl, 4),
                  gen_tokens=int(total_tok), elapsed_s=round(time.time() - step_t0, 1), **comp_means)
        print(f"[grpo] step {step}/{args.max_steps} R={comp_means['reward']:.3f} "
              f"(fp={comp_means['fingerprint']:.3f} fmt={comp_means['format']:.3f} "
              f"cr={comp_means['creativity']:.3f}) kl={total_kl:.4f} "
              f"{time.time()-step_t0:.0f}s", flush=True)

        if step % args.save_steps == 0 or step == args.max_steps:
            ckpt = args.output_dir / f"adapter-step{step:04d}"
            model.save_pretrained(str(ckpt))
            log_event(event="adapter_saved", step=step, path=str(ckpt))

    first = float(np.mean(running[:5])) if len(running) >= 5 else (running[0] if running else 0.0)
    last = float(np.mean(running[-5:])) if len(running) >= 5 else (running[-1] if running else 0.0)
    summary = {
        "status": "ok", "steps": args.max_steps,
        "reward_first5_mean": round(first, 4), "reward_last5_mean": round(last, 4),
        "reward_improved": last > first,
        "final_adapter": str(args.output_dir / f"adapter-step{args.max_steps:04d}"),
        "weights": {"fp": args.w_fp, "fmt": args.w_fmt, "cr": args.w_cr}, "kl_coef": args.kl_coef,
    }
    (args.output_dir / args.summary_name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
