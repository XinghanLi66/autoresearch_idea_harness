#!/usr/bin/env python3
"""Generate MLS task-packet proposals from three arms on one QS pod (closed-loop eval, step 1).

Arms: base (Qwen2.5-32B-Instruct), sft (full-SFT checkpoint-101), rl (checkpoint-101 + GRPO
adapter merged). Protocol matches the recorded V2.5 eval: temperature 0.7, top_p 0.95,
max_new_tokens 1600, one proposal per task from the prepared task-packet messages.

Writes proposals_<arm>.jsonl to --out-dir (on /mnt/3fs).
"""
from __future__ import annotations

import argparse
import gc
import importlib.machinery
import importlib.util
import json
import sys
import time
import types
from pathlib import Path


def _patch_transformers_modeling_layers() -> None:
    if importlib.util.find_spec("transformers.modeling_layers") is not None:
        return
    m = types.ModuleType("transformers.modeling_layers")
    m.__spec__ = importlib.machinery.ModuleSpec("transformers.modeling_layers", loader=None)

    class GradientCheckpointingLayer:
        pass

    m.GradientCheckpointingLayer = GradientCheckpointingLayer
    sys.modules["transformers.modeling_layers"] = m


_patch_transformers_modeling_layers()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def gen_arm(arm: str, model_dir: str, adapter: str | None, packets: list[dict],
            out_path: Path, temperature: float, top_p: float, max_new: int) -> None:
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        print(f"[gen:{arm}] merged adapter {adapter}", flush=True)
    model.eval()
    print(f"[gen:{arm}] loaded in {time.time()-t0:.0f}s", flush=True)

    with out_path.open("w") as f:
        for i, p in enumerate(packets, 1):
            prompt = tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
            enc = {k: v.to(model.device) for k, v in tok(prompt, return_tensors="pt",
                                                         truncation=True, max_length=8192).items()}
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                     temperature=temperature, top_p=top_p,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rec = {"arm": arm, "task": p["task"], "subtask": p.get("subtask"),
                   "pass_metric": p.get("pass_metric"), "proposal": text,
                   "gen_s": round(time.time() - t1, 1), "words": len(text.split())}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[gen:{arm}] {i}/{len(packets)} {p['task']}: {rec['words']}w in {rec['gen_s']}s", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packets", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sft-model", required=True)
    ap.add_argument("--rl-adapter", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="base,sft,rl")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    args = ap.parse_args()

    packets = [json.loads(l) for l in open(args.packets)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    print(f"[gen] {len(packets)} packets, arms={arms}", flush=True)

    specs = {
        "base": (args.base_model, None),
        "sft": (args.sft_model, None),
        "rl": (args.sft_model, args.rl_adapter),
    }
    for arm in arms:
        model_dir, adapter = specs[arm]
        gen_arm(arm, model_dir, adapter, packets, out / f"proposals_{arm}.jsonl",
                args.temperature, args.top_p, args.max_new_tokens)

    summary = {"status": "ok", "arms": arms, "packets": len(packets),
               "files": [str(out / f"proposals_{a}.jsonl") for a in arms]}
    (out / "gen_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
