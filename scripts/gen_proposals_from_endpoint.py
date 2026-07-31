#!/usr/bin/env python3
"""Generate MLS-Bench-Lite proposals from a served (vLLM OpenAI-compatible) checkpoint endpoint.

For each packet in packets.jsonl (30 Lite tasks), call the endpoint's chat/completions with the packet's
[system,user] messages and write {task, proposal} to proposals_<arm>.jsonl. Used to turn a served arm
(base / SFT / RL checkpoint) into the proposals the mlsbench worker then implements.

Usage:
  python scripts/gen_proposals_from_endpoint.py --endpoint http://10.39.6.221:8000/v1 \
    --model proposer-exp09-rl --arm exp09rl --tasks dl-activation-function,ml-clustering-algorithm \
    --out-dir runs/researcher_cot/mls_lite_eval/proposals
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Return the idea only: drop complete <think>...</think> blocks; if an unclosed <think> remains,
    drop from it onward."""
    t = _THINK_RE.sub("", text)
    if "<think>" in t:
        t = t.split("<think>", 1)[0]
    return t.strip()


def is_valid(text: str) -> tuple[bool, str]:
    """A proposal is valid iff: any opened <think> is also CLOSED (not truncated mid-reasoning), and the
    idea (post-<think>) is substantive — has a 'Core idea' anchor and enough content to implement."""
    t = (text or "").strip()
    if "<think>" in t and "</think>" not in t:
        return False, "unclosed <think> (truncated mid-reasoning)"
    idea = strip_think(t)
    if len(idea) < 150:
        return False, f"idea too short ({len(idea)}c after <think>-strip)"
    if "core idea" not in idea.lower():
        return False, "missing 'Core idea:' anchor"
    return True, "ok"


def complete(endpoint: str, model: str, messages: list, max_tokens: int, temperature: float) -> str:
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def complete_text(endpoint: str, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """Base-model path: /completions with a plain text prompt (no chat template)."""
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["text"].strip()


FEWSHOT = (
    "### Example\n"
    "Task: improve the learning-rate schedule for training a small CNN. Editable component: `custom_schedule.py`.\n"
    "Proposal: Replace fixed cosine decay with a curvature-adaptive schedule that shortens the high-LR phase "
    "when the loss landscape sharpens (estimated from a running gradient-variance ratio), spending more steps "
    "at low LR near sharp minima.\n"
    "Core idea: gate the LR decay rate on an online sharpness estimate rather than a fixed step count.\n"
    "Non-trivial crux: sharpness must be estimated cheaply from already-computed gradients, no extra backward pass.\n\n"
    "### Now your task\n")


def as_prompt(messages: list) -> str:
    """Base-model completion prompt: one format exemplar (few-shot) + the task, so the raw base produces a
    real proposal instead of parroting the instruction. Standard/fair base-model elicitation."""
    sysm = next((m["content"] for m in messages if m["role"] == "system"), "")
    usr = next((m["content"] for m in messages if m["role"] == "user"), "")
    return f"{sysm}\n\n{FEWSHOT}{usr}\n\nProposal (end with 'Core idea:' and 'Non-trivial crux:'):\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--packets", default=str(ROOT / "runs/researcher_cot/mls_lite_eval/packets.jsonl"))
    ap.add_argument("--tasks", default=None, help="comma slug subset (default: all in packets)")
    ap.add_argument("--out-dir", default=str(ROOT / "runs/researcher_cot/mls_lite_eval/proposals"))
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="generation cap; must exceed the models' training completion length (~1.3-1.6k tok) "
                         "AND leave room for thinking base models' <think> + the idea (1200 truncated them)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-retries", type=int, default=6,
                    help="regenerate a task until the proposal is valid (closed </think> + real idea); "
                         "keeps the best attempt if none validate")
    ap.add_argument("--api-mode", default="chat", choices=["chat", "completions"],
                    help="chat = instruct/chat models; completions = BASE models (plain text prompt)")
    args = ap.parse_args()

    packets = [json.loads(l) for l in Path(args.packets).open()]
    if args.tasks:
        keep = set(args.tasks.split(","))
        packets = [p for p in packets if p["task"] in keep]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"proposals_{args.arm}.jsonl"
    n = n_valid = 0
    with fp.open("w") as fh:
        for p in packets:
            best, best_len, valid, reason, attempts = "", -1, False, "", 0
            for attempt in range(1, args.max_retries + 1):
                attempts = attempt
                try:
                    if args.api_mode == "completions":
                        text = complete_text(args.endpoint, args.model, as_prompt(p["messages"]),
                                             args.max_tokens, args.temperature)
                    else:
                        text = complete(args.endpoint, args.model, p["messages"],
                                        args.max_tokens, args.temperature)
                except Exception as e:
                    print(f"[warn] {p['task']} attempt {attempt}: {e}", file=sys.stderr); continue
                ok, reason = is_valid(text)
                il = len(strip_think(text))
                if il > best_len:              # keep the attempt with the most idea content
                    best, best_len = text, il
                if ok:
                    valid = True; break
                print(f"  [retry] {args.arm}/{p['task']} attempt {attempt}: {reason}", flush=True)
            if not best:
                print(f"[warn] {p['task']}: no output after {attempts} attempts, skipping", file=sys.stderr); continue
            # store raw generation (audit) + the clean idea that will be injected; injection also strips <think>
            fh.write(json.dumps({"task": p["task"], "arm": args.arm, "proposal": best,
                                 "idea": strip_think(best), "valid": valid, "n_attempts": attempts},
                                ensure_ascii=False) + "\n")
            fh.flush(); n += 1; n_valid += int(valid)
            print(f"  {args.arm}/{p['task']}: {'VALID' if valid else 'INVALID(best-effort)'} "
                  f"idea={best_len}c attempts={attempts}", flush=True)
    print(f"wrote {n} proposals ({n_valid} valid, {n - n_valid} best-effort) -> {fp}")


if __name__ == "__main__":
    main()
