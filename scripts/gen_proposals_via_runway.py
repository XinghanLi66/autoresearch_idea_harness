#!/usr/bin/env python3
"""Generate MLS-Bench-Lite (or MAB) proposals from an API model via the internal RunwayClient.

For frontier-proposer arms that are NOT vLLM-served (fable5=claude-fable-5, gpt55=gpt-5-5), reach them
through RunwayClient.complete() instead of an OpenAI /v1 endpoint. Reuses the packets.jsonl schema, the
validity loop, and the proposals_<arm>.jsonl output schema from gen_proposals_from_endpoint.py so the arm
is a drop-in downstream (dispatch/report unchanged).

Usage:
  python scripts/gen_proposals_via_runway.py --arm fable5 \
    --endpoint google_anthropic --model claude-fable-5 --key-env RUNWAY_FABLE5_API_KEY
  python scripts/gen_proposals_via_runway.py --arm gpt55 \
    --endpoint chat --model gpt-5-5 --key-env RUNWAY_GPT55_API_KEY
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
# reuse the exact validity/think-strip logic used by the vLLM-endpoint gen path
from gen_proposals_from_endpoint import strip_think, is_valid  # noqa: E402
from autoresearch_idea_harness.io import load_config  # noqa: E402
from autoresearch_idea_harness.runway_client import RunwayClient  # noqa: E402

CFG = load_config(str(ROOT / "configs/default.yaml"))
# load .env so MaaS url/key env vars are visible
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def maas_complete(url: str, key: str, model: str, messages: list, max_tokens: int) -> str:
    """POST directly to a full MaaS OpenAI /chat/completions URL (api-key header; GPT-5 needs
    max_completion_tokens + default temperature)."""
    body = json.dumps({"model": model, "messages": messages,
                       "max_completion_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--endpoint", required=True,
                    help="google_anthropic (Anthropic-family) | chat (OpenAI runway) | maas (full MaaS URL)")
    ap.add_argument("--model", required=True, help="e.g. claude-fable-5 | gpt-5-5 | gpt-5.5")
    ap.add_argument("--key-env", default=None, help="env var with the API key (runway endpoints)")
    ap.add_argument("--maas-url-env", default=None, help="env var with the full MaaS chat/completions URL")
    ap.add_argument("--maas-key-env", default=None, help="env var with the MaaS api-key")
    # opus-4.8 (or any) fallback for tasks the primary model refuses (e.g. 风控 → empty responses)
    ap.add_argument("--fallback-endpoint", default=None, help="e.g. google_anthropic")
    ap.add_argument("--fallback-model", default=None, help="e.g. claude-opus-4-8")
    ap.add_argument("--fallback-key-env", default=None, help="e.g. RUNWAY_OPUS48_API_KEY")
    ap.add_argument("--append", action="store_true", help="append to proposals_<arm>.jsonl instead of overwrite")
    ap.add_argument("--packets", default=str(ROOT / "runs/researcher_cot/mls_lite_eval/packets.jsonl"))
    ap.add_argument("--tasks", default=None, help="comma slug subset (default: all in packets)")
    ap.add_argument("--out-dir", default=str(ROOT / "runs/researcher_cot/mls_lite_eval/proposals"))
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omit for frontier models (fable-5/gpt-5 deprecate temperature → send None)")
    ap.add_argument("--max-retries", type=int, default=6)
    args = ap.parse_args()

    client = RunwayClient(CFG, key_env=args.key_env) if args.endpoint != "maas" and args.key_env else None
    maas_url = os.environ.get(args.maas_url_env or "", "")
    maas_key = os.environ.get(args.maas_key_env or "", "")
    if args.endpoint == "maas" and not (maas_url and maas_key):
        sys.exit(f"--endpoint maas needs --maas-url-env + --maas-key-env resolvable in .env")
    fb_client = (RunwayClient(CFG, key_env=args.fallback_key_env)
                 if args.fallback_model and args.fallback_endpoint != "maas" and args.fallback_key_env else None)

    def gen_one(messages, endpoint, model, cli):
        """Run the retry loop for one (model, task); return (best_text, best_len, valid, attempts)."""
        best, best_len, valid, attempts = "", -1, False, 0
        for attempt in range(1, args.max_retries + 1):
            attempts = attempt
            try:
                if endpoint == "maas":
                    text = maas_complete(maas_url, maas_key, model, messages, args.max_tokens)
                else:
                    text = cli.complete(endpoint=endpoint, model=model, messages=messages,
                                        temperature=args.temperature, max_tokens=args.max_tokens,
                                        stream=False).text.strip()
                ok, reason = is_valid(text)
                il = len(strip_think(text))
                if il > best_len:
                    best, best_len = text, il
                if ok:
                    return best, best_len, True, attempts
                print(f"  [retry] {args.arm}/{model} attempt {attempt}: {reason}", flush=True)
            except Exception as e:
                print(f"[warn] {args.arm}/{model} attempt {attempt}: {e}", file=sys.stderr)
        return best, best_len, valid, attempts

    packets = [json.loads(l) for l in Path(args.packets).open()]
    if args.tasks:
        keep = set(args.tasks.split(","))
        packets = [p for p in packets if p["task"] in keep]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"proposals_{args.arm}.jsonl"
    n = n_valid = n_fb = 0
    with fp.open("a" if args.append else "w") as fh:
        for p in packets:
            best, best_len, valid, attempts = gen_one(p["messages"], args.endpoint, args.model, client)
            gen_model = args.model
            # fallback (e.g. opus-4.8) when the primary is empty/refused (风控) or invalid
            if (not valid or best_len < 150) and args.fallback_model:
                print(f"  [fallback] {args.arm}/{p['task']}: primary '{args.model}' failed "
                      f"(valid={valid}, len={best_len}) → trying '{args.fallback_model}'", flush=True)
                fbest, fbest_len, fvalid, fatt = gen_one(p["messages"], args.fallback_endpoint,
                                                         args.fallback_model, fb_client)
                if fbest_len > best_len:
                    best, best_len, valid, attempts, gen_model = fbest, fbest_len, fvalid, fatt, args.fallback_model
                    n_fb += 1
            if not best:
                print(f"[warn] {p['task']}: no output (primary+fallback), skipping", file=sys.stderr); continue
            fh.write(json.dumps({"task": p["task"], "arm": args.arm, "proposal": best,
                                 "idea": strip_think(best), "valid": valid, "n_attempts": attempts,
                                 "gen_model": gen_model}, ensure_ascii=False) + "\n")
            fh.flush(); n += 1; n_valid += int(valid)
            print(f"  {args.arm}/{p['task']}: {'VALID' if valid else 'INVALID(best-effort)'} "
                  f"idea={best_len}c via={gen_model}", flush=True)
    print(f"wrote {n} proposals ({n_valid} valid, {n_fb} via fallback) -> {fp}")


if __name__ == "__main__":
    main()
