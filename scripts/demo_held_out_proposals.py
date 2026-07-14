#!/usr/bin/env python3
"""Creativity demo: run a trained V3 checkpoint on HELD-OUT arXiv papers.

Unlike generate_v3_checkpoint_proposal.py (which conditions on an MLS *task
packet*), this script conditions on a real paper's reading list + research
question -- exactly the `with_research_question` distribution the model was
trained on -- but using papers that were NOT in the 929-article training set.

Held-out pool = papers that (a) have a cached V3 research question, (b) carry
>= --min-refs references with abstracts, and (c) are absent from the strict929
training index. No LLM API calls: the research question is read from the
prompt-property cache and ref abstracts are compacted locally.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config, write_json, write_jsonl
from autoresearch_idea_harness.prompt_properties import (
    build_v3_condition_prompt,
    load_prompt_property_caches,
    select_refs_for_prompt,
)
from autoresearch_idea_harness.training_manifest import (
    PROPOSAL_FORMAT,
    SYSTEM_PROMPT,
    _filter_refs_for_leakage,
    _v1_full_ref_order,
)

DEFAULT_MODEL_DIR = (
    ROOT / "runs" / "training" / "v3_sft_qwen25_32b"
    / "v3_sft_v1sem_cot_strict929_16k" / "checkpoints" / "phase_000_2025-04" / "final"
)
STRICT929_INDEX = ROOT / "runs" / "v3_quality_cache" / "strict929_v4_sharded" / "merged" / "index.jsonl"
PROPOSAL_TAGS = [
    "title", "problem", "gap", "core_idea", "implementation_plan",
    "algorithm_or_system", "training_or_data_recipe", "evaluation_plan",
    "expected_results", "risks_and_limitations",
]


def load_training_ids() -> set[str]:
    ids: set[str] = set()
    if STRICT929_INDEX.exists():
        with STRICT929_INDEX.open() as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["arxiv_id"])
                except Exception:
                    continue
    return ids


def parse_created(value: Any) -> date:
    """Best-effort parse of a record's `created` field to a date (for leakage filtering)."""
    s = str(value or "")[:10]
    try:
        y, m, d = (int(x) for x in s.split("-"))
        return date(y, m, d)
    except Exception:
        return date(2099, 1, 1)  # permissive: keep all refs if date unknown


def select_held_out(
    dataset_path: Path,
    rq_cache,
    training_ids: set[str],
    *,
    n_papers: int,
    month: str | None,
    min_refs: int,
    explicit_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Scan the dataset jsonl and pick deterministic held-out records.

    Selection: not in training_ids, has cached RQ, has >= min_refs refs, optional
    month filter (e.g. "2505"). Deduplicated by category family for diversity,
    then sorted by arxiv_id for determinism.
    """
    explicit = set(explicit_ids or [])
    candidates: dict[str, dict[str, Any]] = {}
    with dataset_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            aid = rec.get("arxiv_id")
            if not aid or aid in training_ids or aid in candidates:
                continue
            refs = rec.get("refs") or []
            if explicit:
                if aid in explicit:
                    candidates[aid] = rec
                continue
            if len(refs) < min_refs or not rq_cache.get(aid):
                continue
            if month and not str(aid).startswith(month):
                continue
            candidates[aid] = rec

    if explicit:
        return [candidates[a] for a in explicit_ids if a in candidates]

    # Diversity: one per primary category, fall back to fill up to n_papers.
    by_id = dict(sorted(candidates.items()))
    chosen: list[dict[str, Any]] = []
    seen_cat: set[str] = set()
    for aid, rec in by_id.items():
        cats = rec.get("categories") or []
        primary = cats[0] if cats else "?"
        if primary in seen_cat:
            continue
        seen_cat.add(primary)
        chosen.append(rec)
        if len(chosen) >= n_papers:
            break
    if len(chosen) < n_papers:  # fill remainder ignoring category
        for aid, rec in by_id.items():
            if rec not in chosen:
                chosen.append(rec)
            if len(chosen) >= n_papers:
                break
    return chosen[:n_papers]


def build_messages_for_record(rec: dict[str, Any], caches) -> tuple[list[dict[str, str]], dict[str, Any]]:
    aid = rec["arxiv_id"]
    refs = list(rec.get("refs") or [])
    target_date = parse_created(rec.get("created"))
    kept, guard = _filter_refs_for_leakage(refs, target_date)
    ordered = _v1_full_ref_order(kept, max_refs=40, seed=42)
    trimmed, sel_meta = select_refs_for_prompt(aid, ordered, caches, top_k=40, mode="all")
    rq_text = caches["research_question"].get(aid)
    user = build_v3_condition_prompt(trimmed, rq_text, PROPOSAL_FORMAT)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    meta = {
        "arxiv_id": aid,
        "title": rec.get("title"),
        "categories": rec.get("categories"),
        "created": rec.get("created"),
        "n_refs_input": len(refs),
        "n_refs_used": len(trimmed),
        "leakage_guard": guard,
        "rq_present": bool(rq_text),
        "rq_chars": len(rq_text or ""),
    }
    return messages, meta


def generate(model_dir: Path, batch: list[tuple[list[dict[str, str]], dict[str, Any]]], args) -> list[dict[str, Any]]:
    """Load the model once and generate for every paper. Left-truncate the prompt
    so the research question + XML schema (at the tail) always survive."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"  # keep the tail (RQ + schema)

    load_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "auto",
                                   "trust_remote_code": args.trust_remote_code}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    model.eval()
    load_s = time.time() - t0
    print(f"[demo] model loaded in {load_s:.1f}s on {model.device}", flush=True)

    results: list[dict[str, Any]] = []
    for i, (messages, meta) in enumerate(batch, 1):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p, pad_token_id=tok.eos_token_id,
            )
        gen_s = time.time() - t1
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        rec = {
            **meta,
            "proposal": text,
            "prompt_tokens": int(enc["input_ids"].shape[1]),
            "new_tokens": int(new_tokens.shape[0]),
            "generation_elapsed_s": round(gen_s, 2),
            "schema_tags_present": sorted(t for t in PROPOSAL_TAGS if f"<{t}>" in text),
        }
        rec["schema_complete"] = len(rec["schema_tags_present"]) == len(PROPOSAL_TAGS)
        results.append(rec)
        print(f"[demo] {i}/{len(batch)} {meta['arxiv_id']}: {rec['new_tokens']} tok in "
              f"{gen_s:.1f}s, schema_complete={rec['schema_complete']}", flush=True)
    return results


def write_report(out_dir: Path, results: list[dict[str, Any]], model_dir: Path) -> None:
    lines = [
        f"# Held-out creativity demo — `{model_dir.parent.parent.parent.name}`",
        "",
        f"Model: `{model_dir}`",
        f"Papers: {len(results)} (held-out: not in strict929 training set)",
        "",
        "| arXiv | category | refs used | prompt tok | new tok | schema complete |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        cat = (r.get("categories") or ["?"])[0]
        lines.append(
            f"| {r['arxiv_id']} | {cat} | {r['n_refs_used']} | {r['prompt_tokens']} | "
            f"{r['new_tokens']} | {'✅' if r['schema_complete'] else '⚠️ ' + str(len(r['schema_tags_present']))+'/10'} |"
        )
    lines.append("")
    for r in results:
        lines += [
            f"## {r['arxiv_id']} — {r.get('title') or ''}",
            f"*categories: {r.get('categories')} · created: {r.get('created')} · "
            f"RQ chars: {r.get('rq_chars')} · dropped future refs: "
            f"{r.get('leakage_guard',{}).get('dropped_future_ref_count')}*",
            "",
            "```xml",
            r["proposal"],
            "```",
            "",
        ]
    (out_dir / "demo_report.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="jsonl of candidate records (need arxiv_id, refs, created); "
                         "defaults to <dataset_dir>/train.jsonl from the config")
    ap.add_argument("--n-papers", type=int, default=5)
    ap.add_argument("--month", default="2505", help="arxiv month prefix filter, e.g. 2505; empty for any")
    ap.add_argument("--min-refs", type=int, default=15)
    ap.add_argument("--arxiv-ids", nargs="*", default=None, help="explicit held-out ids (overrides selection)")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "demo_held_out")
    ap.add_argument("--max-input-tokens", type=int, default=12288)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build prompts only, do not load the model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset is None:
        args.dataset = Path(cfg["dataset_dir"]) / "train.jsonl"
    caches = load_prompt_property_caches(cfg)
    training_ids = load_training_ids()
    print(f"[demo] training_ids(strict929)={len(training_ids)}", flush=True)

    records = select_held_out(
        args.dataset, caches["research_question"], training_ids,
        n_papers=args.n_papers, month=(args.month or None),
        min_refs=args.min_refs, explicit_ids=args.arxiv_ids,
    )
    if not records:
        print("[demo] no held-out records matched the filters", file=sys.stderr)
        sys.exit(1)
    chosen_ids = [r["arxiv_id"] for r in records]
    assert all(a not in training_ids for a in chosen_ids), "held-out invariant violated"
    print(f"[demo] selected held-out papers: {chosen_ids}", flush=True)

    batch = [build_messages_for_record(r, caches) for r in records]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for messages, meta in batch:
        pdir = out_dir / meta["arxiv_id"]
        pdir.mkdir(exist_ok=True)
        write_json(pdir / "messages.json", messages)
        (pdir / "prompt.txt").write_text(
            "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages))

    if args.dry_run:
        write_jsonl(out_dir / "selection.jsonl", [m for _, m in batch])
        print(json.dumps({"status": "dry_run", "selected": chosen_ids,
                          "output_dir": str(out_dir)}, ensure_ascii=False))
        return

    results = generate(args.model_dir, batch, args)
    for r in results:
        (out_dir / r["arxiv_id"] / "proposal.txt").write_text(r["proposal"])
    write_jsonl(out_dir / "results.jsonl", results)
    write_report(out_dir, results, args.model_dir)
    n_ok = sum(1 for r in results if r["schema_complete"])
    print(json.dumps({"status": "ok", "n_papers": len(results),
                      "schema_complete": f"{n_ok}/{len(results)}",
                      "output_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
