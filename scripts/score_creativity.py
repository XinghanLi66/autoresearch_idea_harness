#!/usr/bin/env python3
"""Intrinsic creativity scorer for generated research proposals / CoTs.

Scores each record in a JSONL file with claude-opus-4-8 on a 5-dimension rubric:
  novelty, non_triviality, clarity_of_core, how_arrived, feasibility

Each dimension is scored 1-5 plus a short justification and an overall verdict.

Supports two JSONL formats found in this repo:
  - recent_demo format: records have generations[].cot
  - demo_7b format:     records have generated_cot

Usage:
  python scripts/score_creativity.py \\
      --input runs/researcher_cot/demo_7b_recent/recent_demo.jsonl \\
      --output runs/creativity_eval/recent_demo_creativity.jsonl \\
      --concurrency 4

  python scripts/score_creativity.py \\
      --input runs/researcher_cot/demo_7b/demo.jsonl \\
      --output runs/creativity_eval/demo_7b_creativity.jsonl

  # Held-out proposal results (proposal field):
  python scripts/score_creativity.py \\
      --input runs/demo_held_out/results.jsonl \\
      --text-field proposal \\
      --output runs/creativity_eval/held_out_creativity.jsonl

Options:
  --text-field       Field name containing the text to score. Auto-detected if omitted.
                     For nested paths like generations[].cot, use 'cot' here (the script
                     expands each generation into its own scoring record).
  --cache-dir        Directory for per-record JSON score cache (default: runs/creativity_eval/cache)
  --concurrency      Parallel judge calls (default: 4)
  --model            Model to use for judging (default: claude-opus-4-8)
  --key-env          Env var with the Runway API key (default: RUNWAY_OPUS48_API_KEY)
  --endpoint         Runway endpoint (default: google_anthropic)
  --max-judge-tokens Max tokens for judge response (default: 800)
  --no-cache         Disable cache lookup (re-score everything)
  --dry-run          Print first record's prompt and exit without API calls
  --config           Path to default.yaml (default: configs/default.yaml)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import statistics
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config  # noqa: E402
from autoresearch_idea_harness.runway_client import RunwayClient  # noqa: E402

# ---------------------------------------------------------------------------
# Rubric prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are a STRICT and CONCISE grader of AI-generated research ideas and chain-of-thoughts (CoTs).
Score the given text on five dimensions, each 1-5 (5 = best):

- novelty (1-5): Is the core idea genuinely new relative to the stated context/prior work?
  5 = clearly non-standard, surprising; 1 = textbook rehash.

- non_triviality (1-5): Is the main technical claim non-obvious? Would a competent practitioner
  consider it a real insight rather than something anyone would think of first?
  5 = requires insight/leap; 1 = immediately obvious baseline.

- clarity_of_core (1-5): Is the central idea stated so precisely that a reader can implement it
  without re-reading the paper? Is there a clear mechanism, not just a vague direction?
  5 = crystal-clear mechanism; 1 = vague hand-wave.

- how_arrived (1-5): Does the text explain the reasoning PATH to the idea — what observation,
  analogy, or failure mode led there? Or does the idea appear ex nihilo?
  5 = fully traces the creative path; 1 = idea drops from nowhere.

- feasibility (1-5): Is the proposal technically feasible with existing tools/compute?
  Is the scope calibrated to what can actually be implemented?
  5 = clearly executable, realistic scope; 1 = requires unsolved breakthroughs.

Output STRICT JSON only — no prose outside the JSON:
{
  "novelty": <int 1-5>,
  "non_triviality": <int 1-5>,
  "clarity_of_core": <int 1-5>,
  "how_arrived": <int 1-5>,
  "feasibility": <int 1-5>,
  "justification": "<one or two sentences covering the key strength and weakness>",
  "verdict": "<high_creativity|moderate_creativity|low_creativity>"
}

verdict: "high_creativity" if novelty + non_triviality >= 8 and clarity_of_core >= 3;
         "low_creativity"  if novelty + non_triviality <= 4;
         "moderate_creativity" otherwise.\
"""

DIMENSIONS = ["novelty", "non_triviality", "clarity_of_core", "how_arrived", "feasibility"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env(path: Path) -> None:
    """Load a .env file into os.environ (simple KEY=VALUE parser)."""
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _cache_key(text: str, model: str, endpoint: str) -> str:
    """Stable hash for cache filename."""
    payload = f"{model}|{endpoint}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def parse_json_obj(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction: try raw parse, then strip markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # strip ```json ... ``` or ``` ... ```
    if "```" in text:
        start = text.find("{", text.find("```"))
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    # last resort: find first { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from judge response: {text[:300]}")


# ---------------------------------------------------------------------------
# Record extraction — handle both JSONL formats
# ---------------------------------------------------------------------------

def extract_scoring_records(raw: dict[str, Any], text_field: str | None) -> list[dict[str, Any]]:
    """Return one or more flat records with keys: text, record_id, context, source_rec.

    Handles:
    - recent_demo format: generations[].cot  (2 per record, one per persona)
    - demo_7b format:     generated_cot
    - explicit text_field override
    - held-out results:   proposal
    - generic fallback:   any top-level string field
    """
    results = []

    def make_rec(text: str, rec_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
        return {"text": text, "record_id": rec_id, "context": ctx}

    # --- explicit override ---
    if text_field:
        # Try generations[].text_field (nested list)
        gens = raw.get("generations")
        if gens and isinstance(gens, list):
            for g in gens:
                txt = g.get(text_field, "")
                if txt:
                    ctx = {
                        "arxiv_id": raw.get("arxiv_id", ""),
                        "title": raw.get("title", ""),
                        "role": g.get("role", ""),
                        "researcher": g.get("researcher", ""),
                        "setup": raw.get("setup", "")[:300],
                    }
                    rec_id = f"{raw.get('arxiv_id','?')}_{g.get('role',g.get('researcher',''))}"
                    results.append(make_rec(txt, rec_id, ctx))
            if results:
                return results
        # Direct field
        txt = raw.get(text_field, "")
        if txt:
            ctx = {k: raw.get(k, "") for k in ("researcher", "case_title", "arxiv_id", "title")}
            rec_id = raw.get("arxiv_id") or raw.get("researcher") or str(id(raw))
            results.append(make_rec(txt, rec_id, ctx))
        return results

    # --- auto-detect ---

    # 1. recent_demo: generations[].cot
    gens = raw.get("generations")
    if gens and isinstance(gens, list) and all(isinstance(g, dict) for g in gens):
        for g in gens:
            txt = g.get("cot", "")
            if not txt:
                continue
            ctx = {
                "arxiv_id": raw.get("arxiv_id", ""),
                "title": raw.get("title", ""),
                "role": g.get("role", ""),
                "researcher": g.get("researcher", ""),
                "setup": (raw.get("setup") or "")[:300],
            }
            rec_id = f"{raw.get('arxiv_id','?')}_{g.get('role',g.get('researcher','?'))}"
            results.append(make_rec(txt, rec_id, ctx))
        if results:
            return results

    # 2. demo_7b: generated_cot
    if "generated_cot" in raw:
        ctx = {
            "researcher": raw.get("researcher", ""),
            "case_title": raw.get("case_title", ""),
            "user": (raw.get("user") or "")[:300],
        }
        rec_id = raw.get("researcher") or str(id(raw))
        return [make_rec(raw["generated_cot"], rec_id, ctx)]

    # 3. held-out / demo_held_out: proposal
    if "proposal" in raw:
        ctx = {
            "arxiv_id": raw.get("arxiv_id", ""),
            "title": raw.get("title", ""),
        }
        rec_id = raw.get("arxiv_id") or str(id(raw))
        return [make_rec(raw["proposal"], rec_id, ctx)]

    # 4. generic: first non-empty string field
    for fld in ("cot", "text", "content", "output"):
        if fld in raw and isinstance(raw[fld], str) and raw[fld]:
            ctx = {k: raw.get(k, "") for k in ("researcher", "arxiv_id", "title")}
            return [make_rec(raw[fld], str(id(raw)), ctx)]

    return []


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class CreativityScorer:
    def __init__(
        self,
        cfg: dict[str, Any],
        key_env: str,
        model: str,
        endpoint: str,
        max_judge_tokens: int,
        cache_dir: Path,
        use_cache: bool,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.max_judge_tokens = max_judge_tokens
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self._cfg = cfg
        self._key_env = key_env
        self._tls = threading.local()

    def _client(self) -> RunwayClient:
        if not hasattr(self._tls, "c"):
            self._tls.c = RunwayClient(self._cfg, key_env=self._key_env)
        return self._tls.c

    def score(self, rec: dict[str, Any]) -> dict[str, Any]:
        """Score one extracted record. Returns the record augmented with 'score' dict or 'score_error'."""
        text = rec["text"]
        ctx = rec.get("context", {})

        # Build a context preamble so the judge understands what this text is
        ctx_lines = []
        if ctx.get("title"):
            ctx_lines.append(f"Paper/task: {ctx['title']}")
        if ctx.get("arxiv_id"):
            ctx_lines.append(f"arXiv: {ctx['arxiv_id']}")
        if ctx.get("researcher"):
            ctx_lines.append(f"Researcher persona: {ctx['researcher']}")
        if ctx.get("case_title"):
            ctx_lines.append(f"Case: {ctx['case_title']}")
        if ctx.get("role"):
            ctx_lines.append(f"Role: {ctx['role']}")
        if ctx.get("setup"):
            ctx_lines.append(f"Research setup (excerpt): {ctx['setup']}")
        if ctx.get("user"):
            ctx_lines.append(f"Research prompt (excerpt): {ctx['user']}")
        context_block = "\n".join(ctx_lines) if ctx_lines else ""

        user_msg = (
            (f"Context:\n{context_block}\n\n" if context_block else "")
            + f"Text to score:\n{text}\n\nScore as strict JSON."
        )

        # Cache
        ck = _cache_key(user_msg, self.model, self.endpoint)
        cache_path = self.cache_dir / f"{ck}.json"
        if self.use_cache and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                # Derive verdict if missing (handles old cache entries)
                if not cached.get("verdict"):
                    nov = cached.get("novelty") or 0
                    nont = cached.get("non_triviality") or 0
                    coc = cached.get("clarity_of_core") or 0
                    if nov + nont >= 8 and coc >= 3:
                        cached["verdict"] = "high_creativity"
                    elif nov + nont <= 4:
                        cached["verdict"] = "low_creativity"
                    else:
                        cached["verdict"] = "moderate_creativity"
                return {**rec, "score": cached, "cache_hit": True}
            except Exception:
                pass

        try:
            result = self._client().complete(
                endpoint=self.endpoint,
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=None,  # MUST be None for opus-4-8
                max_tokens=self.max_judge_tokens,
                stream=False,
            )
            parsed = parse_json_obj(result.text)
        except Exception as e:
            print(f"[score_creativity] ERROR {rec['record_id']}: {e}", flush=True)
            return {**rec, "score_error": str(e), "cache_hit": False}

        # Validate dimensions
        for dim in DIMENSIONS:
            if dim not in parsed:
                parsed[dim] = None
            elif parsed[dim] is not None:
                try:
                    parsed[dim] = int(parsed[dim])
                except (ValueError, TypeError):
                    parsed[dim] = None

        # Derive verdict from scores if missing/null (judge sometimes omits it)
        if not parsed.get("verdict"):
            nov = parsed.get("novelty") or 0
            nont = parsed.get("non_triviality") or 0
            coc = parsed.get("clarity_of_core") or 0
            if nov + nont >= 8 and coc >= 3:
                parsed["verdict"] = "high_creativity"
            elif nov + nont <= 4:
                parsed["verdict"] = "low_creativity"
            else:
                parsed["verdict"] = "moderate_creativity"

        if self.use_cache:
            cache_path.write_text(json.dumps(parsed, ensure_ascii=False))

        return {**rec, "score": parsed, "cache_hit": False}


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def compute_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/median per dimension plus verdict distribution."""
    ok = [r for r in scored if "score" in r and r["score"]]
    err = len(scored) - len(ok)

    per_dim: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    verdicts: dict[str, int] = {}
    for r in ok:
        s = r["score"]
        for d in DIMENSIONS:
            v = s.get(d)
            if v is not None:
                per_dim[d].append(float(v))
        verd = s.get("verdict", "unknown")
        verdicts[verd] = verdicts.get(verd, 0) + 1

    stats: dict[str, Any] = {}
    for d in DIMENSIONS:
        vals = per_dim[d]
        if vals:
            stats[d] = {
                "mean": round(statistics.mean(vals), 3),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
        else:
            stats[d] = {"mean": None, "median": None, "min": None, "max": None, "n": 0}

    # Overall creativity: average of novelty + non_triviality + clarity_of_core
    key_dims = ["novelty", "non_triviality", "clarity_of_core"]
    key_vals: list[float] = []
    for r in ok:
        s = r["score"]
        vs = [s.get(d) for d in key_dims if s.get(d) is not None]
        if vs:
            key_vals.append(sum(vs) / len(vs))

    summary = {
        "n_records": len(scored),
        "n_scored": len(ok),
        "n_errors": err,
        "n_cache_hits": sum(1 for r in scored if r.get("cache_hit")),
        "overall_key_creativity_mean": round(statistics.mean(key_vals), 3) if key_vals else None,
        "verdict_distribution": verdicts,
        "per_dimension": stats,
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Intrinsic creativity scorer for research proposals/CoTs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input", "-i", required=True, type=Path,
                    help="Input JSONL file with generated proposals/CoTs")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Output JSONL with scores appended (default: <input_stem>_creativity.jsonl in runs/creativity_eval/)")
    ap.add_argument("--text-field", default=None,
                    help="Field name for the text to score. Auto-detected if omitted.")
    ap.add_argument("--cache-dir", type=Path,
                    default=ROOT / "runs" / "creativity_eval" / "cache",
                    help="Per-record cache dir (default: runs/creativity_eval/cache)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Parallel API calls (default: 4)")
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="Judge model (default: claude-opus-4-8)")
    ap.add_argument("--key-env", default="RUNWAY_OPUS48_API_KEY",
                    help="Env var with Runway API key (default: RUNWAY_OPUS48_API_KEY)")
    ap.add_argument("--endpoint", default="google_anthropic",
                    help="Runway endpoint (default: google_anthropic)")
    ap.add_argument("--max-judge-tokens", type=int, default=800,
                    help="Max tokens for judge response (default: 800)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Disable cache lookup (re-score everything)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print first record's prompt and exit without API calls")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"),
                    help="Path to default.yaml")
    args = ap.parse_args()

    # Load .env
    load_env(ROOT / ".env")

    cfg = load_config(args.config)

    # Resolve output path
    if args.output is None:
        out_dir = ROOT / "runs" / "creativity_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.input.stem
        args.output = out_dir / f"{stem}_creativity.jsonl"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    # Cache dir
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Load input
    raw_recs: list[dict[str, Any]] = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_recs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[score_creativity] WARN: skipping malformed line: {e}", flush=True)

    print(f"[score_creativity] loaded {len(raw_recs)} records from {args.input}", flush=True)

    # Extract scoring records (may expand 1 raw -> N scoring recs for generations[])
    scoring_recs: list[dict[str, Any]] = []
    for raw in raw_recs:
        extracted = extract_scoring_records(raw, args.text_field)
        if not extracted:
            print(f"[score_creativity] WARN: no text extracted from record: {list(raw.keys())}", flush=True)
        scoring_recs.extend(extracted)

    print(f"[score_creativity] {len(scoring_recs)} scoring records extracted", flush=True)

    if not scoring_recs:
        print("[score_creativity] ERROR: no scoring records — check --text-field or input format", file=sys.stderr)
        sys.exit(1)

    # Dry-run: show prompt and exit
    if args.dry_run:
        rec = scoring_recs[0]
        ctx = rec.get("context", {})
        ctx_lines = []
        for k, label in [("title", "Paper/task"), ("arxiv_id", "arXiv"), ("researcher", "Researcher persona"),
                          ("case_title", "Case"), ("role", "Role"), ("setup", "Research setup (excerpt)"),
                          ("user", "Research prompt (excerpt)")]:
            if ctx.get(k):
                ctx_lines.append(f"{label}: {ctx[k]}")
        user_msg = (
            (f"Context:\n" + "\n".join(ctx_lines) + "\n\n" if ctx_lines else "")
            + f"Text to score:\n{rec['text'][:500]}...\n\nScore as strict JSON."
        )
        print("=== DRY RUN ===")
        print("SYSTEM:", JUDGE_SYSTEM[:300], "...")
        print("\nUSER:", user_msg[:600], "...")
        print(f"\n[dry_run] Would score {len(scoring_recs)} records with {args.model} via {args.endpoint}")
        return

    # Score
    scorer = CreativityScorer(
        cfg=cfg,
        key_env=args.key_env,
        model=args.model,
        endpoint=args.endpoint,
        max_judge_tokens=args.max_judge_tokens,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )

    scored: list[dict[str, Any]] = [None] * len(scoring_recs)  # type: ignore[list-item]
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(scorer.score, rec): i for i, rec in enumerate(scoring_recs)}
        done = 0
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                scored[i] = fut.result()
            except Exception as e:
                scored[i] = {**scoring_recs[i], "score_error": str(e), "cache_hit": False}
            done += 1
            rec = scored[i]
            s = rec.get("score") or {}
            dims_str = " ".join(f"{d[0]}={s.get(d,'?')}" for d in DIMENSIONS) if s else f"ERROR: {rec.get('score_error','?')[:60]}"
            print(f"[score_creativity] {done}/{len(scoring_recs)} {rec['record_id']}: {dims_str}", flush=True)

    # Write output JSONL
    with args.output.open("w") as f:
        for rec in scored:
            # Strip raw text from output to keep file compact (keep record_id + context + score)
            out_rec = {k: v for k, v in rec.items() if k != "text"}
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    print(f"[score_creativity] wrote {len(scored)} records to {args.output}", flush=True)

    # Summary
    summary = compute_summary(scored)
    summary_path = args.output.parent / (args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # Print summary
    print("\n=== CREATIVITY SCORE SUMMARY ===")
    print(f"Input: {args.input} ({summary['n_records']} records -> {summary['n_scored']} scored, {summary['n_errors']} errors)")
    print(f"Verdict distribution: {summary['verdict_distribution']}")
    print(f"Overall key-creativity mean (novelty+non_triviality+clarity_of_core avg): {summary['overall_key_creativity_mean']}")
    print("\nPer-dimension stats:")
    for dim in DIMENSIONS:
        st = summary["per_dimension"][dim]
        if st["mean"] is not None:
            print(f"  {dim:20s}: mean={st['mean']:.2f}  median={st['median']:.1f}  "
                  f"min={st['min']:.0f}  max={st['max']:.0f}  n={st['n']}")
        else:
            print(f"  {dim:20s}: no data")
    print(f"\nSummary saved to: {summary_path}")
    print(f"Scored JSONL:      {args.output}")


if __name__ == "__main__":
    main()
