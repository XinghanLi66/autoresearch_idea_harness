#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import iter_jsonl, write_json
from autoresearch_idea_harness.training_manifest import TARGET_SCHEMA_VERSION, TARGET_XML_TAGS


TAG_OPEN_RE = re.compile(r"<([A-Za-z0-9_]+)>")
FORMULA_PATTERNS = [
    re.compile(r"\bif\b.*\belse\b", re.I | re.S),
    re.compile(r"\bfor\b.*\bin\b.*:", re.I | re.S),
    re.compile(r"\breturn\b|\btorch\.|\bF\.|\bnn\.", re.I),
    re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*"),
    re.compile(r"(sigmoid|softmax|tanh|relu|gelu|mish|loss|gradient|schedule|gate)\s*\(", re.I),
    re.compile(r"[+\-*/^].*[A-Za-z0-9_]", re.S),
]
IMPLEMENTATION_TERMS = [
    "module",
    "layer",
    "loss",
    "optimizer",
    "dataset",
    "pipeline",
    "algorithm",
    "architecture",
    "training",
    "inference",
    "preprocessing",
    "ablation",
    "baseline",
    "metric",
]
NOVELTY_TERMS = [
    "novel",
    "unlike",
    "prior",
    "reference",
    "baseline",
    "limitation",
    "gap",
    "instead",
    "delta",
    "different",
]
RISK_TERMS = [
    "fail",
    "falsify",
    "unstable",
    "confound",
    "negative",
    "control",
    "collapse",
    "overfit",
    "variance",
]
GENERIC_PHRASES = [
    "combine the strengths",
    "combines the strengths",
    "combine the advantages",
    "leverages the benefits",
    "novel framework",
    "comprehensive approach",
    "across various",
    "potentially leading to",
    "expected to improve",
    "without overcomplicating",
]
BASELINE_NAMES = [
    "relu",
    "gelu",
    "swish",
    "silu",
    "mish",
    "kaiming",
    "orthogonal",
    "fixup",
    "label smoothing",
    "focal",
    "polyloss",
    "dropblock",
    "stochastic depth",
]


def parse_xmlish(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for tag in sorted(set(TAG_OPEN_RE.findall(text))):
        match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
        if match:
            fields[tag] = match.group(1).strip()
    return fields


def count_terms(text: str, terms: list[str]) -> int:
    lower = text.lower()
    return sum(1 for term in terms if term in lower)


def count_generic_phrases(text: str) -> int:
    lower = text.lower()
    return sum(1 for phrase in GENERIC_PHRASES if phrase in lower)


def baseline_overlap(text: str) -> list[str]:
    lower = text.lower()
    return [name for name in BASELINE_NAMES if name in lower]


def has_formula_like_detail(text: str) -> bool:
    return any(pattern.search(text) for pattern in FORMULA_PATTERNS)


def extract_proposal_block(text: str) -> str:
    match = re.search(r"<proposal>(.*?)</proposal>", text, re.DOTALL)
    if not match:
        return text.strip()
    return f"<proposal>{match.group(1).strip()}</proposal>"


def target_text(row: dict[str, Any]) -> str:
    primary = str(row.get("target_impl_proposal") or row.get("target_proposal") or "")
    if "<proposal>" in primary and "</proposal>" in primary:
        return extract_proposal_block(primary)
    cot = str(row.get("cot_impl_proposal") or "")
    if "<proposal>" in cot and "</proposal>" in cot:
        return extract_proposal_block(cot)
    return primary or cot


def audit_row(row: dict[str, Any], *, require_prompt_version: str | None = None) -> dict[str, Any]:
    text = target_text(row)
    fields = parse_xmlish(text)
    missing_tags = [tag for tag in TARGET_XML_TAGS if tag not in fields]
    empty_tags = [tag for tag in TARGET_XML_TAGS if tag in fields and not fields[tag].strip()]
    prompt_version = str(row.get("synthesis_prompt_version") or "legacy_or_unknown")

    core = fields.get("core_idea", "")
    gap = fields.get("gap", "")
    implementation = fields.get("implementation_plan", "")
    algorithm = fields.get("algorithm_or_system", "")
    recipe = fields.get("training_or_data_recipe", "")
    evaluation = fields.get("evaluation_plan", "")
    risks = fields.get("risks_and_limitations", "")
    full = "\n".join(fields.get(tag, "") for tag in TARGET_XML_TAGS) or text

    prompt_ok = True
    if require_prompt_version:
        prompt_ok = prompt_version == require_prompt_version or prompt_version.startswith(require_prompt_version + "_")

    criteria: dict[str, dict[str, Any]] = {
        "schema_complete": {
            "passed": not missing_tags and not empty_tags,
            "missing_tags": missing_tags,
            "empty_tags": empty_tags,
        },
        "prompt_version": {
            "passed": prompt_ok,
            "prompt_version": prompt_version,
            "required": require_prompt_version,
        },
    }
    criteria["exact_mechanism"] = {
        "passed": has_formula_like_detail(core + "\n" + algorithm + "\n" + implementation),
        "core_chars": len(core),
        "algorithm_chars": len(algorithm),
    }
    criteria["implementation_recipe"] = {
        "passed": len(implementation + recipe) >= 280 and count_terms(implementation + recipe, IMPLEMENTATION_TERMS) >= 3,
        "term_hits": count_terms(implementation + recipe, IMPLEMENTATION_TERMS),
        "chars": len(implementation + recipe),
    }
    criteria["novelty_delta"] = {
        "passed": len(gap + core) >= 220 and count_terms(gap + core, NOVELTY_TERMS) >= 2,
        "term_hits": count_terms(gap + core, NOVELTY_TERMS),
        "chars": len(gap + core),
    }
    criteria["evaluation_ablations"] = {
        "passed": (
            len(evaluation) >= 140
            and "baseline" in evaluation.lower()
            and ("ablation" in evaluation.lower() or "control" in evaluation.lower())
            and ("metric" in evaluation.lower() or "measure" in evaluation.lower() or "benchmark" in evaluation.lower())
        ),
        "chars": len(evaluation),
    }
    criteria["risk_falsification"] = {
        "passed": len(risks) >= 140 and count_terms(risks, RISK_TERMS) >= 2,
        "term_hits": count_terms(risks, RISK_TERMS),
        "chars": len(risks),
    }
    generic_count = count_generic_phrases(full)
    overlaps = baseline_overlap(full)
    criteria["nontriviality_guard"] = {
        "passed": generic_count <= 2 and not (generic_count >= 2 and len(overlaps) >= 4),
        "generic_phrase_count": generic_count,
        "baseline_name_overlap": overlaps,
    }

    weights = {
        "schema_complete": 20,
        "prompt_version": 10,
        "exact_mechanism": 20,
        "implementation_recipe": 15,
        "novelty_delta": 10,
        "evaluation_ablations": 15,
        "risk_falsification": 5,
        "nontriviality_guard": 5,
    }
    score = sum(weight for key, weight in weights.items() if criteria[key]["passed"])
    hard_failure_keys = ["schema_complete", "exact_mechanism", "implementation_recipe", "nontriviality_guard"]
    if require_prompt_version:
        hard_failure_keys.append("prompt_version")
    hard_failures = [key for key in hard_failure_keys if not criteria[key]["passed"]]
    verdict = "pass" if score >= 80 and not hard_failures else "reject"
    return {
        "arxiv_id": row.get("arxiv_id"),
        "sample_id": row.get("sample_id"),
        "created": row.get("created"),
        "title": row.get("title"),
        "score": score,
        "max_score": sum(weights.values()),
        "verdict": verdict,
        "hard_failures": hard_failures,
        "criteria": criteria,
        "field_lengths": {tag: len(fields.get(tag, "")) for tag in TARGET_XML_TAGS},
        "target_chars": len(text),
        "target_schema_version": row.get("target_schema_version") or TARGET_SCHEMA_VERSION,
        "synthesis_prompt_version": prompt_version,
        "synthesis_model": row.get("synthesis_model"),
        "tex_status": row.get("tex_status"),
    }


def audit_cache(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "tex_targets.accepted.jsonl"
    rejected_path = output_dir / "tex_targets.rejected.jsonl"
    rows_path = output_dir / "rows.jsonl"

    rows: list[dict[str, Any]] = []
    scores: list[int] = []
    verdicts = Counter()
    hard_failures = Counter()
    prompt_versions = Counter()
    accepted = 0
    rejected = 0

    with accepted_path.open("w") as accepted_f, rejected_path.open("w") as rejected_f, rows_path.open("w") as rows_f:
        for idx, row in enumerate(iter_jsonl(input_path), 1):
            if args.limit and idx > args.limit:
                break
            audit = audit_row(row, require_prompt_version=args.require_prompt_version)
            scores.append(int(audit["score"]))
            verdicts[str(audit["verdict"])] += 1
            prompt_versions[str(audit["synthesis_prompt_version"])] += 1
            for failure in audit.get("hard_failures") or []:
                hard_failures[str(failure)] += 1
            rows.append(audit)
            rows_f.write(json.dumps(audit, ensure_ascii=False) + "\n")

            out_row = {**row, "target_impl_proposal": target_text(row), "v3_target_audit": audit}
            if audit["verdict"] == "pass":
                accepted += 1
                accepted_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            else:
                rejected += 1
                rejected_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "kind": "target_cache_audit",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "require_prompt_version": args.require_prompt_version,
        "row_count": len(rows),
        "accepted_count": accepted,
        "rejected_count": rejected,
        "score": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": round(statistics.mean(scores), 3) if scores else None,
            "median": round(statistics.median(scores), 3) if scores else None,
            "p25": sorted(scores)[len(scores) // 4] if scores else None,
            "p75": sorted(scores)[(len(scores) * 3) // 4] if scores else None,
        },
        "verdicts": dict(verdicts),
        "hard_failures": dict(hard_failures),
        "prompt_versions": dict(prompt_versions),
        "files": {
            "accepted": str(accepted_path),
            "rejected": str(rejected_path),
            "rows": str(rows_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    lines = [
        "# V3 Target Cache Audit",
        "",
        f"- Input: `{input_path}`",
        f"- Rows: {summary['row_count']}",
        f"- Accepted / rejected: {accepted} / {rejected}",
        f"- Require prompt version: `{args.require_prompt_version}`",
        f"- Score mean/median: {summary['score']['mean']} / {summary['score']['median']}",
        f"- Verdicts: {summary['verdicts']}",
        f"- Hard failures: {summary['hard_failures']}",
        f"- Prompt versions: {summary['prompt_versions']}",
        "",
        "Accepted rows are written to `tex_targets.accepted.jsonl` and can be passed directly to `collate_v3_sft.py`.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a V3 TeX target cache before SFT collate/training.")
    parser.add_argument("--input", required=True, help="Target cache JSONL, usually tex_targets.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-prompt-version", default=None)
    args = parser.parse_args()
    summary = audit_cache(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["accepted_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
