#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import write_json
from autoresearch_idea_harness.training_manifest import TARGET_XML_TAGS


TAG_OPEN_RE = re.compile(r"<([A-Za-z0-9_]+)>")
FORMULA_PATTERNS = [
    re.compile(r"\bif\b.*\belse\b", re.I | re.S),
    re.compile(r"\btorch\.|\bF\.|\bnn\.|\breturn\b"),
    re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*"),
    re.compile(r"(sigmoid|tanh|softplus|gelu|relu|mish|exp|log|sqrt)\s*\(", re.I),
    re.compile(r"[+\-*/^].*[A-Za-z0-9_]", re.S),
]
RAW_IMPLEMENTATION_PATTERNS = [
    re.compile(r"\b(module|architecture|layer|loss|optimizer|dataset|pipeline|algorithm|model|encoder|decoder)\b", re.I),
    re.compile(r"\b(train|fine-?tune|evaluate|benchmark|ablation|metric|baseline)\b", re.I),
    re.compile(r"\b(batch|epoch|learning rate|token|embedding|attention|diffusion|transformer|cnn|unet)\b", re.I),
    re.compile(r"\b(input|output|feature|label|parameter|inference|sampling)\b", re.I),
]
GENERIC_PHRASES = [
    "combines beneficial properties",
    "combining beneficial aspects",
    "hybrid activation function",
    "integrates elements",
    "across various architectures and datasets",
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
    "effective number",
    "balanced softmax",
    "gem",
    "dropblock",
    "confidence penalty",
    "stochastic depth",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_xmlish(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for tag in sorted(set(TAG_OPEN_RE.findall(text))):
        match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
        if match:
            fields[tag] = match.group(1).strip()
    return fields


def has_formula_like_detail(text: str) -> bool:
    return any(pattern.search(text) for pattern in FORMULA_PATTERNS)


def has_raw_implementation_detail(text: str) -> bool:
    hits = sum(1 for pattern in RAW_IMPLEMENTATION_PATTERNS if pattern.search(text))
    has_numbers = bool(re.search(r"\b\d+(\.\d+)?\s*(%|x|k|m|b|epoch|layer|token|step|image|sample|parameter|gpu)?\b", text, re.I))
    return hits >= 3 and has_numbers


def count_generic_phrases(text: str) -> int:
    lower = text.lower()
    return sum(1 for phrase in GENERIC_PHRASES if phrase in lower)


def baseline_overlap(text: str) -> list[str]:
    lower = text.lower()
    return [name for name in BASELINE_NAMES if name in lower]


def score_proposal(proposal: str, task_packet: dict[str, Any], *, allow_raw_target: bool = False) -> dict[str, Any]:
    fields = parse_xmlish(proposal)
    missing_tags = [tag for tag in TARGET_XML_TAGS if tag not in fields]
    empty_tags = [tag for tag in TARGET_XML_TAGS if tag in fields and not fields[tag].strip()]
    text_lower = proposal.lower()
    implementation = fields.get("implementation_plan", proposal if allow_raw_target else "")
    algorithm = fields.get("algorithm_or_system", proposal if allow_raw_target else "")
    core = fields.get("core_idea", proposal if allow_raw_target else "")
    evaluation = fields.get("evaluation_plan", proposal if allow_raw_target else "")
    risks = fields.get("risks_and_limitations", proposal if allow_raw_target else "")
    constraints = task_packet.get("worker_constraints") or {}
    modifiable = [str(x) for x in constraints.get("modifiable_files") or []]
    entrypoint = str(constraints.get("entrypoint") or "")
    pass_metric = task_packet.get("pass_metric")

    criteria: dict[str, dict[str, Any]] = {}
    criteria["schema_complete"] = {
        "passed": allow_raw_target or (not missing_tags and not empty_tags),
        "missing_tags": missing_tags,
        "empty_tags": empty_tags,
        "allow_raw_target": allow_raw_target,
    }
    criteria["single_idea"] = {
        "passed": not re.search(r"\b(option|alternative|idea)\s*[1-3]\b", text_lower)
        and len(re.findall(r"<core_idea>", proposal)) <= 1,
    }
    mechanism_passed = has_formula_like_detail(algorithm) or has_formula_like_detail(core)
    if allow_raw_target:
        mechanism_passed = mechanism_passed or has_raw_implementation_detail(proposal)
    criteria["mechanism_exactness"] = {
        "passed": mechanism_passed,
        "algorithm_chars": len(algorithm),
        "core_chars": len(core),
    }
    criteria["code_level_plan"] = {
        "passed": (
            has_raw_implementation_detail(proposal)
            if allow_raw_target
            else any(name.lower() in text_lower for name in modifiable)
            or "customactivation" in text_lower
            or "editable_region.py" in text_lower
        ),
        "modifiable_files": modifiable,
    }
    criteria["worker_constraints"] = {
        "passed": bool(entrypoint and entrypoint.lower() in text_lower)
        or "run.sh result.json" in text_lower
        or "preserve the exact function signature" in text_lower,
        "entrypoint": entrypoint,
    }
    criteria["evaluation_specificity"] = {
        "passed": bool(pass_metric is not None and str(round(float(pass_metric), 2)) in proposal)
        or bool(evaluation and task_packet.get("metric_name", "").lower() in evaluation.lower()),
        "pass_metric": pass_metric,
        "evaluation_chars": len(evaluation),
    }
    criteria["risk_specificity"] = {
        "passed": len(risks) >= 120 and count_generic_phrases(risks) == 0,
        "risk_chars": len(risks),
    }
    generic_count = count_generic_phrases(proposal)
    baseline_names = baseline_overlap(proposal)
    criteria["nontriviality_guard"] = {
        "passed": generic_count <= 2 and not (
            generic_count >= 2 and len(baseline_names) >= 3 and not criteria["mechanism_exactness"]["passed"]
        ),
        "generic_phrase_count": generic_count,
        "baseline_name_overlap": baseline_names,
    }

    weights = {
        "schema_complete": 15,
        "single_idea": 10,
        "mechanism_exactness": 20,
        "code_level_plan": 15,
        "worker_constraints": 10,
        "evaluation_specificity": 10,
        "risk_specificity": 10,
        "nontriviality_guard": 10,
    }
    score = sum(weight for key, weight in weights.items() if criteria[key]["passed"])
    hard_failures = [
        key
        for key in ("schema_complete", "mechanism_exactness", "code_level_plan", "nontriviality_guard")
        if not criteria[key]["passed"]
    ]
    verdict = "pass" if score >= 75 and not hard_failures else "needs_revision"
    return {
        "score": score,
        "max_score": sum(weights.values()),
        "verdict": verdict,
        "allow_raw_target": allow_raw_target,
        "hard_failures": hard_failures,
        "criteria": criteria,
        "field_lengths": {tag: len(fields.get(tag, "")) for tag in TARGET_XML_TAGS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rule-based local scorer for V3 proposal smoke quality.")
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--task-packet", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-raw-target", action="store_true")
    args = parser.parse_args()
    proposal_path = Path(args.proposal)
    task_packet_path = Path(args.task_packet)
    result = score_proposal(
        proposal_path.read_text(),
        read_json(task_packet_path),
        allow_raw_target=args.allow_raw_target,
    )
    output = Path(args.output) if args.output else proposal_path.parent / "proposal_quality.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
