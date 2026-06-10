from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .io import ensure_src_paths, iter_jsonl, write_json
from .runway_client import RunwayClient
from .training_manifest import SYSTEM_PROMPT, TARGET_SCHEMA_VERSION, TARGET_XML_TAGS


STRICT_TEX_SYNTHESIS_SYSTEM = """\
You are creating high-quality supervised targets for a 32B autoresearch idea
proposal model.

The target must teach the model to infer one concrete, worker-implementable idea
from related work plus a research question. It must not train the model to emit
generic paper summaries, broad survey directions, or superficial combinations of
known methods.

Use the full TeX evidence to reconstruct a prospective proposal with:
1. an exact mechanism: formulas, pseudocode, modules, losses, pipeline stages, or
   implementation-level design choices when the TeX supports them;
2. a clear novelty delta over the references, not just "combine A and B";
3. a reproduction recipe: data, preprocessing, model components, training or
   inference steps, ablations, baselines, and metrics;
4. explicit worker granularity: one idea that could be handed to an implementation
   agent, not a full research agenda;
5. risks and negative controls that would reveal the idea is trivial or unstable.

Do not invent unsupported numeric results. If exact values are unavailable, use
qualitative expectations and say what must be measured.
"""


STRICT_TEX_SYNTHESIS_USER = """\
=== PAPER METADATA ===
arxiv_id: {arxiv_id}
title: {title}

=== ORIGINAL ABSTRACT (old target for comparison; do not merely paraphrase) ===
{abstract}

=== REFERENCE-CONDITIONED PROMPT ===
{ref_block}

=== FULL-TEX EVIDENCE EXCERPTS ===
{tex_excerpt}

=== TASK ===
Write a prospective, implementation-ready target for training a proposal model.
The model should learn to answer the reference-conditioned research question
with one precise idea at worker-implementable granularity.

Hard requirements:
- Make the idea concrete enough that an implementation worker can code it.
- Include at least one exact mechanism detail: formula, pseudocode, module
  architecture, loss definition, scheduling rule, data construction rule, or
  inference algorithm.
- State the novelty delta over the references.
- Include at least two ablations or negative controls.
- Include likely failure modes and what measurements would falsify the idea.
- Avoid generic phrases such as "combine the strengths of prior methods" unless
  the combination is specified as an exact algorithm.

Use this exact XML structure:

<thinking>
[Briefly identify the specific problem, the evidence-supported mechanism, why
it is nontrivial relative to the references, the implementation recipe, the
evaluation/ablation plan, and the main risks.]
</thinking>
<proposal>
<title>Short mechanism-specific title.</title>
<problem>Concrete research problem, not a broad topic.</problem>
<gap>Specific limitation in prior work or current practice, tied to the reference-conditioned prompt.</gap>
<core_idea>One exact technical idea or hypothesis; avoid lists of alternatives.</core_idea>
<implementation_plan>Step-by-step implementation plan with code-level modules, data flow, and integration points.</implementation_plan>
<algorithm_or_system>Formula, pseudocode, loss/objective, architecture, scheduling rule, or system pipeline details.</algorithm_or_system>
<training_or_data_recipe>Datasets, preprocessing, supervision, optimization, inference, or simulation recipe needed to reproduce the idea.</training_or_data_recipe>
<evaluation_plan>Baselines, metrics, at least two ablations or negative controls, and stress tests.</evaluation_plan>
<expected_results>Expected outcomes supported by evidence; do not invent unsupported numbers.</expected_results>
<risks_and_limitations>Failure modes, assumptions, confounders, and measurements that would disprove the idea.</risks_and_limitations>
</proposal>"""


def _v3_target_quality(text: str) -> dict[str, Any]:
    missing_tags = [
        tag
        for tag in TARGET_XML_TAGS
        if f"<{tag}>" not in text or f"</{tag}>" not in text
    ]
    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "required_tags": TARGET_XML_TAGS,
        "missing_v3_tags": missing_tags,
        "has_all_v3_required_tags": not missing_tags,
    }


def _merge_v3_quality(row: dict[str, Any], text: str) -> None:
    quality = dict(row.get("target_quality") or {})
    quality.update(_v3_target_quality(text))
    row["target_quality"] = quality


def _extract_xml_block(text: str, tag: str) -> str:
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    if not match:
        return text.strip()
    return f"<{tag}>{match.group(1).strip()}</{tag}>"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _done_ids(*paths: Path) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            aid = row.get("arxiv_id")
            if aid:
                done.add(str(aid))
    return done


def _sample_to_legacy_record(sample: dict[str, Any]) -> dict[str, Any]:
    refs = []
    for ref in sample.get("condition_packet", {}).get("selected_refs") or []:
        refs.append({
            "arxiv_id": ref.get("arxiv_id"),
            "title": ref.get("title"),
            "year": ref.get("year"),
            "abstract": ref.get("abstract"),
            "ref_key": ref.get("ref_key"),
        })
    return {
        "sample_id": sample.get("sample_id"),
        "manifest_version": sample.get("manifest_version"),
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "arxiv_id": sample.get("arxiv_id"),
        "created": sample.get("created"),
        "chronological_rank": sample.get("chronological_rank"),
        "split": sample.get("split"),
        "title": sample.get("title"),
        "abstract": sample.get("abstract"),
        "categories": sample.get("categories") or [],
        "paper_type": sample.get("paper_type"),
        "category_family": sample.get("category_family"),
        "quality": sample.get("quality") or {},
        "leakage_guard": sample.get("leakage_guard") or {},
        "refs": refs,
        "system": SYSTEM_PROMPT,
        "prompt": sample.get("condition_packet", {}).get("prompt", ""),
    }


def strict_tex_synthesis_messages(
    record: dict[str, Any],
    *,
    abstract: str,
    ref_block: str,
    tex_excerpt: str,
) -> list[dict[str, str]]:
    user = STRICT_TEX_SYNTHESIS_USER.format(
        arxiv_id=record.get("arxiv_id", ""),
        title=record.get("title", ""),
        abstract=abstract,
        ref_block=ref_block,
        tex_excerpt=tex_excerpt,
    )
    return [
        {"role": "system", "content": STRICT_TEX_SYNTHESIS_SYSTEM},
        {"role": "user", "content": user},
    ]


def _mock_target(record: dict[str, Any]) -> dict[str, Any]:
    text = (
        "<thinking>\n"
        "Offline mock target used to verify V3 synthesis and collate plumbing. "
        "It includes the complete V3 schema and a concrete mechanism placeholder.\n"
        "</thinking>\n"
        "<proposal>\n"
        "<title>Evidence-gated residual update rule</title>\n"
        f"<problem>Improve the task suggested by {record.get('title') or record.get('arxiv_id')} with one implementable method.</problem>\n"
        "<gap>The references leave room for a mechanism that decides when a new update should override or preserve the baseline computation.</gap>\n"
        "<core_idea>Add a scalar gate g=sigmoid(MLP([h, delta])) and return h + g * delta, so each example can suppress harmful updates while retaining useful ones.</core_idea>\n"
        "<implementation_plan>Implement the gate inside the editable model block, keep the public function signature unchanged, initialize the final gate bias to -1, and log the gate mean per epoch.</implementation_plan>\n"
        "<algorithm_or_system>For hidden state h and candidate update delta=f(h), compute g=sigmoid(W2*relu(W1*[h, delta])+b2), then y=h+g*delta. Ablate fixed g=1 and fixed g=0.</algorithm_or_system>\n"
        "<training_or_data_recipe>Use the existing dataset, optimizer, schedule, and seeds; add no external data; train the added gate jointly with the baseline model.</training_or_data_recipe>\n"
        "<evaluation_plan>Compare against the baseline metric, fixed-gate ablations, random-gate negative control, and seed variance; report whether the pass threshold is exceeded.</evaluation_plan>\n"
        "<expected_results>Expect improvement only if the learned gate reduces unstable updates without reducing useful capacity; otherwise the fixed-gate ablations should match it.</expected_results>\n"
        "<risks_and_limitations>The gate may collapse, add optimization noise, or exploit seed variance; falsify the idea if gate statistics are constant or ablations perform the same.</risks_and_limitations>\n"
        "</proposal>"
    )
    return {
        **record,
        "target_source": "tex_implementation_mock",
        "tex_status": "mock",
        "cot_impl_proposal": text,
        "target_impl_proposal": text,
        "target_quality": {
            "word_count": len(text.split()),
            "has_all_required_tags": True,
            **_v3_target_quality(text),
            "mock": True,
        },
        "tex_impl_leakage_score": 0.0,
        "synthesis_model": "mock",
        "synthesis_temperature": 0.0,
        "synthesis_prompt_version": "v3_strict_mock",
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "synthesized_at": int(time.time()),
    }


async def synthesize_v3_targets(
    cfg: dict[str, Any],
    manifest_dir: Path,
    output_file: Path,
    skipped_file: Path | None = None,
    errors_file: Path | None = None,
    limit: int | None = None,
    arxiv_id: str | None = None,
    split: str | None = None,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float | None = 0.3,
    concurrency: int = 2,
    max_tex_chars: int = 24000,
    connect_timeout: float = 10.0,
    chunk_timeout: float = 180.0,
    call_timeout: float | None = 420.0,
    dry_run: bool = False,
    mock: bool = False,
) -> dict[str, Any]:
    """Run/resume TeX-grounded synthesis for V3 manifest samples."""
    ts_cfg = cfg.get("target_synthesis", {})
    prompt_version = str(ts_cfg.get("prompt_version") or "v3_strict")
    skipped_file = skipped_file or output_file.with_name(output_file.stem + ".skipped.jsonl")
    errors_file = errors_file or output_file.with_name(output_file.stem + ".errors.jsonl")
    samples = list(iter_jsonl(manifest_dir / "samples.jsonl"))
    done = _done_ids(output_file, skipped_file)
    selected: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("arxiv_id") in done:
            continue
        if arxiv_id and sample.get("arxiv_id") != arxiv_id:
            continue
        if split and sample.get("split") != split:
            continue
        if sample.get("target", {}).get("target_status") == "ready":
            continue
        selected.append(sample)
        if limit is not None and len(selected) >= limit:
            break

    summary = {
        "manifest_dir": str(manifest_dir),
        "output_file": str(output_file),
        "skipped_file": str(skipped_file),
        "errors_file": str(errors_file),
        "selected_count": len(selected),
        "already_done_count": len(done),
        "dry_run": dry_run,
        "mock": mock,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "concurrency": concurrency,
        "max_tex_chars": max_tex_chars,
        "call_timeout": call_timeout,
        "prompt_version": prompt_version,
        "by_split": dict(Counter(s.get("split") for s in selected)),
    }
    if dry_run:
        return summary

    ensure_src_paths(cfg)
    provider = str(ts_cfg.get("provider") or "legacy_proxy")
    endpoint = str(ts_cfg.get("endpoint") or "google_anthropic")
    key_env = str(ts_cfg.get("key_env") or "RUNWAY_OPUS47_API_KEY")
    require_all_target_tags = bool(ts_cfg.get("require_all_target_tags", True))
    from data.synthesize_tex_targets import (
        TEX_SYNTHESIS_SYSTEM,
        TEX_SYNTHESIS_USER,
        build_result_record,
        extract_tex_context,
        make_skip_record,
        synthesize_tex_target_async,
    )
    from data.synthesize_cot import _extract_ref_text

    semaphore = asyncio.Semaphore(concurrency)
    written = skipped = errors = quality_failed = 0
    start = time.time()

    def _synthesize_runway(record: dict[str, Any]) -> dict[str, Any]:
        arxiv_root = Path(cfg["arxiv_root"])
        context = extract_tex_context(str(record.get("arxiv_id") or ""), arxiv_root, max_tex_chars=max_tex_chars)
        if context.get("tex_status") != "ok":
            return make_skip_record(record, context)
        ref_text = _extract_ref_text(record.get("prompt", ""))
        if prompt_version in {"v3_strict", "strict_v3", "strict"}:
            messages = strict_tex_synthesis_messages(
                record,
                abstract=str(record.get("abstract") or ""),
                ref_block=ref_text,
                tex_excerpt=context["tex_excerpt"],
            )
        else:
            user = TEX_SYNTHESIS_USER.format(
                arxiv_id=record.get("arxiv_id", ""),
                title=record.get("title", ""),
                abstract=record.get("abstract", ""),
                ref_block=ref_text,
                tex_excerpt=context["tex_excerpt"],
            )
            messages = [
                {"role": "system", "content": TEX_SYNTHESIS_SYSTEM},
                {"role": "user", "content": user},
            ]
        client = RunwayClient(cfg, key_env=key_env)
        result = client.complete(
            endpoint=endpoint,
            model=str(model or ts_cfg.get("model") or "claude-opus-4-7"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        row = build_result_record(
            record,
            context,
            result.text,
            str(result.model or model or ts_cfg.get("model") or "runway_google_anthropic"),
            temperature if temperature is not None else 0.0,
        )
        row["synthesis_provider"] = "runway_google_anthropic"
        row["synthesis_endpoint"] = endpoint
        row["synthesis_prompt_version"] = prompt_version
        row["target_schema_version"] = TARGET_SCHEMA_VERSION
        _merge_v3_quality(row, result.text)
        if prompt_version in {"v3_strict", "strict_v3", "strict"}:
            row["target_impl_proposal"] = _extract_xml_block(result.text, "proposal")
        row["synthesis_usage"] = result.usage
        row["synthesis_finish_reason"] = result.raw_finish_reason
        return row

    async def _one(sample: dict[str, Any]) -> dict[str, Any]:
        record = _sample_to_legacy_record(sample)
        if mock:
            return _mock_target(record)
        async with semaphore:
            if provider in {"runway_google_anthropic", "runway", "google_anthropic"}:
                task = asyncio.to_thread(_synthesize_runway, record)
                return await asyncio.wait_for(task, timeout=call_timeout) if call_timeout else await task
            task = synthesize_tex_target_async(
                record,
                cfg,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                max_tex_chars=max_tex_chars,
                connect_timeout=connect_timeout,
                chunk_timeout=chunk_timeout,
            )
            return await asyncio.wait_for(task, timeout=call_timeout) if call_timeout else await task

    async def _one_safe(sample: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": await _one(sample)}
        except Exception as exc:
            return {
                "error": {
                    "sample_id": sample.get("sample_id"),
                    "arxiv_id": sample.get("arxiv_id"),
                    "split": sample.get("split"),
                    "error": repr(exc),
                    "timestamp": int(time.time()),
                }
            }

    tasks = [asyncio.create_task(_one_safe(sample)) for sample in selected]
    for fut in asyncio.as_completed(tasks):
        result = await fut
        if "error" in result:
            errors += 1
            _append_jsonl(errors_file, result["error"])
            continue
        row = result["ok"]
        if row.get("tex_status") in {"ok", "mock"} and row.get("target_impl_proposal"):
            tq = row.get("target_quality") or {}
            has_required_tags = tq.get("has_all_v3_required_tags", tq.get("has_all_required_tags"))
            if require_all_target_tags and not has_required_tags:
                quality_failed += 1
                errors += 1
                _append_jsonl(errors_file, {
                    "sample_id": row.get("sample_id"),
                    "arxiv_id": row.get("arxiv_id"),
                    "split": row.get("split"),
                    "error_type": "target_quality_failed",
                    "error": "missing required XML tags in synthesized target",
                    "target_quality": tq,
                    "synthesis_model": row.get("synthesis_model"),
                    "synthesis_provider": row.get("synthesis_provider"),
                    "synthesis_finish_reason": row.get("synthesis_finish_reason"),
                    "target_preview": (row.get("target_impl_proposal") or "")[:600],
                    "timestamp": int(time.time()),
                })
                continue
            _append_jsonl(output_file, row)
            written += 1
        else:
            _append_jsonl(skipped_file, row)
            skipped += 1

    summary.update({
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "quality_failed": quality_failed,
        "elapsed_sec": round(time.time() - start, 3),
    })
    write_json(output_file.with_name(output_file.stem + ".summary.json"), summary)
    return summary
