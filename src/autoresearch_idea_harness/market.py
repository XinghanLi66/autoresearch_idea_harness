from __future__ import annotations

import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import iter_jsonl, stable_id, write_json, write_jsonl
from .prompts import expert_forecast_prompt
from .runway_client import RunwayClient


FIXTURE_PRIORS = {
    "evidence_synthesis": 0.68,
    "conservative_baseline": 0.55,
    "abstract_ablation": 0.38,
    "local_checkpoint_placeholder": 0.12,
    "claude": 0.72,
}


def _private_map(private_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    proposals: dict[str, dict[str, Any]] = {}
    by_packet: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(private_path):
        packet_id = row.get("packet_id")
        by_packet[packet_id] = list(row.get("proposals") or [])
        for proposal in row.get("proposals") or []:
            proposals[proposal["proposal_id"]] = {
                **proposal,
                "packet_id": packet_id,
            }
    return proposals, by_packet


def make_fixture_forecasts(
    private_path: Path,
    output_path: Path,
    experts: int = 3,
) -> int:
    proposals, by_packet = _private_map(private_path)
    rows = []
    now = int(time.time())
    for packet_id, packet_proposals in by_packet.items():
        for expert_idx in range(experts):
            rng = random.Random(f"{packet_id}:{expert_idx}")
            expert_id = f"fixture_expert_{expert_idx + 1:02d}"
            for proposal in packet_proposals:
                generator_id = proposals[proposal["proposal_id"]].get("generator_id")
                base = FIXTURE_PRIORS.get(generator_id, 0.45)
                prob = max(0.01, min(0.99, base + rng.uniform(-0.08, 0.08)))
                rows.append({
                    "forecast_id": stable_id("fcst", packet_id, proposal["proposal_id"], expert_id),
                    "packet_id": packet_id,
                    "proposal_id": proposal["proposal_id"],
                    "expert_id": expert_id,
                    "success_probability": round(prob, 3),
                    "rationale": f"Fixture forecast for {generator_id}.",
                    "created_at": now,
                })
    return write_jsonl(output_path, rows)


def _extract_json_obj(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("expert response did not contain a JSON object")
    return json.loads(match.group(0))


def _as_probability(value: Any) -> float:
    try:
        p = float(value)
    except Exception as exc:
        raise ValueError(f"invalid success_probability: {value!r}") from exc
    if p > 1.0 and p <= 100.0:
        p = p / 100.0
    return max(0.0, min(1.0, p))


def make_llm_forecasts(
    cfg: dict[str, Any],
    expert_batches_path: Path,
    output_path: Path,
    expert_ids: list[str] | None = None,
    limit_packets: int | None = None,
    skip_missing: bool = False,
    skip_errors: bool = False,
) -> dict[str, Any]:
    market_cfg = cfg.get("market", {})
    llm_cfg = cfg.get("llm", {})
    configured = {
        row["id"]: row
        for row in market_cfg.get("llm_experts", [])
        if row.get("id")
    }
    selected_ids = expert_ids or list(configured.keys())
    if not selected_ids:
        raise RuntimeError("No LLM experts configured.")

    rows = []
    errors = []
    skipped_experts = []
    expert_clients: dict[str, tuple[dict[str, Any], RunwayClient]] = {}
    for expert_id in selected_ids:
        if expert_id not in configured:
            raise ValueError(f"Unknown expert id: {expert_id}")
        expert_cfg = configured[expert_id]
        try:
            client = RunwayClient(cfg, key_env=str(expert_cfg["key_env"]))
        except RuntimeError:
            if skip_missing:
                skipped_experts.append(expert_id)
                continue
            raise
        expert_clients[expert_id] = (expert_cfg, client)

    packets_seen = 0
    now = int(time.time())
    for batch in iter_jsonl(expert_batches_path):
        packets_seen += 1
        if limit_packets is not None and packets_seen > limit_packets:
            break
        packet_id = batch["packet_id"]
        expert_view = batch.get("expert_view") or {}
        success_definition = batch.get("success_definition")
        for proposal in batch.get("proposals") or []:
            for expert_id, (expert_cfg, client) in expert_clients.items():
                model_env = expert_cfg.get("model_env")
                model = (
                    os.environ.get(str(model_env))
                    if model_env else None
                ) or str(expert_cfg["model"])
                temp = expert_cfg.get("temperature", llm_cfg.get("expert_temperature", 0.2))
                try:
                    result = client.complete(
                        endpoint=str(expert_cfg.get("endpoint", "chat_completions")),
                        model=model,
                        messages=expert_forecast_prompt(expert_view, proposal, success_definition),
                        temperature=None if temp is None else float(temp),
                        max_tokens=int(expert_cfg.get("max_tokens", llm_cfg.get("expert_max_tokens", 900))),
                        stream=True,
                    )
                    parsed = _extract_json_obj(result.text)
                except Exception as exc:
                    if not skip_errors:
                        raise
                    errors.append({
                        "packet_id": packet_id,
                        "proposal_id": proposal.get("proposal_id"),
                        "expert_id": expert_id,
                        "error": str(exc)[:500],
                    })
                    continue
                rows.append({
                    "forecast_id": stable_id("fcst", packet_id, proposal["proposal_id"], expert_id),
                    "packet_id": packet_id,
                    "proposal_id": proposal["proposal_id"],
                    "expert_id": expert_id,
                    "expert_model_id": model,
                    "success_probability": round(_as_probability(parsed.get("success_probability")), 4),
                    "rationale": str(parsed.get("rationale", "")).strip(),
                    "strengths": parsed.get("strengths") or [],
                    "weaknesses": parsed.get("weaknesses") or [],
                    "baseline_risk": str(parsed.get("baseline_risk", "")).strip(),
                    "usage": result.usage,
                    "created_at": now,
                })
    n = write_jsonl(output_path, rows)
    error_path = output_path.with_suffix(".errors.jsonl")
    if errors:
        write_jsonl(error_path, errors)
    return {
        "forecast_count": n,
        "path": str(output_path),
        "experts": list(expert_clients.keys()),
        "skipped_experts": skipped_experts,
        "error_count": len(errors),
        "error_path": str(error_path) if errors else None,
    }


def aggregate_forecasts(
    private_path: Path,
    forecasts_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    proposals, _ = _private_map(private_path)
    by_proposal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(forecasts_path):
        pid = row.get("proposal_id")
        if pid in proposals:
            by_proposal[pid].append(row)

    proposal_rows = []
    by_model: dict[str, list[float]] = defaultdict(list)
    by_generator: dict[str, list[float]] = defaultdict(list)
    for proposal_id, forecasts in sorted(by_proposal.items()):
        probs = [float(f["success_probability"]) for f in forecasts]
        meta = proposals[proposal_id]
        mean_prob = sum(probs) / len(probs) if probs else 0.0
        proposal_rows.append({
            "packet_id": meta["packet_id"],
            "proposal_id": proposal_id,
            "generator_id": meta.get("generator_id"),
            "model_id": meta.get("model_id"),
            "forecast_count": len(forecasts),
            "mean_success_probability": round(mean_prob, 4),
            "min_success_probability": round(min(probs), 4) if probs else None,
            "max_success_probability": round(max(probs), 4) if probs else None,
        })
        by_model[meta.get("model_id")].append(mean_prob)
        by_generator[meta.get("generator_id")].append(mean_prob)

    def summarize_group(values_by_key: dict[str, list[float]]) -> list[dict[str, Any]]:
        rows = []
        for key, values in values_by_key.items():
            rows.append({
                "id": key,
                "proposal_count": len(values),
                "mean_success_probability": round(sum(values) / len(values), 4) if values else 0.0,
            })
        return sorted(rows, key=lambda r: r["mean_success_probability"], reverse=True)

    summary = {
        "forecast_count": sum(len(v) for v in by_proposal.values()),
        "proposal_count": len(proposal_rows),
        "packet_count": len({r["packet_id"] for r in proposal_rows}),
        "by_generator": summarize_group(by_generator),
        "by_model": summarize_group(by_model),
        "proposals": sorted(
            proposal_rows,
            key=lambda r: r["mean_success_probability"],
            reverse=True,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "proposal_scores.jsonl", summary["proposals"])
    return summary
