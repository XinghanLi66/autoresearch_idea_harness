from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import iter_jsonl, short_text


def _load_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    for row in iter_jsonl(path):
        if row.get(key):
            out[row[key]] = row
    return out


def render_markdown_report(
    packets_path: Path,
    expert_batches_path: Path,
    private_batches_path: Path,
    summary_path: Path,
    output_path: Path,
    limit: int = 5,
) -> None:
    import json

    packets = _load_by_key(packets_path, "packet_id")
    expert_batches = list(iter_jsonl(expert_batches_path)) if expert_batches_path.exists() else []
    private_batches = _load_by_key(private_batches_path, "packet_id")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    proposal_scores = {
        row["proposal_id"]: row
        for row in summary.get("proposals", [])
    }

    lines = [
        "# Autoresearch Idea Harness Market Preview",
        "",
        "## Summary",
        "",
        f"- Packets evaluated: {summary.get('packet_count', 0)}",
        f"- Proposals scored: {summary.get('proposal_count', 0)}",
        f"- Forecasts: {summary.get('forecast_count', 0)}",
        "",
        "## Generator Ranking",
        "",
    ]
    for row in summary.get("by_generator", []):
        lines.append(
            f"- `{row['id']}`: mean success probability {row['mean_success_probability']} "
            f"over {row['proposal_count']} proposals"
        )
    lines.extend(["", "## Packet Preview", ""])

    for batch in expert_batches[:limit]:
        packet_id = batch["packet_id"]
        packet = packets.get(packet_id, {})
        private = private_batches.get(packet_id, {})
        model_by_pid = {
            p["proposal_id"]: p
            for p in private.get("proposals", [])
        }
        lines.extend([
            f"### {packet_id}",
            "",
            f"- Paper type: `{packet.get('paper_type')}`",
            f"- Category family: `{packet.get('category_family')}`",
            f"- Baseline: {packet.get('baseline', {}).get('description', '')}",
            f"- Reference count: {len(batch.get('expert_view', {}).get('references', []))}",
            "",
            "**Evidence references**",
            "",
        ])
        for ref in batch.get("expert_view", {}).get("references", [])[:5]:
            lines.append(
                f"- {ref.get('title')} ({ref.get('year') or 'n.d.'}) "
                f"[{ref.get('evidence_status')}, snippets={len(ref.get('snippets') or [])}]"
            )
            for snippet in (ref.get("snippets") or [])[:2]:
                provenance = snippet.get("provenance") or {}
                source = provenance.get("source") or snippet.get("source") or "unknown source"
                heading = snippet.get("heading") or "untitled section"
                kind = snippet.get("kind") or "snippet"
                text = short_text(snippet.get("text", ""), 360)
                lines.append(f"  - `{kind}` {heading} ({source}): {text}")
        lines.extend(["", "**Anonymous proposals**", ""])
        for proposal in batch.get("proposals", []):
            score = proposal_scores.get(proposal["proposal_id"], {})
            reveal = model_by_pid.get(proposal["proposal_id"], {})
            lines.append(
                f"- Proposal {proposal.get('label')} "
                f"(mean p={score.get('mean_success_probability', 'n/a')}, "
                f"reveal=`{reveal.get('generator_id', 'unknown')}`): "
                f"{short_text(proposal.get('text', ''), 500)}"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
