from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from .io import iter_jsonl, write_json


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open(errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
    }


def _checkpoint_dir_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "config_exists": (path / "config.json").exists(),
        "index_exists": (path / "model.safetensors.index.json").exists(),
        "lora_adapter_exists": (path / "lora_adapter" / "adapter_model.safetensors").exists(),
        "model_shard_count": 0,
    }
    if not path.exists():
        return info
    info["model_shard_count"] = len(list(path.glob("model-*.safetensors")))
    return info


def _quality_counts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"row_count": 0}
    quality = Counter()
    model = Counter()
    source = Counter()
    tex_status = Counter()
    prompt_version = Counter()
    for row in iter_jsonl(path):
        tq = row.get("target_quality") or {}
        quality["has_all_required_tags"] += int(bool(tq.get("has_all_required_tags")))
        quality["has_all_v3_required_tags"] += int(bool(tq.get("has_all_v3_required_tags")))
        quality["has_implementation_terms"] += int(bool(tq.get("has_implementation_terms")))
        quality["has_evaluation_terms"] += int(bool(tq.get("has_evaluation_terms")))
        quality["mock"] += int(bool(tq.get("mock")))
        model[str(row.get("synthesis_model") or "unknown")] += 1
        prompt_version[str(row.get("synthesis_prompt_version") or "legacy_or_unknown")] += 1
        source[str(row.get("target_source") or "unknown")] += 1
        tex_status[str(row.get("tex_status") or "unknown")] += 1
    return {
        "row_count": _count_jsonl(path),
        "quality_true_counts": dict(quality),
        "by_model": dict(model),
        "by_prompt_version": dict(prompt_version),
        "by_target_source": dict(source),
        "by_tex_status": dict(tex_status),
    }


def summarize_manifest_dir(path: Path) -> dict[str, Any] | None:
    manifest = _load_json(path / "manifest.json")
    if not manifest:
        return None
    return {
        "kind": "manifest",
        "name": path.name,
        "path": str(path),
        "candidate_count": manifest.get("candidate_count"),
        "ready_sft_count": manifest.get("ready_sft_count"),
        "synthesis_queue_count": manifest.get("synthesis_queue_count"),
        "unique_selected_ref_count": manifest.get("unique_selected_ref_count"),
        "splits": manifest.get("splits") or {},
        "created_range": manifest.get("created_range") or {},
        "base_model_release_date": manifest.get("base_model_release_date"),
        "target_status": manifest.get("target_status") or {},
        "research_question_status": manifest.get("research_question_status") or {},
        "by_month": manifest.get("by_created_month") or manifest.get("by_month") or {},
        "by_arxiv_month": manifest.get("by_arxiv_month") or {},
        "by_category_family": manifest.get("by_category_family") or {},
        "filters": manifest.get("filters") or {},
        "files": {
            "samples": _file_info(path / "samples.jsonl"),
            "synthesis_queue": _file_info(path / "synthesis_queue.jsonl"),
            "ref_evidence_queue": _file_info(path / "ref_evidence_queue.jsonl"),
        },
    }


def summarize_target_cache(path: Path) -> dict[str, Any] | None:
    target_path = path / "tex_targets.jsonl"
    if not target_path.exists():
        return None
    summary = _load_json(path / "tex_targets.summary.json")
    return {
        "kind": "target_cache",
        "name": path.name,
        "path": str(path),
        "row_count": _count_jsonl(target_path),
        "skipped_count": _count_jsonl(path / "tex_targets.skipped.jsonl"),
        "error_count": _count_jsonl(path / "tex_targets.errors.jsonl"),
        "run_summary": summary,
        "quality": _quality_counts(target_path),
        "files": {
            "targets": _file_info(target_path),
            "skipped": _file_info(path / "tex_targets.skipped.jsonl"),
            "errors": _file_info(path / "tex_targets.errors.jsonl"),
        },
    }


def summarize_target_cache_audit_dir(path: Path) -> dict[str, Any] | None:
    summary = _load_json(path / "summary.json")
    if not summary or summary.get("kind") != "target_cache_audit":
        return None
    return {
        "kind": "target_cache_audit",
        "name": path.name,
        "path": str(path),
        "input": summary.get("input"),
        "row_count": summary.get("row_count"),
        "accepted_count": summary.get("accepted_count"),
        "rejected_count": summary.get("rejected_count"),
        "require_prompt_version": summary.get("require_prompt_version"),
        "score": summary.get("score") or {},
        "verdicts": summary.get("verdicts") or {},
        "hard_failures": summary.get("hard_failures") or {},
        "prompt_versions": summary.get("prompt_versions") or {},
        "files": {
            "summary_json": _file_info(path / "summary.json"),
            "summary_md": _file_info(path / "summary.md"),
            "accepted": _file_info(path / "tex_targets.accepted.jsonl"),
            "rejected": _file_info(path / "tex_targets.rejected.jsonl"),
            "rows": _file_info(path / "rows.jsonl"),
        },
    }


def summarize_sft_dir(path: Path) -> dict[str, Any] | None:
    summary = _load_json(path / "summary.json")
    if not summary and not (path / "train.jsonl").exists():
        return None
    if summary and "collated_count" not in summary and not (path / "train.jsonl").exists():
        return None
    return {
        "kind": "sft",
        "name": path.name,
        "path": str(path),
        "summary": summary,
        "counts": {
            "all": _count_jsonl(path / "all.jsonl"),
            "train": _count_jsonl(path / "train.jsonl"),
            "val": _count_jsonl(path / "val.jsonl"),
            "test": _count_jsonl(path / "test.jsonl"),
            "missing_target": _count_jsonl(path / "missing_target.jsonl"),
        },
        "files": {
            "train_jsonl": _file_info(path / "train.jsonl"),
            "train_parquet": _file_info(path / "train.parquet"),
            "summary": _file_info(path / "summary.json"),
        },
    }


def summarize_sft_target_quality_dir(path: Path) -> dict[str, Any] | None:
    summary = _load_json(path / "summary.json")
    if not summary or summary.get("kind") == "target_cache_audit" or "verdicts" not in summary or "score" not in summary:
        return None
    return {
        "kind": "sft_target_quality",
        "name": path.name,
        "path": str(path),
        "input": summary.get("input"),
        "row_count": summary.get("row_count"),
        "allow_raw_target": summary.get("allow_raw_target"),
        "score": summary.get("score") or {},
        "verdicts": summary.get("verdicts") or {},
        "hard_failures": summary.get("hard_failures") or {},
        "by_category": summary.get("by_category") or {},
        "files": {
            "summary_json": _file_info(path / "summary.json"),
            "summary_md": _file_info(path / "summary.md"),
            "rows": _file_info(path / "rows.jsonl"),
        },
    }


def summarize_run_plan(path: Path) -> dict[str, Any] | None:
    plan = _load_json(path / "run_plan.json")
    if not plan:
        return None
    phases = list(plan.get("phases") or [])
    submission = _load_json(path / "submission.json")
    phase_status = []
    for phase in phases:
        final_dir = Path(str(phase.get("expected_final") or ""))
        phase_status.append({
            "name": phase.get("phase_name"),
            "row_count": phase.get("row_count"),
            "created_range": phase.get("created_range") or {},
            "expected_final": str(final_dir),
            "final_checkpoint": _checkpoint_dir_info(final_dir),
        })
    return {
        "kind": "run_plan",
        "name": path.name,
        "path": str(path),
        "run_id": plan.get("run_id"),
        "base_model_path": plan.get("base_model_path"),
        "base_model_release_date": plan.get("base_model_release_date"),
        "phase_by": plan.get("phase_by"),
        "phase_count": plan.get("phase_count"),
        "train_row_count": plan.get("train_row_count"),
        "train_created_range": plan.get("train_created_range") or {},
        "by_month": plan.get("by_month") or {},
        "training_policy": plan.get("training_policy") or {},
        "phases_preview": phases[:8],
        "phase_status": phase_status[:8],
        "submission": {
            "exists": bool(submission),
            "job_id": submission.get("job_id"),
            "result": submission.get("result"),
            "finished_at": submission.get("finished_at"),
            "workspace_id": submission.get("workspace_id"),
            "resource_id": submission.get("resource_id"),
            "priority": submission.get("priority"),
            "gpus": submission.get("gpus"),
            "latest_status": (submission.get("latest_status") or {}).get("status"),
            "latest_reason_code": (submission.get("latest_status") or {}).get("reason_code"),
            "final_dir": submission.get("final_dir"),
        },
        "files": {
            "run_plan": _file_info(path / "run_plan.json"),
            "launcher": _file_info(path / "run_sft_curriculum.sh"),
            "dlc_skeleton": _file_info(path / "dlc_command_skeleton.sh"),
            "submission": _file_info(path / "submission.json"),
        },
    }


def summarize_proposal_smoke_dir(path: Path) -> dict[str, Any] | None:
    plan = _load_json(path / "run_plan.json")
    if not plan:
        return None
    if "task" not in plan or "subtask" not in plan or "model_dir" not in plan:
        return None
    output_dir = Path(str(plan.get("output_dir") or path / "output"))
    proposal_path = output_dir / "proposal.txt"
    meta_path = output_dir / "meta.json"
    quality_path = output_dir / "proposal_quality.json"
    preflight_path = path / "proposal_smoke_preflight.json"
    submission_path = path / "submission.json"
    submission = _load_json(submission_path)
    return {
        "kind": "proposal_smoke",
        "name": path.name,
        "path": str(path),
        "task": plan.get("task"),
        "subtask": plan.get("subtask"),
        "model_dir": plan.get("model_dir"),
        "ready_to_dry_run": bool(plan.get("ready_to_dry_run")),
        "ready_to_submit": bool(plan.get("ready_to_submit")),
        "submit_blocker": plan.get("submit_blocker"),
        "submit_guard_env": plan.get("submit_guard_env"),
        "errors": plan.get("errors") or [],
        "resources": plan.get("resources") or {},
        "preflight": _load_json(preflight_path),
        "submission": {
            "exists": bool(submission),
            "job_id": submission.get("job_id"),
            "job_name": submission.get("job_name"),
            "submitted_at": submission.get("submitted_at"),
            "workspace_id": submission.get("workspace_id"),
            "resource_id": submission.get("resource_id"),
            "priority": submission.get("priority"),
            "gpus": submission.get("gpus"),
            "latest_status": (submission.get("latest_status") or {}).get("status"),
            "latest_sub_status": (submission.get("latest_status") or {}).get("sub_status"),
            "latest_reason_code": (submission.get("latest_status") or {}).get("reason_code"),
            "checked_at": (submission.get("latest_status") or {}).get("checked_at"),
        },
        "files": {
            "run_plan": _file_info(path / "run_plan.json"),
            "command": _file_info(Path(str((plan.get("files") or {}).get("command") or path / "dlc_command_skeleton.sh"))),
            "dry_run": _file_info(Path(str((plan.get("files") or {}).get("dry_run") or path / "pai_create_job_dry_run.sh"))),
            "submit": _file_info(Path(str((plan.get("files") or {}).get("submit") or path / "pai_create_job.sh"))),
            "preflight": _file_info(preflight_path),
            "submission": _file_info(submission_path),
            "proposal": _file_info(proposal_path),
            "meta": _file_info(meta_path),
            "proposal_quality": _file_info(quality_path),
        },
        "proposal_generated": proposal_path.exists() and proposal_path.stat().st_size > 0,
        "meta": _load_json(meta_path),
        "proposal_quality": _load_json(quality_path),
    }


def summarize_proposal_packet_matrix(path: Path) -> dict[str, Any] | None:
    summary = _load_json(path / "summary.json")
    if not summary:
        return None
    return {
        "kind": "proposal_packet_matrix",
        "name": path.name,
        "path": str(path),
        "task_count": summary.get("task_count"),
        "ok_count": summary.get("ok_count"),
        "warning_count": summary.get("warning_count"),
        "error_count": summary.get("error_count"),
        "min_frontline_refs": summary.get("min_frontline_refs"),
        "max_prompt_tokens": summary.get("max_prompt_tokens"),
        "rows": (summary.get("rows") or [])[:20],
        "warnings": summary.get("warnings") or [],
        "errors": summary.get("errors") or [],
        "files": {
            "summary_json": _file_info(path / "summary.json"),
            "summary_md": _file_info(path / "summary.md"),
        },
    }


def _base_model_registry_status(cfg: dict[str, Any] | None) -> dict[str, Any]:
    registry = ((cfg or {}).get("base_models") or {}).get("registry") or {}
    if not isinstance(registry, dict):
        registry = {}
    entries = []
    valid_32b = 0
    for model_id, record in sorted(registry.items()):
        size = record.get("size_b")
        try:
            size_f = float(size)
        except Exception:
            size_f = None
        is_32b = bool(size_f is not None and 30.0 <= size_f <= 40.0 and record.get("release_date") and record.get("path"))
        valid_32b += int(is_32b and not record.get("smoke_only"))
        entries.append({
            "model_id": model_id,
            "path": record.get("path"),
            "release_date": _json_safe(record.get("release_date")),
            "size_b": size_f,
            "smoke_only": bool(record.get("smoke_only")),
            "formal_32b_ready": bool(is_32b and not record.get("smoke_only")),
        })
    return _json_safe({
        "entry_count": len(entries),
        "formal_32b_ready_count": valid_32b,
        "entries": entries,
    })


def _base_model_discovery_status(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    if not report:
        return {"path": str(path), "exists": False, "candidate_count": 0, "candidates": []}
    candidates = list(report.get("candidates") or [])
    return {
        "path": str(path),
        "exists": True,
        "generated_at": report.get("generated_at"),
        "counts": report.get("counts") or {},
        "candidate_count": len(candidates),
        "candidates": candidates[:8],
        "all_models_truncated": bool(report.get("all_models_truncated")),
    }


def collect_v3_status(
    training_data_root: Path,
    training_root: Path,
    cfg: dict[str, Any] | None = None,
    discovery_report_path: Path | None = None,
) -> dict[str, Any]:
    manifests = []
    targets = []
    target_cache_audits = []
    sfts = []
    sft_target_qualities = []
    plans = []
    proposal_smokes = []
    proposal_packet_matrices = []

    if training_data_root.exists():
        for child in sorted(training_data_root.iterdir()):
            if not child.is_dir():
                continue
            item = summarize_manifest_dir(child)
            if item:
                manifests.append(item)
            item = summarize_target_cache(child)
            if item:
                targets.append(item)
            item = summarize_target_cache_audit_dir(child)
            if item:
                target_cache_audits.append(item)
            item = summarize_sft_dir(child)
            if item:
                sfts.append(item)
            item = summarize_sft_target_quality_dir(child)
            if item:
                sft_target_qualities.append(item)

    if training_root.exists():
        for plan_path in sorted(training_root.rglob("run_plan.json")):
            item = summarize_run_plan(plan_path.parent)
            if item:
                plans.append(item)

    proposal_smoke_root = training_root.parent / "v3_checkpoint_proposal_smoke"
    if proposal_smoke_root.exists():
        for plan_path in sorted(proposal_smoke_root.rglob("run_plan.json")):
            item = summarize_proposal_smoke_dir(plan_path.parent)
            if item:
                proposal_smokes.append(item)
        for summary_path in sorted(proposal_smoke_root.rglob("summary.json")):
            item = summarize_proposal_packet_matrix(summary_path.parent)
            if item:
                proposal_packet_matrices.append(item)

    registry_status = _base_model_registry_status(cfg)
    discovery_path = discovery_report_path or training_root.parent / "reports" / "v3_base_model_candidates.json"
    discovery_status = _base_model_discovery_status(discovery_path)

    return {
        "training_data_root": str(training_data_root),
        "training_root": str(training_root),
        "counts": {
            "manifest_dirs": len(manifests),
            "target_cache_dirs": len(targets),
            "target_cache_audit_dirs": len(target_cache_audits),
            "sft_dirs": len(sfts),
            "sft_target_quality_dirs": len(sft_target_qualities),
            "run_plans": len(plans),
            "proposal_smokes": len(proposal_smokes),
            "proposal_packet_matrices": len(proposal_packet_matrices),
        },
        "base_model_registry": registry_status,
        "base_model_discovery": discovery_status,
        "manifests": manifests,
        "target_caches": targets,
        "target_cache_audits": target_cache_audits,
        "sft_datasets": sfts,
        "sft_target_qualities": sft_target_qualities,
        "run_plans": plans,
        "proposal_smokes": proposal_smokes,
        "proposal_packet_matrices": proposal_packet_matrices,
        "next_actions": infer_next_actions(
            manifests,
            targets,
            target_cache_audits,
            sfts,
            plans,
            registry_status,
            discovery_status,
            proposal_smokes=proposal_smokes,
        ),
    }


def infer_next_actions(
    manifests: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    target_cache_audits: list[dict[str, Any]] | None,
    sfts: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    base_model_registry: dict[str, Any] | None = None,
    base_model_discovery: dict[str, Any] | None = None,
    proposal_smokes: list[dict[str, Any]] | None = None,
) -> list[str]:
    def _latest(items: list[dict[str, Any]], file_key: str) -> dict[str, Any] | None:
        if not items:
            return None
        return max(
            items,
            key=lambda item: float(((item.get("files") or {}).get(file_key) or {}).get("mtime") or 0.0),
        )

    actions = []
    if (base_model_registry or {}).get("formal_32b_ready_count", 0) == 0:
        actions.append("Register a formal 32B base model with path, release_date, and size_b before real V3 manifest/training.")
        if (base_model_discovery or {}).get("candidate_count", 0):
            actions.append("Inspect runs/reports/v3_base_model_candidates.md and manually fill the selected model's trusted release_date.")
        else:
            actions.append("Run scripts/discover_v3_base_models.py or add base_models.search_roots to find local 30B-40B candidates.")
    latest_manifest = _latest(manifests, "samples")
    latest_target = _latest(targets, "targets")
    latest_audit = _latest(target_cache_audits or [], "summary_json")
    latest_sft = _latest(sfts, "summary")
    if not latest_manifest:
        actions.append("Build a V3 manifest with a real base-model release date.")
        return actions
    if latest_manifest.get("base_model_release_date") == "2025-01-01":
        actions.append("Replace the smoke base-model release date with the selected 32B model release date.")
    if not latest_target or int(latest_target.get("row_count") or 0) == 0:
        actions.append("Run a small real TeX target synthesis batch and inspect target granularity.")
    if latest_target and latest_target.get("quality", {}).get("quality_true_counts", {}).get("mock", 0):
        actions.append("Do not train on mock targets; regenerate real TeX target cache.")
    if latest_target and int(latest_target.get("row_count") or 0) > 0:
        q = latest_target.get("quality", {}).get("quality_true_counts", {})
        if int(q.get("mock", 0) or 0) == 0 and int(q.get("has_all_required_tags", 0) or 0) < int(latest_target.get("row_count") or 0):
            actions.append("Latest real target cache has missing XML tags; increase max_tokens or tighten synthesis formatting before scaling.")
        if int(q.get("mock", 0) or 0) == 0 and int(q.get("has_all_v3_required_tags", 0) or 0) < int(latest_target.get("row_count") or 0):
            actions.append("Latest real target cache is not V3-strict; regenerate with target_synthesis.prompt_version=v3_strict before scaling training.")
    if latest_audit and int(latest_audit.get("row_count") or 0) > 0 and int(latest_audit.get("accepted_count") or 0) == 0:
        actions.append("Latest target-cache audit accepted 0 rows; do not collate it into SFT data.")
    if not latest_sft or int((latest_sft.get("summary") or {}).get("collated_count") or 0) == 0:
        actions.append("Collate real target cache into V3 SFT JSONL/Parquet.")
    elif int((latest_sft.get("summary") or {}).get("collated_count") or 0) < 100:
        actions.append("Scale real target synthesis before preparing any formal SFT run; current collated data is smoke-sized.")
    if not plans:
        actions.append("Prepare a chronological 32B SFT run plan after real SFT data exists.")
    latest_plan = _latest(plans, "run_plan") if plans else None
    latest_submission = (latest_plan or {}).get("submission") or {}
    latest_phase_status = (latest_plan or {}).get("phase_status") or []
    latest_final_ready = any(
        bool(((phase.get("final_checkpoint") or {}).get("config_exists")))
        and int(((phase.get("final_checkpoint") or {}).get("model_shard_count") or 0)) > 0
        for phase in latest_phase_status
    )
    if latest_submission.get("result") == "Succeeded" and latest_final_ready:
        generated_smoke = any(item.get("proposal_generated") for item in proposal_smokes or [])
        prepared_smoke = any(item.get("ready_to_dry_run") for item in proposal_smokes or [])
        if generated_smoke:
            actions.append("Inspect trained-checkpoint proposal smoke output for granularity before scaling SFT data.")
        elif prepared_smoke:
            actions.append("After a fresh quota snapshot, run the prepared trained-checkpoint proposal smoke on DLC.")
        else:
            actions.append("Prepare and run a trained-checkpoint proposal smoke on held-out MLS task packets before scaling SFT data.")
    actions.append("After SFT smoke, run MLS proposal sweep and use expert/worker outcomes for preference/RL.")
    return actions


def render_v3_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# V3 Training Status",
        "",
        "## Overview",
        "",
        f"- Training data root: `{status.get('training_data_root')}`",
        f"- Training root: `{status.get('training_root')}`",
        f"- Manifest dirs: {status.get('counts', {}).get('manifest_dirs', 0)}",
        f"- Target cache dirs: {status.get('counts', {}).get('target_cache_dirs', 0)}",
        f"- Target cache audit dirs: {status.get('counts', {}).get('target_cache_audit_dirs', 0)}",
        f"- SFT dirs: {status.get('counts', {}).get('sft_dirs', 0)}",
        f"- SFT target quality dirs: {status.get('counts', {}).get('sft_target_quality_dirs', 0)}",
        f"- Run plans: {status.get('counts', {}).get('run_plans', 0)}",
        f"- Proposal smoke plans: {status.get('counts', {}).get('proposal_smokes', 0)}",
        f"- Proposal packet matrices: {status.get('counts', {}).get('proposal_packet_matrices', 0)}",
        f"- Registered formal 32B base models: {status.get('base_model_registry', {}).get('formal_32b_ready_count', 0)}",
        f"- Discovered 30B-40B candidates: {status.get('base_model_discovery', {}).get('candidate_count', 0)}",
        "",
        "## Next Actions",
        "",
    ]
    for action in status.get("next_actions") or []:
        lines.append(f"- {action}")

    lines.extend(["", "## Base Model Discovery", ""])
    discovery = status.get("base_model_discovery") or {}
    if not discovery.get("exists"):
        lines.append(f"- No discovery report at `{discovery.get('path')}`.")
    else:
        counts = discovery.get("counts") or {}
        lines.append(
            f"- Report: `{discovery.get('path')}` generated={discovery.get('generated_at')} "
            f"configs={counts.get('config_count', 0)} candidates={discovery.get('candidate_count', 0)}"
        )
        for item in discovery.get("candidates") or []:
            lines.append(f"- Candidate `{item.get('model_id_guess')}`: {item.get('size_b')}B, `{item.get('path')}`")

    lines.extend(["", "## Manifests", ""])
    for item in status.get("manifests") or []:
        cr = item.get("created_range") or {}
        lines.append(
            f"- `{item['name']}`: candidates={item.get('candidate_count')} "
            f"ready={item.get('ready_sft_count')} queue={item.get('synthesis_queue_count')} "
            f"refs={item.get('unique_selected_ref_count')} date={cr.get('first')}..{cr.get('last')} "
            f"base_release={item.get('base_model_release_date')}"
        )

    lines.extend(["", "## Target Caches", ""])
    for item in status.get("target_caches") or []:
        q = item.get("quality", {}).get("quality_true_counts", {})
        pv = item.get("quality", {}).get("by_prompt_version", {})
        lines.append(
            f"- `{item['name']}`: rows={item.get('row_count')} skipped={item.get('skipped_count')} "
            f"errors={item.get('error_count')} all_tags={q.get('has_all_required_tags', 0)} "
            f"v3_tags={q.get('has_all_v3_required_tags', 0)} mock={q.get('mock', 0)} "
            f"prompt_versions={pv}"
        )

    lines.extend(["", "## Target Cache Audits", ""])
    for item in status.get("target_cache_audits") or []:
        score = item.get("score") or {}
        lines.append(
            f"- `{item['name']}`: rows={item.get('row_count')} accepted={item.get('accepted_count')} "
            f"rejected={item.get('rejected_count')} mean={score.get('mean')} "
            f"require_prompt={item.get('require_prompt_version')} hard_failures={item.get('hard_failures')} "
            f"prompt_versions={item.get('prompt_versions')}"
        )

    lines.extend(["", "## SFT Datasets", ""])
    for item in status.get("sft_datasets") or []:
        counts = item.get("counts") or {}
        lines.append(
            f"- `{item['name']}`: train={counts.get('train', 0)} val={counts.get('val', 0)} "
            f"test={counts.get('test', 0)} missing_target={counts.get('missing_target', 0)}"
        )

    lines.extend(["", "## SFT Target Quality", ""])
    for item in status.get("sft_target_qualities") or []:
        score = item.get("score") or {}
        lines.append(
            f"- `{item['name']}`: rows={item.get('row_count')} "
            f"mean={score.get('mean')} median={score.get('median')} "
            f"verdicts={item.get('verdicts')} hard_failures={item.get('hard_failures')}"
        )

    lines.extend(["", "## Run Plans", ""])
    for item in status.get("run_plans") or []:
        cr = item.get("train_created_range") or {}
        submission = item.get("submission") or {}
        final_ready_count = sum(
            int(
                bool(((phase.get("final_checkpoint") or {}).get("config_exists")))
                and int(((phase.get("final_checkpoint") or {}).get("model_shard_count") or 0)) > 0
            )
            for phase in item.get("phase_status") or []
        )
        lines.append(
            f"- `{item['name']}`: phases={item.get('phase_count')} rows={item.get('train_row_count')} "
            f"date={cr.get('first')}..{cr.get('last')} base_release={item.get('base_model_release_date')} "
            f"job={submission.get('job_id') or '-'} result={submission.get('result') or submission.get('latest_status') or '-'} "
            f"final_ready={final_ready_count}/{item.get('phase_count') or 0}"
        )
    lines.extend(["", "## Proposal Smokes", ""])
    for item in status.get("proposal_smokes") or []:
        resources = item.get("resources") or {}
        preflight = item.get("preflight") or {}
        submission = item.get("submission") or {}
        quality = item.get("proposal_quality") or {}
        lines.append(
            f"- `{item['name']}`: task={item.get('task')}/{item.get('subtask')} "
            f"job={submission.get('job_id') or '-'} "
            f"status={submission.get('latest_status') or '-'} "
            f"proposal_generated={item.get('proposal_generated')} "
            f"quality={quality.get('verdict') or '-'}:{quality.get('score', '-')} "
            f"hard_failures={','.join(quality.get('hard_failures') or []) or '-'} "
            f"artifact_ready={preflight.get('artifact_ready', '-')} "
            f"launch_ready={preflight.get('launch_ready', '-')} "
            f"ready_to_dry_run={item.get('ready_to_dry_run')} "
            f"ready_to_submit={item.get('ready_to_submit')} "
            f"gpus={resources.get('gpus')} priority={resources.get('priority')} "
            f"guard={item.get('submit_guard_env') or '-'} "
            f"blocker={item.get('submit_blocker') or '-'}"
        )
    lines.extend(["", "## Proposal Packet Matrices", ""])
    for item in status.get("proposal_packet_matrices") or []:
        lines.append(
            f"- `{item['name']}`: tasks={item.get('task_count')} ok={item.get('ok_count')} "
            f"warnings={item.get('warning_count')} errors={item.get('error_count')} "
            f"min_refs={item.get('min_frontline_refs')} max_prompt_tokens={item.get('max_prompt_tokens')}"
        )
    lines.append("")
    return "\n".join(lines)


def write_v3_report(status: dict[str, Any], output_md: Path, output_json: Path | None = None) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_v3_markdown(status))
    if output_json:
        write_json(output_json, status)
