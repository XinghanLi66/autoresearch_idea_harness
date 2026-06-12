#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Static

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.v3_report import collect_v3_status, render_v3_markdown


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _copy_to_clipboard(text: str) -> str:
    for cmd in (
        ["pbcopy"],
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["tmux", "load-buffer", "-w", "-"],
        ["tmux", "load-buffer", "-"],
    ):
        try:
            subprocess.run(cmd, input=text, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "copied"
        except Exception:
            continue
    return "no clipboard backend"


class V3TrainingDashboard(App):
    CSS = """
    DataTable {
        width: 46%;
        height: 100%;
    }
    #content {
        width: 54%;
        height: 100%;
        overflow: auto;
        border-left: solid $accent;
        padding: 1;
    }
    """
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("c", "copy", "Copy"),
        Binding("1", "panel('overview')", "Overview"),
        Binding("2", "panel('detail')", "Detail"),
        Binding("3", "panel('report')", "Report"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, training_data_root: Path, training_root: Path, cfg: dict[str, Any] | None = None, refresh_s: float = 10.0) -> None:
        super().__init__()
        self.training_data_root = training_data_root
        self.training_root = training_root
        self.cfg = cfg or {}
        self.refresh_s = refresh_s
        self.status: dict[str, Any] = {}
        self.items: list[dict[str, Any]] = []
        self.current_index = 0
        self.current_panel = "overview"
        self.current_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="items")
            yield Static(id="content", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#items", DataTable)
        table.cursor_type = "row"
        table.add_columns("Kind", "Name", "Rows", "Ready", "Status")
        self.refresh_status()
        self.set_interval(self.refresh_s, self.refresh_status)

    def action_refresh(self) -> None:
        self.refresh_status()

    def action_copy(self) -> None:
        self.notify(_copy_to_clipboard(self.current_text))

    def action_panel(self, panel: str) -> None:
        self.current_panel = panel
        self.render_panel()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            self.current_index = int(getattr(event.row_key, "value", str(event.row_key)))
        except Exception:
            self.current_index = 0
        self.current_panel = "detail"
        self.render_panel()

    def refresh_status(self) -> None:
        self.status = collect_v3_status(self.training_data_root, self.training_root, cfg=self.cfg)
        self.items = []
        for key in (
            "strict1000_status_items",
            "manifests",
            "target_caches",
            "target_cache_audits",
            "sft_datasets",
            "sft_target_qualities",
            "run_plans",
            "proposal_smokes",
            "proposal_packet_matrices",
            "proposal_batch_qualities",
            "proposal_master_prompts",
            "v2_3_first_report_items",
            "precomputed_worker_eval_plans",
            "precomputed_worker_eval_results",
        ):
            if key == "strict1000_status_items":
                self.items.append(self.status.get("strict1000_status") or {"kind": "strict1000_sft_status", "name": "strict1000_sft"})
            elif key == "v2_3_first_report_items":
                self.items.append(self.status.get("v2_3_first_report") or {"kind": "v2_3_first_report", "name": "v2_3_mls10_modules9"})
            else:
                self.items.extend(self.status.get(key) or [])
        table = self.query_one("#items", DataTable)
        table.clear()
        for idx, item in enumerate(self.items):
            table.add_row(
                str(item.get("kind")),
                str(item.get("name")),
                self._item_rows(item),
                self._item_ready(item),
                self._item_status(item),
                key=str(idx),
            )
        if self.current_index >= len(self.items):
            self.current_index = 0
        self.render_panel()

    def _item_rows(self, item: dict[str, Any]) -> str:
        if item.get("kind") == "manifest":
            return str(item.get("candidate_count") or 0)
        if item.get("kind") == "strict1000_sft_status":
            counts = item.get("sft_counts") or {}
            return str(counts.get("train") or item.get("collated_count") or 0)
        if item.get("kind") == "target_cache":
            return str(item.get("row_count") or 0)
        if item.get("kind") == "target_cache_audit":
            return str(item.get("row_count") or 0)
        if item.get("kind") == "sft":
            return str((item.get("counts") or {}).get("train", 0))
        if item.get("kind") == "sft_target_quality":
            return str(item.get("row_count") or 0)
        if item.get("kind") == "run_plan":
            return str(item.get("train_row_count") or 0)
        if item.get("kind") == "proposal_smoke":
            resources = item.get("resources") or {}
            return f"gpu={resources.get('gpus') or 0}"
        if item.get("kind") == "proposal_packet_matrix":
            return str(item.get("task_count") or 0)
        if item.get("kind") == "proposal_batch_quality":
            return str(item.get("task_count") or 0)
        if item.get("kind") == "proposal_master_prompt":
            return str(item.get("prompt_tokens") or "")
        if item.get("kind") == "v2_3_first_report":
            return str(item.get("samples") or 0)
        if item.get("kind") == "precomputed_worker_eval_plan":
            return str(item.get("selected_count") or 0)
        if item.get("kind") == "precomputed_worker_eval_result":
            return str(item.get("val_metric") or "")
        return ""

    def _item_ready(self, item: dict[str, Any]) -> str:
        if item.get("kind") == "manifest":
            return str(item.get("ready_sft_count") or 0)
        if item.get("kind") == "strict1000_sft_status":
            return f"result={item.get('training_result') or '-'} final={item.get('final_ready_count') or 0}/{item.get('phase_count') or 0}"
        if item.get("kind") == "target_cache":
            q = item.get("quality", {}).get("quality_true_counts", {})
            return f"v3={q.get('has_all_v3_required_tags', 0)} tags={q.get('has_all_required_tags', 0)}"
        if item.get("kind") == "target_cache_audit":
            return f"acc={item.get('accepted_count') or 0}/{item.get('row_count') or 0}"
        if item.get("kind") == "sft":
            counts = item.get("counts") or {}
            return f"val={counts.get('val', 0)} test={counts.get('test', 0)}"
        if item.get("kind") == "sft_target_quality":
            score = item.get("score") or {}
            return f"mean={score.get('mean', '-')}"
        if item.get("kind") == "run_plan":
            ready = 0
            for phase in item.get("phase_status") or []:
                final = phase.get("final_checkpoint") or {}
                ready += int(bool(final.get("config_exists")) and int(final.get("model_shard_count") or 0) > 0)
            return f"final={ready}/{item.get('phase_count') or 0}"
        if item.get("kind") == "proposal_smoke":
            quality = item.get("proposal_quality") or {}
            if quality:
                return f"q={quality.get('score', '-')}/{quality.get('max_score', 100)} {quality.get('verdict', '-')}"
            preflight = item.get("preflight") or {}
            artifact = preflight.get("artifact_ready")
            launch = preflight.get("launch_ready")
            if artifact is None and launch is None:
                return f"dry={int(bool(item.get('ready_to_dry_run')))} submit={int(bool(item.get('ready_to_submit')))}"
            return f"artifact={int(bool(artifact))} launch={int(bool(launch))}"
        if item.get("kind") == "proposal_packet_matrix":
            return f"ok={item.get('ok_count') or 0}/{item.get('task_count') or 0}"
        if item.get("kind") == "proposal_batch_quality":
            return f"qpass={item.get('quality_pass_count') or 0}/{item.get('task_count') or 0} mean={item.get('quality_mean_score', '-')}"
        if item.get("kind") == "proposal_master_prompt":
            files = item.get("files") or {}
            return f"prompt={int(bool((files.get('prompt') or {}).get('exists')))} messages={item.get('message_count') or 0}"
        if item.get("kind") == "v2_3_first_report":
            return f"done={item.get('done') or 0} run={item.get('running') or 0} err={item.get('errors') or 0}"
        if item.get("kind") == "precomputed_worker_eval_plan":
            return f"dry={int(bool(item.get('ready_to_dry_run')))} submit={int(bool(item.get('ready_to_submit')))}"
        if item.get("kind") == "precomputed_worker_eval_result":
            return f"{item.get('result_kind', 'unknown')} passed={item.get('passed')} pass_metric={item.get('pass_metric', '-')}"
        return ""

    def _item_status(self, item: dict[str, Any]) -> str:
        if item.get("kind") == "target_cache":
            q = item.get("quality", {}).get("quality_true_counts", {})
            if q.get("mock", 0):
                return "mock"
            if int(q.get("has_all_v3_required_tags", 0) or 0) < int(item.get("row_count") or 0):
                return "not-v3-strict"
            return "real-or-empty"
        if item.get("kind") == "target_cache_audit":
            if int(item.get("accepted_count") or 0) == 0 and int(item.get("row_count") or 0) > 0:
                return "reject-all"
            failures = item.get("hard_failures") or {}
            if failures:
                return "needs-review"
            return "ok"
        if item.get("kind") == "manifest":
            if item.get("base_model_release_date") == "2025-01-01":
                return "smoke-date"
            return "ok"
        if item.get("kind") == "strict1000_sft_status":
            if not item.get("exists"):
                return "missing"
            result = str(item.get("training_result") or "")
            if result == "Succeeded" and int(item.get("final_ready_count") or 0) > 0:
                return "succeeded"
            return result or "present"
        if item.get("kind") == "run_plan":
            submission = item.get("submission") or {}
            return str(submission.get("result") or submission.get("latest_status") or "planned")
        if item.get("kind") == "sft_target_quality":
            failures = item.get("hard_failures") or {}
            if failures:
                return "needs-review"
            return "ok"
        if item.get("kind") == "proposal_smoke":
            submission = item.get("submission") or {}
            if submission.get("latest_status"):
                return str(submission.get("latest_status"))
            preflight = item.get("preflight") or {}
            if preflight and not preflight.get("artifact_ready"):
                return "preflight-error"
            if preflight and preflight.get("launch_ready"):
                return "launch-ready"
            if item.get("errors"):
                return "error"
            if item.get("proposal_generated"):
                return "generated"
            if item.get("ready_to_submit"):
                return "ready"
            if item.get("ready_to_dry_run"):
                return "needs-quota"
            return "planned"
        if item.get("kind") == "proposal_packet_matrix":
            if int(item.get("error_count") or 0):
                return "error"
            if int(item.get("warning_count") or 0):
                return "warning"
            return "ok"
        if item.get("kind") == "proposal_batch_quality":
            submission = item.get("submission") or {}
            if int(item.get("error_count") or 0):
                return "error"
            if submission.get("result") or submission.get("latest_status"):
                return str(submission.get("result") or submission.get("latest_status"))
            return "ok"
        if item.get("kind") == "proposal_master_prompt":
            files = item.get("files") or {}
            if not (files.get("prompt") or {}).get("exists"):
                return "missing-prompt"
            return "ready"
        if item.get("kind") == "v2_3_first_report":
            return "present" if item.get("exists") else "missing"
        if item.get("kind") == "precomputed_worker_eval_plan":
            submission = item.get("submission") or {}
            if item.get("errors"):
                return "error"
            if submission.get("result") or submission.get("latest_status"):
                return str(submission.get("result") or submission.get("latest_status"))
            if item.get("ready_to_submit"):
                return "ready"
            if item.get("ready_to_dry_run"):
                return "needs-quota"
            return "planned"
        if item.get("kind") == "precomputed_worker_eval_result":
            if item.get("result_kind") in {"error", "fixture", "skipped", "incomplete"}:
                return str(item.get("result_kind"))
            if item.get("error"):
                return "error"
            return str(item.get("worker_status") or "present")
        return "ok"

    def render_panel(self) -> None:
        content = self.query_one("#content", Static)
        if self.current_panel == "overview":
            text = _fmt({
                "counts": self.status.get("counts"),
                "base_model_registry": self.status.get("base_model_registry"),
                "base_model_discovery": self.status.get("base_model_discovery"),
                "strict1000_status": self.status.get("strict1000_status"),
                "v2_3_first_report": {
                    k: v
                    for k, v in (self.status.get("v2_3_first_report") or {}).items()
                    if k != "preview"
                },
                "proposal_batch_qualities": self.status.get("proposal_batch_qualities"),
                "precomputed_worker_eval_plans": self.status.get("precomputed_worker_eval_plans"),
                "precomputed_worker_eval_real_results": self.status.get("precomputed_worker_eval_real_results"),
                "precomputed_worker_eval_results": self.status.get("precomputed_worker_eval_results"),
                "next_actions": self.status.get("next_actions"),
                "training_data_root": str(self.training_data_root),
                "training_root": str(self.training_root),
            })
        elif self.current_panel == "report":
            text = render_v3_markdown(self.status)
        else:
            item = self.items[self.current_index] if self.items else {}
            text = _fmt(item)
        self.current_text = text
        content.update(f"V3 {self.current_panel}\n\n{text}")


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only V3 training data/run dashboard.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--training-data-root", default=str(ROOT / "runs" / "training_data"))
    p.add_argument("--training-root", default=str(ROOT / "runs" / "training"))
    p.add_argument("--refresh", type=float, default=10.0)
    args = p.parse_args()
    V3TrainingDashboard(
        training_data_root=Path(args.training_data_root),
        training_root=Path(args.training_root),
        cfg=load_config(args.config),
        refresh_s=args.refresh,
    ).run()


if __name__ == "__main__":
    main()
