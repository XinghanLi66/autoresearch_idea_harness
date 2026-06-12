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
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Static

from autoresearch_idea_harness.article_cache_inspector import (
    inspect_article_cache,
    render_article_cache_markdown,
)
from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.v3_report import collect_v3_status, render_v3_markdown


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _read(path: str | Path | None, limit: int | None = None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return f"(missing: {p})"
    text = p.read_text(errors="replace")
    return text if limit is None else text[:limit]


def _file_path(item: dict[str, Any], key: str) -> str | None:
    info = (item.get("files") or {}).get(key) or {}
    if isinstance(info, dict):
        return info.get("path")
    return None


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
    #article {
        height: 3;
        border-bottom: solid $accent;
    }
    DataTable {
        width: 48%;
        height: 100%;
    }
    #content {
        width: 52%;
        height: 100%;
        overflow: auto;
        border-left: solid $accent;
        padding: 1;
    }
    """
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("c", "copy", "Copy"),
        Binding("1", "panel('summary')", "Summary"),
        Binding("2", "panel('prompt')", "Prompt"),
        Binding("3", "panel('messages')", "Messages"),
        Binding("4", "panel('proposal')", "Proposal"),
        Binding("5", "panel('worker')", "Worker/Eval"),
        Binding("6", "panel('report')", "Report"),
        Binding("7", "panel('article')", "Article Cache"),
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
        self.current_panel = "summary"
        self.current_text = ""
        self.article_report: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Input(placeholder="Enter arXiv id and press Enter to inspect caches/prompts/TeX details", id="article")
            with Horizontal():
                yield DataTable(id="items")
                yield Static(id="content", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#items", DataTable)
        table.cursor_type = "row"
        table.add_columns("Run", "Kind", "Task", "Status", "Artifacts")
        self.refresh_status()
        self.set_interval(self.refresh_s, self.refresh_status)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        try:
            self.article_report = inspect_article_cache(self.cfg, value)
            self.current_index = 0
            self.current_panel = "summary"
            self.refresh_status()
        except Exception as exc:
            self.current_text = f"Article cache inspection failed for {value}: {exc}"
            self.query_one("#content", Static).update(self.current_text)

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
        self.current_panel = "summary"
        self.render_panel()

    def refresh_status(self) -> None:
        self.status = collect_v3_status(self.training_data_root, self.training_root, cfg=self.cfg)
        self.items = self._build_run_items()
        table = self.query_one("#items", DataTable)
        table.clear()
        for idx, item in enumerate(self.items):
            table.add_row(
                str(item.get("name") or ""),
                str(item.get("kind") or ""),
                self._task_label(item),
                self._status_label(item),
                self._artifact_label(item),
                key=str(idx),
            )
        if self.current_index >= len(self.items):
            self.current_index = 0
        self.render_panel()

    def _build_run_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if self.article_report:
            items.extend(self._build_article_items(self.article_report))
        strict = self.status.get("strict1000_status")
        if strict:
            items.append({**strict, "kind": "training_status", "name": "strict1000_sft"})
        for key in (
            "run_plans",
            "proposal_master_prompts",
            "proposal_batch_qualities",
            "proposal_smokes",
            "precomputed_worker_eval_plans",
            "precomputed_worker_eval_results",
            "v2_3_first_report_items",
        ):
            if key == "v2_3_first_report_items":
                report = self.status.get("v2_3_first_report")
                if report:
                    items.append(report)
            else:
                items.extend(self.status.get(key) or [])
        return items

    def _build_article_items(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        aid = str(report.get("arxiv_id") or "article")
        rows: list[dict[str, Any]] = [
            {
                "kind": "article_overview",
                "name": f"{aid}/overview",
                "article": report,
                "component": "overview",
            },
            {
                "kind": "article_open_question",
                "name": f"{aid}/open_question",
                "article": report,
                "component": "open_question",
            },
        ]
        for strategy, prompt in (report.get("prompt_variants") or {}).items():
            rows.append({
                "kind": "article_prompt",
                "name": f"{aid}/{strategy}",
                "article": report,
                "component": strategy,
                "prompt_variant": prompt,
            })
        for kind, snippets in ((report.get("tex_details") or {}).get("snippets_by_kind") or {}).items():
            rows.append({
                "kind": "article_tex",
                "name": f"{aid}/tex_{kind}",
                "article": report,
                "component": kind,
                "snippets": snippets,
            })
        rows.append({
            "kind": "article_cache_files",
            "name": f"{aid}/cache_files",
            "article": report,
            "component": "cache_files",
        })
        return rows

    def _task_label(self, item: dict[str, Any]) -> str:
        if str(item.get("kind") or "").startswith("article_"):
            return "article cache"
        task = item.get("task")
        subtask = item.get("subtask")
        if task and subtask:
            return f"{task}/{subtask}"
        return str(task or item.get("run_id") or "")

    def _status_label(self, item: dict[str, Any]) -> str:
        kind = item.get("kind")
        if str(kind or "").startswith("article_"):
            if kind == "article_prompt":
                return str((item.get("prompt_variant") or {}).get("status") or "unknown")
            if kind == "article_tex":
                return str(((item.get("article") or {}).get("tex_details") or {}).get("tex_status") or "unknown")
            return "ready"
        if kind == "training_status":
            return str(item.get("training_result") or "present")
        if kind == "run_plan":
            submission = item.get("submission") or {}
            return str(submission.get("result") or submission.get("latest_status") or "planned")
        if kind == "proposal_master_prompt":
            return "ready" if (_file_path(item, "prompt") and Path(_file_path(item, "prompt")).exists()) else "missing-prompt"
        if kind == "proposal_batch_quality":
            submission = item.get("submission") or {}
            return str(submission.get("result") or submission.get("latest_status") or "summarized")
        if kind == "proposal_smoke":
            submission = item.get("submission") or {}
            return str(submission.get("latest_status") or ("generated" if item.get("proposal_generated") else "planned"))
        if kind == "precomputed_worker_eval_plan":
            submission = item.get("submission") or {}
            return str(submission.get("result") or submission.get("latest_status") or "planned")
        if kind == "precomputed_worker_eval_result":
            return f"{item.get('result_kind')} passed={item.get('passed')}"
        if kind == "v2_3_first_report":
            return "present" if item.get("exists") else "missing"
        return "ok"

    def _artifact_label(self, item: dict[str, Any]) -> str:
        kind = item.get("kind")
        if kind == "article_prompt":
            prompt = item.get("prompt_variant") or {}
            meta = prompt.get("metadata") or {}
            counts = meta.get("abstract_source_counts") or {}
            cached = counts.get("cached_abstract_summary", 0)
            raw = counts.get("raw_abstract_under_limit", 0)
            fallback = counts.get("fallback_token_trimmed_abstract", 0)
            return f"chars={prompt.get('chars')} abs={cached}/{raw}/{fallback}"
        if kind == "article_open_question":
            q = (item.get("article") or {}).get("open_question") or {}
            return f"{q.get('tokens')}tok complete={q.get('complete')}"
        if kind == "article_tex":
            return f"snippets={len(item.get('snippets') or [])}"
        if str(kind or "").startswith("article_"):
            return str(item.get("component") or "")
        files = item.get("files") or {}
        names = [k for k, v in files.items() if isinstance(v, dict) and v.get("exists")]
        if names:
            return ",".join(names[:4])
        if item.get("kind") == "proposal_batch_quality":
            return f"tasks={item.get('task_count')} qpass={item.get('quality_pass_count')}"
        if item.get("kind") == "training_status":
            return f"final={item.get('final_ready_count')}/{item.get('phase_count')}"
        return ""

    def render_panel(self) -> None:
        content = self.query_one("#content", Static)
        item = self.items[self.current_index] if self.items else {}
        if str(item.get("kind") or "").startswith("article_") and self.current_panel != "report":
            text = self._panel_article_item(item)
        elif self.current_panel == "summary":
            text = self._panel_summary(item)
        elif self.current_panel == "prompt":
            text = self._panel_prompt(item)
        elif self.current_panel == "messages":
            text = self._panel_messages(item)
        elif self.current_panel == "proposal":
            text = self._panel_proposal(item)
        elif self.current_panel == "worker":
            text = self._panel_worker(item)
        elif self.current_panel == "report":
            text = self._panel_report(item)
        elif self.current_panel == "article":
            text = self._panel_article()
        else:
            text = _fmt(item)
        self.current_text = text
        content.update(text)

    def _panel_summary(self, item: dict[str, Any]) -> str:
        lines = [
            f"# {item.get('name') or '(no run selected)'}",
            "",
            f"- kind: `{item.get('kind')}`",
            f"- status: `{self._status_label(item)}`",
            f"- task: `{self._task_label(item)}`",
            f"- path: `{item.get('path') or item.get('run_root') or ''}`",
            "",
            "## Key Fields",
            "```json",
            _fmt({k: v for k, v in item.items() if k not in {"rows", "selected", "preview", "prompt_preview"}})[:12000],
            "```",
        ]
        if item.get("prompt_preview"):
            lines.extend(["", "## Prompt Preview", "```text", str(item.get("prompt_preview")), "```"])
        return "\n".join(lines)

    def _panel_prompt(self, item: dict[str, Any]) -> str:
        candidates = [
            _file_path(item, "prompt"),
            str(Path(str(item.get("path") or "")) / "prompt.txt") if item.get("path") else None,
            str(Path(str(item.get("path") or "")) / "output" / "prompt.txt") if item.get("path") else None,
        ]
        for path in candidates:
            if path and Path(path).exists():
                return f"# Master Prompt\n`{path}`\n\n```text\n{_read(path)}\n```"
        return "(no master prompt file found for this row)"

    def _panel_messages(self, item: dict[str, Any]) -> str:
        path = _file_path(item, "messages")
        if not path and item.get("path"):
            candidates = [Path(str(item["path"])) / "messages.json", Path(str(item["path"])) / "output" / "messages.json"]
            path = str(next((p for p in candidates if p.exists()), ""))
        if not path:
            return "(no messages.json found for this row)"
        return f"# Messages\n`{path}`\n\n```json\n{_read(path)}\n```"

    def _panel_proposal(self, item: dict[str, Any]) -> str:
        candidates = [
            _file_path(item, "proposal"),
            str(Path(str(item.get("path") or "")) / "proposal.txt") if item.get("path") else None,
            str(Path(str(item.get("path") or "")) / "output" / "proposal.txt") if item.get("path") else None,
        ]
        for path in candidates:
            if path and Path(path).exists():
                return f"# Proposal\n`{path}`\n\n```text\n{_read(path)}\n```"
        return "(no proposal file found for this row)"

    def _panel_worker(self, item: dict[str, Any]) -> str:
        base = Path(str(item.get("path") or ""))
        sections = []
        for label, path in [
            ("Worker Prompt", _file_path(item, "worker_prompt") or str(base / "worker_prompt.txt")),
            ("Worker Log", _file_path(item, "worker_log") or str(base / "worker.log")),
            ("Eval Log", _file_path(item, "eval_log") or str(base / "eval.log")),
            ("Result", _file_path(item, "result") or str(base / "result.json")),
        ]:
            if path and Path(path).exists():
                fence = "json" if str(path).endswith(".json") else "text"
                sections.append(f"# {label}\n`{path}`\n\n```{fence}\n{_read(path)}\n```")
        return "\n\n".join(sections) if sections else "(no worker/eval artifacts found for this row)"

    def _panel_report(self, item: dict[str, Any]) -> str:
        for key in ("summary_md", "latest_md", "report", "summary"):
            path = _file_path(item, key)
            if path and Path(path).exists():
                return f"# Report\n`{path}`\n\n{_read(path)}"
        if item.get("preview"):
            return str(item.get("preview"))
        return render_v3_markdown(self.status)

    def _panel_article(self) -> str:
        if not self.article_report:
            return "Enter an arXiv id above and press Enter. Example: 2504.00302"
        return render_article_cache_markdown(self.article_report, include_prompts=True)

    def _panel_article_item(self, item: dict[str, Any]) -> str:
        report = item.get("article") or {}
        kind = item.get("kind")
        aid = report.get("arxiv_id")
        if kind == "article_overview":
            return "\n".join([
                f"# Article Overview: {aid}",
                "",
                "## Metadata",
                "```json",
                _fmt(report.get("metadata") or {}),
                "```",
                "",
                "## Cache Completeness",
                "```json",
                _fmt(self._article_cache_completeness(report)),
                "```",
            ])
        if kind == "article_open_question":
            q = report.get("open_question") or {}
            return "\n".join([
                f"# Open Question: {aid}",
                "",
                f"- source: `{q.get('source')}`",
                f"- tokens: `{q.get('tokens')}`",
                f"- complete: `{q.get('complete')}`",
                "",
                "## Compact Question",
                q.get("text") or "(missing)",
                "",
                "## Raw Cache",
                "```text",
                q.get("raw") or "",
                "```",
            ])
        if kind == "article_prompt":
            strategy = item.get("component")
            prompt = item.get("prompt_variant") or {}
            return "\n".join([
                f"# Prompt: {aid}/{strategy}",
                "",
                f"- status: `{prompt.get('status')}`",
                f"- chars: `{prompt.get('chars')}`",
                "",
                "## Metadata",
                "```json",
                _fmt(prompt.get("metadata") or {}),
                "```",
                "",
                "## Prompt",
                "```text",
                prompt.get("prompt") or "",
                "```",
            ])
        if kind == "article_tex":
            lines = [f"# TeX Details: {aid}/{item.get('component')}", ""]
            for idx, snippet in enumerate(item.get("snippets") or [], start=1):
                lines.extend([
                    f"## {idx}. {snippet.get('heading') or '(no heading)'}",
                    f"`{snippet.get('source') or ''}`",
                    "",
                    snippet.get("text") or "",
                    "",
                ])
            return "\n".join(lines).rstrip()
        if kind == "article_cache_files":
            return "\n".join([
                f"# Cache Files: {aid}",
                "",
                "```json",
                _fmt(report.get("cache_files") or {}),
                "```",
            ])
        return render_article_cache_markdown(report, include_prompts=True)

    @staticmethod
    def _article_cache_completeness(report: dict[str, Any]) -> dict[str, Any]:
        variants = report.get("prompt_variants") or {}
        prompt_status = {name: item.get("status") for name, item in variants.items()}
        abstract_fallbacks = {}
        for name, item in variants.items():
            counts = (item.get("metadata") or {}).get("abstract_source_counts") or {}
            if counts:
                abstract_fallbacks[name] = counts.get("fallback_token_trimmed_abstract", 0)
        tex_kinds = sorted(((report.get("tex_details") or {}).get("snippets_by_kind") or {}).keys())
        return {
            "open_question_complete": (report.get("open_question") or {}).get("complete"),
            "prompt_status": prompt_status,
            "fallback_abstract_counts": abstract_fallbacks,
            "all_prompt_ref_abstracts_cached_or_raw_under_limit": all(v == 0 for v in abstract_fallbacks.values()),
            "tex_status": (report.get("tex_details") or {}).get("tex_status"),
            "tex_kinds": tex_kinds,
            "has_method_implementation_evaluation_results": all(
                key in tex_kinds for key in ["method", "implementation", "evaluation", "results"]
            ),
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only V3 run/prompt/cache dashboard.")
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
