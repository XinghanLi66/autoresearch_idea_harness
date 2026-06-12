#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Select, Static

from autoresearch_idea_harness.article_cache_inspector import inspect_article_cache
from autoresearch_idea_harness.io import load_config

MAX_TEXT_CHARS = 1_500_000


@dataclass
class DashboardItem:
    item_type: str
    name: str
    status: str
    summary: str
    scope_type: str
    scope_id: str
    path: Path | None = None
    content: str | None = None
    files: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunTab:
    key: str
    title: str
    paths: list[Path] = field(default_factory=list)
    inline: str | None = None


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


def _read(path: Path | None, *, max_chars: int = MAX_TEXT_CHARS) -> str:
    if not path:
        return "(missing)"
    if not path.exists():
        return f"(missing: {path})"
    text = path.read_text(errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... truncated at {max_chars} chars; open file for full content: {path}\n"
    return text


def _read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}


def _file_status(path: Path | None) -> str:
    if not path:
        return "missing"
    return "ready" if path.exists() else "missing"


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


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


def _normalize_query(value: str) -> str:
    value = value.strip()
    if value.startswith("arxiv:"):
        return value.split(":", 1)[1].strip()
    if value.startswith("mls:"):
        return value.split(":", 1)[1].strip()
    return value


def _task_from_packet(path: Path) -> tuple[str | None, str | None, dict[str, Any]]:
    packet = _read_json(path)
    task = packet.get("task") or packet.get("task_id")
    subtask = packet.get("subtask") or packet.get("subtask_id")
    return (str(task) if task else None, str(subtask) if subtask else None, packet)


def _task_matches(task: str | None, subtask: str | None, query: str) -> bool:
    q = query.lower()
    candidates = [str(task or "").lower(), str(subtask or "").lower(), f"{task or ''}/{subtask or ''}".lower()]
    return any(q == c or q in c for c in candidates if c)


def _artifact_files(base: Path) -> dict[str, Path]:
    candidates = {
        "task_packet": [base / "task_packet.json"],
        "master_prompt": [base / "prompt.txt", base / "master_prompt.txt", base / "output" / "prompt.txt"],
        "messages": [base / "messages.json", base / "output" / "messages.json"],
        "proposal": [base / "proposal.txt", base / "output" / "proposal.txt"],
        "expert": [
            base / "expert_forecasts.jsonl",
            base / "forecasts.jsonl",
            base / "market" / "forecasts.jsonl",
            base / "expert_responses.jsonl",
        ],
        "worker_prompt": [base / "worker_prompt.txt", base / "output" / "worker_prompt.txt"],
        "worker_log": [base / "worker.log", base / "output" / "worker.log"],
        "eval_log": [base / "eval.log", base / "workspace" / "eval.log", base / "output" / "eval.log"],
        "result": [base / "result.json", base / "workspace" / "result.json", base / "output" / "result.json"],
        "settlement": [base / "settlement.json", base / "market" / "settlement.json"],
        "summary": [base / "summary.json", base / "report.md", base / "meta.json", base / "proposal_quality.json"],
        "error": [base / "error.json"],
    }
    files: dict[str, Path] = {}
    for name, paths in candidates.items():
        first = _first_existing(paths)
        files[name] = first or paths[0]
    return files


def _has_run_artifacts(files: dict[str, Path]) -> bool:
    return any(files[key].exists() for key in ("worker_prompt", "worker_log", "eval_log", "result", "settlement", "error"))


def _artifact_summary(files: dict[str, Path]) -> str:
    ready = [name for name, path in files.items() if path.exists()]
    return ",".join(ready[:8]) if ready else "no files"


def _status_from_files(files: dict[str, Path]) -> str:
    if files.get("result") and files["result"].exists():
        result = _read_json(files["result"])
        passed = result.get("passed")
        parsed = result.get("_parsed") or {}
        if passed is None:
            passed = parsed.get("passed")
        if passed is not None:
            return f"result pass={passed}"
        return "result"
    if files.get("error") and files["error"].exists():
        return "error"
    if files.get("worker_log") and files["worker_log"].exists():
        return "running/logged"
    if files.get("proposal") and files["proposal"].exists():
        return "proposal"
    if files.get("master_prompt") and files["master_prompt"].exists():
        return "prompt"
    if files.get("task_packet") and files["task_packet"].exists():
        return "packet"
    return "missing"


def _run_title(path: Path, task: str | None, subtask: str | None) -> str:
    task_label = task or "unknown_task"
    if subtask:
        task_label = f"{task_label}/{subtask}"
    parent = path.parent.name
    if parent == "output":
        parent = path.parent.parent.name
    return f"{task_label} :: {parent}/{path.name}"


def _iter_task_packet_paths(runs_root: Path) -> list[Path]:
    patterns = [
        "formal_sweeps/*/*/task_packet.json",
        "v3_precomputed_worker_eval/*/output/*/task_packet.json",
        "v3_precomputed_worker_eval/*/*/task_packet.json",
        "v3_checkpoint_proposal_smoke/*/task_packet.json",
        "v3_checkpoint_proposal_smoke/*/output/task_packet.json",
        "v3_checkpoint_proposal_smoke/task_packet_matrix/*/task_packet.json",
        "v3_checkpoint_proposal_batch/*/*/task_packet.json",
        "v3_checkpoint_proposal_batch/*/output/*/task_packet.json",
        "end_to_end/*/task_packet.json",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(runs_root.glob(pattern))
    return sorted(set(paths))


def _make_run_item(base: Path, query: str) -> DashboardItem | None:
    packet_path = base / "task_packet.json"
    if not packet_path.exists():
        return None
    task, subtask, packet = _task_from_packet(packet_path)
    if not _task_matches(task, subtask, query):
        return None
    files = _artifact_files(base)
    item_type = "run" if _has_run_artifacts(files) else "cache"
    module = packet.get("module_id") or packet.get("proposal_module") or packet.get("generator") or ""
    status = _status_from_files(files)
    summary = _artifact_summary(files)
    if module:
        summary = f"{module} | {summary}"
    return DashboardItem(
        item_type=item_type,
        name=_run_title(base, task, subtask),
        status=status,
        summary=summary,
        scope_type="mls",
        scope_id=str(task or query),
        path=base,
        files=files,
        metadata={"task": task, "subtask": subtask, "module": module},
    )


def discover_mls_items(runs_root: Path, training_data_root: Path, training_root: Path, cfg: dict[str, Any], query: str) -> list[DashboardItem]:
    items: list[DashboardItem] = []
    seen: set[Path] = set()

    for packet_path in _iter_task_packet_paths(runs_root):
        base = packet_path.parent.resolve()
        if base in seen:
            continue
        item = _make_run_item(base, query)
        if item:
            seen.add(base)
            items.append(item)

    items.sort(key=lambda item: (0 if item.item_type == "cache" else 1, item.name, str(item.path or "")))
    return items


def _article_status(item: dict[str, Any]) -> str:
    status = item.get("status")
    if status:
        return str(status)
    text = item.get("text") or item.get("prompt")
    return "ready" if text else "missing"


def discover_article_items(cfg: dict[str, Any], arxiv_id: str) -> list[DashboardItem]:
    cached_report = ROOT / "runs" / "reports" / f"article_cache_{arxiv_id}.full_cache.json"
    report = _read_json(cached_report) if cached_report.exists() else inspect_article_cache(cfg, arxiv_id)
    items: list[DashboardItem] = []
    metadata = report.get("metadata") or {}
    overview = "\n".join(
        [
            f"# Article Overview: {arxiv_id}",
            "",
            "## Metadata",
            "```json",
            _fmt(metadata),
            "```",
            "",
            "## Cache Completeness",
            "```json",
            _fmt(_article_completeness(report)),
            "```",
        ]
    )
    items.append(
        DashboardItem(
            item_type="cache",
            name="overview",
            status="ready" if report.get("found_dataset_record") or report.get("found_classified_row") else "missing",
            summary=str(metadata.get("title") or ""),
            scope_type="arxiv",
            scope_id=arxiv_id,
            content=overview,
            metadata={"component": "overview", "cached_report": str(cached_report) if cached_report.exists() else None},
        )
    )

    q = report.get("open_question") or {}
    question_text = "\n".join(
        [
            f"# Open Question: {arxiv_id}",
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
            q.get("raw") or "(missing)",
            "```",
        ]
    )
    items.append(
        DashboardItem(
            item_type="cache",
            name="open_question",
            status="ready" if q.get("text") else "missing",
            summary=f"{q.get('tokens')} tokens complete={q.get('complete')}",
            scope_type="arxiv",
            scope_id=arxiv_id,
            content=question_text,
            metadata={"component": "open_question"},
        )
    )

    for strategy, prompt in (report.get("prompt_variants") or {}).items():
        meta = prompt.get("metadata") or {}
        counts = meta.get("abstract_source_counts") or {}
        text = "\n".join(
            [
                f"# Prompt Cache: {arxiv_id}/{strategy}",
                "",
                f"- status: `{prompt.get('status') or 'missing'}`",
                f"- chars: `{prompt.get('chars')}`",
                f"- abstract sources cached/raw/fallback: `{counts.get('cached_abstract_summary', 0)}/{counts.get('raw_abstract_under_limit', 0)}/{counts.get('fallback_token_trimmed_abstract', 0)}`",
                "",
                "## Metadata",
                "```json",
                _fmt(meta),
                "```",
                "",
                "## Prompt",
                "```text",
                prompt.get("prompt") or "(missing)",
                "```",
            ]
        )
        items.append(
            DashboardItem(
                item_type="cache",
                name=f"prompt/{strategy}",
                status=str(prompt.get("status") or "missing"),
                summary=f"chars={prompt.get('chars')} abs={counts.get('cached_abstract_summary', 0)}/{counts.get('raw_abstract_under_limit', 0)}/{counts.get('fallback_token_trimmed_abstract', 0)}",
                scope_type="arxiv",
                scope_id=arxiv_id,
                content=text,
                metadata={"component": strategy},
            )
        )

    tex = report.get("tex_details") or {}
    snippets_by_kind = tex.get("snippets_by_kind") or {}
    expected_tex = ["abstract", "problem", "method", "implementation", "algorithm_or_system", "training_or_data_recipe", "evaluation", "results", "risks_and_limitations"]
    for kind in expected_tex:
        snippets = snippets_by_kind.get(kind) or []
        lines = [
            f"# TeX Detail Cache: {arxiv_id}/{kind}",
            "",
            f"- tex_status: `{tex.get('tex_status')}`",
            f"- snippets: `{len(snippets)}`",
            "",
        ]
        if snippets:
            for idx, snippet in enumerate(snippets, start=1):
                lines.extend(
                    [
                        f"## {idx}. {snippet.get('heading') or '(no heading)'}",
                        f"`{snippet.get('source') or ''}`",
                        "",
                        snippet.get("text") or "",
                        "",
                    ]
                )
        else:
            lines.append("(missing)")
        items.append(
            DashboardItem(
                item_type="cache",
                name=f"tex/{kind}",
                status="ready" if snippets else "missing",
                summary=f"snippets={len(snippets)}",
                scope_type="arxiv",
                scope_id=arxiv_id,
                content="\n".join(lines).rstrip() + "\n",
                metadata={"component": kind},
            )
        )

    cache_files = "\n".join(
        [
            f"# Cache Files: {arxiv_id}",
            "",
            "```json",
            _fmt(report.get("cache_files") or {}),
            "```",
        ]
    )
    items.append(
        DashboardItem(
            item_type="cache",
            name="cache_files",
            status="ready",
            summary="prompt/detail cache locations",
            scope_type="arxiv",
            scope_id=arxiv_id,
            content=cache_files,
            metadata={"component": "cache_files"},
        )
    )
    return items


def _article_completeness(report: dict[str, Any]) -> dict[str, Any]:
    variants = report.get("prompt_variants") or {}
    fallback_counts = {}
    for strategy, item in variants.items():
        counts = (item.get("metadata") or {}).get("abstract_source_counts") or {}
        fallback_counts[strategy] = counts.get("fallback_token_trimmed_abstract", 0)
    tex_kinds = sorted(((report.get("tex_details") or {}).get("snippets_by_kind") or {}).keys())
    return {
        "found_dataset_record": report.get("found_dataset_record"),
        "found_classified_row": report.get("found_classified_row"),
        "open_question_complete": (report.get("open_question") or {}).get("complete"),
        "prompt_status": {k: v.get("status") for k, v in variants.items()},
        "fallback_abstract_counts": fallback_counts,
        "tex_status": (report.get("tex_details") or {}).get("tex_status"),
        "tex_kinds": tex_kinds,
    }


def _tab_text(tab: RunTab) -> str:
    if tab.inline is not None:
        return tab.inline
    lines = [f"# {tab.title}", ""]
    existing = [p for p in tab.paths if p.exists()]
    if not existing:
        lines.extend(["(missing)", "", "Expected paths:", "```text", "\n".join(str(p) for p in tab.paths), "```"])
        return "\n".join(lines)
    for path in existing:
        suffix = path.suffix.lower()
        fence = "json" if suffix == ".json" else "text"
        lines.extend([f"`{path}`", "", f"```{fence}", _read(path), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _cache_item_text(item: DashboardItem) -> str:
    if item.content is not None:
        return item.content
    lines = [
        f"# Cache: {item.name}",
        "",
        f"- scope: `{item.scope_type}:{item.scope_id}`",
        f"- status: `{item.status}`",
        f"- path: `{item.path or ''}`",
        "",
        "## Metadata",
        "```json",
        _fmt(item.metadata),
        "```",
        "",
    ]
    if not item.files:
        lines.append("(missing)")
        return "\n".join(lines)
    for label in ("task_packet", "master_prompt", "messages", "proposal", "summary", "error"):
        path = item.files.get(label)
        if not path:
            continue
        title = label.replace("_", " ").title()
        lines.extend([f"## {title}", f"`{path}`", ""])
        if path.exists():
            fence = "json" if path.suffix == ".json" else "text"
            lines.extend([f"```{fence}", _read(path), "```", ""])
        else:
            lines.extend([f"(missing: {path})", ""])
    return "\n".join(lines).rstrip() + "\n"


def _run_tabs(item: DashboardItem) -> list[RunTab]:
    base = item.path or Path()
    files = item.files or _artifact_files(base)
    overview = "\n".join(
        [
            "# Run Overview",
            "",
            f"- name: `{item.name}`",
            f"- status: `{item.status}`",
            f"- scope: `{item.scope_type}:{item.scope_id}`",
            f"- path: `{item.path or ''}`",
            "",
            "## Metadata",
            "```json",
            _fmt(item.metadata),
            "```",
            "",
            "## Files",
            "```json",
            _fmt({name: {"path": str(path), "exists": path.exists()} for name, path in files.items()}),
            "```",
        ]
    )
    return [
        RunTab("overview", "Overview", inline=overview),
        RunTab("packet", "1 Packet", [files.get("task_packet") or base / "task_packet.json"]),
        RunTab("master", "2 Master Prompt", [files.get("master_prompt") or base / "prompt.txt", files.get("messages") or base / "messages.json"]),
        RunTab("expert", "3 Expert / Market", [files.get("expert") or base / "expert_forecasts.jsonl", files.get("settlement") or base / "settlement.json"]),
        RunTab("worker_prompt", "4 Worker Prompt", [files.get("worker_prompt") or base / "worker_prompt.txt"]),
        RunTab("worker_log", "5 Worker Log", [files.get("worker_log") or base / "worker.log"]),
        RunTab("eval_log", "6 Eval Log", [files.get("eval_log") or base / "eval.log"]),
        RunTab("result", "7 Result", [files.get("result") or base / "result.json"]),
        RunTab("summary", "8 Summary / Report", [files.get("summary") or base / "summary.json", files.get("error") or base / "error.json"]),
        RunTab("artifacts", "9 Artifact Index", inline=overview),
    ]


class CacheDetailScreen(Screen[None]):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("b", "back", "Back"),
        Binding("c", "copy", "Copy"),
        Binding("q", "back", "Back"),
    ]

    def __init__(self, item: DashboardItem) -> None:
        super().__init__()
        self.item = item
        self.text = _cache_item_text(item)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"{self.item.scope_type}:{self.item.scope_id} / {self.item.name} [{self.item.status}]", id="detail-title")
        with VerticalScroll(id="detail-scroll"):
            yield Static(self.text, id="detail-body", markup=False)
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_copy(self) -> None:
        self.notify(_copy_to_clipboard(self.text))


class RunDetailScreen(Screen[None]):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("b", "back", "Back"),
        Binding("c", "copy", "Copy"),
        Binding("1", "tab(1)", "Packet"),
        Binding("2", "tab(2)", "Master"),
        Binding("3", "tab(3)", "Expert"),
        Binding("4", "tab(4)", "Worker Prompt"),
        Binding("5", "tab(5)", "Worker Log"),
        Binding("6", "tab(6)", "Eval Log"),
        Binding("7", "tab(7)", "Result"),
        Binding("8", "tab(8)", "Summary"),
        Binding("9", "tab(9)", "Artifacts"),
        Binding("q", "back", "Back"),
    ]

    def __init__(self, item: DashboardItem) -> None:
        super().__init__()
        self.item = item
        self.tabs = _run_tabs(item)
        self.index = 1 if len(self.tabs) > 1 else 0
        self.current_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="run-title")
        yield Static("", id="run-tabs", markup=False)
        with VerticalScroll(id="run-scroll"):
            yield Static("", id="run-body", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self._render()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_copy(self) -> None:
        self.notify(_copy_to_clipboard(self.current_text))

    def action_tab(self, number: int) -> None:
        if number == 9:
            self.index = min(8, len(self.tabs) - 1)
        else:
            self.index = min(max(1, number), len(self.tabs) - 1)
        self._render()

    def _render(self) -> None:
        tab = self.tabs[self.index]
        tab_labels = []
        for idx, candidate in enumerate(self.tabs):
            label = candidate.title
            if idx == self.index:
                label = f"[{label}]"
            tab_labels.append(label)
        self.current_text = _tab_text(tab)
        self.query_one("#run-title", Static).update(f"{self.item.name} / {tab.title}")
        self.query_one("#run-tabs", Static).update("  ".join(tab_labels[1:]))
        self.query_one("#run-body", Static).update(self.current_text)


class V3ScopedDashboard(App[None]):
    CSS = """
    #search-bar {
        height: 3;
        border-bottom: solid $accent;
    }
    #mode {
        width: 16;
        margin: 0 1 0 0;
    }
    #query {
        width: 1fr;
    }
    #hint {
        height: 3;
        padding: 1;
    }
    DataTable {
        height: 1fr;
        width: 100%;
    }
    #detail-title, #run-title {
        height: auto;
        padding: 1;
        border-bottom: solid $accent;
    }
    #run-tabs {
        height: auto;
        padding: 0 1 1 1;
        border-bottom: solid $accent;
    }
    #detail-scroll, #run-scroll {
        height: 1fr;
        padding: 1;
    }
    """
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("c", "copy_row", "Copy Row"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        training_data_root: Path,
        training_root: Path,
        runs_root: Path,
        cfg: dict[str, Any],
        mode: str = "mls",
        query: str | None = None,
    ) -> None:
        super().__init__()
        self.training_data_root = training_data_root
        self.training_root = training_root
        self.runs_root = runs_root
        self.cfg = cfg
        self.mode = mode
        self.query = query or ""
        self.items: list[DashboardItem] = []
        self.current_row = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Horizontal(id="search-bar"):
                yield Select([("MLS", "mls"), ("arXiv", "arxiv")], value=self.mode, id="mode")
                yield Input(value=self.query, placeholder="MLS: task id (dl_lr_schedule) | arXiv: 2504.00302, then Enter", id="query")
            yield Static("Choose MLS or arXiv, enter one ID, press Enter. Results below are scoped only to that task/paper.", id="hint")
            yield DataTable(id="results")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results", DataTable)
        table.cursor_type = "row"
        table.add_columns("Type", "Name", "Status", "Summary", "Path")
        if self.query:
            self._search()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "mode":
            self.mode = str(event.value)
            self.query_one("#query", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "query":
            return
        self.query = _normalize_query(event.value)
        self._search()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            self.current_row = int(getattr(event.row_key, "value", str(event.row_key)))
        except Exception:
            self.current_row = 0
        if self.current_row >= len(self.items):
            return
        item = self.items[self.current_row]
        if item.item_type == "run":
            self.push_screen(RunDetailScreen(item))
        else:
            self.push_screen(CacheDetailScreen(item))

    def action_refresh(self) -> None:
        self.query = _normalize_query(self.query_one("#query", Input).value)
        self._search()

    def action_copy_row(self) -> None:
        if not self.items:
            self.notify("no row")
            return
        item = self.items[min(self.current_row, len(self.items) - 1)]
        self.notify(_copy_to_clipboard(_fmt({
            "type": item.item_type,
            "name": item.name,
            "status": item.status,
            "summary": item.summary,
            "path": str(item.path or ""),
        })))

    def _search(self) -> None:
        query = _normalize_query(self.query)
        table = self.query_one("#results", DataTable)
        table.clear()
        self.items = []
        if not query:
            self.query_one("#hint", Static).update("Enter one MLS task id or one arXiv id.")
            return
        try:
            if self.mode == "arxiv":
                self.items = discover_article_items(self.cfg, query)
            else:
                self.items = discover_mls_items(self.runs_root, self.training_data_root, self.training_root, self.cfg, query)
        except Exception as exc:
            self.items = [
                DashboardItem(
                    item_type="cache",
                    name="search_error",
                    status="error",
                    summary=str(exc),
                    scope_type=self.mode,
                    scope_id=query,
                    content=f"# Search Error\n\n```text\n{exc}\n```",
                )
            ]
        self.current_row = 0
        self.query_one("#hint", Static).update(
            f"{self.mode}:{query} | {len(self.items)} scoped rows. Enter opens details; cache pages scroll; run pages use tabs 1-9."
        )
        for idx, item in enumerate(self.items):
            table.add_row(
                item.item_type,
                item.name,
                item.status,
                item.summary,
                str(item.path or ""),
                key=str(idx),
            )
        if self.items:
            table.move_cursor(row=0)
        self.call_after_refresh(lambda: self.set_focus(table))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only scoped V3 dashboard for MLS tasks and arXiv article caches.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--training-data-root", default=str(ROOT / "runs" / "training_data"))
    parser.add_argument("--training-root", default=str(ROOT / "runs" / "training"))
    parser.add_argument("--runs-root", default=str(ROOT / "runs"))
    parser.add_argument("--mode", choices=["mls", "arxiv"], default="mls")
    parser.add_argument("--query", default=None)
    parser.add_argument("--article-id", default=None, help="Back-compat alias for --mode arxiv --query ID.")
    args = parser.parse_args()
    mode = args.mode
    query = args.query
    if args.article_id:
        mode = "arxiv"
        query = args.article_id
    V3ScopedDashboard(
        training_data_root=Path(args.training_data_root),
        training_root=Path(args.training_root),
        runs_root=Path(args.runs_root),
        cfg=load_config(args.config),
        mode=mode,
        query=query,
    ).run()


if __name__ == "__main__":
    main()
