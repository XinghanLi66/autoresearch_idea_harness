#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Static


PANELS = [
    ("overview", "Overview"),
    ("task_packet", "Task Packet"),
    ("proposals", "Proposals"),
    ("market", "Market"),
    ("gate", "Gate"),
    ("worker_prompt", "Worker Prompt"),
    ("worker_log", "Worker Log"),
    ("dialogue", "Dialogue"),
    ("eval_log", "Eval Log"),
    ("result", "Result"),
    ("report", "Report"),
]


def _read(path: Path, default: str = "") -> str:
    if not path.exists():
        return default or f"(missing: {path.name})"
    return path.read_text(errors="replace")


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _copy_to_clipboard(text: str) -> str:
    cmds = [
        ["pbcopy"],
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["tmux", "load-buffer", "-w", "-"],
        ["tmux", "load-buffer", "-"],
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, input=text, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "copied"
        except Exception:
            continue
    return "no clipboard backend"


class EndToEndDashboard(App):
    CSS = """
    DataTable {
        width: 45%;
        height: 100%;
    }
    #content {
        width: 55%;
        height: 100%;
        overflow: auto;
        border-left: solid $accent;
        padding: 1;
    }
    """
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("c", "copy", "Copy panel"),
        Binding("t", "tail", "Tail"),
        Binding("1", "panel('overview')", "Overview"),
        Binding("2", "panel('task_packet')", "Task"),
        Binding("3", "panel('proposals')", "Proposals"),
        Binding("4", "panel('market')", "Market"),
        Binding("5", "panel('gate')", "Gate"),
        Binding("6", "panel('worker_prompt')", "Prompt"),
        Binding("7", "panel('worker_log')", "Worker Log"),
        Binding("8", "panel('dialogue')", "Dialogue"),
        Binding("9", "panel('eval_log')", "Eval Log"),
        Binding("0", "panel('result')", "Result"),
    ]

    def __init__(self, runs_root: Path, refresh_s: float = 5.0) -> None:
        super().__init__()
        self.runs_root = runs_root
        self.refresh_s = refresh_s
        self.current_run: Path | None = None
        self.current_panel = "overview"
        self.tail_mode = True
        self.current_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="runs")
            yield Static(id="content")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.cursor_type = "row"
        table.add_columns("Run", "Task", "Subtask", "Status", "Metric", "Pass")
        self.refresh_runs()
        self.set_interval(self.refresh_s, self.refresh_runs)

    def action_refresh(self) -> None:
        self.refresh_runs()

    def action_copy(self) -> None:
        self.notify(_copy_to_clipboard(self.current_text))

    def action_tail(self) -> None:
        self.tail_mode = not self.tail_mode
        self.render_panel()

    def action_panel(self, panel: str) -> None:
        self.current_panel = panel
        self.render_panel()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = getattr(event.row_key, "value", str(event.row_key))
        self.current_run = Path(key)
        self.render_panel()

    def refresh_runs(self) -> None:
        table = self.query_one("#runs", DataTable)
        selected = str(self.current_run) if self.current_run else None
        table.clear()
        runs = self._runs()
        for run in runs:
            summary = _json(run / "summary.json")
            wr = summary.get("worker_result") or {}
            metric = wr.get("val_metric")
            task_label = str(summary.get("task") or "")
            module = summary.get("module_id")
            if module:
                task_label = f"{task_label}/{module}"
            table.add_row(
                run.name,
                task_label,
                str(summary.get("subtask") or ""),
                str(wr.get("status") or "running"),
                "" if metric is None else f"{float(metric):.4f}",
                str(wr.get("passed", "")),
                key=str(run),
            )
        if self.current_run is None and runs:
            self.current_run = runs[0]
        elif selected and Path(selected).exists():
            self.current_run = Path(selected)
        self.render_panel()

    def _runs(self) -> list[Path]:
        if not self.runs_root.exists():
            return []
        runs = [
            p for p in self.runs_root.iterdir()
            if p.is_dir() and ((p / "summary.json").exists() or (p / "events.jsonl").exists())
        ]
        return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)

    def render_panel(self) -> None:
        content = self.query_one("#content", Static)
        if self.current_run is None:
            self.current_text = "(no runs)"
            content.update(self.current_text)
            return
        text = self.panel_text(self.current_run, self.current_panel)
        label = dict(PANELS).get(self.current_panel, self.current_panel)
        self.current_text = text
        header = f"[bold]{label}[/bold]  [dim]{self.current_run.name} | tail={self.tail_mode}[/dim]\n\n"
        content.update(header + text)

    def panel_text(self, run: Path, panel: str) -> str:
        summary = _json(run / "summary.json")
        selected = ((summary.get("gate") or {}).get("selected_proposal_id") or "")
        worker_dir = run / "worker_runs" / selected if selected else run / "worker_runs"
        if (run / "worker_prompt.txt").exists() or (run / "worker.log").exists():
            worker_dir = run
        if panel == "overview":
            events = _jsonl(run / "events.jsonl")
            return _fmt({
                "summary": summary,
                "event_count": len(events),
                "last_events": events[-8:],
            })
        if panel == "task_packet":
            return _fmt(_json(run / "task_packet.json"))
        if panel == "proposals":
            rows = _jsonl(run / "proposals" / "proposal_private.jsonl")
            proposal_txt = _read(run / "proposal.txt", "")
            empty = _json(run / "empty_control.json")
            return _fmt({"rows": rows, "proposal_txt": proposal_txt, "empty_control": empty})
        if panel == "market":
            return _fmt(_jsonl(_first_existing(run / "market" / "forecasts.jsonl", run / "expert_forecasts.jsonl")))
        if panel == "gate":
            return _fmt(_json(_first_existing(run / "market" / "gate_summary.json", run / "settlement.json")))
        if panel == "worker_prompt":
            return _read(worker_dir / "worker_prompt.txt")
        if panel == "worker_log":
            return self._maybe_tail(_read(worker_dir / "worker.log"))
        if panel == "dialogue":
            return _read(worker_dir / "master_worker_dialogue.jsonl")
        if panel == "eval_log":
            return self._maybe_tail(_read(worker_dir / "eval.log"))
        if panel == "result":
            return _fmt(_json(worker_dir / "result.json"))
        if panel == "report":
            return _read(run / "report.md")
        return "(unknown panel)"

    def _maybe_tail(self, text: str, chars: int = 20000) -> str:
        if self.tail_mode and len(text) > chars:
            return text[-chars:]
        return text


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only TUI for end-to-end autoresearch runs.")
    p.add_argument("--runs-root", default=str(Path(__file__).resolve().parents[1] / "runs" / "end_to_end"))
    p.add_argument("--refresh", type=float, default=5.0)
    args = p.parse_args()
    EndToEndDashboard(Path(args.runs_root), refresh_s=args.refresh).run()


if __name__ == "__main__":
    main()
