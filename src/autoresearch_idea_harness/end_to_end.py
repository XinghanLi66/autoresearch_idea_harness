from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmarking import get_task, verify_signed_result
from .io import iter_jsonl, stable_id, write_json, write_jsonl
from .prompts import (
    end_to_end_forecast_prompt,
    end_to_end_master_prompt,
    master_advice_prompt,
    worker_prompt,
)
from .runway_client import RunwayClient


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_prob(value: Any) -> float:
    try:
        p = float(value)
    except Exception:
        p = 0.0
    if p > 1.0 and p <= 100.0:
        p /= 100.0
    return max(0.0, min(1.0, p))


def extract_json_obj(text: str) -> dict[str, Any]:
    import re

    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("response did not contain a JSON object")
    return json.loads(match.group(0))


def write_jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class EndToEndOptions:
    task: str
    subtask: str
    n_proposals: int = 3
    experts: list[str] | None = None
    gate_threshold: float = 0.5
    max_workers: int = 1
    worker_mode: str = "claude"  # claude | fixture | skip
    mock_llm: bool = False
    gpu: str = "0"
    worker_timeout: int = 7200
    max_turns: int = 30
    max_master_advice: int = 1
    output_dir: Path | None = None


class EventLogger:
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "events.jsonl"

    def emit(self, event: str, **fields: Any) -> None:
        write_jsonl_append(self.path, {"ts": utc_now(), "event": event, **fields})


class EndToEndRunner:
    def __init__(self, cfg: dict[str, Any], opts: EndToEndOptions) -> None:
        self.cfg = cfg
        self.opts = opts
        run_id = uuid.uuid4().hex[:8]
        root = opts.output_dir or (
            Path(cfg["runs_dir"]) / "end_to_end" / f"{opts.task}_{opts.subtask or 'default'}_{run_id}"
        )
        self.run_dir = root
        self.run_id = run_id
        self.events = EventLogger(root)
        self.experts = opts.experts or [row["id"] for row in cfg.get("market", {}).get("llm_experts", [])]

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events.emit("run_started", run_id=self.run_id, task=self.opts.task, subtask=self.opts.subtask)
        task = get_task(self.cfg, self.opts.task, self.opts.subtask or None)
        task_packet = task.task_packet(self.cfg)
        task_packet["run_id"] = self.run_id
        task_packet["subtask"] = self.opts.subtask
        write_json(self.run_dir / "task_packet.json", task_packet)
        self.events.emit("task_packet_written", path="task_packet.json")

        proposals = self._generate_proposals(task_packet)
        forecasts = self._forecast(task_packet, proposals)
        gate = self._gate(proposals, forecasts)
        worker_result = self._run_selected_worker(task, task_packet, gate, proposals)
        settlement = self._settle(forecasts, worker_result)
        summary = self._summary(task_packet, proposals, forecasts, gate, worker_result, settlement, time.time() - t0)
        write_json(self.run_dir / "summary.json", summary)
        self._render_report(summary, proposals, forecasts, gate, worker_result, settlement)
        self.events.emit("run_done", elapsed_s=round(time.time() - t0, 3), summary_path="summary.json")
        return summary

    def _master_cfg(self) -> dict[str, Any]:
        return (
            self.cfg.get("proposal_batches", {})
            .get("runway_generators", {})
            .get("runway_opus47_proposal", {})
        )

    def _expert_cfg(self, expert_id: str) -> dict[str, Any]:
        for row in self.cfg.get("market", {}).get("llm_experts", []):
            if row.get("id") == expert_id:
                return row
        raise ValueError(f"Unknown expert id: {expert_id}")

    def _call_llm(
        self,
        *,
        role: str,
        call_id: str,
        cfg_row: dict[str, Any],
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        call_dir = self.run_dir / "llm_calls" / call_id
        call_dir.mkdir(parents=True, exist_ok=True)
        request_meta = {
            "role": role,
            "call_id": call_id,
            "endpoint": cfg_row.get("endpoint"),
            "model_env": cfg_row.get("model_env"),
            "model": os.environ.get(str(cfg_row.get("model_env", ""))) or cfg_row.get("model"),
            "key_env": cfg_row.get("key_env"),
            "max_tokens": max_tokens,
            "temperature": cfg_row.get("temperature"),
            "messages": messages,
            "started_at": utc_now(),
        }
        write_json(call_dir / "request.json", request_meta)
        t0 = time.time()
        try:
            client = RunwayClient(self.cfg, key_env=str(cfg_row["key_env"]))
            temp = cfg_row.get("temperature")
            result = client.complete(
                endpoint=str(cfg_row.get("endpoint", "chat_completions")),
                model=str(request_meta["model"]),
                messages=messages,
                temperature=None if temp is None else float(temp),
                max_tokens=max_tokens,
                stream=True,
            )
            response = {
                "text": result.text,
                "model": result.model,
                "usage": result.usage,
                "finish_reason": result.raw_finish_reason,
                "elapsed_s": round(time.time() - t0, 3),
                "completed_at": utc_now(),
            }
            write_json(call_dir / "response.json", response)
            self.events.emit("llm_call_done", role=role, call_id=call_id, elapsed_s=response["elapsed_s"])
            return result.text, response
        except Exception as exc:
            error = {
                "error": str(exc)[:1000],
                "elapsed_s": round(time.time() - t0, 3),
                "completed_at": utc_now(),
            }
            write_json(call_dir / "error.json", error)
            self.events.emit("llm_call_error", role=role, call_id=call_id, error=error["error"])
            raise

    def _generate_proposals(self, task_packet: dict[str, Any]) -> list[dict[str, Any]]:
        self.events.emit("proposal_generation_started", n_proposals=self.opts.n_proposals)
        proposals_dir = self.run_dir / "proposals"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        proposals: list[dict[str, Any]] = []
        for i in range(self.opts.n_proposals):
            if self.opts.mock_llm:
                call_id = f"mock_master_{i:02d}"
                messages = end_to_end_master_prompt(task_packet, i)
                text = self._mock_proposal(task_packet, i)
                response_meta = {"model": "mock-master", "usage": {}, "finish_reason": "mock"}
                self._write_mock_llm_call("master", call_id, messages, text, response_meta)
            else:
                call_id = f"master_proposal_{i:02d}"
                cfg_row = self._master_cfg()
                text, response_meta = self._call_llm(
                    role="master",
                    call_id=call_id,
                    cfg_row=cfg_row,
                    messages=end_to_end_master_prompt(task_packet, i),
                    max_tokens=int(cfg_row.get("max_tokens", self.cfg.get("llm", {}).get("proposal_max_tokens", 2600))),
                )
            proposal_id = stable_id("prop", self.run_id, i, text, length=14)
            proposal = {
                "proposal_id": proposal_id,
                "label": chr(ord("A") + i),
                "generator_id": "runway_opus47_proposal" if not self.opts.mock_llm else "mock_master",
                "model_id": response_meta.get("model"),
                "call_id": call_id,
                "text": text,
            }
            proposals.append(proposal)
            (proposals_dir / f"{proposal_id}.txt").write_text(text)
        write_jsonl(proposals_dir / "proposal_private.jsonl", proposals)
        expert_rows = [
            {"proposal_id": p["proposal_id"], "label": p["label"], "text": p["text"]}
            for p in proposals
        ]
        write_jsonl(proposals_dir / "proposal_expert.jsonl", expert_rows)
        self.events.emit("proposal_generation_done", proposal_count=len(proposals))
        return proposals

    def _mock_proposal(self, task_packet: dict[str, Any], i: int) -> str:
        title = [
            "Parametric gated smooth activation",
            "Piecewise normalized swish activation",
            "Conservative GELU temperature variant",
        ][i % 3]
        return (
            "<proposal>\n"
            f"<title>{title}</title>\n"
            "<problem>Improve the activation function benchmark with a concrete, local change.</problem>\n"
            "<gap>Existing GELU/SiLU/Mish baselines do not adapt the gate strength to the residual scale.</gap>\n"
            "<core_idea>Use a simple smooth gated activation with fixed constants that can be implemented in the editable module.</core_idea>\n"
            "<implementation_plan>Implement the activation as x * sigmoid(1.3 * x) plus a small bounded tanh correction 0.05 * tanh(x), preserving tensor shape and differentiability.</implementation_plan>\n"
            "<evaluation_plan>Run the provided benchmark exactly and compare test_acc against the stated baseline.</evaluation_plan>\n"
            "<expected_results>The smooth gate may improve optimization stability while staying close to known strong activations.</expected_results>\n"
            "<risks_and_limitations>The fixed constants may underperform GELU on CIFAR-10.</risks_and_limitations>\n"
            "</proposal>"
        )

    def _forecast(self, task_packet: dict[str, Any], proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.events.emit("forecast_started", experts=self.experts, proposal_count=len(proposals))
        market_dir = self.run_dir / "market"
        market_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for proposal in proposals:
            for expert_id in self.experts:
                if self.opts.mock_llm:
                    call_id = f"mock_forecast_{expert_id}_{proposal['label']}"
                    messages = end_to_end_forecast_prompt(task_packet, proposal)
                    parsed = self._mock_forecast(proposal, expert_id)
                    response_meta = {"model": f"mock-{expert_id}", "usage": {}}
                    self._write_mock_llm_call(
                        f"expert:{expert_id}",
                        call_id,
                        messages,
                        json.dumps(parsed, ensure_ascii=False),
                        response_meta,
                    )
                else:
                    cfg_row = self._expert_cfg(expert_id)
                    call_id = f"forecast_{expert_id}_{proposal['label']}"
                    text, response_meta = self._call_llm(
                        role=f"expert:{expert_id}",
                        call_id=call_id,
                        cfg_row=cfg_row,
                        messages=end_to_end_forecast_prompt(task_packet, proposal),
                        max_tokens=int(cfg_row.get("max_tokens", self.cfg.get("llm", {}).get("expert_max_tokens", 1600))),
                    )
                    parsed = extract_json_obj(text)
                rows.append({
                    "forecast_id": stable_id("fcst", self.run_id, proposal["proposal_id"], expert_id),
                    "proposal_id": proposal["proposal_id"],
                    "proposal_label": proposal["label"],
                    "expert_id": expert_id,
                    "expert_model_id": response_meta.get("model"),
                    "call_id": call_id,
                    "success_probability": round(clamp_prob(parsed.get("success_probability")), 4),
                    "rationale": str(parsed.get("rationale", "")).strip(),
                    "strengths": parsed.get("strengths") or [],
                    "weaknesses": parsed.get("weaknesses") or [],
                    "baseline_risk": str(parsed.get("baseline_risk", "")).strip(),
                    "usage": response_meta.get("usage") or {},
                    "created_at": utc_now(),
                })
        write_jsonl(market_dir / "forecasts.jsonl", rows)
        self.events.emit("forecast_done", forecast_count=len(rows))
        return rows

    def _mock_forecast(self, proposal: dict[str, Any], expert_id: str) -> dict[str, Any]:
        base = {"A": 0.62, "B": 0.44, "C": 0.36}.get(proposal.get("label"), 0.4)
        if expert_id == "gpt55":
            base -= 0.04
        return {
            "success_probability": max(0.01, min(0.99, base)),
            "rationale": "Mock calibrated forecast for offline pipeline smoke.",
            "strengths": ["Concrete implementation", "Measurable benchmark"],
            "weaknesses": ["Uncertain gain over strong baseline"],
            "baseline_risk": "May not beat the reference baseline.",
        }

    def _write_mock_llm_call(
        self,
        role: str,
        call_id: str,
        messages: list[dict[str, str]],
        text: str,
        response_meta: dict[str, Any],
    ) -> None:
        call_dir = self.run_dir / "llm_calls" / call_id
        call_dir.mkdir(parents=True, exist_ok=True)
        write_json(call_dir / "request.json", {
            "role": role,
            "call_id": call_id,
            "endpoint": "mock",
            "model": response_meta.get("model"),
            "messages": messages,
            "started_at": utc_now(),
        })
        write_json(call_dir / "response.json", {
            "text": text,
            "model": response_meta.get("model"),
            "usage": response_meta.get("usage") or {},
            "finish_reason": response_meta.get("finish_reason", "mock"),
            "elapsed_s": 0.0,
            "completed_at": utc_now(),
        })
        self.events.emit("llm_call_done", role=role, call_id=call_id, elapsed_s=0.0)

    def _gate(self, proposals: list[dict[str, Any]], forecasts: list[dict[str, Any]]) -> dict[str, Any]:
        threshold = self.opts.gate_threshold
        by_pid: dict[str, list[dict[str, Any]]] = {}
        for row in forecasts:
            by_pid.setdefault(row["proposal_id"], []).append(row)
        scores = []
        for proposal in proposals:
            probs = [float(r["success_probability"]) for r in by_pid.get(proposal["proposal_id"], [])]
            mean_p = sum(probs) / len(probs) if probs else 0.0
            scores.append({
                "proposal_id": proposal["proposal_id"],
                "label": proposal["label"],
                "mean_success_probability": round(mean_p, 4),
                "expert_count": len(probs),
                "passed_gate": mean_p >= threshold,
            })
        selected = next((s for s in sorted(scores, key=lambda x: x["mean_success_probability"], reverse=True) if s["passed_gate"]), None)
        forced = False
        if selected is None and scores:
            selected = sorted(scores, key=lambda x: x["mean_success_probability"], reverse=True)[0]
            forced = True
        gate = {
            "gate_threshold": threshold,
            "forced_for_smoke": forced,
            "selected_proposal_id": selected["proposal_id"] if selected else None,
            "selected_label": selected["label"] if selected else None,
            "scores": scores,
        }
        write_json(self.run_dir / "market" / "gate_summary.json", gate)
        self.events.emit("gate_done", selected_proposal_id=gate["selected_proposal_id"], forced_for_smoke=forced)
        return gate

    def _run_selected_worker(
        self,
        task,
        task_packet: dict[str, Any],
        gate: dict[str, Any],
        proposals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected_id = gate.get("selected_proposal_id")
        if not selected_id:
            return {"status": "skipped", "error": "no selected proposal"}
        proposal = next(p for p in proposals if p["proposal_id"] == selected_id)
        if self.opts.worker_mode == "skip":
            self.events.emit("worker_skipped", proposal_id=selected_id)
            return {"status": "skipped", "proposal_id": selected_id}
        worker_dir = self.run_dir / "worker_runs" / selected_id
        worker_dir.mkdir(parents=True, exist_ok=True)
        task.setup_workspace(worker_dir / "workspace")
        if self.opts.worker_mode == "fixture":
            return self._fixture_worker(task, task_packet, proposal, worker_dir)
        return self._claude_worker(task, task_packet, proposal, worker_dir)

    def _fixture_worker(self, task, task_packet: dict[str, Any], proposal: dict[str, Any], worker_dir: Path) -> dict[str, Any]:
        self.events.emit("fixture_worker_started", proposal_id=proposal["proposal_id"])
        prompt = worker_prompt(task_packet, proposal["text"])
        (worker_dir / "worker_prompt.txt").write_text(prompt)
        (worker_dir / "worker.log").write_text("fixture worker: no Claude Code call\n")
        (worker_dir / "eval.log").write_text("fixture eval log\n")
        (worker_dir / "master_worker_dialogue.jsonl").write_text("")
        metric = task.baseline_metric() + task.pass_threshold + 0.01 if not task.lower_is_better else task.baseline_metric() - task.pass_threshold - 0.01
        secret = os.environ.get("BENCHMARK_HMAC_KEY", "benchmark-eval-secret")
        import hashlib as _hl
        import hmac as _hmac

        payload = f"{float(metric):.6f}"
        sig = _hmac.new(secret.encode(), payload.encode(), _hl.sha256).hexdigest()
        result = {"val_metric": metric, "_sig": sig, "fixture": True}
        write_json(worker_dir / "result.json", result)
        return self._parse_worker_result(task, worker_dir, proposal["proposal_id"], elapsed_s=0.0)

    def _claude_worker(self, task, task_packet: dict[str, Any], proposal: dict[str, Any], worker_dir: Path) -> dict[str, Any]:
        advice: list[str] = []
        dialogue_path = worker_dir / "master_worker_dialogue.jsonl"
        dialogue_path.write_text("")
        result: dict[str, Any] = {"status": "error", "error": "worker did not run"}
        for attempt in range(self.opts.max_master_advice + 1):
            prompt_text = worker_prompt(task_packet, proposal["text"], advice=advice)
            (worker_dir / "worker_prompt.txt").write_text(prompt_text)
            self.events.emit("worker_attempt_started", proposal_id=proposal["proposal_id"], attempt=attempt)
            t0 = time.time()
            self._invoke_claude_worker(worker_dir)
            result = self._parse_worker_result(task, worker_dir, proposal["proposal_id"], elapsed_s=time.time() - t0)
            if result.get("status") == "done" and result.get("passed"):
                return result
            if attempt >= self.opts.max_master_advice:
                return result
            question_path = worker_dir / "workspace" / "master_question.txt"
            status = {
                "attempt": attempt,
                "worker_result": result,
                "master_question": question_path.read_text(errors="replace") if question_path.exists() else "",
                "worker_log_tail": self._tail(worker_dir / "worker.log", 6000),
            }
            advice_text = self._ask_master_advice(task_packet, proposal["text"], status, attempt)
            advice.append(advice_text)
            write_jsonl_append(dialogue_path, {
                "ts": utc_now(),
                "attempt": attempt,
                "worker_status": status,
                "master_advice": advice_text,
            })
            (worker_dir / f"master_advice_{attempt + 1:02d}.txt").write_text(advice_text)
        return result

    def _invoke_claude_worker(self, worker_dir: Path) -> None:
        workspace = worker_dir / "workspace"
        prompt_file = worker_dir / "worker_prompt.txt"
        log_file = worker_dir / "worker.log"
        cfg = self.cfg.get("end_to_end", {})
        claude_cmd = str(cfg.get("claude_cmd", "/newcpfs/lxh/claude-home-agent1/run_claude.sh"))
        cmd = [
            claude_cmd,
            "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", str(self.opts.max_turns),
            "--allowedTools", "Bash,Read,Edit,Write",
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.opts.gpu, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        result_path = workspace / "result.json"
        with prompt_file.open() as stdin_f, log_file.open("a") as log_f:
            proc = subprocess.Popen(cmd, stdin=stdin_f, stdout=log_f, stderr=subprocess.STDOUT, cwd=workspace, env=env)
            started = time.time()
            while proc.poll() is None:
                elapsed = time.time() - started
                if result_path.exists() and result_path.stat().st_mtime >= started - 1:
                    self.events.emit("worker_result_detected_while_process_running", elapsed_s=round(elapsed, 3))
                    log_f.write("\n[runner] result.json detected; terminating worker process after result capture\n")
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.events.emit("worker_result_detected_kill_after_grace", elapsed_s=round(time.time() - started, 3))
                        proc.kill()
                        proc.wait()
                    return
                if elapsed >= self.opts.worker_timeout:
                    self.events.emit("worker_timeout", elapsed_s=round(elapsed, 3), worker_timeout=self.opts.worker_timeout)
                    proc.kill()
                    proc.wait()
                    log_f.write(f"\n[runner] worker timeout after {self.opts.worker_timeout}s\n")
                    return
                time.sleep(5)

    def _ask_master_advice(
        self,
        task_packet: dict[str, Any],
        proposal_text: str,
        worker_status: dict[str, Any],
        attempt: int,
    ) -> str:
        if self.opts.mock_llm:
            return "Mock advice: keep the implementation minimal and rerun the exact benchmark command."
        cfg_row = self._master_cfg()
        text, _ = self._call_llm(
            role="master_advice",
            call_id=f"master_advice_{attempt + 1:02d}",
            cfg_row=cfg_row,
            messages=master_advice_prompt(task_packet, proposal_text, worker_status),
            max_tokens=800,
        )
        return text.strip()

    def _parse_worker_result(self, task, worker_dir: Path, proposal_id: str, elapsed_s: float) -> dict[str, Any]:
        result_path = worker_dir / "workspace" / "result.json"
        if not result_path.exists():
            result_path = worker_dir / "result.json"
        if not result_path.exists():
            return {
                "status": "error",
                "proposal_id": proposal_id,
                "error": "result.json not written",
                "elapsed_s": round(elapsed_s, 3),
            }
        try:
            raw = json.loads(result_path.read_text())
        except Exception as exc:
            return {"status": "error", "proposal_id": proposal_id, "error": f"invalid result.json: {exc}", "elapsed_s": round(elapsed_s, 3)}
        if result_path.parent.name == "workspace":
            shutil.copy2(result_path, worker_dir / "result.json")
        for name in ("eval.log", "editable_region.py", "baseline.py", "submission.csv"):
            src = worker_dir / "workspace" / name
            if src.exists() and src.is_file():
                shutil.copy2(src, worker_dir / name)
        metric, error = verify_signed_result(raw)
        if metric is None:
            return {
                "status": "error",
                "proposal_id": proposal_id,
                "error": error,
                "raw_result": raw,
                "elapsed_s": round(elapsed_s, 3),
            }
        improvement = task.improvement(metric)
        passed = task.passed(metric)
        return {
            "status": "done",
            "proposal_id": proposal_id,
            "val_metric": metric,
            "baseline_metric": task.baseline_metric(),
            "improvement": round(improvement, 6),
            "passed": passed,
            "raw_result": raw,
            "elapsed_s": round(elapsed_s, 3),
        }

    def _settle(self, forecasts: list[dict[str, Any]], worker_result: dict[str, Any]) -> list[dict[str, Any]]:
        if worker_result.get("status") != "done":
            settlement: list[dict[str, Any]] = []
            write_json(self.run_dir / "market" / "settlement.json", {"settled": False, "reason": worker_result.get("error")})
            return settlement
        outcome = 1.0 if worker_result.get("passed") else 0.0
        selected = worker_result.get("proposal_id")
        rows = []
        for row in forecasts:
            if row.get("proposal_id") != selected:
                continue
            p = max(1e-6, min(1 - 1e-6, float(row["success_probability"])))
            brier = (p - outcome) ** 2
            log_score = -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))
            rows.append({
                "expert_id": row["expert_id"],
                "proposal_id": selected,
                "outcome": int(outcome),
                "success_probability": row["success_probability"],
                "brier_score": round(brier, 6),
                "log_score": round(log_score, 6),
                "settled_at": utc_now(),
            })
        write_json(self.run_dir / "market" / "settlement.json", {"settled": True, "rows": rows})
        reliability_path = Path(self.cfg["runs_dir"]) / "expert_reliability.jsonl"
        for row in rows:
            write_jsonl_append(reliability_path, {**row, "run_id": self.run_id, "task": self.opts.task})
        return rows

    def _summary(
        self,
        task_packet: dict[str, Any],
        proposals: list[dict[str, Any]],
        forecasts: list[dict[str, Any]],
        gate: dict[str, Any],
        worker_result: dict[str, Any],
        settlement: list[dict[str, Any]],
        elapsed_s: float,
    ) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": task_packet.get("task"),
            "subtask": task_packet.get("subtask"),
            "task_type": task_packet.get("task_type"),
            "baseline_metric": task_packet.get("baseline_metric"),
            "pass_threshold": task_packet.get("pass_threshold"),
            "pass_metric": task_packet.get("pass_metric"),
            "proposal_count": len(proposals),
            "forecast_count": len(forecasts),
            "gate": gate,
            "worker_result": worker_result,
            "settlement": settlement,
            "elapsed_s": round(elapsed_s, 3),
            "paths": {
                "events": "events.jsonl",
                "task_packet": "task_packet.json",
                "report": "report.md",
            },
        }

    def _render_report(
        self,
        summary: dict[str, Any],
        proposals: list[dict[str, Any]],
        forecasts: list[dict[str, Any]],
        gate: dict[str, Any],
        worker_result: dict[str, Any],
        settlement: list[dict[str, Any]],
    ) -> None:
        by_pid = {p["proposal_id"]: p for p in proposals}
        selected = by_pid.get(gate.get("selected_proposal_id"), {})
        lines = [
            "# End-to-End Autoresearch Run",
            "",
            f"- Run ID: `{self.run_id}`",
            f"- Task: `{summary.get('task')}` / `{summary.get('subtask')}`",
            f"- Baseline: `{summary.get('baseline_metric')}`",
            f"- Pass metric: `{summary.get('pass_metric')}`",
            f"- Selected proposal: `{gate.get('selected_label')}` `{gate.get('selected_proposal_id')}`",
            f"- Forced smoke fallback: `{gate.get('forced_for_smoke')}`",
            "",
            "## Market",
            "",
        ]
        for score in gate.get("scores", []):
            lines.append(
                f"- Proposal {score['label']}: mean p={score['mean_success_probability']} "
                f"gate={score['passed_gate']}"
            )
        lines.extend(["", "## Selected Proposal", "", selected.get("text", "(none)"), "", "## Worker Result", ""])
        lines.append(json.dumps(worker_result, indent=2, ensure_ascii=False))
        if settlement:
            lines.extend(["", "## Expert Settlement", ""])
            for row in settlement:
                lines.append(
                    f"- `{row['expert_id']}` p={row['success_probability']} "
                    f"outcome={row['outcome']} brier={row['brier_score']} log={row['log_score']}"
                )
        lines.extend(["", "## Forecasts", ""])
        for row in forecasts:
            lines.append(
                f"- `{row['expert_id']}` on {row['proposal_label']}: "
                f"p={row['success_probability']} - {row.get('rationale', '')}"
            )
        (self.run_dir / "report.md").write_text("\n".join(lines))

    @staticmethod
    def _tail(path: Path, chars: int) -> str:
        if not path.exists():
            return ""
        text = path.read_text(errors="replace")
        return text[-chars:]


def run_end_to_end(cfg: dict[str, Any], opts: EndToEndOptions) -> dict[str, Any]:
    return EndToEndRunner(cfg, opts).run()
