from __future__ import annotations

import gc
import hashlib
import hmac
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmarking import MLS_SUBTASKS, get_task, verify_signed_result
from .end_to_end import clamp_prob, extract_json_obj
from .io import project_root, stable_id, write_json
from .prompts import end_to_end_forecast_prompt, end_to_end_master_prompt, worker_prompt
from .runway_client import RunwayClient


V2_3_TASKS = [
    "dl_lr_schedule",
    "dl_activation_function",
    "cv_data_augmentation",
    "dl_weight_initialization",
    "cv_classification_loss",
    "cv_sample_weighting",
    "cv_pooling_aggregation",
    "cv_multitask_loss",
    "dl_regularization",
    "dl_residual_connection",
]

V2_3_NEW_CALIBRATION_TASKS = [
    "dl_weight_initialization",
    "cv_classification_loss",
    "cv_sample_weighting",
    "cv_pooling_aggregation",
    "cv_multitask_loss",
    "dl_regularization",
    "dl_residual_connection",
]

V2_3_MODULES: dict[str, dict[str, Any]] = {
    "empty_worker_only": {"kind": "empty", "strategy": "worker_only", "model": "worker_only_control"},
    "opus47_master": {"kind": "runway", "generator_id": "runway_opus47_proposal"},
    "qwen25_7b_base_with_research_question": {
        "kind": "checkpoint",
        "strategy": "with_research_question",
        "checkpoint": "/newcpfs/lxh/agentic-training/proposal_rl/runs/model_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    },
    "exp09_top_k_related_work": {
        "kind": "checkpoint",
        "strategy": "top_k_related_work",
        "checkpoint": "/newcpfs/lxh/agentic-training/proposal_rl/runs/exps/exp09_top_k_refs_sft_rl_20260515_150827/rl/final",
    },
    "exp11_top_k_related_work": {
        "kind": "checkpoint",
        "strategy": "top_k_related_work",
        "checkpoint": "/newcpfs/lxh/agentic-training/proposal_rl/runs/exps/exp11_topk_rw_sft_rl_20260514_044424/rl/final",
    },
    "exp12_with_research_question": {
        "kind": "checkpoint",
        "strategy": "with_research_question",
        "checkpoint": "/newcpfs/lxh/agentic-training/proposal_rl/runs/exps/exp12_research_q_sft_rl_20260510_193056/rl/final",
    },
    "exp13_top_k_refs": {
        "kind": "checkpoint",
        "strategy": "top_k_refs",
        "checkpoint": "/newcpfs/lxh/agentic-training/proposal_rl/runs/exps/exp13_full_refs_sft_rl_20260510_192856/rl/final",
    },
    "exp16_top_k_refs": {
        "kind": "checkpoint",
        "strategy": "top_k_refs",
        "checkpoint": "/newcpfs/lxh/agentic-training/proposal_rl/runs/exps/exp16_full_refs_20x800_sft_rl_20260511_032856/rl/final",
    },
    "exp17_with_research_question": {
        "kind": "checkpoint",
        "strategy": "with_research_question",
        "checkpoint": "/newcpfs/lxh/agentic-training/proposal_rl/runs/exps/exp17_top_k_refs_ppl_rl_20260520_023410/rl/final",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def slug(text: str, max_len: int = 72) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_")
    return (text or "run")[:max_len]


def brier_and_log(prob: float, outcome: float) -> tuple[float, float]:
    p = max(1e-6, min(1 - 1e-6, float(prob)))
    brier = (p - outcome) ** 2
    log_score = -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))
    return brier, log_score


def calibration_bucket(prob: float) -> str:
    lo = int(max(0, min(9, math.floor(prob * 10)))) * 10
    hi = lo + 10
    return f"{lo:02d}-{hi:02d}"


@dataclass
class SweepOptions:
    root: Path
    tasks: list[str]
    modules: list[str]
    n_samples: int
    experts: list[str]
    mode: str = "formal"  # formal | calibration | dry-run
    worker_mode: str = "claude"  # claude | fixture | skip
    mock_llm: bool = False
    mock_experts: bool = False
    gpu: str = "0"
    worker_timeout: int = 7200
    max_turns: int = 30
    max_master_advice: int = 1
    free_hparams: bool = False  # allow worker to choose own training hyperparameters
    force: bool = False
    write_thresholds: bool = False
    max_new_tokens: int = 2048
    temperature: float = 0.7
    shard_index: int = 0
    shard_count: int = 1
    result_wait_timeout: int = 7200
    no_eval_wait_timeout: int = 180


class LocalCheckpointModule:
    def __init__(self, cfg: dict[str, Any], module_id: str, module_cfg: dict[str, Any], opts: SweepOptions) -> None:
        self.cfg = cfg
        self.module_id = module_id
        self.module_cfg = module_cfg
        self.opts = opts
        self.model = None
        self.tokenizer = None
        self.device = None

    def generate(self, task_packet: dict[str, Any], sample_dir: Path) -> tuple[str, dict[str, Any]]:
        if self.opts.mock_llm:
            text = mock_proposal(task_packet, self.module_id)
            return text, {"model": f"mock:{self.module_id}", "usage": {}, "call_id": "mock_checkpoint"}
        self._ensure_loaded()
        assert self.model is not None and self.tokenizer is not None and self.device is not None
        record = checkpoint_prompt_record(task_packet, self.module_id, self.module_cfg.get("strategy", "full_refs"))
        write_json(sample_dir / "prompt_record.json", record)
        t0 = time.time()
        text = self._generate_from_record(record)
        elapsed = round(time.time() - t0, 3)
        if not text:
            raise RuntimeError(f"{self.module_id} returned empty proposal")
        return text, {
            "model": str(self.module_cfg.get("checkpoint")),
            "usage": {},
            "elapsed_s": elapsed,
            "call_id": f"checkpoint_{self.module_id}",
        }

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import sys

        proposal_root = Path(self.cfg.get("proposal_rl_root", project_root().parent / "proposal_rl"))
        for path in (proposal_root, proposal_root / "scripts"):
            s = str(path)
            if s not in sys.path:
                sys.path.insert(0, s)
        import yaml
        from probe import load_model_and_tokenizer

        cfg_path = proposal_root / "configs" / "base.yaml"
        legacy_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        checkpoint = Path(str(self.module_cfg["checkpoint"]))
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(checkpoint, None, legacy_cfg)

    def _generate_from_record(self, record: dict[str, Any]) -> str:
        import torch

        messages = [
            {"role": "system", "content": record["system"]},
            {"role": "user", "content": record["prompt"]},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.opts.max_new_tokens,
                do_sample=True,
                temperature=self.opts.temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class FormalSweepRunner:
    def __init__(self, cfg: dict[str, Any], opts: SweepOptions) -> None:
        self.cfg = cfg
        self.opts = opts
        self.root = opts.root
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.jsonl"
        self.events_path = self.root / "events.jsonl"
        self.checkpoint_modules: dict[str, LocalCheckpointModule] = {}

    def run(self) -> dict[str, Any]:
        self._write_sweep_manifest()
        self._event(
            "sweep_started",
            mode=self.opts.mode,
            tasks=self.opts.tasks,
            modules=self.opts.modules,
            shard_index=self.opts.shard_index,
            shard_count=self.opts.shard_count,
        )
        if self.opts.mode == "dry-run":
            summary = self._dry_run_summary()
            self._write_shard_json("summary.json", summary)
            if self.opts.shard_count == 1:
                write_json(self.root / "summary.json", summary)
            return summary
        try:
            total = 0
            for module_id in self.opts.modules:
                try:
                    for task_name in self.opts.tasks:
                        subtask = primary_subtask(task_name)
                        for sample_idx in range(self.opts.n_samples):
                            ordinal = total
                            total += 1
                            if not self._sample_belongs_to_shard(ordinal):
                                continue
                            self.run_sample(task_name, subtask, module_id, sample_idx)
                finally:
                    module = self.checkpoint_modules.pop(module_id, None)
                    if module is not None:
                        module.close()
            self._write_matrices()
            if self.opts.mode == "calibration" and self.opts.write_thresholds:
                self._write_calibration_thresholds()
            self._write_reports()
            summary = self._sweep_summary(total, self._assigned_sample_count(total))
            self._write_shard_json("summary.json", summary)
            if self.opts.shard_count == 1:
                write_json(self.root / "summary.json", summary)
            self._event("sweep_done", total_samples=total, assigned_samples=summary["assigned_samples_planned"])
            return summary
        finally:
            for module in self.checkpoint_modules.values():
                module.close()

    def _sample_belongs_to_shard(self, ordinal: int) -> bool:
        if self.opts.shard_count <= 1:
            return True
        return ordinal % self.opts.shard_count == self.opts.shard_index

    def _assigned_sample_count(self, total: int) -> int:
        if self.opts.shard_count <= 1:
            return total
        return sum(1 for ordinal in range(total) if self._sample_belongs_to_shard(ordinal))

    def _shard_dir(self) -> Path:
        return self.root / "shards" / f"shard_{self.opts.shard_index:03d}_of_{self.opts.shard_count:03d}"

    def _write_shard_json(self, name: str, obj: dict[str, Any]) -> None:
        if self.opts.shard_count <= 1:
            write_json(self.root / name, obj)
            return
        write_json(self._shard_dir() / name, obj)

    def run_sample(self, task_name: str, subtask: str, module_id: str, sample_idx: int) -> dict[str, Any]:
        sample_id = f"{slug(task_name)}__{slug(subtask)}__{slug(module_id)}__s{sample_idx:02d}"
        sample_dir = self.root / sample_id
        if sample_dir.exists() and not self.opts.force and (sample_dir / "summary.json").exists():
            return read_json(sample_dir / "summary.json")
        sample_dir.mkdir(parents=True, exist_ok=True)
        events_path = sample_dir / "events.jsonl"

        def event(event_name: str, **fields: Any) -> None:
            row = {"ts": utc_now(), "sample_id": sample_id, "event": event_name, **fields}
            write_jsonl_append(events_path, row)
            write_jsonl_append(self.events_path, row)

        event("sample_started", task=task_name, subtask=subtask, module=module_id, sample_idx=sample_idx)
        task = get_task(self.cfg, task_name, subtask)
        task_packet = task.task_packet(self.cfg)
        task_packet.update({
            "run_id": sample_id,
            "sample_id": sample_id,
            "subtask": subtask,
            "module_id": module_id,
            "sweep_root": str(self.root),
        })
        write_json(sample_dir / "task_packet.json", task_packet)

        try:
            proposal = self._generate_proposal(module_id, task_packet, sample_dir, sample_idx, event)
            forecasts = self._forecast(task_packet, proposal, sample_dir, event)
            worker_result = self._run_worker(task, task_packet, proposal, sample_dir, event)
            settlement = self._settle(task_packet, proposal, forecasts, worker_result, sample_dir, event)
            summary = self._sample_summary(task_packet, proposal, forecasts, worker_result, settlement)
            write_json(sample_dir / "summary.json", summary)
            self._register_sample(summary)
            event("sample_done", status=worker_result.get("status"), passed=worker_result.get("passed"))
            return summary
        except Exception as exc:
            error = {"error": str(exc)[:2000], "ts": utc_now()}
            write_json(sample_dir / "error.json", error)
            summary = {
                "sample_id": sample_id,
                "task": task_name,
                "subtask": subtask,
                "module_id": module_id,
                "worker_result": {"status": "error", "error": error["error"]},
                "error": error,
            }
            write_json(sample_dir / "summary.json", summary)
            self._register_sample(summary)
            event("sample_error", error=error["error"])
            return summary

    def _generate_proposal(
        self,
        module_id: str,
        task_packet: dict[str, Any],
        sample_dir: Path,
        sample_idx: int,
        event,
    ) -> dict[str, Any]:
        module_cfg = V2_3_MODULES[module_id]
        kind = module_cfg["kind"]
        proposal_id = stable_id("prop", task_packet["sample_id"], module_id, sample_idx, length=14)
        if kind == "empty":
            proposal = {
                "proposal_id": proposal_id,
                "label": "EMPTY",
                "module_id": module_id,
                "generator_id": module_id,
                "model_id": "worker_only_control",
                "text": "",
                "empty_control": True,
            }
            write_json(sample_dir / "empty_control.json", proposal)
            self._write_proposal_compat(sample_dir, proposal)
            event("proposal_empty_control_written", proposal_id=proposal_id)
            return proposal
        if kind == "runway":
            cfg_row = self._runway_generator_cfg(str(module_cfg["generator_id"]))
            call_id = f"proposal_{module_id}_{sample_idx:02d}"
            text, response = self._call_llm(
                sample_dir=sample_dir,
                role="proposal",
                call_id=call_id,
                cfg_row=cfg_row,
                messages=end_to_end_master_prompt(task_packet, sample_idx),
                max_tokens=int(cfg_row.get("max_tokens", self.cfg.get("llm", {}).get("proposal_max_tokens", 2600))),
            )
            model_id = response.get("model")
        elif kind == "checkpoint":
            module = self.checkpoint_modules.get(module_id)
            if module is None:
                module = LocalCheckpointModule(self.cfg, module_id, module_cfg, self.opts)
                self.checkpoint_modules[module_id] = module
            text, meta = module.generate(task_packet, sample_dir)
            call_id = str(meta.get("call_id", f"checkpoint_{module_id}"))
            model_id = meta.get("model")
            write_json(sample_dir / "llm_calls" / call_id / "response.json", {
                "text": text,
                "model": model_id,
                "usage": meta.get("usage") or {},
                "elapsed_s": meta.get("elapsed_s"),
                "completed_at": utc_now(),
            })
        else:
            raise ValueError(f"Unknown module kind: {kind}")
        proposal = {
            "proposal_id": proposal_id,
            "label": module_id,
            "module_id": module_id,
            "generator_id": module_id,
            "strategy": module_cfg.get("strategy"),
            "model_id": model_id,
            "call_id": call_id,
            "text": text,
            "empty_control": False,
        }
        (sample_dir / "proposal.txt").write_text(text)
        self._write_proposal_compat(sample_dir, proposal)
        event("proposal_written", proposal_id=proposal_id, module=module_id)
        return proposal

    def _write_proposal_compat(self, sample_dir: Path, proposal: dict[str, Any]) -> None:
        proposals_dir = sample_dir / "proposals"
        proposals_dir.mkdir(exist_ok=True)
        write_jsonl_append(proposals_dir / "proposal_private.jsonl", proposal)
        write_jsonl_append(proposals_dir / "proposal_expert.jsonl", {
            "proposal_id": proposal["proposal_id"],
            "label": proposal.get("label"),
            "empty_control": proposal.get("empty_control", False),
            "text": proposal.get("text", ""),
        })

    def _forecast(self, task_packet: dict[str, Any], proposal: dict[str, Any], sample_dir: Path, event) -> list[dict[str, Any]]:
        market_dir = sample_dir / "market"
        prompts_dir = sample_dir / "expert_prompts"
        responses_dir = sample_dir / "expert_responses"
        market_dir.mkdir(exist_ok=True)
        prompts_dir.mkdir(exist_ok=True)
        responses_dir.mkdir(exist_ok=True)
        rows = []
        for expert_id in self.opts.experts:
            messages = end_to_end_forecast_prompt(task_packet, proposal)
            write_json(prompts_dir / f"{expert_id}.json", {
                "expert_id": expert_id,
                "messages": messages,
                "model_cfg": self._safe_expert_cfg(expert_id),
                "created_at": utc_now(),
            })
            if self.opts.mock_experts or self.opts.mock_llm:
                parsed = mock_forecast(proposal, expert_id)
                response = {"text": json.dumps(parsed), "model": f"mock:{expert_id}", "usage": {}, "elapsed_s": 0.0}
            else:
                cfg_row = self._expert_cfg(expert_id)
                text, response = self._call_llm(
                    sample_dir=sample_dir,
                    role=f"expert:{expert_id}",
                    call_id=f"forecast_{expert_id}",
                    cfg_row=cfg_row,
                    messages=messages,
                    max_tokens=int(cfg_row.get("max_tokens", self.cfg.get("llm", {}).get("expert_max_tokens", 1600))),
                )
                parsed = extract_json_obj(text)
                response = {**response, "text": text}
            write_json(responses_dir / f"{expert_id}.json", {
                "expert_id": expert_id,
                "raw_response": response,
                "parsed": parsed,
                "created_at": utc_now(),
            })
            row = forecast_row(task_packet, proposal, expert_id, parsed, response)
            rows.append(row)
            write_jsonl_append(sample_dir / "expert_forecasts.jsonl", row)
            write_jsonl_append(market_dir / "forecasts.jsonl", row)
        event("forecast_done", forecast_count=len(rows))
        return rows

    def _run_worker(self, task, task_packet: dict[str, Any], proposal: dict[str, Any], sample_dir: Path, event) -> dict[str, Any]:
        if self.opts.worker_mode == "skip":
            result = {"status": "skipped", "proposal_id": proposal["proposal_id"]}
            write_json(sample_dir / "result.json", result)
            event("worker_skipped")
            return result
        workspace = sample_dir / "workspace"
        task.setup_workspace(workspace)
        prompt_text = worker_prompt(task_packet, proposal.get("text", ""), free_hparams=self.opts.free_hparams)
        (sample_dir / "worker_prompt.txt").write_text(prompt_text)
        (sample_dir / "master_worker_dialogue.jsonl").write_text("")
        if self.opts.worker_mode == "fixture":
            return self._fixture_worker(task, proposal, sample_dir, event)
        event("worker_started", proposal_id=proposal["proposal_id"])
        t0 = time.time()
        self._invoke_claude_worker(sample_dir, event)
        self._wait_for_worker_result(sample_dir, event)
        result = self._parse_worker_result(task, proposal["proposal_id"], sample_dir, time.time() - t0)
        if result.get("status") != "done":
            write_json(sample_dir / "error.json", {"ts": utc_now(), **result})
        event("worker_done", status=result.get("status"), passed=result.get("passed"), metric=result.get("val_metric"))
        return result

    def _fixture_worker(self, task, proposal: dict[str, Any], sample_dir: Path, event) -> dict[str, Any]:
        metric = task.pass_metric() + (0.01 if not task.lower_is_better else -0.01)
        secret = os.environ.get("BENCHMARK_HMAC_KEY", "benchmark-eval-secret")
        payload = f"{float(metric):.6f}"
        sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        result_json = {"val_metric": metric, "_sig": sig, "fixture": True}
        write_json(sample_dir / "workspace" / "result.json", result_json)
        (sample_dir / "worker.log").write_text("fixture worker\n")
        (sample_dir / "eval.log").write_text("fixture eval\n")
        result = self._parse_worker_result(task, proposal["proposal_id"], sample_dir, 0.0)
        event("fixture_worker_done", metric=metric, passed=result.get("passed"))
        return result

    def _invoke_claude_worker(self, sample_dir: Path, event=None) -> None:
        workspace = sample_dir / "workspace"
        prompt_file = sample_dir / "worker_prompt.txt"
        log_file = sample_dir / "worker.log"
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
                    if event:
                        event("worker_result_detected_while_process_running", elapsed_s=round(elapsed, 3))
                    log_f.write("\n[runner] result.json detected; terminating worker process after result capture\n")
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        if event:
                            event("worker_result_detected_kill_after_grace", elapsed_s=round(time.time() - started, 3))
                        proc.kill()
                        proc.wait()
                    return
                if elapsed >= self.opts.worker_timeout:
                    if event:
                        event("worker_timeout", elapsed_s=round(elapsed, 3), worker_timeout=self.opts.worker_timeout)
                    proc.kill()
                    proc.wait()
                    log_f.write(f"\n[runner] worker timeout after {self.opts.worker_timeout}s\n")
                    return
                time.sleep(5)

    def _wait_for_worker_result(self, sample_dir: Path, event) -> None:
        workspace = sample_dir / "workspace"
        result_path = workspace / "result.json"
        if result_path.exists() or self.opts.result_wait_timeout <= 0:
            return
        eval_log = workspace / "eval.log"
        saw_eval = eval_log.exists()
        start = time.time()
        deadline = start + self.opts.result_wait_timeout
        no_eval_deadline = start + self.opts.no_eval_wait_timeout
        event(
            "worker_result_wait_started",
            result_wait_timeout=self.opts.result_wait_timeout,
            no_eval_wait_timeout=self.opts.no_eval_wait_timeout,
        )
        while time.time() < deadline:
            if result_path.exists():
                event("worker_result_wait_done", elapsed_s=round(time.time() - start, 3))
                return
            if eval_log.exists():
                saw_eval = True
            if not saw_eval and time.time() >= no_eval_deadline:
                event("worker_result_wait_no_eval_log", elapsed_s=round(time.time() - start, 3))
                return
            time.sleep(10)
        event("worker_result_wait_timeout", elapsed_s=round(time.time() - start, 3))

    def _parse_worker_result(self, task, proposal_id: str, sample_dir: Path, elapsed_s: float) -> dict[str, Any]:
        workspace = sample_dir / "workspace"
        result_path = workspace / "result.json"
        if not result_path.exists():
            return {"status": "error", "proposal_id": proposal_id, "error": "result.json not written", "elapsed_s": round(elapsed_s, 3)}
        try:
            raw = json.loads(result_path.read_text())
        except Exception as exc:
            return {"status": "error", "proposal_id": proposal_id, "error": f"invalid result.json: {exc}", "elapsed_s": round(elapsed_s, 3)}
        shutil.copy2(result_path, sample_dir / "result.json")
        for name in ("eval.log", "editable_region.py", "baseline.py", "submission.csv"):
            src = workspace / name
            if src.exists() and src.is_file():
                shutil.copy2(src, sample_dir / name)
        metric, error = verify_signed_result(raw)
        if metric is None:
            return {"status": "error", "proposal_id": proposal_id, "error": error, "raw_result": raw, "elapsed_s": round(elapsed_s, 3)}
        result = {
            "status": "done",
            "proposal_id": proposal_id,
            "val_metric": metric,
            "baseline_metric": task.baseline_metric(),
            "pass_metric": task.pass_metric(),
            "improvement": round(task.improvement(metric), 6),
            "passed": task.passed(metric),
            "raw_result": raw,
            "elapsed_s": round(elapsed_s, 3),
        }
        write_json(sample_dir / "result.json", {**raw, "_parsed": result})
        return result

    def _settle(
        self,
        task_packet: dict[str, Any],
        proposal: dict[str, Any],
        forecasts: list[dict[str, Any]],
        worker_result: dict[str, Any],
        sample_dir: Path,
        event,
    ) -> dict[str, Any]:
        if worker_result.get("status") != "done":
            settlement = {"settled": False, "reason": worker_result.get("error"), "rows": []}
            write_json(sample_dir / "settlement.json", settlement)
            write_json(sample_dir / "market" / "settlement.json", settlement)
            return settlement
        outcome = 1.0 if worker_result.get("passed") else 0.0
        rows = []
        for forecast in forecasts:
            p = float(forecast["success_probability"])
            brier, log_score = brier_and_log(p, outcome)
            row = {
                "sample_id": task_packet["sample_id"],
                "task": task_packet["task"],
                "subtask": task_packet["subtask"],
                "module_id": proposal["module_id"],
                "proposal_id": proposal["proposal_id"],
                "expert_id": forecast["expert_id"],
                "expert_model_id": forecast.get("expert_model_id"),
                "success_probability": p,
                "outcome": int(outcome),
                "brier_score": round(brier, 6),
                "log_score": round(log_score, 6),
                "calibration_bucket": calibration_bucket(p),
                "val_metric": worker_result.get("val_metric"),
                "pass_metric": worker_result.get("pass_metric"),
                "settled_at": utc_now(),
            }
            rows.append(row)
            write_jsonl_append(self.root / "expert_reliability.jsonl", row)
        settlement = {
            "settled": True,
            "outcome": int(outcome),
            "rank_correlation": None,
            "rank_correlation_note": "single-proposal sample; sweep-level ranking computed in reports",
            "rows": rows,
        }
        write_json(sample_dir / "settlement.json", settlement)
        write_json(sample_dir / "market" / "settlement.json", settlement)
        event("settlement_done", outcome=int(outcome), rows=len(rows))
        return settlement

    def _call_llm(
        self,
        *,
        sample_dir: Path,
        role: str,
        call_id: str,
        cfg_row: dict[str, Any],
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        call_dir = sample_dir / "llm_calls" / call_id
        call_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "role": role,
            "call_id": call_id,
            "endpoint": cfg_row.get("endpoint"),
            "model_env": cfg_row.get("model_env"),
            "model": os.environ.get(str(cfg_row.get("model_env", ""))) or cfg_row.get("model"),
            "key_env": cfg_row.get("key_env"),
            "temperature": cfg_row.get("temperature"),
            "max_tokens": max_tokens,
            "messages": messages,
            "started_at": utc_now(),
        }
        write_json(call_dir / "request.json", request)
        total_t0 = time.time()
        llm_cfg = self.cfg.get("llm", {})
        max_retries = int(cfg_row.get("max_retries", llm_cfg.get("max_retries", 4)))
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 2):
            attempt_t0 = time.time()
            try:
                client = RunwayClient(self.cfg, key_env=str(cfg_row["key_env"]))
                temp = cfg_row.get("temperature")
                result = client.complete(
                    endpoint=str(cfg_row.get("endpoint", "chat_completions")),
                    model=str(request["model"]),
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
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "elapsed_s": round(time.time() - total_t0, 3),
                    "attempt_elapsed_s": round(time.time() - attempt_t0, 3),
                    "completed_at": utc_now(),
                }
                write_json(call_dir / "response.json", response)
                return result.text, response
            except Exception as exc:
                last_error = exc
                error = {
                    "error": str(exc)[:2000],
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "elapsed_s": round(time.time() - total_t0, 3),
                    "attempt_elapsed_s": round(time.time() - attempt_t0, 3),
                    "completed_at": utc_now(),
                }
                write_json(call_dir / f"error_attempt_{attempt:02d}.json", error)
                if attempt > max_retries:
                    write_json(call_dir / "error.json", error)
                    raise
                time.sleep(min(60.0, 2.0 ** attempt) + random.random())
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM call failed without an exception")

    def _runway_generator_cfg(self, generator_id: str) -> dict[str, Any]:
        row = self.cfg.get("proposal_batches", {}).get("runway_generators", {}).get(generator_id)
        if not row:
            raise ValueError(f"Unknown runway generator: {generator_id}")
        return row

    def _expert_cfg(self, expert_id: str) -> dict[str, Any]:
        for row in self.cfg.get("market", {}).get("llm_experts", []):
            if row.get("id") == expert_id:
                return row
        raise ValueError(f"Unknown expert id: {expert_id}")

    def _safe_expert_cfg(self, expert_id: str) -> dict[str, Any]:
        row = dict(self._expert_cfg(expert_id))
        row.pop("api_key", None)
        return row

    def _sample_summary(
        self,
        task_packet: dict[str, Any],
        proposal: dict[str, Any],
        forecasts: list[dict[str, Any]],
        worker_result: dict[str, Any],
        settlement: dict[str, Any],
    ) -> dict[str, Any]:
        mean_p = sum(float(f["success_probability"]) for f in forecasts) / len(forecasts) if forecasts else None
        return {
            "run_id": task_packet["sample_id"],
            "sample_id": task_packet["sample_id"],
            "task": task_packet["task"],
            "subtask": task_packet["subtask"],
            "task_type": task_packet["task_type"],
            "module_id": proposal["module_id"],
            "proposal_id": proposal["proposal_id"],
            "empty_control": proposal.get("empty_control", False),
            "baseline_metric": task_packet.get("baseline_metric"),
            "pass_threshold": task_packet.get("pass_threshold"),
            "pass_metric": task_packet.get("pass_metric"),
            "forecast_count": len(forecasts),
            "mean_success_probability": round(mean_p, 4) if mean_p is not None else None,
            "gate": {
                "selected_proposal_id": proposal["proposal_id"],
                "selected_label": proposal.get("label"),
                "scores": [{
                    "proposal_id": proposal["proposal_id"],
                    "label": proposal.get("label"),
                    "mean_success_probability": round(mean_p or 0.0, 4),
                    "passed_gate": True,
                }],
                "forced_for_sweep": True,
            },
            "worker_result": worker_result,
            "settlement": settlement,
            "paths": {
                "task_packet": "task_packet.json",
                "proposal": "empty_control.json" if proposal.get("empty_control") else "proposal.txt",
                "expert_forecasts": "expert_forecasts.jsonl",
                "worker_prompt": "worker_prompt.txt",
                "worker_log": "worker.log",
                "eval_log": "eval.log",
                "result": "result.json",
                "settlement": "settlement.json",
            },
        }

    def _register_sample(self, summary: dict[str, Any]) -> None:
        write_jsonl_append(self.registry_path, {
            "ts": utc_now(),
            "sample_id": summary.get("sample_id"),
            "task": summary.get("task"),
            "subtask": summary.get("subtask"),
            "module_id": summary.get("module_id"),
            "status": (summary.get("worker_result") or {}).get("status"),
            "passed": (summary.get("worker_result") or {}).get("passed"),
            "val_metric": (summary.get("worker_result") or {}).get("val_metric"),
            "path": summary.get("sample_id"),
        })

    def _write_sweep_manifest(self) -> None:
        write_json(self.root / "sweep_manifest.json", {
            "version": "v2.3",
            "created_at": utc_now(),
            "mode": self.opts.mode,
            "tasks": self.opts.tasks,
            "modules": self.opts.modules,
            "module_configs": {k: V2_3_MODULES[k] for k in self.opts.modules},
            "n_samples": self.opts.n_samples,
            "experts": self.opts.experts,
            "worker_mode": self.opts.worker_mode,
            "shard_index": self.opts.shard_index,
            "shard_count": self.opts.shard_count,
            "artifact_contract": [
                "task_packet.json",
                "proposal.txt or empty_control.json",
                "expert_forecasts.jsonl",
                "expert_prompts/<expert_id>.json",
                "expert_responses/<expert_id>.json",
                "worker_prompt.txt",
                "worker.log",
                "eval.log",
                "result.json or error.json",
                "settlement.json",
            ],
        })

    def _write_matrices(self) -> None:
        rows = [json.loads(line) for line in self.registry_path.read_text().splitlines()] if self.registry_path.exists() else []
        task_matrix: dict[str, dict[str, Any]] = {}
        module_matrix: dict[str, dict[str, Any]] = {}
        for row in rows:
            task = row.get("task")
            module = row.get("module_id")
            for matrix, key in ((task_matrix, task), (module_matrix, module)):
                if key not in matrix:
                    matrix[key] = {"samples": 0, "done": 0, "passed": 0, "errors": 0, "metrics": []}
                cell = matrix[key]
                cell["samples"] += 1
                if row.get("status") == "done":
                    cell["done"] += 1
                elif row.get("status") == "error":
                    cell["errors"] += 1
                if row.get("passed"):
                    cell["passed"] += 1
                if row.get("val_metric") is not None:
                    cell["metrics"].append(float(row["val_metric"]))
        for matrix in (task_matrix, module_matrix):
            for cell in matrix.values():
                metrics = cell.pop("metrics")
                cell["mean_metric"] = round(sum(metrics) / len(metrics), 4) if metrics else None
                cell["best_metric"] = round(max(metrics), 4) if metrics else None
        write_json(self.root / "task_matrix.json", task_matrix)
        write_json(self.root / "module_matrix.json", module_matrix)

    def _write_calibration_thresholds(self) -> None:
        rows = [json.loads(line) for line in self.registry_path.read_text().splitlines()] if self.registry_path.exists() else []
        by_task: dict[str, list[float]] = {}
        for row in rows:
            if row.get("status") == "done" and row.get("val_metric") is not None:
                by_task.setdefault(row["task"], []).append(float(row["val_metric"]))
        thresholds_path = project_root() / "configs" / "v2_3_thresholds.json"
        current = read_json(thresholds_path)
        task_thresholds = current.get("task_thresholds", {})
        for task_name, values in by_task.items():
            subtask = primary_subtask(task_name)
            best = max(values)
            task_thresholds.setdefault(task_name, {})
            task_thresholds[task_name][subtask] = {
                "baseline_metric": round(best, 4),
                "pass_threshold": 0.01,
                "pass_metric": round(best + 0.01, 4),
                "source": "v2_3_empty_worker_only_calibration",
                "sample_count": len(values),
                "updated_at": utc_now(),
            }
        write_json(thresholds_path, {"task_thresholds": task_thresholds})
        shutil.copy2(thresholds_path, self.root / "v2_3_thresholds.json")

    def _write_reports(self) -> None:
        self._write_matrices()
        task_matrix = read_json(self.root / "task_matrix.json")
        module_matrix = read_json(self.root / "module_matrix.json")
        lines = [
            "# V2.3 Formal Sweep Report",
            "",
            f"- Root: `{self.root}`",
            f"- Mode: `{self.opts.mode}`",
            f"- Tasks: {len(self.opts.tasks)}",
            f"- Modules: {len(self.opts.modules)}",
            f"- Samples per cell: {self.opts.n_samples}",
            "",
            "## Modules",
            "",
        ]
        for module_id, row in sorted(module_matrix.items()):
            lines.append(
                f"- `{module_id}`: samples={row.get('samples')} done={row.get('done')} "
                f"passed={row.get('passed')} errors={row.get('errors')} best={row.get('best_metric')}"
            )
        lines.extend(["", "## Tasks", ""])
        for task_id, row in sorted(task_matrix.items()):
            lines.append(
                f"- `{task_id}`: samples={row.get('samples')} done={row.get('done')} "
                f"passed={row.get('passed')} errors={row.get('errors')} best={row.get('best_metric')}"
            )
        (self.root / "formal_sweep_report.md").write_text("\n".join(lines))
        (self.root / "expert_calibration_report.md").write_text(render_expert_calibration(self.root))

    def _sweep_summary(self, total: int, assigned: int | None = None) -> dict[str, Any]:
        return {
            "version": "v2.3",
            "mode": self.opts.mode,
            "root": str(self.root),
            "total_samples_planned": total,
            "assigned_samples_planned": assigned if assigned is not None else total,
            "shard_index": self.opts.shard_index,
            "shard_count": self.opts.shard_count,
            "tasks": self.opts.tasks,
            "modules": self.opts.modules,
            "registry": "registry.jsonl",
            "task_matrix": "task_matrix.json",
            "module_matrix": "module_matrix.json",
            "expert_reliability": "expert_reliability.jsonl",
            "formal_sweep_report": "formal_sweep_report.md",
            "expert_calibration_report": "expert_calibration_report.md",
        }

    def _dry_run_summary(self) -> dict[str, Any]:
        rows = []
        ordinal = 0
        for module_id in self.opts.modules:
            for task_name in self.opts.tasks:
                sample_indices = []
                for sample_idx in range(self.opts.n_samples):
                    if self._sample_belongs_to_shard(ordinal):
                        sample_indices.append(sample_idx)
                    ordinal += 1
                if sample_indices:
                    rows.append({
                        "task": task_name,
                        "subtask": primary_subtask(task_name),
                        "module_id": module_id,
                        "samples": len(sample_indices),
                        "sample_indices": sample_indices,
                    })
        return {
            "version": "v2.3",
            "mode": "dry-run",
            "cell_count": len(rows),
            "sample_count": sum(row["samples"] for row in rows),
            "shard_index": self.opts.shard_index,
            "shard_count": self.opts.shard_count,
            "cells": rows,
        }

    def _event(self, event_name: str, **fields: Any) -> None:
        write_jsonl_append(self.events_path, {"ts": utc_now(), "event": event_name, **fields})


def primary_subtask(task_name: str) -> str:
    subtasks = MLS_SUBTASKS.get(task_name)
    if not subtasks:
        raise ValueError(f"No primary subtask registered for {task_name}")
    return subtasks[0]


def checkpoint_prompt_record(task_packet: dict[str, Any], module_id: str, strategy: str) -> dict[str, Any]:
    research_question = (
        f"What concrete method change for `{task_packet['task']}` / `{task_packet.get('subtask')}` "
        f"is most likely to exceed pass metric {task_packet.get('pass_metric')}?"
    )
    strategy_note = {
        "top_k_refs": "Use the most relevant task evidence and frontline papers only.",
        "top_k_related_work": "Use concise related-work synthesis plus top references from the task evidence.",
        "with_research_question": f"Answer this explicit research question: {research_question}",
    }.get(strategy, "Use the full task evidence.")
    system = (
        "You are an idea proposal model. Generate one XML research proposal with "
        "problem, gap, key insight, approach, expected contributions. Be concrete "
        "and implementable for the benchmark worker."
    )
    prompt = (
        f"Prompt strategy: {strategy}\n"
        f"Module: {module_id}\n\n"
        f"{strategy_note}\n\n"
        "Task packet JSON:\n"
        f"{json.dumps(task_packet, ensure_ascii=False, indent=2)}\n\n"
        "Generate a structured research proposal using this exact format:\n"
        "<proposal>\n"
        "<problem>...</problem>\n"
        "<gap>...</gap>\n"
        "<key_insight>...</key_insight>\n"
        "<approach>...</approach>\n"
        "<expected_contributions>...</expected_contributions>\n"
        "</proposal>"
    )
    return {"system": system, "prompt": prompt, "strategy": strategy, "task": task_packet["task"], "module_id": module_id}


def mock_proposal(task_packet: dict[str, Any], module_id: str) -> str:
    return (
        "<proposal>\n"
        f"<title>Mock {module_id} proposal</title>\n"
        f"<problem>Improve {task_packet.get('task')} on {task_packet.get('subtask')}.</problem>\n"
        "<gap>The current baseline may leave a small optimization or generalization gap.</gap>\n"
        "<core_idea>Use a conservative smooth variant that preserves the task interface.</core_idea>\n"
        "<implementation_plan>Modify only editable_region.py with a minimal, deterministic implementation.</implementation_plan>\n"
        "<evaluation_plan>Run bash run.sh result.json and compare against pass_metric.</evaluation_plan>\n"
        "<expected_results>Small but measurable accuracy improvement.</expected_results>\n"
        "<risks_and_limitations>The change may not beat a calibrated worker-only threshold.</risks_and_limitations>\n"
        "</proposal>"
    )


def mock_forecast(proposal: dict[str, Any], expert_id: str) -> dict[str, Any]:
    if proposal.get("empty_control"):
        p = 0.18
    else:
        base = 0.46 + (int(hashlib.sha1(proposal["module_id"].encode()).hexdigest()[:2], 16) % 20) / 100
        p = base - (0.04 if expert_id == "gpt55" else 0.0)
    return {
        "success_probability": max(0.01, min(0.99, p)),
        "rationale": "Mock forecast for offline V2.3 pipeline validation.",
        "feasibility": 0.65,
        "novelty_over_worker_best": 0.0 if proposal.get("empty_control") else 0.45,
        "expected_delta": 0.02,
        "risk": 0.55,
        "confidence": 0.5,
        "strengths": ["Concrete artifact path", "Benchmark-defined outcome"],
        "weaknesses": ["Uncertain improvement over calibrated worker-only ceiling"],
        "baseline_risk": "The idea may not exceed the worker-only pass threshold.",
    }


def forecast_row(
    task_packet: dict[str, Any],
    proposal: dict[str, Any],
    expert_id: str,
    parsed: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        "forecast_id": stable_id("fcst", task_packet["sample_id"], proposal["proposal_id"], expert_id),
        "sample_id": task_packet["sample_id"],
        "task": task_packet["task"],
        "subtask": task_packet.get("subtask"),
        "module_id": proposal["module_id"],
        "proposal_id": proposal["proposal_id"],
        "empty_control": proposal.get("empty_control", False),
        "expert_id": expert_id,
        "expert_model_id": response.get("model"),
        "success_probability": round(clamp_prob(parsed.get("success_probability")), 4),
        "rationale": str(parsed.get("rationale", "")).strip(),
        "feasibility": round(clamp_prob(parsed.get("feasibility")), 4),
        "novelty_over_worker_best": round(clamp_prob(parsed.get("novelty_over_worker_best")), 4),
        "expected_delta": safe_float(parsed.get("expected_delta")),
        "risk": round(clamp_prob(parsed.get("risk")), 4),
        "confidence": round(clamp_prob(parsed.get("confidence")), 4),
        "strengths": parsed.get("strengths") or [],
        "weaknesses": parsed.get("weaknesses") or [],
        "baseline_risk": str(parsed.get("baseline_risk", "")).strip(),
        "usage": response.get("usage") or {},
        "latency_s": response.get("elapsed_s"),
        "created_at": utc_now(),
    }


def safe_float(value: Any) -> float | None:
    try:
        return round(float(value), 6)
    except Exception:
        return None


def render_expert_calibration(root: Path) -> str:
    path = root / "expert_reliability.jsonl"
    if not path.exists():
        return "# Expert Calibration\n\nNo settled forecasts yet.\n"
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    by_expert: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_expert.setdefault(row["expert_id"], []).append(row)
    lines = ["# Expert Calibration", ""]
    for expert_id, items in sorted(by_expert.items()):
        mean_brier = sum(float(x["brier_score"]) for x in items) / len(items)
        mean_log = sum(float(x["log_score"]) for x in items) / len(items)
        accuracy = sum(int(x["outcome"] == (float(x["success_probability"]) >= 0.5)) for x in items) / len(items)
        lines.append(
            f"- `{expert_id}`: n={len(items)} mean_brier={mean_brier:.4f} "
            f"mean_log={mean_log:.4f} threshold_accuracy={accuracy:.3f}"
        )
    lines.extend(["", "## Calibration Buckets", ""])
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["expert_id"], row["calibration_bucket"]), []).append(row)
    for (expert_id, bucket), items in sorted(buckets.items()):
        mean_p = sum(float(x["success_probability"]) for x in items) / len(items)
        rate = sum(float(x["outcome"]) for x in items) / len(items)
        lines.append(f"- `{expert_id}` {bucket}: n={len(items)} mean_p={mean_p:.3f} pass_rate={rate:.3f}")
    return "\n".join(lines) + "\n"


def run_formal_sweep(cfg: dict[str, Any], opts: SweepOptions) -> dict[str, Any]:
    return FormalSweepRunner(cfg, opts).run()
