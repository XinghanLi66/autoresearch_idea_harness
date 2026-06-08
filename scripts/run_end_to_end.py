#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.end_to_end import EndToEndOptions, run_end_to_end
from autoresearch_idea_harness.io import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Run the end-to-end autoresearch pipeline.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--task", default="dl_activation_function")
    p.add_argument("--subtask", default="resnet20-cifar10")
    p.add_argument("--n-proposals", type=int, default=3)
    p.add_argument("--experts", default="opus47,gpt55")
    p.add_argument("--gate-threshold", type=float, default=0.5)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--worker-mode", choices=["claude", "fixture", "skip"], default="claude")
    p.add_argument("--mock-llm", action="store_true")
    p.add_argument("--gpu", default="0")
    p.add_argument("--worker-timeout", type=int, default=7200)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-master-advice", type=int, default=1)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    experts = [x.strip() for x in args.experts.split(",") if x.strip()]
    opts = EndToEndOptions(
        task=args.task,
        subtask=args.subtask,
        n_proposals=args.n_proposals,
        experts=experts,
        gate_threshold=args.gate_threshold,
        max_workers=args.max_workers,
        worker_mode=args.worker_mode,
        mock_llm=args.mock_llm,
        gpu=args.gpu,
        worker_timeout=args.worker_timeout,
        max_turns=args.max_turns,
        max_master_advice=args.max_master_advice,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    summary = run_end_to_end(cfg, opts)
    print({
        "run_id": summary["run_id"],
        "task": summary["task"],
        "subtask": summary["subtask"],
        "selected": summary["gate"].get("selected_proposal_id"),
        "worker_status": summary["worker_result"].get("status"),
        "passed": summary["worker_result"].get("passed"),
        "report": summary["paths"]["report"],
    })


if __name__ == "__main__":
    main()
