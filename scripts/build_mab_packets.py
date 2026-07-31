#!/usr/bin/env python3
"""Build proposal-elicitation packets for the 5 MLAgentBench (MAB) starter tasks.

One packet per task = the prompt fed to a checkpoint (base/SFT/RL arm) to elicit ONE research idea for
that task's editable component (`train.py`). The model's output (its Core-idea CoT) becomes the proposal
that the MAB worker (mab_worker_edit.py) then implements by rewriting train.py. The same packet is used
across all arms so the only variable is the model. Reads research_problem.txt + metric direction straight
from the MLAgentBench checkout (no GPU/API).

Schema matches the MLS-Bench-Lite packets (build_mls_lite_packets.py) so gen_proposals_from_endpoint.py
works unmodified against this file.

Out: runs/researcher_cot/mab_eval/packets.jsonl
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _mab_common as mab  # noqa: E402

SYSTEM = ("You are a world-class AI researcher. You are given a machine-learning task with a specific "
          "editable code component. Reason step by step in your own distinctive style toward ONE genuinely "
          "novel, non-obvious idea to improve that component. Make the creative core unmistakably clear and "
          "explain how you arrived at it. Focus on ideas and mechanisms, not full implementation code.")


def main() -> None:
    tasks = json.loads((ROOT / "docs/eval/mab_tasks.json").read_text())
    out = ROOT / "runs/researcher_cot/mab_eval/packets.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for slug in tasks["slugs"]:
            spec = tasks["tasks"][slug]
            ef = spec.get("edit_file", "train.py")
            desc = (mab.task_dir(slug) / "scripts" / "research_problem.txt").read_text(errors="ignore")[:2600]
            user = (f"{desc}\n\nThe editable component is `{ef}`. Propose ONE concrete, novel idea to "
                    "improve it. Conclude with two lines: 'Core idea:' and 'Non-trivial crux:'.")
            fh.write(json.dumps({
                "task": slug, "setting": None, "metric": spec.get("metric"),
                "lower_is_better": spec.get("lower_is_better", False),
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} packets -> {out}")


if __name__ == "__main__":
    main()
