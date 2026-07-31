#!/usr/bin/env python3
"""Build proposal-elicitation packets for the 30 MLS-Bench-Lite tasks.

One packet per task = the prompt fed to a checkpoint (base/SFT/RL arm) to elicit ONE research idea for
that task's editable component. The model's output (its Core-idea CoT) becomes the proposal that the
mlsbench worker implements via --extra-context. Same packet is used across all arms so the only variable
is the model. Reads task_description.md + editable file + metric direction straight from the GitHub
MLS-Bench checkout (no GPU/API).

Out: runs/researcher_cot/mls_lite_eval/packets.jsonl
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _mls_lite_common as mls  # noqa: E402

SYSTEM = ("You are a world-class AI researcher. You are given a machine-learning task with a specific "
          "editable code component. Reason step by step in your own distinctive style toward ONE genuinely "
          "novel, non-obvious idea to improve that component. Make the creative core unmistakably clear and "
          "explain how you arrived at it. Focus on ideas and mechanisms, not full implementation code.")


def edit_file(slug: str) -> str:
    cfg = json.loads((mls.task_dir(slug) / "config.json").read_text())
    files = cfg.get("files") or []
    return str(files[0].get("filename")) if files and isinstance(files[0], dict) else "(the editable region)"


def main() -> None:
    lite = json.loads((ROOT / "docs/eval/mls_bench_lite_tasks.json").read_text())
    out = ROOT / "runs/researcher_cot/mls_lite_eval/packets.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for slug in lite["slugs"]:
            setting = (mls.visible_settings(slug) or [None])[0]
            dirs = mls.score_spec_directions(slug)
            metric_cols = [c for c in dirs if setting and c.endswith(setting)] or list(dirs)
            metric = metric_cols[0] if metric_cols else None
            desc = (mls.task_dir(slug) / "task_description.md").read_text(errors="ignore")[:2600]
            ef = edit_file(slug)
            user = (f"{desc}\n\nThe editable component is `{ef}` (evaluation setting: {setting}). "
                    "Propose ONE concrete, novel idea to improve it. Conclude with two lines: "
                    "'Core idea:' and 'Non-trivial crux:'.")
            fh.write(json.dumps({
                "task": slug, "setting": setting, "metric": metric,
                "lower_is_better": (not dirs.get(metric, True)) if metric else None,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} packets -> {out}")


if __name__ == "__main__":
    main()
