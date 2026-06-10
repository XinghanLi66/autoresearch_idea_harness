#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.v3_report import collect_v3_status, write_v3_report


def main() -> None:
    p = argparse.ArgumentParser(description="Render a V3 training data/run status report.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument(
        "--training-data-root",
        default=str(ROOT / "runs" / "training_data"),
    )
    p.add_argument(
        "--training-root",
        default=str(ROOT / "runs" / "training"),
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "runs" / "reports" / "v3_training_status.md"),
    )
    p.add_argument("--json-output", default=None)
    args = p.parse_args()

    output = Path(args.output)
    json_output = Path(args.json_output) if args.json_output else output.with_suffix(".json")
    status = collect_v3_status(Path(args.training_data_root), Path(args.training_root), cfg=load_config(args.config))
    write_v3_report(status, output_md=output, output_json=json_output)
    print({"output": str(output), "json_output": str(json_output), "counts": status.get("counts")})


if __name__ == "__main__":
    main()
