#!/usr/bin/env python3
"""Prepare HuggingFace datasets for lm-evaluation-harness offline evaluation.

Ensures the datasets used by lm-eval benchmarks (hellaswag, arc_easy, piqa,
winogrande) are present in the offline HF cache, keyed by the EXACT
(dataset_path, dataset_name) that lm-eval's task YAMLs resolve — so that
`datasets.load_dataset(...)` finds them under HF_DATASETS_OFFLINE=1:

    hellaswag  -> Rowan/hellaswag        (config default)   -> Rowan___hellaswag
    piqa       -> baber/piqa             (config default)   -> baber___piqa
    arc_easy   -> allenai/ai2_arc        (config ARC-Easy)  -> allenai___ai2_arc/ARC-Easy
    winogrande -> allenai/winogrande     (config winogrande_xl) -> allenai___winogrande/winogrande_xl

The previous version also listed BARE legacy names ("piqa", "hellaswag",
"winogrande").  Those are script-based / trust_remote_code datasets that lm-eval
does NOT use; loading "piqa" offline raises
    "Couldn't find a dataset script at .../piqa/piqa.py"
and (online, non-interactive) raises
    "The repository for piqa contains custom code which must be executed",
which aborted this prepare step.  Only the namespaced names above are needed.

This script defaults to OFFLINE and loads from the vendored cache
({data_root}/lm-eval-datasets), because it is re-run on every baseline on the
(offline) eval farm.  To (re)populate the cache from the Hub on a machine with
internet, run:
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
        python vendor/data_scripts/lm-evaluation-harness/prepare_datasets.py --data-root vendor/data

Output: {data_root}/lm-eval-datasets/  (~200 MB)
"""

import argparse
import os
import sys
from pathlib import Path

# Default to offline so the vendored cache is used (and no Hub round-trip / no
# "custom code must be executed" prompt).  setdefault lets an explicit
# HF_HUB_OFFLINE=0 in the environment re-enable downloads for (re)population.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# (dataset_path, dataset_name) exactly as lm-eval's task YAMLs resolve them.
DATASETS = [
    ("Rowan/hellaswag", None),
    ("baber/piqa", None),
    ("allenai/ai2_arc", "ARC-Easy"),
    ("allenai/winogrande", "winogrande_xl"),
]


def main():
    parser = argparse.ArgumentParser(description="Prepare lm-eval datasets")
    parser.add_argument(
        "--data-root", type=str, default="vendor/data",
        help="Root directory for data storage",
    )
    args = parser.parse_args()

    cache_dir = Path(args.data_root) / "lm-eval-datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    print(f"Preparing datasets in: {cache_dir} (offline={offline})")

    from datasets import load_dataset

    for name, config in DATASETS:
        try:
            print(f"  Loading {name} (config={config})...")
            load_dataset(name, config, cache_dir=str(cache_dir), trust_remote_code=True)
            print(f"  OK: {name}")
        except Exception as e:
            print(f"  FAIL: {name}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nAll datasets ready at {cache_dir}")


if __name__ == "__main__":
    main()
