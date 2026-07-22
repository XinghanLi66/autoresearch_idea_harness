#!/usr/bin/env python3
"""Emit a standardized HOST_METRICS line for read-only video external validation.

This script intentionally does not define leaderboard metrics. It converts
already-produced FAR/VBench sidecar artifacts into one compact `HOST_METRICS:`
line so task logs, journals, and future hidden tests can all speak the same
surface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent.parent
CODEBASE_ROOT = TASK_DIR.parent.parent.parent
sys.path.insert(0, str(CODEBASE_ROOT / "compat" / "vbench"))

from summarize_vbench_long_results import summarize as summarize_vbench  # type: ignore


PAIR_RE = re.compile(r"^([a-zA-Z0-9_]+)=(.+)$")


def parse_pairs(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        match = PAIR_RE.match(value)
        if not match:
            raise ValueError(f"expected name=path, got: {value}")
        parsed[match.group(1)] = Path(match.group(2))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-vbench",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="candidate/readout VBench result root(s), e.g. imaging_quality=/path/to/output",
    )
    parser.add_argument(
        "--reference-vbench",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="reference/control VBench result root(s), e.g. imaging_quality=/path/to/output",
    )
    parser.add_argument("--candidate-psnr", type=float)
    parser.add_argument("--reference-psnr", type=float)
    parser.add_argument("--candidate-ssim", type=float)
    parser.add_argument("--reference-ssim", type=float)
    args = parser.parse_args()

    candidate_roots = parse_pairs(args.candidate_vbench)
    reference_roots = parse_pairs(args.reference_vbench)

    payload: dict[str, float | str] = {}
    for metric_name, root in sorted(candidate_roots.items()):
        summary = summarize_vbench(root)
        score = summary.get("best_score")
        if isinstance(score, (int, float)):
            payload[f"host_vbench_long_{metric_name}_candidate"] = float(score)
        payload[f"host_vbench_long_{metric_name}_candidate_root"] = str(root)
    for metric_name, root in sorted(reference_roots.items()):
        summary = summarize_vbench(root)
        score = summary.get("best_score")
        if isinstance(score, (int, float)):
            payload[f"host_vbench_long_{metric_name}_reference"] = float(score)
        payload[f"host_vbench_long_{metric_name}_reference_root"] = str(root)

    shared_names = sorted(set(candidate_roots) & set(reference_roots))
    for metric_name in shared_names:
        cand_key = f"host_vbench_long_{metric_name}_candidate"
        ref_key = f"host_vbench_long_{metric_name}_reference"
        if cand_key in payload and ref_key in payload:
            payload[f"host_vbench_long_{metric_name}_gap"] = float(payload[cand_key]) - float(payload[ref_key])

    if args.candidate_psnr is not None:
        payload["host_decoded_psnr_candidate"] = args.candidate_psnr
    if args.reference_psnr is not None:
        payload["host_decoded_psnr_reference"] = args.reference_psnr
    if args.candidate_psnr is not None and args.reference_psnr is not None:
        payload["host_decoded_psnr_gap"] = args.candidate_psnr - args.reference_psnr

    if args.candidate_ssim is not None:
        payload["host_decoded_ssim_candidate"] = args.candidate_ssim
    if args.reference_ssim is not None:
        payload["host_decoded_ssim_reference"] = args.reference_ssim
    if args.candidate_ssim is not None and args.reference_ssim is not None:
        payload["host_decoded_ssim_gap"] = args.candidate_ssim - args.reference_ssim

    print("HOST_METRICS:", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
