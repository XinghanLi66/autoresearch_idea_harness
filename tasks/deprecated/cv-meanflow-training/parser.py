"""Task-specific output parser for cv-meanflow-training.

Extracts FID from TEST_METRICS output line.

Expected format:
    TEST_METRICS: fid=12.34, best_fid=11.50

Metrics are suffixed by the test command label so visible and hidden training
scales are scored independently.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the cv-meanflow-training task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}
        label = cmd_label.replace("-", "_")

        for line in raw_output.splitlines():
            if "TEST_METRICS:" not in line:
                continue

            fid_match = re.search(r"fid=([\d.]+)", line)
            best_match = re.search(r"best_fid=([\d.]+)", line)

            if fid_match:
                fid = float(fid_match.group(1))
                metrics[f"fid_{label}"] = fid

                best_fid = float(best_match.group(1)) if best_match else fid
                metrics[f"best_fid_{label}"] = best_fid

                feedback_parts.append(
                    f"{cmd_label}: FID: {fid:.2f}, Best FID: {best_fid:.2f}"
                )

        if feedback_parts:
            feedback = "Training results:\n" + "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)
