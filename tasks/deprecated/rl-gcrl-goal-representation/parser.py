"""Task-specific output parser for rl-gcrl-goal-representation.

Handles training + evaluation output from the GCIVL goal representation task:

Training feedback: lines matching
    TRAIN_METRICS step=N key=val key=val ...

Evaluation feedback: lines matching
    TEST_METRICS step=N success_rate=X.XXXX

Metrics are keyed by environment name, e.g. success_rate_antmaze_large_navigate_v0.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the rl-gcrl-goal-representation task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics.
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse evaluation metrics (success rate).
        eval_feedback, eval_metrics = self._parse_eval_metrics(raw_output, cmd_label)
        if eval_feedback:
            feedback_parts.append(eval_feedback)
        metrics.update(eval_metrics)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract TRAIN_METRICS lines and return a summary of the last few."""
        lines = []
        for line in output.splitlines():
            if line.strip().startswith("TRAIN_METRICS "):
                lines.append(line.strip())

        if not lines:
            return ""

        summary_lines = lines[-5:]
        return "Training metrics (last steps):\n" + "\n".join(summary_lines)

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract TEST_METRICS lines and return feedback + metrics.

        Expected format: TEST_METRICS step=N success_rate=X.XXXX
        """
        scores: list[float] = []
        eval_lines: list[str] = []

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+step=(\d+)\s+success_rate=(-?[\d.]+(?:e[+-]?\d+)?|nan|inf|-inf)",
                line, re.IGNORECASE
            )
            if match:
                step = int(match.group(1))
                raw = match.group(2).lower()
                score = float(raw)
                eval_lines.append(line.strip())
                if not (score != score or abs(score) == float("inf")):
                    scores.append(score)

        metrics: dict = {}
        feedback = ""

        if scores:
            final_score = scores[-1]
            metric_key = "success_rate_" + cmd_label.replace("-", "_")
            metrics[metric_key] = final_score

            feedback = f"Evaluation ({cmd_label}):\n" + "\n".join(eval_lines[-3:])
            feedback += f"\nFinal success rate: {final_score:.4f}"

        return feedback, metrics
