"""Task-specific output parser for rl-offline-policy.

Training feedback: lines matching
    TRAIN_METRICS step=N critic_loss=X q_mean=X actor_loss=X ...

Evaluation feedback: lines matching
    TEST_METRICS step=N success_rate=X.XXXX

Leaderboard metric: success_rate (from the final evaluation).
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the rl-offline-policy task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

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
        lines = []
        for line in output.splitlines():
            if line.strip().startswith("TRAIN_METRICS "):
                lines.append(line.strip())
        if not lines:
            return ""
        summary_lines = lines[-5:]
        return "Training metrics (last steps):\n" + "\n".join(summary_lines)

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract TEST_METRICS lines with success_rate."""
        success_rates: list[float] = []
        eval_lines: list[str] = []

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+step=(\d+)\s+success_rate=([\d.]+(?:e[+-]?\d+)?)",
                line,
            )
            if match:
                eval_lines.append(line.strip())
                rate = float(match.group(2))
                if not (rate != rate or abs(rate) == float("inf")):
                    success_rates.append(rate)

        metrics: dict = {}
        feedback = ""

        if success_rates:
            final_rate = success_rates[-1]
            metric_key = "success_rate_" + cmd_label.replace("-", "_")
            metrics[metric_key] = final_rate

            feedback = f"Evaluation ({cmd_label}):\n" + "\n".join(eval_lines[-3:])
            feedback += f"\nFinal success rate: {final_rate:.4f}"

        return feedback, metrics
