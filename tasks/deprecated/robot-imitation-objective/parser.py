"""Task-specific output parser for robot-imitation-objective.

Training feedback: lines matching
    TRAIN_METRICS step=S loss=L

Evaluation feedback: lines matching
    TEST_METRICS step=S success_rate=R

Leaderboard metric: success_rate_<label> (from the last TEST_METRICS line).
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for robot-imitation-objective task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

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

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        success_rates: list[float] = []
        test_lines: list[str] = []

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+step=(\d+)\s+success_rate=([\d.]+)", line
            )
            if match:
                test_lines.append(line.strip())
                success_rates.append(float(match.group(2)))

        metrics: dict = {}
        feedback = ""

        if success_rates:
            final_sr = success_rates[-1]
            metrics[f"success_rate_{cmd_label}"] = final_sr

            feedback = "Evaluation results:\n" + "\n".join(test_lines[-3:])
            feedback += f"\nFinal success rate: {final_sr:.4f}"

        return feedback, metrics
