"""Task-specific output parser for robot-policy-conditioning.

Training feedback: lines matching
    TRAIN_METRICS step=S loss=L
    TRAIN_METRICS step=S success_rate=R best_success=B

Final metric: line matching
    TEST_METRICS success_rate=X.XXXX

Leaderboard metric: success_rate_<label> (higher is better).
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the robot-policy-conditioning task."""

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

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [
            l.strip()
            for l in output.splitlines()
            if l.strip().startswith("TRAIN_METRICS ")
        ]
        if not lines:
            return ""
        return "Training metrics (last steps):\n" + "\n".join(lines[-8:])

    def _parse_test_metrics(self, output: str, cmd_label: str = "") -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+success_rate=([\d.]+)", line
            )
            if match:
                success_rate = float(match.group(1))
                metric_key = f"success_rate_{cmd_label}" if cmd_label else "success_rate"
                metrics[metric_key] = success_rate
                feedback = f"Final success rate: {success_rate:.4f}"

        return feedback, metrics
