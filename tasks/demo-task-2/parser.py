"""Task-specific output parser for demo-task-2.

Training feedback: lines matching
    TRAIN_METRICS epoch=E batch=B loss=L

Evaluation feedback: lines matching
    TEST_METRICS loss=L accuracy=A

Leaderboard metrics: test_accuracy (from the last epoch's test).
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the demo-task-2 (MNIST lr-scheduler) task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse test metrics
        test_feedback, test_metrics = self._parse_test_metrics(raw_output)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

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

        # Return last 5 training metric lines as feedback
        summary_lines = lines[-5:]
        return "Training metrics (last steps):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, ) -> tuple[str, dict]:
        """Extract TEST_METRICS lines and return feedback + metrics.

        Expected format: TEST_METRICS loss=X.XXXX accuracy=XX.XX
        """
        accuracies: list[float] = []
        losses: list[float] = []
        test_lines: list[str] = []

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+loss=([\d.]+)\s+accuracy=([\d.]+)", line
            )
            if match:
                test_lines.append(line.strip())
                losses.append(float(match.group(1)))
                accuracies.append(float(match.group(2)))

        metrics: dict = {}
        feedback = ""

        if accuracies:
            final_accuracy = accuracies[-1]
            final_loss = losses[-1]
            metrics["test_accuracy"] = final_accuracy
            metrics["test_loss"] = final_loss

            feedback = "Test evaluation:\n" + "\n".join(test_lines[-3:])
            feedback += f"\nFinal test accuracy: {final_accuracy:.2f}%"

        return feedback, metrics
