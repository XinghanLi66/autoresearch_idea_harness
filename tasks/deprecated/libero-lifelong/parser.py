"""Task-specific output parser for libero-lifelong.

Training feedback: lines matching
    TRAIN_METRICS task=T epoch=E loss=L
    [info] Epoch: E | train loss: L | time: T
    [info] Epoch: E | succ: S +/- C | best succ: B | ...

Evaluation feedback: lines matching
    EVAL_METRICS after_task=T avg_success=S
    [Task N succ.] S1 | S2 | ...

Leaderboard metric: avg_final_success (from TEST_METRICS).
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the libero-lifelong (continual robot learning) task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_fb = self._parse_train_metrics(raw_output)
        if train_fb:
            feedback_parts.append(train_fb)

        # Parse evaluation metrics
        eval_fb = self._parse_eval_metrics(raw_output)
        if eval_fb:
            feedback_parts.append(eval_fb)

        # Parse final test metrics
        test_fb, test_metrics = self._parse_test_metrics(raw_output)
        if test_fb:
            feedback_parts.append(test_fb)
        metrics.update(test_metrics)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract training progress and return a summary."""
        lines = []
        for line in output.splitlines():
            stripped = line.strip()
            # Match our injected TRAIN_METRICS
            if stripped.startswith("TRAIN_METRICS "):
                lines.append(stripped)
            # Match LIBERO's native epoch info
            elif stripped.startswith("[info] Epoch:"):
                lines.append(stripped)

        if not lines:
            return ""

        # Return last 10 lines as feedback
        summary = lines[-10:]
        return "Training progress (recent):\n" + "\n".join(summary)

    def _parse_eval_metrics(self, output: str) -> str:
        """Extract evaluation metrics after each task."""
        eval_lines = []
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("EVAL_METRICS "):
                eval_lines.append(stripped)
            elif "[Task" in stripped and "succ.]" in stripped:
                eval_lines.append(stripped)

        if not eval_lines:
            return ""

        # Return last 5 eval summaries
        summary = eval_lines[-5:]
        return "Evaluation results:\n" + "\n".join(summary)

    def _parse_test_metrics(self, output: str) -> tuple[str, dict]:
        """Extract TEST_METRICS for leaderboard."""
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+avg_final_success=([\d.]+)", line
            )
            if match:
                avg_success = float(match.group(1))
                metrics["avg_final_success"] = avg_success
                feedback = f"Final average success rate: {avg_success:.4f}"

        # Also collect per-task results
        task_results = []
        for line in output.splitlines():
            match = re.search(
                r"TASK_METRICS\s+task=(\d+)\s+success_rate=([\d.]+)", line
            )
            if match:
                task_id = int(match.group(1))
                rate = float(match.group(2))
                task_results.append(f"  Task {task_id}: {rate:.4f}")

        if task_results:
            feedback += "\nPer-task success rates:\n" + "\n".join(task_results)

        return feedback, metrics
