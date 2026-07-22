"""Task-specific output parser for humanoid-ppo-extractor.

Training feedback: lines matching
    TRAIN_METRICS timestep=T mean_reward=R mean_length=L

Evaluation feedback: lines matching
    TEST_METRICS mean_reward=R std_reward=S

Leaderboard metric: mean_reward (from final evaluation).
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the humanoid-ppo-extractor task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse test metrics
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

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract TEST_METRICS lines and return feedback + metrics.

        Expected format: TEST_METRICS mean_reward=X.XX std_reward=X.XX
        """
        metrics: dict = {}
        feedback = ""
        label = cmd_label.replace("-", "_")

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+mean_reward=([\d.-]+)\s+std_reward=([\d.-]+)",
                line,
            )
            if match:
                mean_reward = float(match.group(1))
                std_reward = float(match.group(2))
                metrics["mean_reward"] = mean_reward
                metrics["std_reward"] = std_reward
                metrics[f"mean_reward_{label}"] = mean_reward
                metrics[f"std_reward_{label}"] = std_reward
                feedback = (
                    f"Final evaluation ({cmd_label}):\n"
                    f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
                )

        return feedback, metrics
