"""Task-specific output parser for nanochat LLM pretraining tasks.

Handles output from nanochat base_train.py:
- Training feedback: step NNNNN/NNNNN ... loss: X.XXXXXX ...
- Validation: Step NNNNN | Validation bpb: X.XXXXXX
- Final metric: Minimum validation bpb: X.XXXXXX
Metrics are keyed by model size label, e.g. val_bpb_depth-4.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for nanochat LLM pretraining tasks."""

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

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract training step logs (last 5 lines with loss info)."""
        lines = []
        for line in output.splitlines():
            stripped = line.strip()
            # Match nanochat training lines like: step 00100/00500 (20.00%) | loss: 5.123456 ...
            if re.search(r"step \d+/\d+.*loss:", stripped):
                lines.append(stripped)
        if not lines:
            return ""
        return "Training metrics (last steps):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract validation bpb metrics from nanochat output."""
        metrics: dict = {}
        feedback = ""

        val_bpb_values = []
        min_val_bpb = None

        for line in output.splitlines():
            # Match: Step 00500 | Validation bpb: 1.234567
            m = re.search(r"Validation bpb:\s+([\d.]+)", line)
            if m:
                val_bpb_values.append(float(m.group(1)))

            # Match: Minimum validation bpb: 1.234567
            m = re.search(r"Minimum validation bpb:\s+([\d.]+)", line)
            if m:
                min_val_bpb = float(m.group(1))

        # Use minimum validation bpb as the primary metric
        if min_val_bpb is not None:
            metrics[f"val_bpb_{cmd_label}"] = min_val_bpb
        elif val_bpb_values:
            # Fallback: use the last reported val_bpb
            metrics[f"val_bpb_{cmd_label}"] = val_bpb_values[-1]

        if metrics:
            parts = [f"{k}={v:.6f}" for k, v in metrics.items()]
            feedback = f"Final metrics ({cmd_label}): " + ", ".join(parts)

            # Also include val_bpb progression if available
            if val_bpb_values:
                progression = ", ".join(f"{v:.6f}" for v in val_bpb_values)
                feedback += f"\nValidation bpb progression: {progression}"

        return feedback, metrics
