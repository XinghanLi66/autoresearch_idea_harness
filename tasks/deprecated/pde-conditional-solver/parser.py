"""Task-specific output parser for pde-conditional-solver.
Handles output from Neural-Solver-Library exp_dynamic_conditional:
- Training feedback: TRAIN_METRICS epoch=E train_loss_step=L test_loss_full=T
- Test feedback: rel_err:{value}
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the pde-conditional-solver task."""

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
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training metrics (last epochs):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(r"rel_err:([\d.eE+-]+)", line)
            if match:
                rel_err = float(match.group(1))
                metrics[f"rel_err_{cmd_label}"] = rel_err
                feedback = f"Test results ({cmd_label}):\n  Relative L2 Error: {rel_err:.6f}"

        return feedback, metrics
