"""Task-specific output parser for llm-sft-loss.

Handles combined train+eval output:
- Training: TRAIN_METRICS lines from LLaMA-Factory SFT trainer
- Evaluation: lm-evaluation-harness table output with acc/acc_norm metrics

Metrics extracted per eval task (hellaswag, arc_challenge, piqa):
    <task>_acc, <task>_acc_norm (where available)
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

# Tasks we expect from lm-eval
_EVAL_TASKS = ["hellaswag", "arc_challenge", "piqa"]


class Parser(OutputParser):
    """Parser for the llm-sft-loss task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        if cmd_label == "metamathqa-sft":
            train_feedback = self._parse_train_metrics(raw_output)
            if train_feedback:
                feedback_parts.append(train_feedback)
        elif cmd_label == "lm-eval":
            eval_feedback, eval_metrics = self._parse_lm_eval(raw_output)
            if eval_feedback:
                feedback_parts.append(eval_feedback)
            metrics.update(eval_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract TRAIN_METRICS lines and return a summary of the last few."""
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS:")]
        if not lines:
            return ""
        return "Training metrics (last steps):\n" + "\n".join(lines[-5:])

    def _parse_lm_eval(self, output: str) -> tuple[str, dict]:
        """Parse lm-evaluation-harness table output for acc/acc_norm metrics.

        The table format is:
        |    Tasks     |Version|Filter|n-shot|  Metric   |   |Value |   |Stderr|
        |--------------|------:|------|-----:|-----------|---|-----:|---|-----:|
        |hellaswag     |      1|none  |     0|acc        |↑  |0.3120|±  |0.0046|
        |              |       |none  |     0|acc_norm   |↑  |0.3980|±  |0.0049|
        """
        metrics: dict = {}
        feedback_lines = []

        # Match table rows: | task | version | filter | n-shot | metric | arrow | value | ± | stderr |
        row_pattern = re.compile(
            r"\|\s*(\w[\w_]*)\s*\|"   # task name
            r"[^|]*\|"                # version
            r"[^|]*\|"                # filter
            r"[^|]*\|"                # n-shot
            r"\s*(acc(?:_norm)?)\s*\|" # metric name (acc or acc_norm)
            r"[^|]*\|"                # arrow
            r"\s*([\d.]+)\s*\|"       # value
        )

        # Also match continuation rows where task name is empty
        # (second metric for same task)
        cont_pattern = re.compile(
            r"\|\s*\|"               # empty task cell
            r"[^|]*\|"              # version
            r"[^|]*\|"              # filter
            r"[^|]*\|"              # n-shot
            r"\s*(acc(?:_norm)?)\s*\|" # metric name
            r"[^|]*\|"              # arrow
            r"\s*([\d.]+)\s*\|"     # value
        )

        current_task = None
        for line in output.splitlines():
            match = row_pattern.search(line)
            if match:
                task_name = match.group(1).strip()
                metric_name = match.group(2).strip()
                value = float(match.group(3))
                current_task = task_name
                metric_key = f"{task_name}_{metric_name}"
                metrics[metric_key] = value
                feedback_lines.append(f"  {task_name}/{metric_name}: {value:.4f}")
                continue

            cont_match = cont_pattern.search(line)
            if cont_match and current_task:
                metric_name = cont_match.group(1).strip()
                value = float(cont_match.group(2))
                metric_key = f"{current_task}_{metric_name}"
                metrics[metric_key] = value
                feedback_lines.append(f"  {current_task}/{metric_name}: {value:.4f}")

        feedback = ""
        if feedback_lines:
            feedback = "Evaluation results:\n" + "\n".join(feedback_lines)

        return feedback, metrics
