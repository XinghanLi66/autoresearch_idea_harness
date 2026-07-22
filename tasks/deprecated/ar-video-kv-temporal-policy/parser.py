"""Parser for ar-video-kv-temporal-policy."""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Extracts replay summaries and final benchmark metrics."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        trace_feedback = self._parse_trace_metrics(raw_output)
        if trace_feedback:
            feedback_parts.append(trace_feedback)

        final_feedback, final_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if final_feedback:
            feedback_parts.append(final_feedback)
        metrics.update(final_metrics)

        host_feedback, host_metrics = self._parse_host_metrics(raw_output, cmd_label)
        if host_feedback:
            feedback_parts.append(host_feedback)
        metrics.update(host_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_trace_metrics(self, output: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip().startswith("TRACE_METRICS:")]
        if not lines:
            return ""
        return "Trace summary:\n" + "\n".join(lines[-3:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            if "TEST_METRICS:" not in line:
                continue
            pairs = re.findall(r"(\w+)=([-+]?[\d.]+(?:e[+-]?\d+)?)", line, re.IGNORECASE)
            for key, raw in pairs:
                metrics[f"{key}_{cmd_label}"] = float(raw)
            if metrics:
                parts = [f"{key}={value:.4f}" for key, value in sorted(metrics.items())]
                feedback = f"Final metrics ({cmd_label}): " + ", ".join(parts)

        return feedback, metrics

    def _parse_host_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        lines = [line.strip() for line in output.splitlines() if line.strip().startswith("HOST_METRICS:")]
        if not lines:
            return "", {}

        payload = json.loads(lines[-1].split(":", 1)[1].strip())
        metrics = {
            f"diag_{key}_{cmd_label}": float(value)
            for key, value in sorted(payload.items())
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return "Host validation:\n" + "\n".join(lines[-3:]), metrics
