"""Parser for the shared-train HPT controller task."""

import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


class Parser(OutputParser):
    """Extract the final validation score from UPT console logs."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        safe_label = _normalize(cmd_label)
        initial = self._extract_dict_metric(raw_output, "Initial validation metrics", safe_label)
        final = self._extract_dict_metric(raw_output, "Final validation metrics", safe_label)
        if final is None:
            final = self._extract_step_metric(raw_output, safe_label)

        feedback_parts = []
        step_lines = [line.strip() for line in raw_output.splitlines() if line.strip().startswith("step:")]
        if step_lines:
            feedback_parts.append("Recent training lines:\n" + "\n".join(step_lines[-3:]))

        metrics = {}
        if final is not None:
            metric_key = f"test_score_{safe_label}"
            metrics[metric_key] = final
            feedback_parts.append(f"Scoring rule ({cmd_label}): final_validation={final:.6f}")
            if initial is not None:
                delta = final - initial
                feedback_parts.append(
                    f"Selected metric ({cmd_label}): {metric_key}={final:.6f} (delta={delta:+.6f})"
                )
            else:
                feedback_parts.append(f"Selected metric ({cmd_label}): {metric_key}={final:.6f}")

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _extract_dict_metric(self, output: str, prefix: str, safe_label: str) -> float | None:
        for line in reversed(output.splitlines()):
            if prefix not in line:
                continue
            try:
                _, payload = line.split(":", 1)
                payload = payload.strip()
                if payload.startswith('"') and payload.endswith('"'):
                    payload = payload[1:-1]
                payload = re.sub(r"np\.float64\(([^)]+)\)", r"\1", payload)
                metric_dict = ast.literal_eval(payload)
            except Exception:
                continue
            for key, value in metric_dict.items():
                if key.startswith("val/test_score/") and _normalize(key.split("/")[-1]) == safe_label:
                    return float(value)
        return None

    def _extract_step_metric(self, output: str, safe_label: str) -> float | None:
        pattern = re.compile(r"val/test_score/([A-Za-z0-9_\-]+):([0-9]+(?:\.[0-9]+)?)")
        for line in reversed(output.splitlines()):
            matches = pattern.findall(line)
            for source, value in matches:
                if _normalize(source) == safe_label:
                    return float(value)
        return None
