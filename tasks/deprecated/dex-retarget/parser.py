"""Task-specific output parser for dex-retarget.

Evaluation feedback: lines matching
    RETARGET_METRICS mpjpe_mm=X.XXXX smoothness=X.XXXXXX fps=X.X

Leaderboard metrics: <label>_mpjpe_mm (primary, lower is better), <label>_smoothness, <label>_fps.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the dex-retarget (hand retargeting optimizer) task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse retargeting metrics
        retarget_feedback, retarget_metrics = self._parse_retarget_metrics(
            raw_output, cmd_label
        )
        if retarget_feedback:
            feedback_parts.append(retarget_feedback)
        metrics.update(retarget_metrics)

        # Parse per-link errors
        link_feedback = self._parse_link_errors(raw_output)
        if link_feedback:
            feedback_parts.append(link_feedback)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_retarget_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract RETARGET_METRICS lines, prefixed by cmd_label."""
        metrics: dict = {}
        feedback = ""
        prefix = cmd_label.replace("-", "_")

        for line in output.splitlines():
            match = re.search(
                r"RETARGET_METRICS\s+mpjpe_mm=([\d.]+)\s+smoothness=([\d.]+)\s+fps=([\d.]+)",
                line,
            )
            if match:
                metrics[f"{prefix}_mpjpe_mm"] = float(match.group(1))
                metrics[f"{prefix}_smoothness"] = float(match.group(2))
                metrics[f"{prefix}_fps"] = float(match.group(3))
                feedback = (
                    f"[{cmd_label}] Retargeting results:\n"
                    f"  MPJPE: {metrics[f'{prefix}_mpjpe_mm']:.4f} mm\n"
                    f"  Smoothness: {metrics[f'{prefix}_smoothness']:.6f}\n"
                    f"  FPS: {metrics[f'{prefix}_fps']:.1f}"
                )

        return feedback, metrics

    def _parse_link_errors(self, output: str) -> str:
        """Extract per-link error lines for detailed feedback."""
        lines = []
        capture = False
        for line in output.splitlines():
            if "Per-link mean errors" in line:
                capture = True
                lines.append(line.strip())
                continue
            if capture:
                stripped = line.strip()
                if stripped and stripped[0].isalpha():
                    lines.append("  " + stripped)
                else:
                    capture = False

        return "\n".join(lines) if lines else ""
