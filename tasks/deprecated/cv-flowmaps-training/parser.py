"""Task-specific output parser for cv-flowmaps-training.

Metrics use ``fid_<label>`` / ``best_fid_<label>`` (e.g. ``fid_train_small``) so the
three ``train_*`` benchmarks do not overwrite each other in one ``test()`` run.
"""

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


def _norm_label(cmd_label: str) -> str:
    return cmd_label.replace("-", "_")


class Parser(OutputParser):
    """Parser for flow-maps CIFAR-10 FID (TEST_METRICS format)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts: list[str] = []
        metrics: dict = {}
        suffix = _norm_label(cmd_label)
        fid_key = f"fid_{suffix}"
        best_key = f"best_fid_{suffix}"

        for line in raw_output.splitlines():
            if "TEST_METRICS:" not in line:
                continue

            fid_m = re.search(r"fid=([\d.]+|nan|inf|-inf)", line, re.IGNORECASE)
            best_m = re.search(r"best_fid=([\d.]+|nan|inf|-inf)", line, re.IGNORECASE)

            if fid_m:
                raw = fid_m.group(1).lower()
                if raw == "nan" or raw in ("inf", "-inf"):
                    fid = float("nan")
                else:
                    fid = float(raw)
                metrics[fid_key] = fid

                if best_m:
                    braw = best_m.group(1).lower()
                    if braw == "nan" or braw in ("inf", "-inf"):
                        best_fid = float("nan")
                    else:
                        best_fid = float(braw)
                else:
                    best_fid = fid
                metrics[best_key] = best_fid

                bf_s = f"{best_fid:.2f}" if best_fid == best_fid else "nan"
                if fid == fid:  # not NaN
                    fb = f"FID ({cmd_label}): {fid:.2f}, Best FID: {bf_s}"
                else:
                    fb = f"FID ({cmd_label}): nan, Best FID: {bf_s}"
                feedback_parts.append(fb)

        if feedback_parts:
            feedback = "Training results:\n" + "\n".join(feedback_parts)
        else:
            feedback = raw_output[-4000:]

        def _is_nan_val(v) -> bool:
            try:
                return isinstance(v, float) and math.isnan(v)
            except (TypeError, ValueError):
                try:
                    return bool(math.isnan(float(v)))
                except (TypeError, ValueError):
                    return False

        if metrics and _is_nan_val(metrics.get(fid_key)):
            warn_lines = [
                ln
                for ln in raw_output.splitlines()
                if "Warning:" in ln
                or "MLS-Bench final FID failed" in ln
                or "FID computation failed" in ln
            ]
            if warn_lines:
                tail = warn_lines[-40:]
                feedback += (
                    "\n\n--- FID debug (from container stderr/stdout) ---\n"
                    + "\n".join(tail)
                )

        return ParseResult(feedback=feedback, metrics=metrics)
