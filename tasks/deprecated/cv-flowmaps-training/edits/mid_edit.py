"""Mid-edit operations for cv-flowmaps-training.

Applied to the flow-maps workspace after pre_edit, before the agent starts.
Replaces the editable loss region with a scaffold template so rigorous_codebase
starts from a non-paper baseline.

Creates ``scripts/bench_env.sh`` (default slurm_id=0 / LSD) so each baseline can
replace it with the PSD variants without changing the train_* scripts.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "losses_template.py"
_LOSSES_TEMPLATE = _TEMPLATE_PATH.read_text()

_BENCH_ENV_DEFAULT = """# MLS-Bench: cifar10_bench slurm_id (0=LSD, 1=PSD uniform, 2=PSD midpoint)
export FLOWMAPS_BENCH_SLURM_ID=0
"""

OPS = [
    {
        "op": "replace",
        "file": "flow-maps/py/common/losses.py",
        "start_line": 36,
        "end_line": 300,
        "content": _LOSSES_TEMPLATE,
    },
    {
        "op": "create",
        "file": "scripts/bench_env.sh",
        "content": _BENCH_ENV_DEFAULT,
    },
]
