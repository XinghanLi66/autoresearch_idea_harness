"""Uniform (linear) schedule baseline — evenly spaced time steps."""

_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

_UNIFORM_SCHEDULE = """\
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    return torch.linspace(t_max, t_min, n + 1).to(device)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 310,  # +9 for pre_edit OP 6d NFE instrumentation (was 301)
        "end_line": 320,    # was 311
        "content": _UNIFORM_SCHEDULE,
    },
]
