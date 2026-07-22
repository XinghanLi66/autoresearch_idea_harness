"""Karras schedule baseline — power-law noise schedule from Karras et al. (2022).

Concentrates steps at higher noise levels using inverse power-law ramp
with rho=7, following the EDM paper.
"""

_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

_KARRAS_SCHEDULE = """\
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    rho = 7.0
    ramp = torch.linspace(0, 1, n + 1)
    min_inv_rho = t_min ** (1 / rho)
    max_inv_rho = t_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas[-1] = t_min  # ensure exact terminal value
    return sigmas.to(device)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 310,  # +9 for pre_edit OP 6d NFE instrumentation (was 301)
        "end_line": 320,    # was 311
        "content": _KARRAS_SCHEDULE,
    },
]
