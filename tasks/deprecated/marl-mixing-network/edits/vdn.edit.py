"""VDN baseline — rigorous codebase edit ops.

Replaces the default weighted-sum mixer with VDN (pure summation).
VDN simply sums individual agent Q-values with no learned parameters.
Q_tot = sum(Q_i). No state conditioning.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "epymarl/src/modules/mixers/custom.py"

# ── 1. Replace CustomMixer class (lines 13-49) ──────────────────────────

_VDN_CLASS = """\
class CustomMixer(nn.Module):
    \"\"\"VDN: Value Decomposition Network — simple sum of agent Q-values.\"\"\"

    def __init__(self, args):
        super(CustomMixer, self).__init__()

    def forward(self, agent_qs, states):
        return th.sum(agent_qs, dim=2, keepdim=True)
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 13,
        "end_line": 49,
        "content": _VDN_CLASS,
    },
]
