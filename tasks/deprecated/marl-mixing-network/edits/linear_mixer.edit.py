"""Linear mixer baseline -- rigorous codebase edit ops.

Replaces the default mixer with a state-agnostic learnable weighted sum.
This is a naive baseline between VDN (fixed sum) and QMIX (state-conditioned
hypernetwork mixer): it learns per-agent weights and a bias, but it does not
condition on the global state.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "epymarl/src/modules/mixers/custom.py"

_LINEAR_MIXER_CLASS = """\
class CustomMixer(nn.Module):
    \"\"\"Linear monotonic mixer with learned per-agent weights.\"\"\"

    def __init__(self, args):
        super(CustomMixer, self).__init__()
        self.n_agents = args.n_agents
        self.w = nn.Parameter(th.ones(self.n_agents))
        self.b = nn.Parameter(th.zeros(1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        w = th.abs(self.w).view(1, self.n_agents, 1).expand(agent_qs.size(0), -1, -1)
        q_tot = th.bmm(agent_qs, w) + self.b
        return q_tot.view(bs, -1, 1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 13,
        "end_line": 49,
        "content": _LINEAR_MIXER_CLASS,
    },
]
