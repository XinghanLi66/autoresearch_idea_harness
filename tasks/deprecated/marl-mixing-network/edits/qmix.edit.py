"""QMIX baseline — rigorous codebase edit ops.

Replaces the default weighted-sum mixer with QMIX hypernetwork mixer.
Uses state-conditioned hypernetworks to generate mixing weights with
absolute value constraint for monotonicity. Two-layer mixing network
with state-dependent bias and V(s) baseline.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "epymarl/src/modules/mixers/custom.py"

# ── 1. Replace CustomMixer class (lines 13-49) ──────────────────────────

_QMIX_CLASS = """\
class CustomMixer(nn.Module):
    \"\"\"QMIX: Monotonic mixing via state-conditioned hypernetworks.\"\"\"

    def __init__(self, args):
        super(CustomMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        hypernet_embed = getattr(args, "hypernet_embed", 64)
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim),
        )
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)
        y = th.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1, 1)
        return q_tot
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 13,
        "end_line": 49,
        "content": _QMIX_CLASS,
    },
]
