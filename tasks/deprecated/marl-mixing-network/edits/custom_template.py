import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ── Custom imports (editable) ────────────────────────────────────────────


# ======================================================================
# EDITABLE — Custom mixing network
# ======================================================================
class CustomMixer(nn.Module):
    """Custom mixing network for cooperative MARL value decomposition.

    Combines individual agent Q-values into a joint Q_tot value.

    Args:
        args: Namespace with attributes:
            - n_agents (int): number of agents
            - state_shape (tuple/int): global state dimension
            - mixing_embed_dim (int): suggested hidden dim (default 32)

    Interface:
        forward(agent_qs, states) -> q_tot
            agent_qs: shape (batch, T, n_agents)
            states:   shape (batch, T, state_dim)
            q_tot:    shape (batch, T, 1)
    """

    def __init__(self, args):
        super(CustomMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        # Default: learnable weighted sum (not state-conditioned)
        self.w = nn.Parameter(th.ones(self.n_agents))
        self.b = nn.Parameter(th.zeros(1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # Weighted sum with non-negative weights (monotonicity)
        w = th.abs(self.w).view(1, self.n_agents, 1).expand(agent_qs.size(0), -1, -1)
        q_tot = th.bmm(agent_qs, w) + self.b
        q_tot = q_tot.view(bs, -1, 1)
        return q_tot
