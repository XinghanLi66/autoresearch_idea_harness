"""GeoBFN baseline — Geometric Bayesian Flow Networks.
Replaces editable section with EGNN + Bayesian flow approach.
Reference: Song et al., "Unified Generative Modeling of 3D Molecules via Bayesian Flow Networks" (ICLR 2024)
"""

_FILE = "Uni-3DAR/custom_mol3d.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — GeoBFN: Geometric Bayesian Flow Network
# =====================================================================

class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        edge_input = 2 * hidden_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.node_norm = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def forward(self, h, x, mask, edge_attr=None):
        B, N, D = h.shape
        dx = x.unsqueeze(2) - x.unsqueeze(1)
        dist_sq = (dx ** 2).sum(-1, keepdim=True)
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([hi, hj, dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        att = self.att_mlp(m_ij)
        m_ij = m_ij * att
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1) * (1.0 - torch.eye(N, device=h.device).unsqueeze(0))
        m_ij = m_ij * pair_mask.unsqueeze(-1)
        num_neighbors = pair_mask.sum(dim=2, keepdim=True).clamp(min=1)
        coord_weights = torch.tanh(self.coord_mlp(m_ij))
        x = x + ((dx * coord_weights).sum(dim=2) / num_neighbors) * mask.unsqueeze(-1)
        agg = m_ij.sum(dim=2) / num_neighbors
        h = self.node_norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)) * mask.unsqueeze(-1))
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=6):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, h, x, mask):
        h = self.embed(h)
        for layer in self.layers:
            h, x = layer(h, x, mask)
        return self.out(h), x


class StructureGenerator(nn.Module):
    \"\"\"GeoBFN: Geometric Bayesian Flow Network.

    Uses Bayesian flow updates on coordinates and atom types.
    At each step t, maintains Bayesian parameters (mu, rho) for coordinates
    and category probabilities for atom types, then uses the network to
    predict the target from these parameters.
    \"\"\"

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_layers = getattr(config, 'num_layers', 6)
        self.num_timesteps = getattr(config, 'num_timesteps', 1000)
        self.max_atom_type = MAX_ATOM_TYPE
        self.sigma1 = 0.001  # Final noise level

        # Embeddings
        self.type_embed = nn.Linear(MAX_ATOM_TYPE + 1, self.hidden_dim)
        self.coord_param_embed = nn.Linear(6, self.hidden_dim)  # mu(3) + log_rho(3)
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        in_dim = self.hidden_dim * 3
        self.score_net = EGNN(in_dim, self.hidden_dim, self.num_layers)

        self.coord_head = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        self.type_head = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, MAX_ATOM_TYPE + 1),
        )

        # BFN schedule: rho(t) = t^2 / sigma1^2
        ts = torch.linspace(0, 1, self.num_timesteps + 1)
        self.register_buffer('ts', ts)

    def _center_positions(self, pos, mask):
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        com = (pos * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)
        return (pos - com) * mask.unsqueeze(-1)

    def _get_rho(self, t):
        return t ** 2 / (self.sigma1 ** 2)

    def compute_loss(self, batch):
        if batch['dataset_type'] == 'molecule':
            pos = batch['positions']
        else:
            pos = batch['frac_coords']
        atom_types = batch['atom_types']
        mask = batch['mask']
        B, N = atom_types.shape

        if batch['dataset_type'] == 'molecule':
            pos = self._center_positions(pos, mask)

        # Sample random time
        t = torch.rand(B, device=pos.device)

        # Bayesian flow parameters for coordinates
        rho = self._get_rho(t).view(B, 1, 1)
        mu = rho * pos / (1 + rho)
        noise = torch.randn_like(pos) * mask.unsqueeze(-1)
        if batch['dataset_type'] == 'molecule':
            noise = self._center_positions(noise, mask)
        sigma_mu = torch.sqrt(rho / (1 + rho) ** 2)
        mu_noisy = mu + sigma_mu * noise
        log_rho = torch.log(rho.expand(-1, N, 3).clamp(min=1e-8))

        # BFN for atom types
        type_onehot = F.one_hot(atom_types.clamp(0, self.max_atom_type),
                                self.max_atom_type + 1).float()
        type_noise = torch.randn_like(type_onehot) * mask.unsqueeze(-1) * (1 - t.view(B, 1, 1))
        type_param = type_onehot * t.view(B, 1, 1) + type_noise * 0.1

        # Network prediction
        h_coord = self.coord_param_embed(torch.cat([mu_noisy, log_rho], dim=-1))
        h_type = self.type_embed(type_param)
        h_time = self.time_embed(t.unsqueeze(-1)).unsqueeze(1).expand(-1, N, -1)
        h_input = torch.cat([h_coord, h_type, h_time], dim=-1)

        h_out, x_out = self.score_net(h_input, mu_noisy, mask)
        pos_pred = self.coord_head(h_out)
        type_pred = self.type_head(h_out)

        # Losses
        coord_loss = F.mse_loss(pos_pred * mask.unsqueeze(-1), pos * mask.unsqueeze(-1))
        type_loss = F.cross_entropy(
            type_pred.reshape(-1, self.max_atom_type + 1),
            atom_types.reshape(-1), ignore_index=0,
        )

        return coord_loss + 0.5 * type_loss

    @torch.no_grad()
    def sample(self, n_samples, atom_counts, device, **kwargs):
        dataset_type = kwargs.get('dataset_type', 'molecule')
        max_atoms = int(atom_counts.max().item())
        arange = torch.arange(max_atoms, device=device).unsqueeze(0)
        mask = (arange < atom_counts.unsqueeze(1)).float()

        # Initialize Bayesian parameters
        mu = torch.zeros(n_samples, max_atoms, 3, device=device)
        rho = torch.ones(n_samples, max_atoms, 3, device=device) * 1e-6
        type_probs = torch.ones(n_samples, max_atoms, self.max_atom_type + 1,
                                device=device) / (self.max_atom_type + 1)
        # Log-evidence accumulator for discrete BFN type update
        log_theta = torch.zeros(n_samples, max_atoms, self.max_atom_type + 1, device=device)

        for i in range(self.num_timesteps):
            t = self.ts[i].item()
            t_next = self.ts[i + 1].item()
            t_tensor = torch.full((n_samples,), t, device=device)

            log_rho = torch.log(rho.clamp(min=1e-8))
            h_coord = self.coord_param_embed(torch.cat([mu, log_rho], dim=-1))
            h_type = self.type_embed(type_probs)
            h_time = self.time_embed(t_tensor.unsqueeze(-1)).unsqueeze(1).expand(-1, max_atoms, -1)
            h_input = torch.cat([h_coord, h_type, h_time], dim=-1)

            h_out, _ = self.score_net(h_input, mu, mask)
            x_pred = self.coord_head(h_out) * mask.unsqueeze(-1)
            type_logits = self.type_head(h_out)

            # Bayesian update
            rho_next = self._get_rho(torch.tensor(t_next, device=device))
            rho_prev = self._get_rho(torch.tensor(t, device=device))
            delta_rho = (rho_next - rho_prev).clamp(min=0)

            # Sender sigma: clamp to prevent explosion near t=0
            sender_sigma = (1.0 / (delta_rho.clamp(min=1e-8).sqrt())).clamp(max=10.0)
            y = x_pred + torch.randn_like(x_pred) * sender_sigma * mask.unsqueeze(-1)
            if dataset_type == 'molecule':
                y = self._center_positions(y, mask)

            rho_new = rho + delta_rho
            mu = (rho * mu + delta_rho * y) / rho_new.clamp(min=1e-8)
            rho = rho_new

            # BFN discrete update: accumulate evidence in log space
            beta_t = (t_next ** 2 - t ** 2) / max(self.sigma1 ** 2, 1e-8)
            log_theta = log_theta + beta_t * F.log_softmax(type_logits, dim=-1) * mask.unsqueeze(-1)
            type_probs = F.softmax(log_theta, dim=-1) * mask.unsqueeze(-1)
            mu = mu * mask.unsqueeze(-1)

        # Final output
        pos = mu
        type_probs[:, :, 0] = -float('inf')  # exclude padding index
        atom_types = type_probs.argmax(dim=-1) * mask.long()

        if dataset_type == 'crystal':
            pos = pos % 1.0

        result = {'atom_types': atom_types}
        if dataset_type == 'molecule':
            result['positions'] = pos
        else:
            result['frac_coords'] = pos
            lattice = torch.eye(3, device=device).unsqueeze(0).expand(n_samples, -1, -1) * 5.0
            result['lattice'] = lattice

        return result

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 42,
        "end_line": 328,
        "content": _CONTENT,
    },
]
