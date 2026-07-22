"""EDM baseline — Equivariant Diffusion Models.
Replaces editable section with EGNN-based equivariant DDPM.
Reference: Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D" (ICML 2022)
"""

_FILE = "Uni-3DAR/custom_mol3d.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — EDM: Equivariant Diffusion Model
# =====================================================================

class GaussianFourierProjection(nn.Module):
    \"\"\"Gaussian Fourier time embedding.\"\"\"
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t.unsqueeze(-1) * self.W * 2 * math.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class EGNNLayer(nn.Module):
    \"\"\"E(n) Equivariant Graph Neural Network layer with attention.\"\"\"

    def __init__(self, hidden_dim, edge_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        edge_input = 2 * hidden_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
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

        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        eye_mask = 1.0 - torch.eye(N, device=h.device).unsqueeze(0)
        pair_mask = pair_mask * eye_mask
        m_ij = m_ij * pair_mask.unsqueeze(-1)

        # Normalize by neighbor count
        num_neighbors = pair_mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B, N, 1)

        coord_weights = torch.tanh(self.coord_mlp(m_ij))
        x_update = (dx * coord_weights).sum(dim=2)
        x = x + (x_update / num_neighbors) * mask.unsqueeze(-1)

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
    \"\"\"EDM: Equivariant Diffusion Model.

    Uses EGNN to predict noise (epsilon) in an equivariant denoising process.
    Coordinates: E(3)-equivariant diffusion with zero center-of-mass subspace.
    Atom types: Continuous diffusion on one-hot embeddings with argmax decoding.
    \"\"\"

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_layers = getattr(config, 'num_layers', 6)
        self.num_timesteps = getattr(config, 'num_timesteps', 1000)
        self.max_atom_type = MAX_ATOM_TYPE

        # Embeddings
        self.atom_type_continuous_dim = MAX_ATOM_TYPE + 1
        self.type_embed = nn.Linear(self.atom_type_continuous_dim, self.hidden_dim)
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        # Score network
        in_dim = self.hidden_dim * 2  # type + time
        self.score_net = EGNN(in_dim, self.hidden_dim, self.num_layers)

        # Output heads
        self.eps_head_coord = nn.Linear(in_dim * 2, 3)
        self.eps_head_type = nn.Linear(in_dim * 2, self.atom_type_continuous_dim)

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def _center_positions(self, pos, mask):
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        com = (pos * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)
        return (pos - com) * mask.unsqueeze(-1)

    def _one_hot(self, types, num_classes):
        return F.one_hot(types.clamp(0, num_classes - 1), num_classes).float()

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

        # One-hot atom types for continuous diffusion
        type_onehot = self._one_hot(atom_types, self.atom_type_continuous_dim)

        t = torch.randint(0, self.num_timesteps, (B,), device=pos.device)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)

        # Noise coordinates
        eps_pos = torch.randn_like(pos) * mask.unsqueeze(-1)
        if batch['dataset_type'] == 'molecule':
            eps_pos = self._center_positions(eps_pos, mask)
        pos_noisy = sqrt_alpha * pos + sqrt_one_minus_alpha * eps_pos

        # Noise atom types (continuous)
        eps_type = torch.randn_like(type_onehot) * mask.unsqueeze(-1)
        type_noisy = sqrt_alpha * type_onehot + sqrt_one_minus_alpha * eps_type

        # Network input
        h_type = self.type_embed(type_noisy)
        h_time = self.time_embed(t.float() / self.num_timesteps).unsqueeze(1).expand(-1, N, -1)
        h_input = torch.cat([h_type, h_time], dim=-1)

        h_out, x_out = self.score_net(h_input, pos_noisy, mask)

        # x0 parameterization: EGNN outputs denoised coordinates directly
        x0_pred = x_out * mask.unsqueeze(-1)
        h_cat = torch.cat([h_out, h_input], dim=-1)
        eps_type_pred = self.eps_head_type(h_cat)

        # Losses: predict clean positions (x0), not noise
        coord_loss = F.mse_loss(x0_pred, pos * mask.unsqueeze(-1))
        type_loss = F.mse_loss(eps_type_pred * mask.unsqueeze(-1), eps_type * mask.unsqueeze(-1))

        return coord_loss + 0.25 * type_loss

    @torch.no_grad()
    def sample(self, n_samples, atom_counts, device, **kwargs):
        dataset_type = kwargs.get('dataset_type', 'molecule')
        max_atoms = int(atom_counts.max().item())
        arange = torch.arange(max_atoms, device=device).unsqueeze(0)
        mask = (arange < atom_counts.unsqueeze(1)).float()

        # Start from noise
        pos = torch.randn(n_samples, max_atoms, 3, device=device) * mask.unsqueeze(-1)
        type_cont = torch.randn(n_samples, max_atoms, self.atom_type_continuous_dim,
                                device=device) * mask.unsqueeze(-1)

        if dataset_type == 'molecule':
            pos = self._center_positions(pos, mask)

        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]

            h_type = self.type_embed(type_cont)
            h_time = self.time_embed(t.float() / self.num_timesteps).unsqueeze(1).expand(-1, max_atoms, -1)
            h_input = torch.cat([h_type, h_time], dim=-1)

            h_out, x_out = self.score_net(h_input, pos, mask)
            x0_pred = x_out * mask.unsqueeze(-1)
            h_cat = torch.cat([h_out, h_input], dim=-1)
            eps_type_pred = self.eps_head_type(h_cat)

            # DDPM reverse step (x0 parameterization for coords)
            # Convert x0 prediction to noise: eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)
            sqrt_ab = math.sqrt(alpha_bar_t.item())
            sqrt_1mab = math.sqrt(1 - alpha_bar_t.item())
            eps_pos_pred = (pos - sqrt_ab * x0_pred) / max(sqrt_1mab, 1e-8)

            coeff1 = 1.0 / math.sqrt(alpha_t.item())
            coeff2 = beta_t.item() / sqrt_1mab

            pos = coeff1 * (pos - coeff2 * eps_pos_pred)
            # Type still uses epsilon parameterization
            type_cont = coeff1 * (type_cont - coeff2 * eps_type_pred)

            if t_idx > 0:
                noise_pos = torch.randn_like(pos) * mask.unsqueeze(-1)
                noise_type = torch.randn_like(type_cont) * mask.unsqueeze(-1)
                if dataset_type == 'molecule':
                    noise_pos = self._center_positions(noise_pos, mask)
                sigma = math.sqrt(beta_t.item())
                pos = pos + sigma * noise_pos
                type_cont = type_cont + sigma * noise_type

            if dataset_type == 'molecule':
                pos = self._center_positions(pos, mask)
            else:
                pos = pos * mask.unsqueeze(-1)
            type_cont = type_cont * mask.unsqueeze(-1)

        # Decode atom types (exclude index 0 = padding)
        type_cont[:, :, 0] = -float('inf')
        atom_types = type_cont.argmax(dim=-1) * mask.long()

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
