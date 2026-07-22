"""GeoLDM baseline — Geometric Latent Diffusion Model.
Replaces editable section with equivariant VAE + latent diffusion.
Reference: Xu et al., "Geometric Latent Diffusion Models for 3D Molecule Generation" (ICML 2023)
"""

_FILE = "Uni-3DAR/custom_mol3d.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — GeoLDM: Geometric Latent Diffusion Model
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
    def __init__(self, in_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, h, x, mask):
        h = self.embed(h)
        for layer in self.layers:
            h, x = layer(h, x, mask)
        return self.out(h), x


class EquivariantEncoder(nn.Module):
    \"\"\"Equivariant VAE encoder.\"\"\"
    def __init__(self, hidden_dim, latent_dim, num_layers=3):
        super().__init__()
        self.atom_embed = nn.Embedding(MAX_ATOM_TYPE + 1, hidden_dim, padding_idx=0)
        self.egnn = EGNN(hidden_dim, hidden_dim, num_layers)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, atom_types, pos, mask):
        h = self.atom_embed(atom_types)
        h_out, x_out = self.egnn(h, pos, mask)
        mu = self.mu_head(h_out)
        logvar = self.logvar_head(h_out)
        # Use EGNN equivariant coordinate output (not an invariant linear head)
        return mu, logvar, x_out


class EquivariantDecoder(nn.Module):
    \"\"\"Equivariant VAE decoder.\"\"\"
    def __init__(self, hidden_dim, latent_dim, num_layers=3):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.egnn = EGNN(hidden_dim, hidden_dim, num_layers)
        self.type_head = nn.Linear(hidden_dim, MAX_ATOM_TYPE + 1)
        self.coord_head = nn.Linear(hidden_dim, 3)

    def forward(self, z, x, mask):
        h = self.latent_proj(z)
        h_out, x_out = self.egnn(h, x, mask)
        type_logits = self.type_head(h_out)
        coord_delta = self.coord_head(h_out)
        return type_logits, x_out + coord_delta * mask.unsqueeze(-1)


class LatentDiffusion(nn.Module):
    \"\"\"Diffusion model in latent space.\"\"\"
    def __init__(self, hidden_dim, latent_dim, num_layers=4, num_timesteps=1000):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        in_dim = hidden_dim * 2
        self.egnn = EGNN(in_dim, hidden_dim, num_layers)
        self.out_proj = nn.Linear(in_dim * 2, latent_dim)
        self.num_timesteps = num_timesteps

        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def forward(self, z, x, mask, t):
        B, N = mask.shape
        h_z = self.latent_proj(z)
        h_t = self.time_embed(t.float().unsqueeze(-1) / self.num_timesteps).unsqueeze(1).expand(-1, N, -1)
        h_input = torch.cat([h_z, h_t], dim=-1)
        h_out, x_out = self.egnn(h_input, x, mask)
        h_cat = torch.cat([h_out, h_input], dim=-1)
        eps_z = self.out_proj(h_cat)
        # Use equivariant coordinate output: eps_x = x_out - x is equivariant
        eps_x = (x_out - x) * mask.unsqueeze(-1)
        return eps_z, eps_x


class StructureGenerator(nn.Module):
    \"\"\"GeoLDM: Geometric Latent Diffusion Model.

    Two-stage: (1) equivariant VAE maps structures to latent space,
    (2) diffusion model generates in latent space.
    \"\"\"

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.latent_dim = self.hidden_dim // 2
        self.num_layers = getattr(config, 'num_layers', 6)
        self.num_timesteps = getattr(config, 'num_timesteps', 1000)
        self.max_atom_type = MAX_ATOM_TYPE
        self.kl_weight = 0.1

        enc_layers = max(self.num_layers // 2, 2)
        self.encoder = EquivariantEncoder(self.hidden_dim, self.latent_dim, enc_layers)
        self.decoder = EquivariantDecoder(self.hidden_dim, self.latent_dim, enc_layers)
        self.diffusion = LatentDiffusion(
            self.hidden_dim, self.latent_dim,
            num_layers=self.num_layers, num_timesteps=self.num_timesteps,
        )

    def _center_positions(self, pos, mask):
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        com = (pos * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)
        return (pos - com) * mask.unsqueeze(-1)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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

        # VAE encode
        mu, logvar, x_enc = self.encoder(atom_types, pos, mask)
        z = self._reparameterize(mu, logvar) * mask.unsqueeze(-1)
        # Use EGNN equivariant coordinate output (keeps gradient for encoder)
        x_init = x_enc * mask.unsqueeze(-1)
        type_logits, pos_recon = self.decoder(z, x_init, mask)

        recon_type = F.cross_entropy(
            type_logits.reshape(-1, self.max_atom_type + 1),
            atom_types.reshape(-1), ignore_index=0,
        )
        recon_coord = F.mse_loss(pos_recon * mask.unsqueeze(-1), pos * mask.unsqueeze(-1))
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
        kl = (kl * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1)

        vae_loss = recon_coord + 1.0 * recon_type + self.kl_weight * kl

        # Diffusion loss: joint on z AND x
        # Detach both targets so diffusion loss doesn't backprop into encoder
        z_clean = mu.detach()
        x_clean = x_enc.detach() * mask.unsqueeze(-1)
        t = torch.randint(0, self.num_timesteps, (B,), device=pos.device)

        sqrt_a = self.diffusion.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_1ma = self.diffusion.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)

        eps_z = torch.randn_like(z_clean) * mask.unsqueeze(-1)
        z_noisy = sqrt_a * z_clean + sqrt_1ma * eps_z
        eps_x = torch.randn_like(x_clean) * mask.unsqueeze(-1)
        if batch['dataset_type'] == 'molecule':
            eps_x = self._center_positions(eps_x, mask)
        x_noisy = sqrt_a * x_clean + sqrt_1ma * eps_x

        eps_z_pred, eps_x_pred = self.diffusion(z_noisy, x_noisy, mask, t)
        diff_z = F.mse_loss(eps_z_pred * mask.unsqueeze(-1), eps_z * mask.unsqueeze(-1))
        diff_x = F.mse_loss(eps_x_pred * mask.unsqueeze(-1), eps_x * mask.unsqueeze(-1))

        return vae_loss + diff_z + diff_x

    @torch.no_grad()
    def sample(self, n_samples, atom_counts, device, **kwargs):
        dataset_type = kwargs.get('dataset_type', 'molecule')
        max_atoms = int(atom_counts.max().item())
        arange = torch.arange(max_atoms, device=device).unsqueeze(0)
        mask = (arange < atom_counts.unsqueeze(1)).float()

        z = torch.randn(n_samples, max_atoms, self.latent_dim, device=device) * mask.unsqueeze(-1)
        x = torch.randn(n_samples, max_atoms, 3, device=device) * mask.unsqueeze(-1)
        if dataset_type == 'molecule':
            x = self._center_positions(x, mask)

        # Joint denoise z and x
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            alpha_t = self.diffusion.alphas[t_idx]
            alpha_bar_t = self.diffusion.alphas_cumprod[t_idx]
            beta_t = self.diffusion.betas[t_idx]

            eps_z_pred, eps_x_pred = self.diffusion(z, x, mask, t)

            coeff1 = 1.0 / math.sqrt(alpha_t.item())
            coeff2 = beta_t.item() / math.sqrt(1 - alpha_bar_t.item())
            z = coeff1 * (z - coeff2 * eps_z_pred)
            x = coeff1 * (x - coeff2 * eps_x_pred)

            if t_idx > 0:
                sigma = math.sqrt(beta_t.item())
                z = z + sigma * torch.randn_like(z) * mask.unsqueeze(-1)
                noise_x = torch.randn_like(x) * mask.unsqueeze(-1)
                if dataset_type == 'molecule':
                    noise_x = self._center_positions(noise_x, mask)
                x = x + sigma * noise_x

            z = z * mask.unsqueeze(-1)
            if dataset_type == 'molecule':
                x = self._center_positions(x, mask)
            else:
                x = x * mask.unsqueeze(-1)

        # Decode from denoised z + x
        type_logits, pos_recon = self.decoder(z, x, mask)
        type_logits[:, :, 0] = -float('inf')  # exclude padding index
        atom_types = type_logits.argmax(dim=-1) * mask.long()
        pos = pos_recon * mask.unsqueeze(-1)

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
