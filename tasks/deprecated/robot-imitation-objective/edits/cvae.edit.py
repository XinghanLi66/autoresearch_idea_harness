"""CVAE baseline -- Conditional Variational Autoencoder.

Encoder maps (actions, condition) -> latent z.
Decoder (the backbone ChiUNet1d) maps (z + condition) -> actions.
Loss = reconstruction + KL divergence.
Single-step generation at inference.

Replaces lines 74-129 of train_custom.py (CONFIG + ImitationObjective class).

Encoder + adapter parameters are returned via extra_params() and optimized
jointly with the backbone by the main optimizer.
"""

_FILE = "CleanDiffuser/train_custom.py"

_REPLACEMENT = """\
CONFIG = {
    "latent_dim": 64,
    "kl_weight": 0.01,
}


class _CVAEEncoder(nn.Module):
    \"\"\"Encoder: maps (actions, condition) -> (mu, logvar) of latent z.\"\"\"

    def __init__(self, act_dim, horizon, cond_dim, latent_dim):
        super().__init__()
        input_dim = act_dim * horizon + cond_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, actions_flat, condition):
        x = torch.cat([actions_flat, condition], dim=-1)
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class _ConditionAdapter(nn.Module):
    \"\"\"Project latent z into condition space and add to obs condition.\"\"\"

    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, cond_dim),
        )

    def forward(self, z, condition):
        return condition + self.proj(z)


class ImitationObjective:
    \"\"\"CVAE: Conditional Variational Autoencoder.

    Encoder: (actions, condition) -> latent z with KL regularization.
    Decoder: the backbone ChiUNet1d, with latent z projected and added
    to the observation condition. Timestep fixed to 0.
    \"\"\"

    def __init__(self, network: nn.Module, network_ema: nn.Module,
                 act_dim: int, horizon: int, device: str, config: dict):
        self.network = network
        self.network_ema = network_ema
        self.act_dim = act_dim
        self.horizon = horizon
        self.device = device
        self.config = config

        self.latent_dim = config.get("latent_dim", 64)
        self.kl_weight = config.get("kl_weight", 0.01)

        # Get cond_dim from network's global condition encoder
        cond_dim = network.global_cond_encoder.in_features
        self._encoder = _CVAEEncoder(
            act_dim, horizon, cond_dim, self.latent_dim).to(device)
        self._adapter = _ConditionAdapter(
            self.latent_dim, cond_dim).to(device)

    def compute_loss(self, network: nn.Module, actions: torch.Tensor,
                     condition: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        actions_flat = actions.reshape(B, -1)
        mu, logvar = self._encoder(actions_flat, condition)

        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)

        decoder_cond = self._adapter(z, condition)

        t = torch.zeros(B, device=self.device, dtype=torch.long)
        actions_pred = network(torch.zeros_like(actions), t, decoder_cond)

        recon_loss = ((actions_pred - actions) ** 2).mean()
        kl_loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean()
        total_loss = recon_loss + self.kl_weight * kl_loss

        return total_loss

    def extra_params(self):
        \"\"\"Return encoder + adapter parameters for joint optimization.\"\"\"
        return list(self._encoder.parameters()) + list(self._adapter.parameters())

    def sample(self, network_ema: nn.Module, condition: torch.Tensor,
               prior_shape: tuple) -> torch.Tensor:
        B = prior_shape[0]
        z = torch.randn(B, self.latent_dim, device=self.device)
        decoder_cond = self._adapter(z, condition)

        t = torch.zeros(B, device=self.device, dtype=torch.long)
        actions = network_ema(torch.zeros(prior_shape, device=self.device), t, decoder_cond)
        return actions
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 74,
        "end_line": 129,
        "content": _REPLACEMENT,
    },
]
