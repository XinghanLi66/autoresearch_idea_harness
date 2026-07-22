"""DDPM baseline -- denoising diffusion probabilistic model.

Forward process: add Gaussian noise according to cosine schedule.
Network predicts the noise (epsilon-prediction). Full DDPM reverse chain
for sampling.

Replaces lines 74-129 of train_custom.py (CONFIG + ImitationObjective class).
"""

_FILE = "CleanDiffuser/train_custom.py"

_REPLACEMENT = """\
CONFIG = {
    "diffusion_steps": 100,
    "sample_steps": 100,
    "beta_schedule": "cosine",
}


def _cosine_beta_schedule(T, s=0.008):
    \"\"\"Cosine noise schedule from Nichol & Dhariwal.\"\"\"
    import math
    steps = torch.arange(T + 1, dtype=torch.float64)
    alpha_bar = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


class ImitationObjective:
    \"\"\"DDPM: Denoising Diffusion Probabilistic Model.

    Forward process adds Gaussian noise via cosine schedule.
    Network predicts the added noise (epsilon-prediction).
    Inference uses the full DDPM reverse chain.
    \"\"\"

    def __init__(self, network: nn.Module, network_ema: nn.Module,
                 act_dim: int, horizon: int, device: str, config: dict):
        self.network = network
        self.network_ema = network_ema
        self.act_dim = act_dim
        self.horizon = horizon
        self.device = device
        self.config = config

        T = config.get("diffusion_steps", 100)
        self.T = T

        betas = _cosine_beta_schedule(T).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def compute_loss(self, network: nn.Module, actions: torch.Tensor,
                     condition: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        eps = torch.randn_like(actions)
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1)
        x_t = actions * alpha_bar_t.sqrt() + eps * (1 - alpha_bar_t).sqrt()
        eps_pred = network(x_t, t, condition)
        loss = ((eps_pred - eps) ** 2).mean()
        return loss

    def sample(self, network_ema: nn.Module, condition: torch.Tensor,
               prior_shape: tuple) -> torch.Tensor:
        B = prior_shape[0]
        x_t = torch.randn(prior_shape, device=self.device)

        for t_val in reversed(range(self.T)):
            t = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            alpha_bar_t = self.alpha_bar[t_val]
            alpha_t = self.alphas[t_val]
            beta_t = self.betas[t_val]

            eps_pred = network_ema(x_t, t, condition)

            mean = (1.0 / alpha_t.sqrt()) * (
                x_t - beta_t / (1 - alpha_bar_t).sqrt() * eps_pred)

            if t_val > 0:
                alpha_bar_prev = self.alpha_bar[t_val - 1]
                sigma = (beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt()
                x_t = mean + sigma * torch.randn_like(x_t)
            else:
                x_t = mean

        return x_t
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
