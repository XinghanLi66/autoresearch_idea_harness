"""Flow Matching baseline -- conditional flow matching (rectified flow).

Linear interpolation path between noise and data.
Network predicts velocity field. Euler ODE solver at inference.

Replaces lines 74-129 of train_custom.py (CONFIG + ImitationObjective class).
"""

_FILE = "CleanDiffuser/train_custom.py"

_REPLACEMENT = """\
CONFIG = {
    "sample_steps": 10,
}


class ImitationObjective:
    \"\"\"Flow Matching (Rectified Flow): conditional flow matching objective.

    Training: sample t ~ U(0,1), form x_t = (1-t)*x_0 + t*noise,
    network predicts velocity v = x_0 - noise.
    Inference: Euler integration from noise (t=1) to data (t=0).
    \"\"\"

    def __init__(self, network: nn.Module, network_ema: nn.Module,
                 act_dim: int, horizon: int, device: str, config: dict):
        self.network = network
        self.network_ema = network_ema
        self.act_dim = act_dim
        self.horizon = horizon
        self.device = device
        self.config = config
        self.sample_steps = config.get("sample_steps", 10)

    def compute_loss(self, network: nn.Module, actions: torch.Tensor,
                     condition: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        t = torch.rand(B, device=self.device)
        t_expand = t.view(B, 1, 1)

        x1 = torch.randn_like(actions)
        x_t = t_expand * x1 + (1 - t_expand) * actions
        target_vel = actions - x1

        pred_vel = network(x_t, t, condition)
        loss = ((pred_vel - target_vel) ** 2).mean()
        return loss

    def sample(self, network_ema: nn.Module, condition: torch.Tensor,
               prior_shape: tuple) -> torch.Tensor:
        B = prior_shape[0]
        x_t = torch.randn(prior_shape, device=self.device)

        dt = 1.0 / self.sample_steps
        for i in reversed(range(self.sample_steps)):
            t_val = (i + 1) / self.sample_steps
            t = torch.full((B,), t_val, device=self.device, dtype=torch.float32)
            vel = network_ema(x_t, t, condition)
            x_t = x_t + dt * vel

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
