"""MSE Regression baseline -- direct behavioral cloning.

The simplest baseline: network directly predicts clean actions from condition.
Timestep is set to 0 (unused). Loss = MSE between predicted and true actions.
Single-step inference: just call the network.

Replaces lines 74-129 of train_custom.py (CONFIG + ImitationObjective class).
"""

_FILE = "CleanDiffuser/train_custom.py"

_REPLACEMENT = """\
CONFIG = {}


class ImitationObjective:
    \"\"\"MSE Regression: direct behavioral cloning.

    The network predicts actions directly. Timestep is fixed to 0.
    Loss = MSE(predicted_actions, true_actions).
    Inference: single forward pass.
    \"\"\"

    def __init__(self, network: nn.Module, network_ema: nn.Module,
                 act_dim: int, horizon: int, device: str, config: dict):
        self.network = network
        self.network_ema = network_ema
        self.act_dim = act_dim
        self.horizon = horizon
        self.device = device
        self.config = config

    def compute_loss(self, network: nn.Module, actions: torch.Tensor,
                     condition: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        t = torch.zeros(B, device=self.device, dtype=torch.long)
        predicted = network(actions * 0.0, t, condition)
        loss = ((predicted - actions) ** 2).mean()
        return loss

    def sample(self, network_ema: nn.Module, condition: torch.Tensor,
               prior_shape: tuple) -> torch.Tensor:
        B = prior_shape[0]
        t = torch.zeros(B, device=self.device, dtype=torch.long)
        dummy_input = torch.zeros(prior_shape, device=self.device)
        actions = network_ema(dummy_input, t, condition)
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
