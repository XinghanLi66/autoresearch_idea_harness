"""BCQ baseline for the rl-offline-discrete task.

Matches d3rlpy DiscreteBCQ reproduction (reproductions/offline/discrete_bcq.py):
  - batch_size=32, lr=5e-5, optim_eps=3.125e-4
  - action_flexibility=0.3, beta=0.01
  - Hard target update every 2000 steps
  - DoubleDQN TD backup with BCQ action filtering
  - QR-QFunction with 200 quantiles (quantile Huber loss)
  - Shared encoder (Q-head + BC-head)
  - Reward clipping [-1, 1] handled by FIXED buffer
  - Observation scaling /255.0 handled by FIXED buffer
"""

import math

_FILE = "d3rlpy/atari_offline/custom_atari.py"

_BCQ_IMPL = """\
class QNetwork(nn.Module):
    \"\"\"Dual-head QR-DQN + BC network for DiscreteBCQ with shared encoder.

    - q_head: quantile regression Q-values (action_dim * n_quantiles)
    - bc_head: behavioral cloning logits (action_dim)
    forward() returns mean Q-values (compatible with FIXED evaluate).
    \"\"\"
    def __init__(self, observation_shape, action_dim, feature_dim=512, n_quantiles=200):
        super().__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.encoder = NatureDQNEncoder(observation_shape[0], feature_dim)
        self.q_head = nn.Linear(feature_dim, action_dim * n_quantiles)
        self.bc_head = nn.Linear(feature_dim, action_dim)

    def forward(self, obs):
        \"\"\"Returns mean Q-values (B, action_dim) — compatible with FIXED evaluate.\"\"\"
        features = self.encoder(obs)
        quantiles = self.q_head(features).view(-1, self.action_dim, self.n_quantiles)
        return quantiles.mean(dim=-1)

    def forward_quantiles(self, obs):
        \"\"\"Returns (quantiles (B, A, N), bc_logits (B, A)).\"\"\"
        features = self.encoder(obs)
        quantiles = self.q_head(features).view(-1, self.action_dim, self.n_quantiles)
        bc_logits = self.bc_head(features)
        return quantiles, bc_logits


class OfflineAlgorithm:
    \"\"\"DiscreteBCQ with QR-DQN: batch-constrained Q-learning + quantile regression.\"\"\"
    def __init__(self, observation_shape, action_dim, config, device, buffer):
        import math as _math
        from copy import deepcopy
        self.device = device
        self.config = config
        self.buffer = buffer
        self.action_dim = action_dim
        self.action_flexibility = 0.3
        self.log_action_flexibility = _math.log(0.3)
        self.beta = 0.01
        self.n_quantiles = 200
        self.target_update_interval = 2000
        self._step = 0
        # Override config to match d3rlpy DiscreteBCQ reproduction
        config.batch_size = 32
        config.learning_rate = 5e-5
        config.optim_eps = 1e-2 / 32  # 3.125e-4
        config.max_timesteps = 50_000_000 // 4  # 12.5M, matching d3rlpy
        config.eval_freq = 125_000  # matching d3rlpy n_steps_per_epoch
        self.model = QNetwork(observation_shape, action_dim, n_quantiles=self.n_quantiles).to(device)
        self.target_model = deepcopy(self.model)
        # Quantile midpoints: tau_i = (i + 0.5) / N
        taus = torch.arange(self.n_quantiles, dtype=torch.float32, device=device)
        self._taus = ((taus + 0.5) / self.n_quantiles).view(1, 1, -1)

    def parameters(self):
        return self.model.parameters()

    def _quantile_huber_loss(self, current, target):
        \"\"\"QR-DQN quantile Huber loss (matches d3rlpy _quantile_huber_loss).\"\"\"
        delta = target.unsqueeze(1) - current.unsqueeze(2)  # (B, N, N)
        huber = torch.where(delta.abs() <= 1.0, 0.5 * delta ** 2, delta.abs() - 0.5)
        weight = (self._taus - (delta < 0).float()).abs()
        return (weight * huber).sum(dim=2).mean()

    def _select_action_bcq(self, q_mean, bc_logits):
        \"\"\"BCQ action filtering (matches d3rlpy inner_predict_best_action).\"\"\"
        log_probs = F.log_softmax(bc_logits, dim=-1)
        ratio = log_probs - log_probs.max(dim=-1, keepdim=True).values
        mask = (ratio > self.log_action_flexibility).float()
        normalized_q = q_mean - q_mean.min(dim=-1, keepdim=True).values + 1e-5
        return (normalized_q * mask).argmax(dim=-1)

    def train_step(self, obs, actions, rewards, next_obs, dones):
        # DoubleDQN target with BCQ action selection
        with torch.no_grad():
            next_quant_all, next_bc = self.model.forward_quantiles(next_obs)
            next_q_mean = next_quant_all.mean(dim=-1)  # (B, A)
            next_actions = self._select_action_bcq(next_q_mean, next_bc)
            target_quant_all, _ = self.target_model.forward_quantiles(next_obs)
            next_quant = target_quant_all[torch.arange(len(next_actions)), next_actions]  # (B, N)
            target_quant = rewards.unsqueeze(-1) + self.config.gamma * (1 - dones.unsqueeze(-1)) * next_quant

        quant_all, bc_logits = self.model.forward_quantiles(obs)
        current_quant = quant_all[torch.arange(len(actions)), actions]  # (B, N)

        td_loss = self._quantile_huber_loss(current_quant, target_quant)
        # Imitator loss: NLL + beta * logits^2
        bc_nll = F.cross_entropy(bc_logits, actions)
        bc_reg = self.beta * (bc_logits ** 2).mean()
        bc_loss = bc_nll + bc_reg
        loss = td_loss + bc_loss

        return loss, {
            "loss": loss.item(),
            "td_loss": td_loss.item(),
            "bc_loss": bc_loss.item(),
            "q_mean": current_quant.mean().item(),
        }

    def after_gradient_step(self, optimizer):
        self._step += 1
        if self._step % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, obs):
        quant_all, bc_logits = self.model.forward_quantiles(obs)
        q_mean = quant_all.mean(dim=-1)
        return self._select_action_bcq(q_mean, bc_logits).item()

    def begin_episode(self):
        pass

    def observe(self, reward):
        pass
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 220,
        "end_line": 258,
        "content": _BCQ_IMPL,
    },
]
