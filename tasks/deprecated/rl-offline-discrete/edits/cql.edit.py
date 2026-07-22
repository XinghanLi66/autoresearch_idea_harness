"""CQL baseline for the rl-offline-discrete task.

Matches d3rlpy DiscreteCQL reproduction (reproductions/offline/discrete_cql.py):
  - batch_size=32, lr=5e-5, optim_eps=3.125e-4
  - alpha=4.0 for conservative loss
  - Hard target update every 2000 steps
  - DoubleDQN TD backup (action from online, Q from target)
  - QR-QFunction with 200 quantiles (quantile Huber loss)
  - Conservative loss: logsumexp(mean_Q) - mean_Q(s, a_data)
  - Reward clipping [-1, 1] handled by FIXED buffer
  - Observation scaling /255.0 handled by FIXED buffer
"""

_FILE = "d3rlpy/atari_offline/custom_atari.py"

_CQL_IMPL = """\
class QNetwork(nn.Module):
    \"\"\"Quantile Regression Q-network (QR-DQN) for DiscreteCQL.
    Uses NatureDQNEncoder with n_quantiles output per action.
    forward() returns mean Q-values (compatible with FIXED evaluate).
    \"\"\"
    def __init__(self, observation_shape, action_dim, feature_dim=512, n_quantiles=200):
        super().__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.encoder = NatureDQNEncoder(observation_shape[0], feature_dim)
        self.head = nn.Linear(feature_dim, action_dim * n_quantiles)

    def forward(self, obs):
        \"\"\"Returns mean Q-values (B, action_dim) — compatible with FIXED evaluate.\"\"\"
        return self.forward_quantiles(obs).mean(dim=-1)

    def forward_quantiles(self, obs):
        \"\"\"Returns full quantile distribution (B, action_dim, n_quantiles).\"\"\"
        features = self.encoder(obs)
        return self.head(features).view(-1, self.action_dim, self.n_quantiles)


class OfflineAlgorithm:
    \"\"\"DiscreteCQL with QR-DQN: conservative Q-learning + quantile regression.\"\"\"
    def __init__(self, observation_shape, action_dim, config, device, buffer):
        from copy import deepcopy
        self.device = device
        self.config = config
        self.buffer = buffer
        self.action_dim = action_dim
        self.alpha = 4.0
        self.n_quantiles = 200
        self.target_update_interval = 2000
        self._step = 0
        # Override config to match d3rlpy DiscreteCQL reproduction
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
        \"\"\"QR-DQN quantile Huber loss (matches d3rlpy _quantile_huber_loss).
        current: (B, N) — current quantile predictions for chosen action
        target:  (B, N) — target quantile values
        \"\"\"
        delta = target.unsqueeze(1) - current.unsqueeze(2)  # (B, N, N)
        huber = torch.where(delta.abs() <= 1.0, 0.5 * delta ** 2, delta.abs() - 0.5)
        weight = (self._taus - (delta < 0).float()).abs()
        return (weight * huber).sum(dim=2).mean()

    def train_step(self, obs, actions, rewards, next_obs, dones):
        # DoubleDQN target: online selects action, target provides quantiles
        with torch.no_grad():
            next_q_mean = self.model(next_obs)  # (B, A) mean Q-values
            best_actions = next_q_mean.argmax(dim=-1)  # (B,)
            next_quant_all = self.target_model.forward_quantiles(next_obs)  # (B, A, N)
            next_quant = next_quant_all[torch.arange(len(best_actions)), best_actions]  # (B, N)
            target_quant = rewards.unsqueeze(-1) + self.config.gamma * (1 - dones.unsqueeze(-1)) * next_quant

        # Current quantile predictions for chosen actions
        quant_all = self.model.forward_quantiles(obs)  # (B, A, N)
        current_quant = quant_all[torch.arange(len(actions)), actions]  # (B, N)

        td_loss = self._quantile_huber_loss(current_quant, target_quant)

        # CQL conservative loss on mean Q-values
        q_values = quant_all.mean(dim=-1)  # (B, A)
        q_pred = q_values[torch.arange(len(actions)), actions]  # (B,)
        cql_loss = (torch.logsumexp(q_values, dim=-1) - q_pred).mean()

        loss = td_loss + self.alpha * cql_loss

        return loss, {
            "loss": loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "q_mean": q_pred.mean().item(),
        }

    def after_gradient_step(self, optimizer):
        self._step += 1
        if self._step % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, obs):
        return torch.argmax(self.model(obs), dim=-1).item()

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
        "content": _CQL_IMPL,
    },
]
