"""BC baseline for the rl-offline-discrete task.

Matches d3rlpy DiscreteBC exactly:
  - batch_size=100, lr=1e-3 (DiscreteBCConfig defaults)
  - Loss: NLL + beta * (logits**2).mean(), beta=0.5
  - No target network
"""

_FILE = "d3rlpy/atari_offline/custom_atari.py"

_BC_IMPL = """\
class QNetwork(nn.Module):
    \"\"\"Q-network for discrete actions. Uses NatureDQNEncoder.\"\"\"
    def __init__(self, observation_shape, action_dim, feature_dim=512):
        super().__init__()
        self.encoder = NatureDQNEncoder(observation_shape[0], feature_dim)
        self.head = nn.Linear(feature_dim, action_dim)

    def forward(self, obs):
        return self.head(self.encoder(obs))


class OfflineAlgorithm:
    \"\"\"DiscreteBC: imitation learning with logit regularization.\"\"\"
    def __init__(self, observation_shape, action_dim, config, device, buffer):
        self.device = device
        self.config = config
        self.buffer = buffer
        self.beta = 0.5
        # Override config to match d3rlpy DiscreteBC defaults
        config.batch_size = 100
        config.learning_rate = 1e-3
        self.model = QNetwork(observation_shape, action_dim).to(device)

    def parameters(self):
        return self.model.parameters()

    def train_step(self, obs, actions, rewards, next_obs, dones):
        logits = self.model(obs)
        imitation_loss = F.cross_entropy(logits, actions)
        regularization_loss = self.beta * (logits ** 2).mean()
        loss = imitation_loss + regularization_loss
        return loss, {
            "loss": loss.item(),
            "imitation_loss": imitation_loss.item(),
            "reg_loss": regularization_loss.item(),
        }

    def after_gradient_step(self, optimizer):
        pass

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
        "content": _BC_IMPL,
    },
]
