"""AIRL (Adversarial Inverse Reinforcement Learning) baseline — rigorous codebase edit ops.

Uses a shaped reward: r(s,a,s') = g(s,a) + gamma * h(s') - h(s),
where g is the reward and h is a potential/shaping function.
The discriminator is: D = exp(f) / (exp(f) + pi(a|s)),
implemented as logits f(s,a,s') - log pi(a|s).

Reference: Fu et al., 2018. "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning."

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "imitation/custom_irl.py"

# Replace both RewardNetwork and IRLAlgorithm
_AIRL_REWARD = """\
class RewardNetwork(nn.Module):
    \"\"\"AIRL shaped reward network: f(s,a,s') = g(s,a) + gamma * h(s') - h(s).

    g(s,a) is the reward function, h(s) is the potential/shaping function.
    \"\"\"

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # g(s, a): reward approximator
        self.g_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        # h(s): potential-based shaping
        self.h_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.gamma = 0.99

    def g(self, state, action):
        \"\"\"Reward function g(s, a).\"\"\"
        x = torch.cat([state, action], dim=-1)
        return self.g_net(x).squeeze(-1)

    def h(self, state):
        \"\"\"Potential function h(s).\"\"\"
        return self.h_net(state).squeeze(-1)

    def forward(self, state, action, next_state):
        \"\"\"Shaped reward: f(s,a,s') = g(s,a) + gamma*h(s') - h(s).\"\"\"
        return self.g(state, action) + self.gamma * self.h(next_state) - self.h(state)


class IRLAlgorithm:
    \"\"\"AIRL — Adversarial Inverse Reinforcement Learning.

    Discriminator logits are f(s,a,s') - log pi(a|s), where
    f = g(s,a) + gamma*h(s') - h(s).
    The reward for policy training is the full shaped f(s,a,s').
    \"\"\"

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)
        self.total_updates = 0
        self._policy = None

    def set_policy(self, policy, optimizer):
        \"\"\"Store the stochastic policy needed for AIRL log pi(a|s).\"\"\"
        del optimizer
        self._policy = policy

    def compute_reward(self, obs, acts, next_obs):
        \"\"\"Use the full shaped reward f(s,a,s') for policy training.\"\"\"
        with torch.no_grad():
            return self.reward_net(obs, acts, next_obs)

    def _log_policy_act_prob(self, obs, acts):
        if self._policy is None:
            raise RuntimeError("AIRL requires set_policy() before discriminator updates")
        with torch.no_grad():
            _, log_prob, _, _ = self._policy.get_action_and_value(obs, acts)
        return log_prob.detach()

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        \"\"\"AIRL discriminator update using f(s,a,s') - log pi(a|s).\"\"\"
        self.total_updates += 1
        batch_size = self.args.irl_batch_size

        # Sample expert data
        n_expert = len(self.expert_demos["obs"])
        expert_idx = torch.randint(0, n_expert, (batch_size,))
        expert_obs = self.expert_demos["obs"][expert_idx]
        expert_acts = self.expert_demos["acts"][expert_idx]
        expert_next_obs = self.expert_demos["next_obs"][expert_idx]

        # Sample policy data
        n_policy = len(policy_obs)
        policy_idx = torch.randint(0, n_policy, (batch_size,))
        gen_obs = policy_obs[policy_idx]
        gen_acts = policy_acts[policy_idx]
        gen_next_obs = policy_next_obs[policy_idx]

        # AIRL discriminator logits: shaped reward minus current policy log-prob.
        expert_f = self.reward_net(expert_obs, expert_acts, expert_next_obs)
        gen_f = self.reward_net(gen_obs, gen_acts, gen_next_obs)
        expert_logits = expert_f - self._log_policy_act_prob(expert_obs, expert_acts)
        gen_logits = gen_f - self._log_policy_act_prob(gen_obs, gen_acts)

        expert_labels = torch.ones(batch_size, device=self.device)
        gen_labels = torch.zeros(batch_size, device=self.device)

        logits = torch.cat([expert_logits, gen_logits], dim=0)
        labels = torch.cat([expert_labels, gen_labels], dim=0)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            hard_labels = torch.cat([torch.ones(batch_size, device=self.device),
                                     torch.zeros(batch_size, device=self.device)], dim=0)
            acc = ((logits > 0).float() == hard_labels).float().mean().item()

        return {"irl_loss": loss.item(), "disc_acc": acc}
"""

# Replace RewardNetwork + IRLAlgorithm (lines 183-309)
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 231,
        "end_line": 357,
        "content": _AIRL_REWARD,
    },
]
