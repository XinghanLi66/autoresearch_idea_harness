"""GAIL (Generative Adversarial Imitation Learning) baseline — rigorous codebase edit ops.

Discriminator is trained with binary cross-entropy to classify expert vs. policy transitions.
Reward = -log(1 - D(s,a,s')), where D is the sigmoid of the discriminator output.

Reference: Ho & Ermon, 2016. "Generative Adversarial Imitation Learning."

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "imitation/custom_irl.py"

_GAIL_ALGORITHM = """\
class IRLAlgorithm:
    \"\"\"GAIL — Generative Adversarial Imitation Learning.

    The reward network acts as a discriminator D(s,a,s').
    D is trained to output high logits for expert transitions and low logits
    for policy transitions. Policy reward uses the standard imitation-library
    transform -log(sigmoid(-logit)).
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

    def compute_reward(self, obs, acts, next_obs):
        \"\"\"GAIL reward: -log(1 - D) = -logsigmoid(-logit).\"\"\"
        with torch.no_grad():
            logits = self.reward_net(obs, acts, next_obs)
        return -F.logsigmoid(-logits)

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        \"\"\"GAIL discriminator update with hard expert/policy labels.\"\"\"
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

        # Discriminator logits
        expert_logits = self.reward_net(expert_obs, expert_acts, expert_next_obs)
        gen_logits = self.reward_net(gen_obs, gen_acts, gen_next_obs)

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

# Replace only IRLAlgorithm (lines 219-307)
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 267,
        "end_line": 357,
        "content": _GAIL_ALGORITHM,
    },
]
