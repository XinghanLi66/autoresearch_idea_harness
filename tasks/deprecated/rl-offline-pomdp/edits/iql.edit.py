"""IQL baseline for the rl-offline-pomdp task.

Replaces the EDITABLE section (Model + OfflineAlgorithm) in custom_nethack_pomdp.py
with Implicit Q-Learning: dual Q-networks + V-network + advantage-weighted policy.
"""

_FILE = "katakomba/algorithms/small_scale/custom_nethack_pomdp.py"

_IQL_IMPL = """\
class Model(nn.Module):
    def __init__(self, backbone: ModelBackbone):
        super().__init__()
        self.backbone = backbone
        self.num_actions = backbone.num_actions
        rnn_dim = backbone.rnn_hidden_dim
        self.qf1 = nn.Linear(rnn_dim + self.num_actions, 1)
        self.qf2 = nn.Linear(rnn_dim + self.num_actions, 1)
        self.vf = nn.Linear(rnn_dim, 1)
        self.policy = nn.Linear(rnn_dim, self.num_actions)

    def forward(self, obs, state=None, actions=None):
        core_output, new_state = self.backbone(obs, state)
        logits = self.policy(core_output)
        vf = self.vf(core_output).squeeze(-1)
        if actions is not None:
            sa = torch.cat([core_output, F.one_hot(actions, self.num_actions)], dim=-1)
            q1 = self.qf1(sa).squeeze(-1)
            q2 = self.qf2(sa).squeeze(-1)
            return logits, vf, q1, q2, new_state
        return logits, vf, None, None, new_state

    @torch.no_grad()
    def vec_act(self, obs, state=None, device="cpu"):
        inputs = {
            "tty_chars": torch.tensor(obs["tty_chars"][:, None], device=device),
            "screen_image": torch.tensor(obs["screen_image"][:, None], device=device),
            "prev_actions": torch.tensor(obs["prev_actions"][:, None], dtype=torch.long, device=device)
        }
        logits, *_, new_state = self(inputs, state=state)
        actions = torch.argmax(logits.squeeze(1), dim=-1)
        return actions.cpu().numpy(), new_state


class OfflineAlgorithm:
    def __init__(self, backbone, config, device):
        self.device = device
        self.config = config
        self.expectile_tau = 0.8
        self.temperature = 1.0
        self.model = Model(backbone).to(device)
        self.target_model = deepcopy(self.model)

    def parameters(self):
        return self.model.parameters()

    def train_step(self, batch, prev_actions, rnn_states):
        rnn_state = rnn_states.get("rnn_state", None)
        target_rnn_state = rnn_states.get("target_rnn_state", None)
        next_rnn_state = rnn_states.get("next_rnn_state", None)

        obs = {
            "screen_image": batch["screen_image"][:, :-1].contiguous(),
            "tty_chars": batch["tty_chars"][:, :-1].contiguous(),
            "prev_actions": torch.cat([prev_actions.long(), batch["actions"][:, :-2].long()], dim=1)
        }
        next_obs = {
            "screen_image": batch["screen_image"][:, 1:].contiguous(),
            "tty_chars": batch["tty_chars"][:, 1:].contiguous(),
            "prev_actions": batch["actions"][:, :-1].long()
        }
        actions = batch["actions"][:, :-1].long()

        with torch.no_grad():
            _, _, tq1, tq2, new_target_state = self.target_model(obs, actions=actions, state=target_rnn_state)
            target_q = torch.minimum(tq1, tq2)

        logits, v_pred, q1, q2, new_rnn_state = self.model(obs, actions=actions, state=rnn_state)
        advantage = target_q - v_pred
        value_loss = asymmetric_l2_loss(advantage, self.expectile_tau)

        with torch.no_grad():
            _, next_v, _, _, new_next_state = self.model(next_obs, state=next_rnn_state)
            next_q = batch["rewards"][:, :-1] + (1 - batch["dones"][:, :-1]) * self.config.gamma * next_v
        td_loss = (F.mse_loss(q1, next_q) + F.mse_loss(q2, next_q)) / 2

        weights = torch.exp(self.temperature * advantage.clamp(max=100.0))
        log_probs = Categorical(logits=logits).log_prob(actions)
        actor_loss = torch.mean(-log_probs * weights.detach())

        loss = value_loss + td_loss + actor_loss

        new_rnn_state = [a.detach() for a in new_rnn_state]
        new_target_state = [a.detach() for a in new_target_state]
        new_next_state = [a.detach() for a in new_next_state]

        return loss, {
            "rnn_state": new_rnn_state,
            "target_rnn_state": new_target_state,
            "next_rnn_state": new_next_state,
        }, {
            "loss": loss.item(), "td_loss": td_loss.item(),
            "value_loss": value_loss.item(), "actor_loss": actor_loss.item(),
            "next_v": next_v.mean().item(), "q_target": next_q.mean().item()
        }

    def after_gradient_step(self):
        soft_update(self.target_model, self.model, tau=self.config.tau)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 278,
        "end_line": 364,
        "content": _IQL_IMPL,
    },
]
