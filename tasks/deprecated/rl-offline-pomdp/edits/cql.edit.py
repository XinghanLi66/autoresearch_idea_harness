"""CQL baseline for the rl-offline-pomdp task.

Replaces the EDITABLE section (Model + OfflineAlgorithm) in custom_nethack_pomdp.py
with Conservative Q-Learning: Q-network head, TD loss + CQL penalty, target network.
"""

_FILE = "katakomba/algorithms/small_scale/custom_nethack_pomdp.py"

_CQL_IMPL = """\
class Model(nn.Module):
    def __init__(self, backbone: ModelBackbone):
        super().__init__()
        self.backbone = backbone
        self.num_actions = backbone.num_actions
        self.head = nn.Linear(backbone.rnn_hidden_dim, self.num_actions)

    def forward(self, inputs, state=None):
        core_output, new_state = self.backbone(inputs, state)
        q_values = self.head(core_output).view(
            core_output.shape[0], core_output.shape[1], self.num_actions
        )
        return q_values, new_state

    @torch.no_grad()
    def vec_act(self, obs, state=None, device="cpu"):
        inputs = {
            "tty_chars": torch.tensor(obs["tty_chars"][:, None], device=device),
            "screen_image": torch.tensor(obs["screen_image"][:, None], device=device),
            "prev_actions": torch.tensor(obs["prev_actions"][:, None], dtype=torch.long, device=device)
        }
        q_values, new_state = self(inputs, state)
        actions = torch.argmax(q_values.squeeze(1), dim=-1)
        return actions.cpu().numpy(), new_state


class OfflineAlgorithm:
    def __init__(self, backbone, config, device):
        self.device = device
        self.config = config
        self.alpha = 2.0
        self.model = Model(backbone).to(device)
        self.target_model = deepcopy(self.model)

    def parameters(self):
        return self.model.parameters()

    def train_step(self, batch, prev_actions, rnn_states):
        rnn_state = rnn_states.get("rnn_state", None)
        target_rnn_state = rnn_states.get("target_rnn_state", None)

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

        with torch.no_grad():
            next_q, new_target_state = self.target_model(next_obs, state=target_rnn_state)
            next_q_max = next_q.max(dim=-1).values
            targets = batch["rewards"][:, :-1] + self.config.gamma * (1 - batch["dones"][:, :-1]) * next_q_max

        q_values, new_rnn_state = self.model(obs, state=rnn_state)
        q_pred = q_values.gather(-1, batch["actions"][:, :-1].long().unsqueeze(-1)).squeeze(-1)

        td_loss = F.mse_loss(q_pred, targets) * self.alpha
        cql_loss = (torch.logsumexp(q_values, dim=-1) - q_pred).mean()
        loss = td_loss + cql_loss

        new_rnn_state = [a.detach() for a in new_rnn_state]
        new_target_state = [a.detach() for a in new_target_state]

        return loss, {"rnn_state": new_rnn_state, "target_rnn_state": new_target_state}, {
            "loss": loss.item(), "td_loss": td_loss.item(), "cql_loss": cql_loss.item(),
            "q_target": targets.mean().item()
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
        "content": _CQL_IMPL,
    },
]
