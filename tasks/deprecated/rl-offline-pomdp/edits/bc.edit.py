"""BC baseline for the rl-offline-pomdp task.

Replaces the EDITABLE section (Model + OfflineAlgorithm) in custom_nethack_pomdp.py
with behavioral cloning: single policy head, cross-entropy loss.
"""

_FILE = "katakomba/algorithms/small_scale/custom_nethack_pomdp.py"

_BC_IMPL = """\
class Model(nn.Module):
    def __init__(self, backbone: ModelBackbone):
        super().__init__()
        self.backbone = backbone
        self.num_actions = backbone.num_actions
        self.head = nn.Linear(backbone.rnn_hidden_dim, self.num_actions)

    def forward(self, inputs, state=None):
        core_output, new_state = self.backbone(inputs, state)
        logits = self.head(core_output)
        return logits, new_state

    @torch.no_grad()
    def vec_act(self, obs, state=None, device="cpu"):
        inputs = {
            "tty_chars": torch.tensor(obs["tty_chars"][:, None], device=device),
            "screen_image": torch.tensor(obs["screen_image"][:, None], device=device),
            "prev_actions": torch.tensor(obs["prev_actions"][:, None], dtype=torch.long, device=device)
        }
        logits, new_state = self(inputs, state)
        actions = torch.argmax(logits.squeeze(1), dim=-1)
        return actions.cpu().numpy(), new_state


class OfflineAlgorithm:
    def __init__(self, backbone, config, device):
        self.device = device
        self.config = config
        self.model = Model(backbone).to(device)

    def parameters(self):
        return self.model.parameters()

    def train_step(self, batch, prev_actions, rnn_states):
        rnn_state = rnn_states.get("rnn_state", None)
        obs = {
            "screen_image": batch["screen_image"][:, :-1].contiguous(),
            "tty_chars": batch["tty_chars"][:, :-1].contiguous(),
            "prev_actions": torch.cat([prev_actions, batch["actions"][:, :-2].long()], dim=1)
        }
        logits, new_rnn_state = self.model(obs, state=rnn_state)
        new_rnn_state = [a.detach() for a in new_rnn_state]
        dist = Categorical(logits=logits)
        loss = -dist.log_prob(batch["actions"][:, :-1]).mean()
        return loss, {"rnn_state": new_rnn_state}, {"loss": loss.item()}

    def after_gradient_step(self):
        pass
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 278,
        "end_line": 364,
        "content": _BC_IMPL,
    },
]
