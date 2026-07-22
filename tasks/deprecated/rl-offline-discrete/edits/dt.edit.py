"""Decision Transformer baseline for the rl-offline-discrete task.

Matches d3rlpy DiscreteDecisionTransformer reproduction:
  - batch_size=128, lr=6e-4, context_size=30 (pong: batch=512, context=50)
  - GPT-2 backbone: 6 layers, 8 heads, GELU activation
  - GlobalPositionEncoding, tanh embed activation
  - AdamW with betas=(0.9, 0.95), weight_decay=0.1
  - LR schedule: linear warmup + cosine decay
  - Softmax action sampling (temperature=1.0)
  - Target returns: breakout=90, pong=20, qbert=2500
  - Observation /255.0 (pixel scaler), no reward scaling
  - 5 epochs over dataset (n_steps = dataset_size // batch_size * 5)
"""

_FILE = "d3rlpy/atari_offline/custom_atari.py"

_DT_IMPL = """\
import math as _math
from collections import deque


# ── GPT-2 Transformer components (matching d3rlpy) ──────────────

def _create_attention_mask(context_size):
    mask = torch.ones(context_size, context_size, dtype=torch.float32)
    return torch.tril(mask).view(1, 1, context_size, context_size)


class _CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, context_size, attn_dropout, resid_dropout):
        super().__init__()
        self._num_heads = num_heads
        self._k = nn.Linear(embed_size, embed_size)
        self._q = nn.Linear(embed_size, embed_size)
        self._v = nn.Linear(embed_size, embed_size)
        self._proj = nn.Linear(embed_size, embed_size)
        self._attn_dropout = nn.Dropout(attn_dropout)
        self._proj_dropout = nn.Dropout(resid_dropout)
        self.register_buffer("_mask", _create_attention_mask(context_size))

    def forward(self, x, attention_mask):
        B, T, _ = x.shape
        shape = (B, T, self._num_heads, -1)
        k = self._k(x).view(shape).transpose(1, 2)
        q = self._q(x).view(shape).transpose(1, 2)
        v = self._v(x).view(shape).transpose(1, 2)
        att = torch.matmul(q, k.transpose(2, 3)) / _math.sqrt(k.shape[-1])
        att = att.masked_fill(self._mask[..., :T, :T] == 0, -10000)
        att = att + attention_mask.view(B, 1, 1, T) * -10000
        att = F.softmax(att, dim=-1)
        att = self._attn_dropout(att)
        out = torch.matmul(att, v).transpose(1, 2).reshape(B, T, -1)
        return self._proj_dropout(self._proj(out))


class _Block(nn.Module):
    def __init__(self, embed_size, num_heads, context_size, attn_dropout, resid_dropout):
        super().__init__()
        self._attn = _CausalSelfAttention(embed_size, num_heads, context_size, attn_dropout, resid_dropout)
        self._mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size), nn.GELU(),
            nn.Linear(4 * embed_size, embed_size), nn.Dropout(resid_dropout),
        )
        self._ln1 = nn.LayerNorm(embed_size, eps=0.003)
        self._ln2 = nn.LayerNorm(embed_size, eps=0.003)

    def forward(self, x, attention_mask):
        x = x + self._attn(self._ln1(x), attention_mask)
        x = x + self._mlp(self._ln2(x))
        return x


class _GPT2(nn.Module):
    def __init__(self, embed_size, num_heads, context_size, num_layers, attn_dropout, resid_dropout, embed_dropout):
        super().__init__()
        self._blocks = nn.ModuleList([
            _Block(embed_size, num_heads, context_size, attn_dropout, resid_dropout)
            for _ in range(num_layers)
        ])
        self._ln = nn.LayerNorm(embed_size, eps=0.003)
        self._drop = nn.Dropout(embed_dropout)

    def forward(self, x, attention_mask):
        x = self._drop(x)
        for block in self._blocks:
            x = block(x, attention_mask)
        return self._ln(x)


# ── Decision Transformer model ──────────────────────────────────

class QNetwork(nn.Module):
    \"\"\"DiscreteDecisionTransformer matching d3rlpy implementation.

    GPT-2 backbone with (rtg, state, action) token interleaving.
    forward(obs) returns logits for compatibility (used as fallback).
    \"\"\"
    def __init__(self, observation_shape, action_dim, embed_size=128,
                 num_heads=8, num_layers=6, context_size=30,
                 attn_dropout=0.1, resid_dropout=0.1, embed_dropout=0.1,
                 max_timestep=2345):
        super().__init__()
        self.action_dim = action_dim
        self.context_size = context_size
        self.embed_size = embed_size
        # Encoder: Nature DQN CNN -> embed_size (matching PixelEncoderFactory feature_size=128)
        self.encoder = NatureDQNEncoder(observation_shape[0], embed_size)
        # Embeddings
        self.action_embed = nn.Embedding(action_dim, embed_size)
        self.rtg_embed = nn.Linear(1, embed_size)
        # Global position encoding (matching d3rlpy GlobalPositionEncoding)
        self.global_pos_embed = nn.Embedding(max_timestep + 1, embed_size)
        self.block_pos_embed = nn.Parameter(torch.zeros(1, 3 * context_size, embed_size))
        # GPT-2
        self.gpt2 = _GPT2(embed_size, num_heads, 3 * context_size, num_layers,
                           attn_dropout, resid_dropout, embed_dropout)
        # Output head
        self.output_head = nn.Linear(embed_size, action_dim, bias=False)
        # Embed activation (tanh, matching d3rlpy embed_activation_type)
        self.embed_activation = nn.Tanh()
        # Init weights (matching d3rlpy _init_weights)
        self.apply(self._init_weights)
        # Re-init encoder after apply (encoder has its own init)
        self.encoder = NatureDQNEncoder(observation_shape[0], embed_size)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, obs):
        \"\"\"Fallback: returns logits from single obs (not used in DT).\"\"\"
        features = self.encoder(obs)
        return self.output_head(features)

    def forward_dt(self, obs_seq, action_seq, rtg_seq, timesteps, attention_mask):
        \"\"\"Full DT forward pass.
        obs_seq: (B, T, C, H, W), action_seq: (B, T), rtg_seq: (B, T, 1),
        timesteps: (B, T), attention_mask: (B, T) where 1=masked, 0=valid.
        Returns: (probs (B, T, A), logits (B, T, A))
        \"\"\"
        B, T = obs_seq.shape[:2]
        # Encode observations
        flat_obs = obs_seq.reshape(B * T, *obs_seq.shape[2:])
        state_emb = self.encoder(flat_obs).view(B, T, -1)
        action_emb = self.action_embed(action_seq.long())
        rtg_emb = self.rtg_embed(rtg_seq)
        # Position encoding (matching d3rlpy GlobalPositionEncoding)
        global_emb = self.global_pos_embed(timesteps[:, -1:])  # (B, 1, N)
        block_emb = self.block_pos_embed[:, :T, :]  # (1, T, N) from (1, 3*C, N)
        # Stack: (rtg, state, action) interleaved
        h = torch.stack([rtg_emb, state_emb, action_emb], dim=1)  # (B, 3, T, N)
        h = self.embed_activation(h)
        # Same block_emb broadcast to all 3 token types (matching d3rlpy)
        pos_emb = global_emb.unsqueeze(1) + block_emb.unsqueeze(1)  # (B, 1, T, N)
        h = h + pos_emb
        h = h.transpose(1, 2).reshape(B, 3 * T, -1)  # (B, 3T, N)
        # Attention mask: repeat for 3 tokens per step
        attn_mask = torch.stack([attention_mask] * 3, dim=1).transpose(1, 2).reshape(B, 3 * T)
        # For inference, drop last action token
        if not self.training:
            h = h[:, :-1, :]
            attn_mask = attn_mask[:, :-1]
        h = self.gpt2(h, attn_mask)
        # Extract state token outputs (index 1 of every 3)
        logits = self.output_head(h[:, 1::3, :])
        return F.softmax(logits, dim=-1), logits


# ── Trajectory sampling from buffer ──────────────────────────────

class _TrajectoryBuffer:
    \"\"\"Extracts episodes from AtariReplayBuffer for trajectory sampling.\"\"\"
    def __init__(self, buffer, context_size, gamma=1.0):
        self.context_size = context_size
        self.gamma = gamma
        # Build episode list
        self.episodes = []
        ep_start = 0
        for i in range(buffer.size):
            if buffer.terminals[i] > 0.5 or i == buffer.size - 1:
                self.episodes.append((ep_start, i + 1))
                ep_start = i + 1
        self.buffer = buffer

    def sample(self, batch_size, device):
        \"\"\"Sample trajectory batch. Returns dict with tensors on device.\"\"\"
        C = self.context_size
        obs_batch = []
        act_batch = []
        rtg_batch = []
        ts_batch = []
        mask_batch = []
        for _ in range(batch_size):
            ep_idx = np.random.randint(len(self.episodes))
            ep_start, ep_end = self.episodes[ep_idx]
            ep_len = ep_end - ep_start
            # Random start within episode
            if ep_len <= C:
                start = ep_start
                length = ep_len
            else:
                offset = np.random.randint(ep_len - C + 1)
                start = ep_start + offset
                length = C
            # Compute returns-to-go for this episode slice
            rewards = self.buffer.rewards[start:start+length]
            rtg = np.zeros(length, dtype=np.float32)
            rtg[-1] = rewards[-1]
            for t in range(length - 2, -1, -1):
                rtg[t] = rewards[t] + self.gamma * rtg[t + 1]
            # Get stacked observations
            obs_list = [self.buffer._get_stacked_obs(start + t) for t in range(length)]
            obs = np.stack(obs_list).astype(np.float32) / 255.0  # (L, 4, 84, 84)
            actions = self.buffer.actions[start:start+length]
            timesteps = np.arange(start - self.episodes[ep_idx][0],
                                  start - self.episodes[ep_idx][0] + length) + 1
            # Pad to context_size
            pad = C - length
            if pad > 0:
                obs = np.concatenate([np.zeros((pad, *obs.shape[1:]), dtype=np.float32), obs], axis=0)
                actions = np.concatenate([np.zeros(pad, dtype=np.int64), actions])
                rtg = np.concatenate([np.zeros(pad, dtype=np.float32), rtg])
                timesteps = np.concatenate([np.zeros(pad, dtype=np.int64), timesteps])
                mask = np.concatenate([np.ones(pad, dtype=np.float32), np.zeros(length, dtype=np.float32)])
            else:
                mask = np.zeros(C, dtype=np.float32)
            obs_batch.append(obs)
            act_batch.append(actions)
            rtg_batch.append(rtg)
            ts_batch.append(timesteps)
            mask_batch.append(mask)
        return {
            "observations": torch.tensor(np.stack(obs_batch), device=device),
            "actions": torch.tensor(np.stack(act_batch), dtype=torch.long, device=device),
            "returns_to_go": torch.tensor(np.stack(rtg_batch), device=device).unsqueeze(-1),
            "timesteps": torch.tensor(np.stack(ts_batch), dtype=torch.long, device=device),
            "masks": torch.tensor(np.stack(mask_batch), device=device),
        }


# ── Algorithm ────────────────────────────────────────────────────

# Per-game configs (matching d3rlpy reproduction)
_GAME_CONFIGS = {
    "breakout": {"context_size": 30, "batch_size": 128, "target_return": 90},
    "pong": {"context_size": 50, "batch_size": 512, "target_return": 20},
    "qbert": {"context_size": 30, "batch_size": 128, "target_return": 2500},
    "seaquest": {"context_size": 30, "batch_size": 128, "target_return": 1450},
}


class OfflineAlgorithm:
    \"\"\"DiscreteDecisionTransformer: sequence modeling for offline RL.\"\"\"
    def __init__(self, observation_shape, action_dim, config, device, buffer):
        self.device = device
        self.config = config
        self.action_dim = action_dim
        # Per-game config
        game = getattr(config, "game", "breakout")
        gc = _GAME_CONFIGS.get(game, _GAME_CONFIGS["breakout"])
        self.context_size = gc["context_size"]
        self.target_return = gc["target_return"]
        config.batch_size = gc["batch_size"]
        config.learning_rate = 6e-4
        self.clip_grad_norm = 1.0
        # Match d3rlpy: train for 5 epochs over the dataset
        n_steps_per_epoch = buffer.size // gc["batch_size"]
        config.max_timesteps = n_steps_per_epoch * 5
        config.eval_freq = n_steps_per_epoch  # eval every epoch
        # Compute max_timestep from buffer episodes
        max_ts = 0
        ep_start = 0
        for i in range(buffer.size):
            if buffer.terminals[i] > 0.5 or i == buffer.size - 1:
                max_ts = max(max_ts, i - ep_start + 1)
                ep_start = i + 1
        # Build model
        self.model = QNetwork(
            observation_shape, action_dim,
            embed_size=128, num_heads=8, num_layers=6,
            context_size=self.context_size,
            attn_dropout=0.1, resid_dropout=0.1, embed_dropout=0.1,
            max_timestep=max_ts,
        ).to(device)
        # Trajectory buffer
        self.traj_buffer = _TrajectoryBuffer(buffer, self.context_size, gamma=1.0)
        self.buffer = buffer
        # LR schedule state (matching d3rlpy reproduction exactly)
        self._tokens = 0
        self._warmup_tokens = 512 * 20  # 10240, matching d3rlpy
        self._final_tokens = 2 * buffer.size * self.context_size * 3
        self._initial_lr = config.learning_rate
        # Eval state
        self._obs_history = deque(maxlen=self.context_size)
        self._act_history = deque(maxlen=self.context_size)
        self._rtg_history = deque(maxlen=self.context_size)
        self._ts_history = deque(maxlen=self.context_size)
        self._timestep = 1
        self._return_rest = self.target_return

    def parameters(self):
        return self.model.parameters()

    def create_optimizer(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith('.bias') or 'ln' in name or 'embed' in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self._initial_lr,
            betas=(0.9, 0.95),
        )

    def train_step(self, obs, actions, rewards, next_obs, dones):
        \"\"\"Ignores transition input; samples trajectory batch from buffer.\"\"\"
        batch = self.traj_buffer.sample(self.config.batch_size, self.device)
        _, logits = self.model.forward_dt(
            batch["observations"], batch["actions"],
            batch["returns_to_go"], batch["timesteps"], batch["masks"],
        )
        loss = F.cross_entropy(
            logits.reshape(-1, self.action_dim),
            batch["actions"].reshape(-1),
        )
        self._last_n_valid = int((1.0 - batch["masks"]).sum().item())
        return loss, {"loss": loss.item()}

    def after_gradient_step(self, optimizer):
        \"\"\"Update LR schedule (matching d3rlpy warmup + cosine decay).\"\"\"
        self._tokens += getattr(self, '_last_n_valid', self.config.batch_size * self.context_size)
        if self._tokens < self._warmup_tokens:
            lr_mult = self._tokens / max(1, self._warmup_tokens)
        else:
            progress = (self._tokens - self._warmup_tokens) / max(
                1, self._final_tokens - self._warmup_tokens)
            lr_mult = max(0.1, 0.5 * (1.0 + _math.cos(_math.pi * progress)))
        new_lr = lr_mult * self._initial_lr
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

    def select_action(self, obs):
        \"\"\"Stateful action selection (matching d3rlpy StatefulTransformerWrapper).\"\"\"
        self._obs_history.append(obs.squeeze(0).cpu().numpy())
        self._rtg_history.append(self._return_rest)
        self._ts_history.append(min(self._timestep, self.model.global_pos_embed.num_embeddings - 1))
        C = len(self._obs_history)
        obs_seq = torch.tensor(np.stack(list(self._obs_history)), device=self.device).unsqueeze(0)
        # Pad action history with 0 for current step
        acts = list(self._act_history)
        acts.append(0)
        act_seq = torch.tensor([acts[-C:]], dtype=torch.long, device=self.device)
        rtg_seq = torch.tensor([list(self._rtg_history)], dtype=torch.float32, device=self.device).unsqueeze(-1)
        ts_seq = torch.tensor([list(self._ts_history)], dtype=torch.long, device=self.device)
        mask = torch.zeros(1, C, device=self.device)
        with torch.no_grad():
            probs, logits = self.model.forward_dt(obs_seq, act_seq, rtg_seq, ts_seq, mask)
        # Softmax sampling (temperature=1.0, matching d3rlpy SoftmaxTransformerActionSampler)
        last_logits = logits[0, -1]
        action_probs = F.softmax(last_logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        # Update action history
        self._act_history.append(action)
        self._timestep += 1
        return action

    def begin_episode(self):
        \"\"\"Reset stateful history for new evaluation episode.\"\"\"
        self._obs_history.clear()
        self._act_history.clear()
        self._rtg_history.clear()
        self._ts_history.clear()
        self._act_history.append(0)  # pad action
        self._timestep = 1
        self._return_rest = self.target_return

    def observe(self, reward):
        \"\"\"Update return-to-go after observing reward.\"\"\"
        self._return_rest -= reward
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 220,
        "end_line": 258,
        "content": _DT_IMPL,
    },
]
