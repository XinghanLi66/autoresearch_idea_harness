"""Baseline edit: GEM loss (beta=0.7).

Generalized Entropy Minimization: uses an auxiliary soft distribution
q = softmax(logits / beta) for contrastive weighting of hard examples.
"""

OPS = [
    {
        "op": "replace",
        "file": "LLaMA-Factory/src/llamafactory/train/sft/custom_sft_loss.py",
        "start_line": 15,
        "end_line": 63,
        "content": """\
# ===== EDITABLE SECTION START =====
# GEM (Generalized Entropy Minimization) loss baseline (beta=0.7)

def custom_loss_func(
    outputs,
    labels,
    num_items_in_batch=None,
):
    \"\"\"GEM loss: auxiliary soft distribution for contrastive hard-example weighting.\"\"\"
    import torch
    import torch.nn.functional as F

    IGNORE_INDEX = -100
    BETA = 0.7

    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0))

    logits = logits.float()
    vocab_size = logits.size(-1)

    # Causal shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1).to(flat_logits.device)

    valid_mask = flat_labels != IGNORE_INDEX
    if not valid_mask.any():
        return torch.tensor(0.0, device=flat_logits.device, dtype=flat_logits.dtype)

    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]

    # Auxiliary soft distribution q = softmax(logits / beta)
    q = F.softmax(valid_logits / BETA, dim=-1)

    # Log-probabilities of the model
    log_p = F.log_softmax(valid_logits, dim=-1)

    # Target log-probs (one-hot indexed)
    target_log_p = log_p.gather(1, valid_labels.unsqueeze(1)).squeeze(1)

    # q-weighted log-probs: sum over vocab of q * log_p
    q_weighted_log_p = (q * log_p).sum(dim=-1)

    # Contrastive: target log-prob vs q-weighted average
    contrast = target_log_p - q_weighted_log_p

    # Sigmoid weighting: emphasize hard examples (where contrast is negative/small)
    with torch.no_grad():
        weight = torch.sigmoid(-contrast)

    # Per-token CE
    ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction="none")

    weighted_loss = weight * ce_loss

    if num_items_in_batch is not None:
        loss = weighted_loss.sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    else:
        loss = weighted_loss.mean()

    return loss

# ===== EDITABLE SECTION END =====""",
    },
]
