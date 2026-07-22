"""Baseline edit: Focal loss (gamma=2.0)."""

OPS = [
    {
        "op": "replace",
        "file": "LLaMA-Factory/src/llamafactory/train/sft/custom_sft_loss.py",
        "start_line": 15,
        "end_line": 63,
        "content": """\
# ===== EDITABLE SECTION START =====
# Focal loss baseline (gamma=2.0)

def custom_loss_func(
    outputs,
    labels,
    num_items_in_batch=None,
):
    \"\"\"Focal loss: down-weight easy tokens with (1-p_t)^gamma factor.\"\"\"
    import torch
    import torch.nn.functional as F

    IGNORE_INDEX = -100
    GAMMA = 2.0

    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0))

    logits = logits.float()
    vocab_size = logits.size(-1)

    # Causal shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(shift_logits.device)

    # Per-token CE (unreduced)
    ce_loss = F.cross_entropy(
        shift_logits, shift_labels,
        ignore_index=IGNORE_INDEX, reduction="none",
    )

    # p_t = probability of the correct token = exp(-ce)
    with torch.no_grad():
        p_t = torch.exp(-ce_loss)
        focal_weight = (1.0 - p_t) ** GAMMA

    focal_loss = focal_weight * ce_loss

    # Mask ignored tokens
    valid_mask = shift_labels != IGNORE_INDEX

    if num_items_in_batch is not None:
        loss = focal_loss[valid_mask].sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    else:
        loss = focal_loss[valid_mask].mean()

    return loss

# ===== EDITABLE SECTION END =====""",
    },
]
