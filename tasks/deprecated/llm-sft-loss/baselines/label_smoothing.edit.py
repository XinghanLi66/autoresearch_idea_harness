"""Baseline edit: Cross-entropy with label smoothing (alpha=0.1)."""

OPS = [
    {
        "op": "replace",
        "file": "LLaMA-Factory/src/llamafactory/train/sft/custom_sft_loss.py",
        "start_line": 15,
        "end_line": 63,
        "content": """\
# ===== EDITABLE SECTION START =====
# Label smoothing cross-entropy baseline (smoothing=0.1)

def custom_loss_func(
    outputs,
    labels,
    num_items_in_batch=None,
):
    \"\"\"Cross-entropy with label smoothing (alpha=0.1).\"\"\"
    import torch
    import torch.nn.functional as F

    IGNORE_INDEX = -100
    LABEL_SMOOTHING = 0.1

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

    if num_items_in_batch is not None:
        loss = F.cross_entropy(
            shift_logits, shift_labels,
            ignore_index=IGNORE_INDEX, reduction="sum",
            label_smoothing=LABEL_SMOOTHING,
        )
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    else:
        loss = F.cross_entropy(
            shift_logits, shift_labels,
            ignore_index=IGNORE_INDEX,
            label_smoothing=LABEL_SMOOTHING,
        )

    return loss

# ===== EDITABLE SECTION END =====""",
    },
]
