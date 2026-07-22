"""Custom SFT loss function for LLaMA-Factory supervised fine-tuning.

Replace the default cross-entropy loss with a custom loss function.
This module is imported by the SFT trainer and used as compute_loss_func.
"""

from typing import Optional

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

# ===== EDITABLE SECTION START =====
# Lines below (until EDITABLE SECTION END) are the region the agent may modify.

def custom_loss_func(
    outputs: "torch.Tensor",
    labels: "torch.Tensor",
    num_items_in_batch: Optional["torch.Tensor"] = None,
) -> torch.Tensor:
    """Compute a custom SFT loss given model outputs and labels.

    Args:
        outputs: Model output dict. Use outputs.get("logits") to retrieve
            the raw logits tensor of shape (batch, seq_len, vocab_size).
        labels: Target token IDs of shape (batch, seq_len). Positions with
            value -100 (IGNORE_INDEX) should be excluded from the loss.
            NOTE: Labels are NOT pre-shifted. You must perform causal shift:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
        num_items_in_batch: Optional scalar for gradient accumulation
            normalization. When provided, compute sum of per-token losses
            divided by this value instead of mean.

    Returns:
        Scalar loss tensor.

    Hints for possible approaches:
        - Standard cross-entropy (F.cross_entropy with ignore_index=-100)
        - Focal loss: down-weight easy tokens via (1 - p_t)^gamma factor
        - GEM (Generalized Entropy Minimization): auxiliary soft distribution
          q = softmax(logits / beta) for contrastive weighting
        - Label smoothing: F.cross_entropy(..., label_smoothing=alpha)
        - Entropy regularization: add H(p) term to encourage exploration
        - DFT-style reweighting: weight tokens by their prediction confidence
    """
    raise NotImplementedError(
        "Implement your custom SFT loss function here. "
        "See the docstring above for the interface specification."
    )


# placeholder lines to maintain line count for edit range alignment
#
#
#
#
#
#
#

# ===== EDITABLE SECTION END =====
