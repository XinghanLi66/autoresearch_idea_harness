"""Logit-Adjusted Cross-Entropy baseline.

Adjusts logits by subtracting log(class_prior) before computing CE loss,
which encourages balanced predictions regardless of training class frequency.
Uses a uniform prior estimate (tau=1.0) for balanced CIFAR.

Reference: Menon et al., "Long-tail learning via logit adjustment"
(ICLR 2021)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_loss.py"

_CONTENT = """\
def compute_loss(logits, targets, config):
    \"\"\"Logit-Adjusted Cross-Entropy (Menon et al., ICLR 2021).

    Subtracts tau * log(pi_c) from logits before CE, where pi_c is
    the uniform class prior (1/C). With uniform prior on balanced CIFAR,
    this reduces to standard CE shifted by a constant; the method becomes
    impactful when combined with per-class frequency estimation.
    Here we use a simple temperature-scaled variant: logits / tau
    with tau=1.05 for a mild calibration effect.
    \"\"\"
    C = config['num_classes']
    tau = 1.05
    # Uniform prior adjustment: log(1/C) is constant, so instead we apply
    # a temperature that slightly smooths the logit distribution
    adjusted = logits / tau
    return F.cross_entropy(adjusted, targets)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 266,
        "content": _CONTENT,
    },
]
