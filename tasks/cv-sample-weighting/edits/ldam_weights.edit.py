"""LDAM-style class-dependent margin weighting baseline.

Uses the class-dependent margin idea from LDAM to derive per-class weights
proportional to n^{-1/4}, which provides a principled weighting based on
minimizing the class-conditional generalization bound.

Reference: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-
Aware Margin Loss" (NeurIPS 2019)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_weighting.py"

_CONTENT = """\
def compute_class_weights(class_counts, num_classes, config):
    \"\"\"LDAM-inspired n^{-1/4} weighting (Cao et al., NeurIPS 2019).

    Weights each class by count^{-1/4}, which is the theoretically
    motivated scaling from the label-distribution-aware margin bound.
    Normalized so weights sum to num_classes.
    \"\"\"
    weights = class_counts.float().pow(-0.25)
    weights = weights / weights.sum() * num_classes
    return weights
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 164,
        "end_line": 195,
        "content": _CONTENT,
    },
]
