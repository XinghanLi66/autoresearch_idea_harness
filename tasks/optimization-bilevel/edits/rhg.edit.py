"""Official RHG / ITD baseline for optimization-bilevel.

Reference:
- RHG/data_hyper_clean_rhg.py
"""

_FILE = "penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py"

_CONTENT = '''\
def get_toy_strategy() -> ToyStrategy:
    return ToyStrategy(
        method="v_pbgd",
        gams=(10.0,),
        alpha0=0.1,
    )


def get_hyperclean_strategy(net: str) -> HypercleanStrategy:
    if net == "linear":
        return HypercleanStrategy(
            method="rhg",
            lr=0.001,
            lr_inner=0.1,
            outer_itr=100,
            T=500,
            K=500,
            reg=0.0,
            eval_interval=1,
        )
    if net == "mlp":
        return HypercleanStrategy(
            method="rhg",
            lr=0.001,
            lr_inner=0.4,
            outer_itr=100,
            T=500,
            K=500,
            reg=0.0,
            eval_interval=1,
        )
    raise ValueError(f"Unsupported network: {net}")
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 216,
        "end_line": 253,
        "content": _CONTENT,
    },
]
