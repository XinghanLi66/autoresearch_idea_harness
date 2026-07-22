"""Official G-PBGD baseline for optimization-bilevel.

Reference:
- G-PBGD/data_hyper_clean_gpbgd.py
"""

_FILE = "penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py"

_CONTENT = '''\
def get_toy_strategy() -> ToyStrategy:
    return ToyStrategy(
        method="g_pbgd",
        gams=(10.0,),
        alpha0=0.1,
    )


def get_hyperclean_strategy(net: str) -> HypercleanStrategy:
    if net == "linear":
        return HypercleanStrategy(
            method="g_pbgd",
            lrx=0.3,
            lry=0.5,
            gamma_init=0.0,
            gamma_max=37.0,
            gamma_argmax_step=5_000,
            outer_itr=40_000,
            reg=0.0,
            eval_interval=10,
        )
    if net == "mlp":
        return HypercleanStrategy(
            method="g_pbgd",
            lrx=0.5,
            lry=0.5,
            gamma_init=0.0,
            gamma_max=37.0,
            gamma_argmax_step=30_000,
            outer_itr=50_000,
            reg=0.0,
            eval_interval=10,
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
