"""Official V-PBGD baseline for optimization-bilevel.

Reference:
- V-PBGD/toy/toy.py
- V-PBGD/data-hyper-cleaning/data_hyper_clean.py
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
            method="v_pbgd",
            lrx=0.1,
            lry=0.1,
            lr_inner=0.01,
            gamma_init=0.0,
            gamma_max=0.2,
            gamma_argmax_step=30_000,
            outer_itr=40_000,
            inner_itr=1,
            reg=0.0,
            eval_interval=10,
        )
    if net == "mlp":
        return HypercleanStrategy(
            method="v_pbgd",
            lrx=0.1,
            lry=0.01,
            lr_inner=0.01,
            gamma_init=0.0,
            gamma_max=0.1,
            gamma_argmax_step=10_000,
            outer_itr=80_000,
            inner_itr=1,
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
