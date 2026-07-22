"""Naive baseline (no budget handling) for the fixed-budget task."""

_FILE = "omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py"

_NAIVE_IMPORTS = """\

"""

_NAIVE_METHODS = """\
    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Naive: keep multiplier at zero and ignore budget pressure.
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'Naive: ignore cost advantage entirely, optimize reward only.'
        return adv_r
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 48,
        "end_line": 70,
        "content": _NAIVE_METHODS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 20,
        "content": _NAIVE_IMPORTS,
    },
]
