"""PPO-Lag baseline for the fixed-budget task."""

_FILE = "omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py"

_PPO_LAG_IMPORTS = """\
from omnisafe.common.lagrange import Lagrange
"""

_PPO_LAG_METHODS = """\
    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        self._lagrange.update_lagrange_multiplier(Jc)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'PPO-Lag: penalize cost advantage using the learned multiplier.'
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 48,
        "end_line": 70,
        "content": _PPO_LAG_METHODS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 20,
        "content": _PPO_LAG_IMPORTS,
    },
]
