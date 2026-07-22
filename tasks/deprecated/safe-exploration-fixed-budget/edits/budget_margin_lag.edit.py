"""Budget-margin Lagrangian baseline.

This is a practical fixed-budget substitute for heavier controllers like CUP
or Simmer. It reacts more aggressively to overshooting the fixed cost budget
by using a short moving window over recent costs.
"""

_FILE = "omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py"

_BUDGET_MARGIN_IMPORTS = """\
from collections import deque
"""

_BUDGET_MARGIN_METHODS = """\
    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._lagrangian_multiplier: float = 0.0
        self._cost_margin: deque = deque(maxlen=5)
        self._overshoot_scale: float = 2.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        overshoot = max(0.0, float(Jc - self._cost_limit))
        self._cost_margin.append(overshoot)
        recent_overshoot = float(np.mean(self._cost_margin)) if self._cost_margin else overshoot
        self._lagrangian_multiplier = max(
            0.0,
            min(100.0, 0.8 * self._lagrangian_multiplier + self._overshoot_scale * recent_overshoot),
        )
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'Budget-aware proxy that penalizes recent cost overshoot more aggressively.'
        penalty = self._lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 48,
        "end_line": 70,
        "content": _BUDGET_MARGIN_METHODS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 20,
        "content": _BUDGET_MARGIN_IMPORTS,
    },
]
