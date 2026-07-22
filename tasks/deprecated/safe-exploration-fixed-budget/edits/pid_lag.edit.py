"""PID-Lag baseline for the fixed-budget task."""

_FILE = "omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py"

_PID_LAG_IMPORTS = """\
from collections import deque
"""

_PID_LAG_METHODS = """\
    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._pid_kp: float = 0.1
        self._pid_ki: float = 0.01
        self._pid_kd: float = 0.01
        self._pid_i: float = 0.0
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_ds: deque = deque(maxlen=10)
        self._cost_ds.append(0.0)
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        delta = float(Jc - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        self._delta_p = 0.95 * self._delta_p + 0.05 * delta
        self._cost_d = 0.95 * self._cost_d + 0.05 * float(Jc)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._lagrangian_multiplier = max(0.0, min(pid_o, 100.0))
        self._cost_ds.append(self._cost_d)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'PID-Lag: combine advantages using PID-controlled multiplier.'
        penalty = self._lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 48,
        "end_line": 70,
        "content": _PID_LAG_METHODS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 20,
        "content": _PID_LAG_IMPORTS,
    },
]
