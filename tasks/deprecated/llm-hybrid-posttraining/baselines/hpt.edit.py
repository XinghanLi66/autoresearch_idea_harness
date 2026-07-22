"""Original HPT controller from Unify-Post-Training."""

_FILE = "Unify-Post-Training/hpt/verl/verl/mix_src/mix_trainer.py"

_HPT_CONTROLLER = """\
    def select_on_off_ada_balance(self, on_solve_num: int):
        if self.config.trainer.unify_strategy == 'switch':
            on_add_num = 0
            if on_solve_num <= self.config.trainer.switch_gate:
                on_remove_num = 8
                off_add_num = 1
            elif on_solve_num <= self.config.trainer.switch_gate_off:
                on_remove_num = 8
                off_add_num = -1
            else:
                on_remove_num = 0
                off_add_num = 0

            return on_remove_num, on_add_num, off_add_num

        if self.config.trainer.unify_strategy == 'soft':
            on_remove_num = 0
            on_add_num = 0
            off_add_num = 1

            return on_remove_num, on_add_num, off_add_num
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 394,
        "end_line": 414,
        "content": _HPT_CONTROLLER,
    }
]
