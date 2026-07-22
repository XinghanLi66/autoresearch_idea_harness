"""Pure-GRPO limit on the shared HPT scaffold.

Never switch a question away from its on-policy rollouts.
"""

_FILE = "Unify-Post-Training/hpt/verl/verl/mix_src/mix_trainer.py"

_GRPO_CONTROLLER = """\
    def select_on_off_ada_balance(self, on_solve_num: int):
        del on_solve_num
        return 0, 0, 0
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 394,
        "end_line": 414,
        "content": _GRPO_CONTROLLER,
    }
]

