"""Pure-SFT limit on the shared HPT scaffold.

For every question, drop all on-policy attempts and replace them with one
off-policy demonstration example.
"""

_FILE = "Unify-Post-Training/hpt/verl/verl/mix_src/mix_trainer.py"

_SFT_CONTROLLER = """\
    def select_on_off_ada_balance(self, on_solve_num: int):
        del on_solve_num
        return self.config.actor_rollout_ref.rollout.n_verify, 0, 1
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 394,
        "end_line": 414,
        "content": _SFT_CONTROLLER,
    }
]

