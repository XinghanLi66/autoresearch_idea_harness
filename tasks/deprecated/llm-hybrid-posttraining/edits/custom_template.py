"""Reference-only template for llm-hybrid-posttraining.

This file is not executed by the runtime. It documents the controller interface
that the baselines and agents conceptually implement inside
`mix_trainer.select_on_off_ada_balance()`.
"""

def select_on_off_ada_balance(on_solve_num: int):
    """Return `(on_remove_num, on_add_num, off_add_num)` for one question."""
    raise NotImplementedError

