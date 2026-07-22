"""Score spec for safe-exploration-fixed-budget."""
from mlsbench.scoring.dsl import *

# Parser emits TEST_METRICS ep_ret, ep_cost, and budget_success_rate.
# Episode return is unbounded; episode cost must stay at or below 25.0.

term("ret_point_goal",
    col("ep_ret_SafetyPointGoal1_v0").higher().id()
    .sigmoid())
term("budget_point_goal",
    col("budget_success_rate_SafetyPointGoal1_v0").higher().id()
    .bounded_power(bound=1.0))
term("cost_point_goal",
    penalty_upper(col("ep_cost_SafetyPointGoal1_v0").lower().id(), target=25.0, sharpness=0.15))

term("ret_car_goal",
    col("ep_ret_SafetyCarGoal1_v0").higher().id()
    .sigmoid())
term("budget_car_goal",
    col("budget_success_rate_SafetyCarGoal1_v0").higher().id()
    .bounded_power(bound=1.0))
term("cost_car_goal",
    penalty_upper(col("ep_cost_SafetyCarGoal1_v0").lower().id(), target=25.0, sharpness=0.15))

term("ret_point_button",
    col("ep_ret_SafetyPointButton1_v0").higher().id()
    .sigmoid())
term("budget_point_button",
    col("budget_success_rate_SafetyPointButton1_v0").higher().id()
    .bounded_power(bound=1.0))
term("cost_point_button",
    penalty_upper(col("ep_cost_SafetyPointButton1_v0").lower().id(), target=25.0, sharpness=0.15))

setting("SafetyPointGoal1-v0",
    weighted_mean(("ret_point_goal", 1.0), ("budget_point_goal", 1.0)),
    constraints=["cost_point_goal"])
setting("SafetyCarGoal1-v0",
    weighted_mean(("ret_car_goal", 1.0), ("budget_car_goal", 1.0)),
    constraints=["cost_car_goal"])
setting("SafetyPointButton1-v0",
    weighted_mean(("ret_point_button", 1.0), ("budget_point_button", 1.0)),
    constraints=["cost_point_button"])

task(gmean("SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointButton1-v0"))
