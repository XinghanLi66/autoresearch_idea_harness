Let me start from what I actually want to happen, not from copying an existing method. My target is: at each iteration, the gradient estimate should have low variance, so the update is a good direction. But variance reduction is a tradeoff -- too much reduction costs computation, too little leaves me stuck. So the real question is how to allocate the budget between full gradient computations and variance-reduced steps.

The constraint that `compute_full_gradient` can be called at most once per epoch is the lever here. A naive interpretation is "I get one full gradient per epoch." But I don't want to pin myself to a single frequency; I want the algorithm to decide when it's worth recomputing the snapshot. That suggests a *triggered* snapshot update rather than a fixed schedule.

Now let me map out the forces. If I keep the snapshot stale, the control variate degrades and variance grows. But recomputing the snapshot is expensive. So I need a criterion that detects when variance has risen enough to justify the cost. A natural signal is the *change in the objective*. If the loss hasn't moved much since the last snapshot, the gradient landscape probably hasn't changed much either, so I don't need a new snapshot. Conversely, if the loss has dropped significantly, the model has moved far enough that the old snapshot is likely outdated. 

That gives me a decision rule: recompute the full gradient when the loss improvement since the last snapshot exceeds some threshold. Let me name it: Loss-Guided Snapshot Update (LGSU).

Why does this cut against the consensus? Most variance reduction methods fix the snapshot frequency (e.g. every k iterations in SVRG) or make it depend on iteration count. Those are static schedules; they assume a prior on how quickly the landscape changes. My intuition is that the landscape's evolution is better measured by the objective itself, not by wall-clock time. The loss is a direct proxy for the amount of movement, so it should drive the update timing.

How do I keep myself honest before writing a line? Two experiments:
1. Compare LGSU against a fixed-frequency baseline (say, compute full gradient every sqrt(n) iterations). If the fixed schedule wins, my trigger hypothesis is wrong and I should revert.
2. Check whether the chosen threshold matters: sweep it and see if there's a sweet spot. If performance is flat over a wide range, the trigger is robust and the exact value is secondary. If it's very sensitive, the trigger is fragile and I'd prefer a simpler, less-tuned approach.

Core idea: Use a triggered snapshot update driven by loss improvement (Loss-Guided Snapshot Update), recomputing the full gradient only when the objective has dropped by more than a threshold, instead of a fixed schedule.

Non-trivial crux: The decision rule is based on the *loss change* as a proxy for landscape movement, not on iteration count -- but the threshold needs to be validated rather than assumed, since a poorly-chosen threshold could make the trigger either too aggressive or too conservative.