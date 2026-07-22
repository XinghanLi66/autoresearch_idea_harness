**Mocking:** World-class AI researcher, thinking through a real model design problem

Let me start not from the code but from what the numbers actually mean. I want to design a custom time-series model, and I have three reference implementations to reason about. First, I need to understand what the reference models are doing and where I can add value.

DLinear is a linear auto-regressive baseline: it does a linear projection over the lagged input and a linear projection of the lagged output. PatchTST is the transformer version with fixed-position embeddings, self-attention over time, and a linear head. iTransformer is the identity variant: it uses a time-agnostic linear projection of the input, then a time-dependent linear projection of the output, with no explicit attention.

What do I observe? In all three, the output projection is a linear layer of the *entire* sequence. That's the same across the board. That's suspicious. I'm modeling time. I expect local, recent patterns to matter more than distant ones. A linear projection of the *whole* sequence gives equal weight to a point far back in time and a point one step before the prediction. That's a global, flat weighting. In real forecasting problems, the recent neighborhood is where the signal lives.

So my working hypothesis is: the output projection should be a *local* aggregation over a small window near the target point, not a global one. That would be a non-trivial crux — local windows introduce edge effects (you can't aggregate beyond the sequence) and a window size is a hyperparameter that has to be tuned, unlike a single global projection.

Now, how do I operationalize this? I want to replace the single linear output layer with a small windowed aggregation. Two candidates: 1) a 1D convolution with kernel size equal to the window size and stride matching the target interval (so each output point aggregates a fixed number of preceding points), or 2) a moving average over a small fixed window. Convolution is more flexible, so I'll lean on that. I'll keep the same time-agnostic input projection from iTransformer and the same time-dependent output projection, but I'll replace the global linear layer with a small conv layer.

I should be precise about the change and where it lands. I'll keep the input projection as a linear layer over the whole sequence (same as iTransformer), and I'll keep the output projection as a linear layer that depends on time (same as DLinear and iTransformer). The one new, local piece is the convolutional windowed aggregation. That's the only real structural change relative to the references.

How do I keep myself honest? I won't decide on the window size ahead of time — I'll tune it, and I'll watch for edge effects when the window approaches the sequence boundary. And I'll evaluate the change across all three datasets to see whether local aggregation actually helps before I declare it a win.

Core idea: Replace the global linear output projection with a small convolutional window that aggregates only the most recent local neighborhood, to give recent data preferential weight in time-sensitive forecasting.

Non-trivial crux: The windowed aggregation introduces edge effects near sequence boundaries and adds a tunable hyperparameter (window size), which must be validated against the global-projection baseline rather than assumed to be better.