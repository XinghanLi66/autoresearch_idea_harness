# Pilot CoT 20 — Andrej Karpathy · first_principles

- researcher: **Andrej Karpathy**  ·  style: 工程实证·偏执debug·打开黑盒·Software2.0

- case: Karpathy 案例 3：从 Tesla 视觉栈的工程现实 → 抽象出 "Software 2.0" 概念框架

- route: first_principles  ·  case score: 7/10

- lengths (qwen2.5-32b tokens): user(setup) **201** · assistant(cot) **1102** · whole **1360**

- fact-check changes: 3

## user (setup)

```plaintext
It's around 2017. You're leading a large production vision stack (lane detection, object classification, planning heuristics) that has historically been thousands of lines of hand-written C++/Python: explicit thresholds, geometric fits, if-else logic. Over the past couple of years you've watched neural networks quietly eat chunk after chunk of that codebase — a hand-tuned module gets replaced by 'dataset + architecture + loss + gradient descent,' and the replacement usually works better. The prevailing framing in the community is that neural nets are 'just another kind of classifier / function approximator' you bolt onto an otherwise conventional program. You suspect that framing is undercounting something. Reason from first principles about what a trained network fundamentally *is* relative to ordinary code, what must be true about the class of problems where this substitution keeps winning, and produce a genuinely novel conceptual framing (plus the operational decision rule it implies). Don't just describe a trick — derive the invariant.
```

## assistant (cot)

```plaintext
Let me not start from 'does this net beat that baseline.' Let me start from what a trained network *is*, as an object, and reason until a framing is forced.

Strip away the ML vocabulary. When I write ordinary code, I — a human — search the space of programs by hand, one line at a time, guided by my model of the task. The artifact is a sequence of explicit instructions. Now look at what gradient descent produces: an architecture plus a loss plus a dataset define a constrained optimization over a giant parameter space, and the optimizer returns a point in that space. That point *is* a program — it takes inputs, computes, returns outputs. The weights aren't a classifier hyperparameter; they are the source code, in a different representation. So the honest statement is: the weight tensor is a compiled program, and gradient descent is the compiler. The dataset is the specification. That's not a metaphor I'm reaching for — it's just re-labeling what already happened.

Once I accept that, calling it 'just another classifier' feels like a category error. There are two ways to obtain a program: hand-write instructions (call it 1.0), or specify a loss+data and let the optimizer search program-space (call it 2.0). Same output type — a callable program — different production process.

Now, the deductive part: what *must* be true about a 2.0 program regardless of task? I can reason about it before running anything. Its computation is a fixed graph of matmuls and elementwise nonlinearities, so (a) one forward pass has fixed, predictable compute and memory — no data-dependent branching blowups; (b) the whole stack is computationally homogeneous, which means hardware and compilers only have to be good at a tiny instruction set; (c) the artifact is a flat blob of numbers, so it's arguably more portable than a dependency-tangled 1.0 codebase; (d) two such programs can be concatenated and the seam optimized away by continuing to descend the joint loss — something you almost never get with two hand-written modules. None of these are empirical claims about accuracy; they follow from the *form* of the artifact. That's what makes me think the distinction is real and not hype: the properties are structural. (I should keep in mind the caveats — a very large weight blob is not automatically 'more portable,' and end-to-end joint optimization has its own difficulties; these are tendencies, not laws.)

So why is this cutting against consensus? Because everyone is arguing about whether nets are accurate enough, an empirical question, while I suspect the *mode of programming* has changed. The paranoid engineer in me actually likes this reframing precisely because it reorganizes where bugs live. In 1.0 you debug logic; in 2.0 the 'source' is the dataset, so a mislabeled or leaking dataset is a bug in the program, and it can fail *silently* — loss looks fine, the compiled weights are just quietly wrong. That's consistent with how these things tend to fail without complaining.

But I refuse to over-claim. When does 2.0 actually win, and when should I stay in 1.0? Try to deduce it from what the compiler needs. Gradient descent can only search toward a spec it can *measure*. So the plausible necessary conditions are: it's cheaper to accumulate data than to write the rules explicitly, AND you can cheaply *evaluate* whether an output is correct enough to shape the loss. If you can't evaluate correctness, you can't specify the program, and no amount of optimization saves you. That gives a crisp decision rule: Software 2.0 automates what you can verify. Verifiability is the operational variable, not 'is it a perception task.'

How do I keep myself honest instead of falling in love with the framing? I don't assert 2.0 always beats 1.0. The test is per-task and boring: pick a module, ask 'can I cheaply generate labeled data and cheaply score outputs?' If yes, try replacing it and hold everything else fixed; the decision rule is — if the learned program doesn't clearly beat the hand-written one on the metric I *could already compute*, keep the 1.0 code, because 1.0 is inspectable and 2.0 can fail silently. And for any task where I can't define an evaluation, the framing implies I *shouldn't* hand it to gradient descent at all — that's a falsifiable boundary, not a slogan.

**Core idea:** A trained neural network is not 'just a classifier' — it is a program (weights = compiled code) produced by a different compiler (gradient descent) from a different source (loss + dataset); this is a second way of writing software, 'Software 2.0,' whose structural properties (fixed-cost homogeneous compute, portability, weldable modules, but silent failure) follow from the artifact's form.
**Non-trivial crux:** The framing only pays off through its boundary condition — 2.0 wins exactly where it's cheaper to collect data than write rules AND you can cheaply *evaluate* correctness; without a measurable spec you cannot compile a program, so unverifiable tasks must stay in 1.0.
```

## fact-check changes

```plaintext
- Softened absolutized claims into tendencies (e.g., portability, joint welding, silent failure) and added a caveat noting large weight blobs aren't automatically more portable and end-to-end optimization has difficulties (category d).
- Replaced overly certain phrasings like 'is a category error' / 'the framing predicts' / 'convinces me' with intuition-marked language ('feels like', 'I suspect', 'the framing implies') to avoid presenting hindsight framing as decisive contemporaneous proof (category c).
- Changed 'FLOPs' to plain 'compute' in one spot for English-only clarity; no non-English words remained in body (the Chinese-language case label was only in the prompt header, not in setup/cot).
```