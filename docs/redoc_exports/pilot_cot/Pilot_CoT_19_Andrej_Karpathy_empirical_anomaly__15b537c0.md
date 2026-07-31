# Pilot CoT 19 — Andrej Karpathy · empirical_anomaly

- researcher: **Andrej Karpathy**  ·  style: 工程实证·偏执debug·打开黑盒·Software2.0

- case: Karpathy 案例 3：从 Tesla 视觉栈的工程现实 → 抽象出 "Software 2.0" 概念框架

- route: empirical_anomaly  ·  case score: 7/10

- lengths (qwen2.5-32b tokens): user(setup) **208** · assistant(cot) **1006** · whole **1271**

- fact-check changes: 4

## user (setup)

```plaintext
It's around 2017. You've spent the last few years shipping production computer-vision systems — first research, now leading a large perception stack for a self-driving product. Your codebase used to be full of hand-written logic: lane-line fitting, hand-tuned thresholds for whether a blob is a car, cascades of if-statements for edge cases. Over the past couple of years you've watched something strange happen in your own repo: chunks of that carefully engineered C++/Python keep getting deleted and replaced by 'collect more labeled data + train a bigger net.' Each time, the neural version is smaller in code, easier to ship, and often just... better. The prevailing framing in the field is that neural nets are 'a very good classifier,' one more tool in the ML toolbox. But something about what you're seeing in the actual diffs doesn't fit that story. Articulate what's really going on here — propose a genuinely new way to think about it that changes how teams should invest.
```

## assistant (cot)

```plaintext
Let me start from what I can actually see in the repo, not from any theory. Concrete, reproducible observation: over the past many months, the lines-of-code count for the perception stack is trending *down* while capability goes *up*. That's weird. Normally more capability = more code. Here the code is being *deleted*. What replaces a deleted lane-fitting heuristic is not more code — it's a bigger training set and a slightly different loss. That's the datum that won't fit the standard story.

The standard story everyone repeats: 'a neural net is just another classifier.' Fine. But if it were *just* a classifier, I'd expect it to live *inside* my program as one function call, and the rest of the C++ scaffolding to stay roughly constant. That's not what's happening. The scaffolding is evaporating. The net isn't a function *in* my program; it's eating the program. When I stare at that honestly, the 'just a classifier' framing is explaining away the thing I'm actually watching in the diffs.

So let me not hand-wave it. Let me ask the paranoid-debugger question: what is the *real* object here, and what would falsify my interpretation? My interpretation: the weights are the program. The engineer no longer writes the instructions; the engineer writes the *spec* (architecture + loss + dataset) and gradient descent *searches* the weight space for a program that satisfies it. If that's true, it should make concrete, checkable claims about how the whole engineering practice behaves — and if those fail, I should drop the framing.

Claims I'd hold myself to:
- Where do the bugs move? If weights are the program, then bugs shouldn't live in branches anymore — they should live in the *dataset*. Mislabeled data = miscompiled program. I've already lived this: label/image misalignment silently poisons a model with loss still going down. That's the kind of thing that fits — bugs migrating to data.
- Where does engineering effort move? If the source code is now the dataset, then the highest-leverage work stops being 'write cleverer code' and becomes 'curate, clean, label, version the data.' Test: watch where my best engineers spend their time on projects that succeed vs. stall. My intuition is it's data infra and labeling, but I'd treat that as the decision variable to observe, not as something already proven.
- Runtime shape: a 1.0 program has wildly data-dependent branching and unpredictable latency/memory; a 2.0 program is basically a fixed pile of matmuls + ReLUs, so FLOPs and memory are roughly constant and predictable per forward pass. That's checkable — and it's a genuine engineering *advantage* the classifier framing never surfaces.

Now the leap, and why it cuts against consensus. The consensus treats deep learning as a modeling advance. I think it's a *software* advance — a second way of producing programs. Call it Software 2.0. This is uncomfortable because it demotes the thing we're all proud of (writing code) and elevates the thing we treat as grunt work (managing data). It says the compiler is gradient descent and the object file is a blob of weights.

But I have to keep the framing honest, including its failure modes — that's the whole point of not being a hero. A 2.0 program is opaque: it can fail *silently*, with metrics looking fine while the model has learned a shortcut or leaked information. It has adversarial examples and inherits dataset bias. So the framing must come with its own boundary condition — a rule of thumb for *which* problems to hand to 2.0. The rule I'd propose: 2.0 is attractive where it's easier to accumulate data than to write the program explicitly, *and* where you can cheaply evaluate correctness. Software 2.0 automates what you can verify. Where you can't verify, you're flying blind on an uninterpretable artifact — keep that in 1.0.

So the observation forces the reframe, the reframe makes checkable operational claims, and it carries its own caveat about verifiability. That's the shape I trust.

**Core idea:** Neural nets aren't just better classifiers — they're a new way of *writing software*: you specify architecture + loss + dataset, and gradient descent compiles a program out of weight-space. The dataset becomes the source code, so engineering effort and bugs migrate from code to data.

**Non-trivial crux:** The reframe only holds where data is cheaper than explicit code AND correctness is cheaply verifiable — Software 2.0 automates what you can verify; outside that boundary you're shipping an opaque program that can fail silently, so it must stay in Software 1.0.
```

## fact-check changes

```plaintext
- Translated the Chinese fragments in the case label context; kept output fully in English.
- Softened fabricated-confirmation language: changed 'That matches — bugs migrated to data, exactly as the framing predicts' to an observation that fits rather than a confirmed prediction; changed 'predictions' framing to 'checkable claims/observations' to avoid asserting results not yet measured.
- Replaced the overly specific '~18 months' with 'the past many months' to avoid fabricating a precise figure.
- Rephrased the decision rule from 'wins exactly where' to 'is attractive where'/'rule of thumb' to avoid over-absolutizing a heuristic as a proven law, while preserving the crux.
```