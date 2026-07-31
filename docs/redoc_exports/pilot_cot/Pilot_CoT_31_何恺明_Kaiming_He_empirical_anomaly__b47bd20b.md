# Pilot CoT 31 — 何恺明 Kaiming He · empirical_anomaly

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例4："实验前先预测结果"——用 pre-hoc/post-hoc 对抗"研究的过拟合"

- route: empirical_anomaly  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **221** · assistant(cot) **1043** · whole **1321**

- fact-check changes: 2

## user (setup)

```plaintext
You are advising junior researchers who keep producing results that look brilliant in the paper but evaporate when someone else reproduces them, or when the method meets a new dataset next year. Everyone already agrees on the local hygiene: run controlled ablations, tune baselines fairly, report the test protocol. Yet a subtler failure keeps recurring and nobody has a crisp name for it. Here is the concrete, reproducible pattern you keep observing: a team runs a batch of experiments, some results come out surprising, they invent a clean mechanistic story that explains those numbers, write it into the introduction as an 'insight' — and the insight repeatedly fails to transfer. Meanwhile teams that got equally good numbers but by luck report equally confident insights. The standard explanations ('they didn't tune enough', 'noisy seeds', 'weak baseline') don't fully account for it, because the numbers themselves are real and the ablations are clean. Something about the process of arriving at the conclusion is broken, not the individual experiments. Propose a genuinely novel framing and an operational discipline that attacks this at its root.
```

## assistant (cot)

```plaintext
Let me start from the thing that won't fit the standard story. I keep seeing papers with clean single-factor ablations, tuned baselines, honest protocols — all the local hygiene is there — and the headline *insight* still fails to reproduce on a new dataset next year. That's the anomaly. My first instinct is the cheap explanations: bad seeds, weak baselines, someone forgot to shuffle. But I've watched cases where the ablation grid is genuinely clean and the numbers are genuinely reproducible on *that* setup, and the conclusion still rots. So I have to peel the trivial causes off. The failure is not in any single experiment. That's what makes it interesting — the bug lives in the *process that manufactures the conclusion from a batch of experiments*, not in the experiments.

Now I want to push the vague 'insights don't transfer' into a sharp contradiction. Suppose the researcher genuinely understood the mechanism. Then the insight, being about the mechanism and not about this particular data, *should* hold on new data. It doesn't. So either they didn't understand the mechanism, or — and this is the construction I want — the insight was never evidence of understanding in the first place; it was *fit to the data after seeing it*. Once I say it that way, the machine that makes this happen is obvious to me, because it's a machine I already know intimately. Research itself is an optimization: you keep adjusting the method, the settings, the story, guided by the results you can see. Research is like gradient descent in a chaotic landscape. And any optimizer that keeps consuming the same batch of observations to shape its output *overfits* that batch. The post-hoc insight is exactly the overfitting term. That's the leap: the disease of non-transferring insights is `research overfitting`, and I can name it in the language of the very field we work in.

The moment I have train/test in my head, the fix is forced. If research is fitting, it has a test set, and the test set is *not* the experiments on my desk — those are the training set I've already been optimizing against. The real test set is the future: new data, a new task, someone else reproducing it. So the discipline is: treat the future as a held-out set, and never let a story I invented *after* seeing the numbers count as validation. The concrete, minimal move — and I want it minimal, less is more, a rule you can actually keep — is to write down your predicted outcome *before* you run the experiment. Then every result splits into two kinds of signal with no ambiguity: pre-hoc, where the result matched the prediction I committed to in advance — that is real evidence I understood the mechanism, like validating on a held-out set; and post-hoc, where I was surprised and only then constructed an explanation — that is a hypothesis, not a conclusion, and it must be re-tested on an independent setup before it earns a place in the intro.

Why this cuts against the common wisdom: everyone optimizes for clean ablations and fair baselines, and those are necessary — but they only sanitize each single measurement. They do nothing against overfitting *across* the batch, because you can have impeccable ablations and still be curve-fitting a narrative to the results you happened to see. My rule attacks the layer nobody instruments: the temporal order between hypothesis and observation.

How do I keep myself honest, given I can't run a clean experiment on 'is this discipline good'? I won't claim a measured win — the payoff is long-horizon self-discipline and won't show up in one run. The decision rule instead: any insight I want to publish must carry a pre-hoc tag *or* have survived a fresh, independent setup (new data / new task / external repro); everything else stays labeled post-hoc and out of the conclusions. And the reason I can even afford this brutal filter is curiosity over publication — most of my predictions turn out wrong, and only a small fraction survive to become a paper. If you accept that most of your predictions will be wrong, you can hold the line of only letting the honest predictions speak.

**Core idea:** Reframe doing research itself as an optimization that overfits the batch of experiments you keep staring at; the true test set is the future, so commit a written prediction *before* each run and let only prediction-matching (pre-hoc) results support conclusions, quarantining after-the-fact (post-hoc) explanations for independent re-validation.

**Non-trivial crux:** Clean per-experiment hygiene (fair ablations, tuned baselines) does not prevent research overfitting, because the leak is temporal — a story fit to already-seen results — and can only be closed by fixing the order (predict, then observe); the caveat is that in pure pilot exploration, before any hypothesis exists, forcing predictions is pointless, but the moment you draw a conclusion the pre-hoc-plus-independent-repro step is mandatory.
```

## fact-check changes

```plaintext
- Changed 'Research is SGD' to 'Research is like gradient descent' — softened the metaphor as a simile (intuition framing) rather than an asserted identity, and avoided the abbreviation.
- Replaced fabricated precise statistic 'disappointed roughly 95% of the time and finish the paper in the other 5%' with a non-fabricated qualitative statement ('most of my predictions turn out wrong, and only a small fraction survive'), since the specific numbers are not a verified quantity.
```