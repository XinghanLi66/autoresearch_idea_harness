# Pilot CoT 33 — 何恺明 Kaiming He · analogy_transfer

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例4："实验前先预测结果"——用 pre-hoc/post-hoc 对抗"研究的过拟合"

- route: analogy_transfer  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **178** · assistant(cot) **1155** · whole **1390**

- fact-check changes: 4

## user (setup)

```plaintext
You've spent years running large-scale deep learning experiments, and you keep noticing a quiet failure mode in how good researchers — including yourself — draw conclusions. You run a batch of experiments, tweak the method, reselect settings, and eventually a clean story emerges that explains the numbers. But when you or someone else revisits it on new data or a new task, the 'insight' sometimes doesn't hold. This isn't about a specific model; it's about the epistemics of doing research itself. There's a body of statistical learning theory that has extremely precise language for exactly this kind of trap, developed for models — not for researchers. You're preparing a talk for young researchers and want to give them a single, executable discipline, not vague advice like 'be rigorous.' Propose a genuinely novel framing and a concrete practice that attacks this self-deception at its root.
```

## assistant (cot)

```plaintext
Let me start from the thing that actually bothers me, because it's a reversal of what people assume rigor means. Most people think the danger in research is a wrong experiment. I don't think so. The more insidious danger is a *correct* experiment plus a story you assembled afterward. You ran ten things, five surprised you, and by the time you write the intro you have a beautiful mechanism that 'explains' all of them. That story feels like understanding. But you never held anything out. You fit your narrative to the very numbers you're using to justify it.

Now — I already have precise language for this, but not in the place I need it. In statistical learning we've beaten to death the difference between training error and test error. We know that if you keep touching the same data — reselecting features, retuning, peeking — your training error keeps dropping while your generalization silently rots. We even have a name for the mechanism: overfitting. And we know the *cure* is not cleverness, it's discipline: a held-out test set you do not touch, and honesty about what the optimizer is really doing in a messy landscape.

So here is the transfer I want to make explicit: treat *doing research* as itself an optimization process. Research is like SGD in a chaotic landscape — you take noisy steps, you follow gradients of intuition, you land somewhere. The moment I say that out loud, the whole train/test apparatus wants to come along for the ride. If research is optimization, then research can overfit. And if it can overfit, then the entire question becomes: what is the test set of a research project?

Let me be careful about *what transfers and what breaks*, because sloppy analogies are how you fool yourself twice. What transfers cleanly: the notion of overfitting to a finite sample, the idea that repeated touching of the same data inflates apparent performance, and the idea that the only honest measure of generalization is data the process never influenced. What does NOT transfer: in ML the test set is a fixed dataset I can literally hold in a file and evaluate once. In research there is no such file. My 'training set' isn't the data — it's the *batch of experiments and their observed outcomes* that I've been staring at and reacting to. So the analogy breaks exactly at 'where is the held-out set,' and I have to reconstruct it.

Where is it, then? It's the future. New data, new tasks, someone else reproducing it. The real test set is the part of the world my current tinkering could not have contaminated. That's the reframing I trust: *the future is the real test set.*

But a slogan isn't a discipline. In ML, held-out evaluation is enforced *mechanically* — you split before you train. What's the analogous mechanical move for a researcher? Here's the leap: write down your prediction of an experiment's outcome *before* you run it. That prewritten prediction is the one thing the result cannot retro-fit. It converts every experiment into a labeled signal. If the result matches what I predicted from my stated mechanism — call it pre-hoc — that's stronger evidence I understood something, because I committed before observing. If the result surprised me and I only invented the explanation afterward — post-hoc — that explanation is suspect by construction; it might just be me fitting noise. Post-hoc insights don't get to enter the conclusion as established; they become hypotheses that must be re-tested in a fresh setting, on the actual future test set.

Why does this cut against how people work? Because it's less, not more. No new tool, no fancy statistics — just a prewritten sentence. And it costs you your favorite pleasure: the tidy story. People resist it because most predictions will be wrong. I'm fine with that — being disappointed most of the time is exactly what you'd expect if research is genuinely hard, and it's what lets me only admit pre-hoc evidence into the paper.

How do I keep myself honest about the framing itself? I don't want to oversell it as a theorem — it's an analogy, and its payoff is long-run self-discipline, not a single-experiment win. So the decision rule is behavioral: before a conclusion goes into a paper, ask whether it rests on pre-hoc or post-hoc evidence, and refuse to ship any post-hoc 'insight' as a claim until it survives an independent test on new data / new task / independent reproduction. Caveat, because I don't want to strangle exploration: pure pilot phases, where you don't even have a hypothesis yet, are exempt — there's nothing to predict. The discipline binds only when you're about to claim you understand something.

**Core idea:** Treat research as an optimization process that overfits to the batch of experiments you keep staring at; since research's real held-out test set is the future (new data, tasks, independent reproduction), enforce it mechanically by writing down each experiment's predicted outcome beforehand and letting only pre-hoc (prediction-confirmed) results support conclusions, while post-hoc explanations are demoted to hypotheses requiring independent re-verification.

**Non-trivial crux:** The train/test analogy transfers in spirit but breaks at implementation — there is no literal held-out file, so the 'test set' must be reconstructed as the uncontaminated future, and the prewritten prediction is the only instrument that makes an outcome truly held-out; without that ex-ante commitment, any 'held-out' claim is just relabeled post-hoc storytelling.
```

## fact-check changes

```plaintext
- Softened 'Research is SGD' to 'Research is like SGD' to mark it as analogy, not literal claim.
- Changed 'genuine evidence' to 'stronger evidence' to avoid over-absolutizing the epistemic status of a single pre-hoc match.
- Replaced fabricated specific 'disappointed about 95% of the time' figure with a qualitative statement, since it's an invented number.
- Clarified that post-hoc insights are demoted from 'established' claims rather than barred entirely, restoring the real exception.
```