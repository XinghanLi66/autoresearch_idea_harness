# Pilot CoT 01 — Richard Sutton · empirical_anomaly

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例1：RL 在 80 年代被学界抛弃 → 从动物心理学"翻译"出 TD Learning

- route: empirical_anomaly  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **233** · assistant(cot) **985** · whole **1275**

- fact-check changes: 2

## user (setup)

```plaintext
It's the early 1980s. Reinforcement learning by trial-and-error is out of fashion; symbolic AI and supervised learning dominate. You're working on the credit-assignment problem for an agent that acts over long sequences and only occasionally receives reward. There's a concrete, reproducible phenomenon you keep staring at: when a predictor is trained to estimate future reward and later gets corrected by the actual outcome, most standard schemes treat the true delayed outcome as the only legitimate learning target — you must wait until the end, then hand back the observed return as a supervised label. The prevailing story says that's how it has to work: the environment (or a teacher) provides the target, and everything in between is just error to be assigned. But you keep noticing something in the data from conditioning-style experiments and from your own prediction runs: the predictions themselves shift step-by-step in ways that carry information long before the final outcome arrives, and successive predictions disagree with each other in a systematic, exploitable way. The standard 'wait for the true label' framing quietly discards that disagreement. Propose a genuinely novel way to drive learning in this setting.
```

## assistant (cot)

```plaintext
Start with the thing that won't sit still. I run a predictor that estimates how much reward is coming, and I watch its output as time steps forward. The standard account says: the only real signal is the final outcome; everything before it is provisional, to be corrected once the truth lands. Fine. But then explain to me the measurement I keep seeing — between step t and step t+1, before any outcome has arrived, the prediction itself has already moved, and that movement is not noise. It correlates with what actually happens next. People wave this away as 'the estimate warming up,' as an artifact of an untrained predictor. I don't buy it. If the change from one prediction to the next carries reliable information, then throwing it away until the end is leaving a real learning signal on the floor.

So let me take the agent's first-person view seriously. What can this system actually measure, moment to moment, from its own experience? Not the final return — that's far away, and in a continuing task it may never arrive at all. What it *can* measure at every step is: its own prediction now, its own prediction one step later, and whatever reward it just received. Those are all quantities inside the experience stream. The consensus target — the true delayed outcome supplied as a label — is precisely the thing the agent often cannot get its hands on in time. If I insist on it, I've asked the agent to chase something it can't measure until it's too late to matter. That offends me on principle.

Here's the leap. Turn the disagreement between successive predictions *into the target itself*. Don't wait for the final truth. At each step, treat the difference between the prediction now and the prediction a step later (plus the reward just seen) as the error that drives learning. The later prediction is a better-informed guess than the earlier one — it has seen one more step of reality — so let the earlier one move toward it. The learning signal becomes the temporal *difference* between two of the agent's own estimates. Each prediction is not a fixed label to be matched; it is a claim that the very next moment of experience will immediately test and correct. Prediction becomes self-verifying: the stream of experience audits every estimate, one step at a time, with no external teacher handing down labels.

Why does this cut against the room? Because the field's instinct is that learning targets must be *true*, i.e. grounded in actual outcomes, and bootstrapping a prediction off another prediction looks like building on sand — you're learning from a guess. I understand the discomfort. But the guess is anchored to real reward at every step, and errors are corrected continuously as later experience arrives, so the sand is constantly re-poured by reality. This is the honest thing: the agent learns only from what it can itself measure and itself check.

Where did the courage to trust the measurement come from? Not from AI fashion — from going back to the classical source, animal learning and classical conditioning, where you plainly see behavior driven by changes in expectation, by the moment prediction and reality diverge. Work in that tradition already frames conditioning around the discrepancy between expected and received reinforcement rather than the raw outcome. That's obvious to psychologists and to ordinary people. When something is that obvious and only unfashionable, it's worth taking seriously and pressing on.

How do I keep myself honest? I won't declare victory by rhetoric. I set up the comparison directly: prediction driven by successive-difference errors versus the wait-for-the-outcome scheme, on the same experience. The decision rule: if bootstrapping off the next prediction gives faster, more accurate, more sample-efficient prediction — especially in long-horizon or non-terminating tasks where the true return is unavailable — then the disagreement was signal, not artifact, and I keep the difference-driven rule. If it merely matches the supervised form at greater complexity, I have no case and I drop it. And the honest caveat: bootstrapping introduces bias from leaning on current estimates; in settings where the true outcome is cheaply available and horizons are short, the plain outcome target may still be the safer bet.

**Core idea:** Drive learning from the temporal difference between an agent's own successive predictions (plus observed reward), rather than from the final delayed outcome — making each prediction a self-verifying claim that the next step of experience immediately tests and corrects, with no external teacher.

**Non-trivial crux:** The learning target is one prediction bootstrapped off the *next* prediction, not the true return; this is legitimate only because each step is anchored to real reward and continuously re-corrected by experience — but it trades variance for bias, so where the true outcome is cheap and horizons short, the plain outcome target can still win.
```

## fact-check changes

```plaintext
- (a) Softened the closing line of the conditioning paragraph from an over-confident 'usually right and just waiting to be recognized' to a claim-of-motivation ('worth taking seriously and pressing on'), avoiding hindsight vindication.
- (c) Grounded the conditioning appeal in the actual contemporaneous framing (expectation-discrepancy / prediction-error models of classical conditioning) rather than a vague appeal to obviousness, marking it as the intuition source it was.
```