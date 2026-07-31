# Pilot CoT 06 — Richard Sutton · analogy_transfer

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例2：能力太多、公式化各异 → 抛出 "Reward is Enough" 极简统一假设

- route: analogy_transfer  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **145** · assistant(cot) **962** · whole **1164**

- fact-check changes: 2

## user (setup)

```plaintext
The field treats intelligence as a bundle of distinct capabilities — perception, language, planning, social reasoning, imitation, generalization — and the dominant practice is to invent a separate problem formulation for each one. Perception gets its own objective, language another, planning yet another. The complexity keeps piling up, and there is no principled story of how these abilities relate to each other or where they come from. You believe general methods that scale with computation eventually dominate hand-crafted per-capability machinery, and you have spent decades studying how agents learn from their own experience. Propose a genuinely novel organizing hypothesis for the origin of these diverse abilities — something simple, unifying, and testable — rather than adding one more formulation.
```

## assistant (cot)

```plaintext
Let me start not from the AI literature but from a place I keep returning to when the field's fashions get too clever: animal behavior, and behind it, the cybernetics tradition of goal-directed systems. What strikes me about a rat, a squirrel, a human child is that nobody handed them a 'perception module' and a separate 'social module.' They are creatures embedded in an environment, and they behave. The organizing principle that ethology and reinforcement psychology kept circling — Thorndike's law of effect, the whole trial-and-error tradition — is that behavior gets shaped by consequences. There is one thing the animal is pushing toward: some scalar sense of better-or-worse in its own experience. That is the source mechanism I want to carry over.

So the transfer is explicit: take from biology the idea that a single, scalar reward signal, acting through an agent that maximizes it while interacting with a sufficiently rich world, might be enough to produce the whole zoo of adaptive behavior. In the animal it produces foraging, navigation, recognition of kin and threat, communication signals, planning of routes. The question I want to force onto AI: what actually transfers, and what breaks?

What might transfer is the causal direction. In biology the capabilities are downstream of the pressure to do well, not designed in. A predator that can predict prey trajectories does so because prediction served reward; the perception is in service of the maximization, not a separate given. If that direction is real, then in an artificial agent too, perception, memory, even language could be instrumental sub-skills that a reward-maximizing learner is forced to acquire because they help it score. That is exactly the opposite of adding a formulation per skill — it says the skills would be emergent consequences of one objective in a rich enough environment.

Now, what might break in the transfer, because I refuse to smuggle an analogy in whole. Evolution gave the animal a reward signal already grounded in its own sensations — hunger, pain, warmth. It is measurable by the agent from the inside. Any artificial version has to obey the same discipline: the reward and the goal must be anchored in signals the agent itself can sense and verify, not in some abstract objective only an outside human can judge. If I let the reward be something the agent cannot measure, I have broken the very thing that made the biological case work. Second, the animal case implicitly requires 'a sufficiently rich environment' and 'a sufficiently powerful learner' — those are load-bearing conditions, and I must state them as conditions, not hide them.

Why does this cut against the consensus? Because the field is betting that we should engineer each capability with human-designed priors — that we are cleverer than the discovery process. My whole reading of the last decades is that general methods riding on more computation tend to win over hand-built domain machinery. Search and learning are the two things that scale without bound. A single-objective learner is precisely such a general method; a pile of per-capability formulations is precisely the hand-built thing that keeps getting overtaken. I state this as my working conviction, not a settled result.

Now the honesty rule, and here the analogy also disciplines how I present the idea. I am not claiming this is proven — this is a hypothesis, an intuition to be pressed, not a theorem. I want it stated as a maximally simple, unifying, and falsifiable hypothesis — 'reward is enough' to drive all these abilities — and then I want to invite the field to attack it, not to protect it. A unifying theory that cannot be attacked is worthless. The decision rule I would hold myself to: if for some capability no reward-maximization account can, even in principle, produce it in a rich environment, that is a genuine counterexample and the hypothesis narrows or dies. And a sharper test of whether any such agent is really learning rather than mimicking a distribution: does it get surprised by outcomes and adjust based on what happens? If it never updates from experience, it has no world model, and 'reward is enough' would be an empty slogan for it.

**Core idea:** Rather than a separate formulation per capability, hypothesize that perception, language, planning, social intelligence and the rest all emerge as byproducts of a single agent maximizing one scalar reward in a sufficiently rich environment — transferring the causal direction of reward-shaped behavior from animal learning into AI.

**Non-trivial crux:** The claim only holds under load-bearing conditions (rich enough environment, powerful enough learner) and only if the reward is grounded in signals the agent can itself measure and verify; and it earns its keep by being stated as a falsifiable hypothesis to be attacked, not a defended belief.
```

## fact-check changes

```plaintext
- (a/c) Softened assertions that the causal transfer 'is' true or proven into hypothesized/conditional framing ('might transfer', 'would be'), keeping it as a stated conjecture consistent with 'Reward is Enough' being an argued hypothesis rather than an empirical finding.
- (d) Tempered the over-absolutized 'win by a large margin' / 'only two things' claim into a working conviction stated as such, restoring appropriate caveat.
```