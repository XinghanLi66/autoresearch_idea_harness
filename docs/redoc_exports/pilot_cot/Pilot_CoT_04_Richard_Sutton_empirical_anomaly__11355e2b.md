# Pilot CoT 04 — Richard Sutton · empirical_anomaly

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例2：能力太多、公式化各异 → 抛出 "Reward is Enough" 极简统一假设

- route: empirical_anomaly  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **157** · assistant(cot) **949** · whole **1163**

- fact-check changes: 2

## user (setup)

```plaintext
The field builds intelligence one faculty at a time. Perception gets its own problem formulation, language another, planning another, imitation another, social behavior yet another. Each new capability arrives with its own hand-crafted objective and its own machinery, and the pile grows. You keep noticing something that the standard division-of-labor story does not explain: when you actually watch reward-driven agents interact with rich environments, they keep producing behaviors nobody explicitly wrote an objective for — rudimentary perception, memory use, planning-like sequencing, even proto-social maneuvers against other agents — as side effects of just trying to get more reward. The consensus explains these away as coincidences, or as things the environment designer secretly baked in. Propose a genuinely novel way to think about where all these different capabilities come from.
```

## assistant (cot)

```plaintext
Let me start from the thing that won't sit still. I keep looking at reward-maximizing agents in sufficiently rich settings, and they keep exhibiting capabilities that nobody formulated a separate objective for. A predator agent develops something that functions like perception and prediction; an agent in a world with other agents starts doing things that look social; give it a long enough horizon and you see planning-shaped behavior emerge. The standard story says: that's incidental, the real intelligence is in the specialized machinery we hand-design per faculty. But that explanation is doing a lot of quiet work. Every time I try to attribute the emergent competence to something the designer inserted, I can't find the insertion point. The capability shows up in the measured behavior, and the only thing I actually put in was the scalar reward and the environment's richness.

So the contradiction is this: the mainstream program adds one formulation per capability, yet the capabilities I actually observe appear *without* their dedicated formulation, as long as reward and environment are present. If I take the observation seriously instead of explaining it away, the parsimony runs the other direction. Ten formulations that keep getting pre-empted by one is not ten theories — it's one theory the field refuses to write down.

I want to be honest about my own bias here, because the history of AI keeps teaching the same lesson. Every time we hand-write the domain knowledge to win the current benchmark — chess, speech, vision — the general method that just leverages search and learning at scale eventually overtakes it, and by a large margin. The per-faculty-formulation habit smells like the same mistake at the level of *objectives*: we are hand-specifying what each capability is, instead of letting a single scalable objective discover them. The question I always ask about a method — give it far more computation, does it automatically get better? — a bag of separate faculty formulations does not; a single reward-maximization objective in an ever-richer environment plausibly does. That asymmetry is the tell. I should be clear this is intuition guiding the hypothesis, not a proven mechanism.

The leap, then: I'll hypothesize that maximization of a single scalar reward, by an agent capable enough, interacting with an environment rich enough, is *sufficient* to drive the emergence of all these capabilities — perception, language, memory, generalization, social intelligence — as instrumental byproducts of the pursuit of reward. Not that reward is *nice*, not that it *helps* — that it is *enough*. I want the strong, minimal, unified form on purpose.

Why cut against consensus this hard? Because a unified minimal principle is itself a form of judgment: if one objective can account for the phenomena, using ten is a failure of understanding, not a virtue of engineering. And because this is the agent's-first-person view I always insist on — the agent only has what it can measure. Reward is exactly a signal the agent can measure and be surprised by and adjust to. The specialized faculty objectives are mostly designer conveniences the agent never perceives directly.

Now, keeping myself honest. I must not smuggle in confirmation. The value of this claim is entirely in whether it can be *attacked*. So I state it as a falsifiable hypothesis and put it on the table for others to break, rather than protecting it. The decision rule: the claim is sufficiency, so a single clean counterexample — a capability that demonstrably *cannot* arise from reward maximization in any sufficiently rich environment, no matter the agent's power — would defeat it. And I must flag the real caveats: 'sufficiently rich environment' and 'sufficiently powerful agent' are load-bearing; this is a claim about what is *possible in principle to emerge*, not a recipe guaranteeing it emerges efficiently, and it says nothing about *which* scalar reward. If I can't specify tests that could kill it, it isn't a theory, it's a private belief — and a unified theory that can't be attacked is worthless.

**Core idea:** Instead of a separate hand-designed formulation per capability, hypothesize that maximizing a single scalar reward — given a sufficiently rich environment and a sufficiently capable learning agent — is enough to drive all the varied capabilities of intelligence to emerge as instrumental byproducts.

**Non-trivial crux:** The claim is *sufficiency in principle*, stated in falsifiable form to invite refutation; it leans entirely on 'sufficiently rich environment' and 'capable enough agent,' and it does not tell you which reward to choose or guarantee that emergence is efficient — so it must be attacked, not protected.
```

## fact-check changes

```plaintext
- Replaced 'seventy years of AI history' and 'a hundred times the compute' with non-fabricated phrasing; the bitter-lesson intuition is about search and learning scaling, not a specific numeric compute multiplier.
- Marked the scaling/bitter-lesson asymmetry explicitly as intuition guiding the hypothesis rather than a proven decisive mechanism, per hindsight/over-claim integrity.
```