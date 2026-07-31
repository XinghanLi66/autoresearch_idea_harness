# Pilot CoT 05 — Richard Sutton · first_principles

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例2：能力太多、公式化各异 → 抛出 "Reward is Enough" 极简统一假设

- route: first_principles  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **162** · assistant(cot) **1000** · whole **1219**

- fact-check changes: 3

## user (setup)

```plaintext
The field carves intelligence into a growing list of separate faculties — perception, language, knowledge, planning, imitation, social skill, generalization — and hands each its own bespoke problem formulation, its own objective, its own machinery. The pile grows, and with it the engineering weight and the theoretical debt: nobody can say how these faculties relate, or where any of them comes from. You are convinced this stacking of special cases is a symptom, not a solution. Rather than adding one more formulation for the next capability, ask whether there is a single, minimal principle from which all of these must follow. State the boldest, simplest, most falsifiable hypothesis you can defend by deductive argument about what could, in principle, be sufficient to give rise to all of intelligence — before appealing to any experiment.
```

## assistant (cot)

```plaintext
Let me reason from what must be true, on paper, before I let any benchmark tell me anything.

Start with a construction. Put an agent inside an environment. Strip away everything the designer wishes were there and keep only what the agent can actually touch: a stream of sensations, actions it can emit, and consequences that return through the same stream. That is the whole ledger. Anything I want the agent to acquire must be expressible as some function computed over this experience, because there is nothing else. This is my agent-first-person invariant, and I hold to it: I refuse to smuggle in any quantity the agent cannot itself measure.

Now the deductive move. Consider a single scalar signal — a reward — defined on that experience stream, and consider an agent whose one directive is to maximize its cumulative value over time. Ask: what capabilities would be *implied* by doing that well in a sufficiently rich world? I want to argue by entailment, not by luck.

Walk the list the field keeps splitting apart. To maximize reward across a long horizon in a complex environment, the agent must anticipate consequences of actions before taking them — that pushes it toward a predictive model. Prediction over a partially observed world pushes toward internal state that summarizes the past usefully — that is perception and memory, arising not as a separate module but as whatever compression best serves the return. If the environment contains other agents whose behavior affects the reward, then modeling and influencing them is not optional decoration — social intelligence would be entailed. If symbols in the environment carry information that changes the achievable return, then representing and manipulating them is compelled — language, again, as a servant of the objective, not a faculty bolted on. Generalization is forced because no finite agent can enumerate a rich world; it must reuse structure to keep earning reward under novelty.

Each of these I reach by asking "what is the maximizing agent *obliged* to do here?" I want to be careful: these are arguments of plausibility and entailment, not yet demonstrated facts — the construction motivates the claim, it does not prove it. Stated as a conjecture: reward maximization, in a sufficiently rich environment, may be *enough* to summon the whole zoo of abilities as byproducts. One objective, not ten formulations.

Why does this cut against consensus? Because the consensus bets that a human, hand-designing the right formulation per capability, beats the general method. My reading of the field's history says otherwise: across the decades the methods that scale with computation — search and learning — have tended to overtake hand-crafted knowledge. Adding one bespoke objective per faculty looks like the losing side of that bet. And it obscures the very thing I care about: the discovering process itself. If I hard-code a capability, I can no longer see how it could have been *discovered*.

But a construction is not a proof of sufficiency in the real world, and I must be honest about the gap. "Sufficiently rich environment" and "sufficiently powerful learning" are load-bearing conditions; I cannot let them quietly become escape hatches. So two disciplines. First: the hypothesis is worth stating only if it is falsifiable — I should state it publicly, in its strongest, most attackable form, and invite the field to break it, not to admire it. A unified theory that cannot be attacked is worthless. Second, the decision rule: the claim is meant to *lose* if someone exhibits a capability that provably cannot arise from any reward-maximization in any rich-enough environment. That is the test I hand my critics.

One caveat I keep, because my own invariant demands it: the reward must be a signal the agent can actually measure from its own experience. A reward only an external judge can evaluate would break the argument — there would be no learnable signal, and "enough" would be a fiction. My own rule stands: don't ask the agent to achieve what it cannot measure.

So the leap is not a new mechanism. It is a refusal to keep adding formulations, replaced by a single minimal one — and the willingness to bet, and to be proven wrong, that maximization does the rest.

**Core idea:** All the seemingly distinct abilities of intelligence — perception, knowledge, language, social skill, generalization — can be understood as byproducts that plausibly emerge when a single agent maximizes one scalar reward in a sufficiently rich environment; so seek one minimal objective, not a formulation per faculty.

**Non-trivial crux:** The sufficiency conjecture rides entirely on two load-bearing conditions — a rich-enough environment and a reward the agent can measure *from its own experience*; if the reward is only evaluable by an external judge, there is no learnable signal and the argument collapses. It must be stated in falsifiable form and offered up to be attacked, not defended.
```

## fact-check changes

```plaintext
- (a/c) Softened deductive 'necessity/entailed/forces' language where it overstated a proof; reframed the sufficiency statement as an explicitly falsifiable conjecture the researcher offers to be broken, matching the actual 'Reward is Enough' hypothesis framing rather than a demonstrated result.
- (d) Recast the 'seven decades... overtake by a large margin' claim as a general historical tendency rather than an absolute quantified outcome, and made the entailment arguments read as plausibility/motivation rather than settled fact.
- (e) Removed reference to 'Sutton's rule' in third person / normalized voice to first-person self-reference; all text already English.
```