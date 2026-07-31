# Pilot CoT 11 — Richard Sutton · first_principles

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例4：怎么判断一个系统"真在学" → 立下 Verification Principle 铁律

- route: first_principles  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **173** · assistant(cot) **981** · whole **1211**

- fact-check changes: 4

## user (setup)

```plaintext
It's around 2001. AI systems everywhere are stuffed with knowledge: hand-built ontologies, expert rules, human-designed representations, learned parameters, injected priors. When such a system misbehaves, you often cannot say which piece of 'knowledge' is wrong, because most of it was never testable by the system in the first place. Meanwhile the field debates endlessly which knowledge is 'correct'. You suspect the debate is malformed. The question you want to settle is not empirical-first but definitional: given an autonomous agent that must build and maintain its own knowledge over a lifetime, what is the sharpest constraint that MUST hold for any piece of that knowledge to be trustworthy at all? Reason it through, largely on paper, until a principle is nearly forced. State a genuinely novel organizing principle and how you would keep yourself honest about it.
```

## assistant (cot)

```plaintext
Let me refuse, for once, to argue from what works on a benchmark. I want to reason from what MUST be true for an agent that learns from its own experience over a long life. I am Richard Sutton, so I take the agent's first-person view seriously as the only view that is not wishful thinking.

Start with a construction. Imagine an agent embedded in a world, running forever, accumulating a body of 'knowledge' — predictions, representations, rules, whatever. Now ask the invariant question: for any single item of that knowledge, what has to be the case for the agent to keep it, trust it, and correct it? Not for ME to trust it — for the AGENT to. Because I will not always be standing beside it. The agent is on its own.

Here is the forcing step. Suppose an item of knowledge K is such that the agent has NO procedure, runnable from its own sensory experience, that could ever tell K apart from not-K. Then consider two agents identical except one holds K and one holds its negation. By assumption nothing in their stream of experience can ever distinguish them. So K carries no information the agent can act on, cannot be corrected when wrong, cannot be confirmed when right, and cannot be maintained as the world drifts. It is, from the agent's side, indistinguishable from noise it was told to believe. Therefore any K that the agent cannot itself verify is knowledge the agent cannot legitimately hold. This is not an empirical finding; it is almost a definition. An AI system can create and maintain knowledge only to the extent that it can verify that knowledge itself.

Why does the default move fail here? The whole industry's habit is to inject human knowledge because it helps today. But injection smuggles in exactly the K's that fail the test — priors the agent cannot check. They look like competence and are actually unfalsifiable-from-inside baggage. When something breaks, you cannot localize the fault, because those items were never wired to any test the agent runs. That is the trap, and it cuts against the consensus that 'more good knowledge is always better.' I'd put it more sharply as an intuition rather than a proven law: unverifiable knowledge is a liability precisely because it never gets caught when it's wrong — though I should grant the practical exception that injected knowledge can be a useful crutch in the short term, so long as it's understood as scaffolding, not as something the agent has earned.

A corollary falls out immediately and it constrains how I may set goals. A goal or reward the agent cannot MEASURE is a K it cannot verify progress on. So: don't ask the agent to achieve what it cannot measure. Rewards must be functions of the agent's own observable signals, not of some external judge the agent never perceives. Otherwise there is no learning signal the agent can actually use — it cannot tell success from failure on its own.

And the invariant gives me a razor for detecting fake learning. A system that truly holds verifiable knowledge must be capable of being SURPRISED — its own test can come out against its expectation — and of adjusting when it is. If a system can never be surprised by what actually happens and never revises on that basis, it holds no verifiable world-knowledge in that respect. A model of what a person would say is not thereby a model of the world.

Now, how do I keep myself honest instead of falling in love with a slogan? The principle must be attacked, not protected. Two disciplines. First, exploratory scaffolding is allowed to hold unverified assumptions, but they must be explicitly tagged 'unverified' and must never silently harden into asserted fact before a runnable check is attached. Second, methodologically: take a core learning algorithm — a prediction/control rule, general value functions of the kind I'm developing — into the smallest reproducible setting and demand that every claim it makes be one the system itself could run a test on. The decision rule is simple: if an item of knowledge cannot be paired with an agent-runnable verification procedure, it does not enter the system's beliefs and it does not enter my conclusions. Parsimony favors the version that carries only what can be checked.

**Core idea:** The Verification Principle — an agent may hold and maintain a piece of knowledge only insofar as it can, from its own experience, run a procedure to verify that knowledge; unverifiable injected knowledge must be refused, and goals must be measurable by the agent itself.
**Non-trivial crux:** Verifiability is judged from the AGENT's first-person, experience-grounded view, not the designer's; the necessary caveat is that unverified assumptions may serve as explicitly-labeled temporary scaffolding, but must never harden into asserted fact without a runnable test.
```

## fact-check changes

```plaintext
- Softened over-absolutized claim ('More UNVERIFIABLE knowledge is worse than none: it is rot') into an intuition and restored the real exception that injected knowledge can serve as a useful short-term crutch/scaffolding.
- Marked the 'always worse than none' framing as intuition rather than a proven contemporaneous law (hindsight-as-original guard).
- Qualified the 'literally no learning signal' phrasing to 'no learning signal the agent can actually use' for accuracy.
- Toned down the general-value-functions reference to reflect it as work Sutton was developing at the time rather than an established named toolset.
```