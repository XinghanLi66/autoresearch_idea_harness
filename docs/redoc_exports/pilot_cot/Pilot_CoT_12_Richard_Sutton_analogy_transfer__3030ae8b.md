# Pilot CoT 12 — Richard Sutton · analogy_transfer

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例4：怎么判断一个系统"真在学" → 立下 Verification Principle 铁律

- route: analogy_transfer  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **154** · assistant(cot) **980** · whole **1191**

- fact-check changes: 2

## user (setup)

```plaintext
It is around 2001. AI systems are stuffed with knowledge and representations, and much of it is hand-supplied by designers. There is no principled way to say which of that knowledge the system can actually trust, or to tell whether a learning system is genuinely learning from experience versus merely reproducing patterns its designers baked in. Meanwhile, engineering practice everywhere else — software, control, science — has long relied on some notion of testing claims before depending on them. You want a single, sharp organizing principle for building AI agents that learn open-endedly and keep their own knowledge honest. Propose a genuinely novel foundational principle, being explicit about what mechanism you are borrowing from another field, what transfers cleanly, and what breaks when you carry it into AI.
```

## assistant (cot)

```plaintext
Let me start where I trust myself most: from the agent's own point of view. An agent only has what it can measure and check for itself. Everything else — every fact I as the designer type in — is my hallucination projected onto it. So the real question is not "how do I give the agent good knowledge?" but "how does the agent decide, on its own, which of its knowledge is any good?"

And here is where I want to borrow a mechanism from outside AI rather than invent one from scratch. Look at engineering and at the philosophy of science. In software engineering nothing is trusted until it is tested; a claim you cannot exercise with an automated test is not part of the shipped system, it is a hope. In empirical science, Popper's whole move is that a statement earns its status as knowledge only if it is falsifiable — only if there is a procedure that could show it wrong. That is the source mechanism: knowledge-as-testable-claim. Correctness is not conferred by the authority of whoever asserted it; it is conferred by the existence of a check that the system itself can run.

Now let me be precise, because a sloppy analogy poisons everything. What transfers? The core discipline: attach to every assertion a verification procedure, and refuse to depend on assertions that lack one. What breaks? In software the test is written and run by a human outside the system; in science the falsifying experiment is performed by the community of scientists. If I import it that way, I've smuggled the designer back in — I'd be verifying the agent's knowledge for it. That is exactly the failure I'm trying to kill. So the transfer has to be tightened: the verifier cannot be me, and cannot be any external judge the agent can't invoke. The test must be one the agent can itself run, from its own sensorimotor experience, and the success criterion must be built only out of signals the agent can actually measure. Don't ask the agent to achieve what it cannot measure. That is the part of the analogy that does NOT carry over unchanged, and it's the whole game.

Why does this cut against the fashion? Because the field's instinct is the opposite: hand-code more knowledge, richer ontologies, expert rules — anything that makes the system look smart today. My principle disqualifies most of that in one stroke: if the agent can't test it, it shouldn't be trusted, no matter how true I'm sure it is. That feels like throwing away useful information. It is not. Unverifiable knowledge is knowledge you can never debug — when the system fails you can't even say which planted "fact" was wrong. And it fits my longer intuition: I suspect that in the long run only search and learning scale with computation, and both are precisely processes that generate and check candidate answers against experience. A verification-first agent is one whose knowledge-making loop is intrinsically scalable, because bigger compute means more checking, not more of my typing. I'll hold that as a conjecture, not a settled result.

How do I keep myself honest here — practice what I preach? I will not assert this principle works; I'll state the decision rule it implies and how I'd test it. First: take a root learning algorithm — prediction and control expressed as value-like quantities grounded entirely in the agent's own observation stream — and get it right in the smallest reproducible setting before any talk of scale, staying honest about whether it scales rather than assuming it. Second, a diagnostic that follows directly from the principle: a system that truly learns from experience must be able to be surprised — its own prediction fails against what actually happens — and must adjust based on what happens. If a system is never surprised and never revises, it is reproducing a distribution, not verifying anything against the world. So the rule is: for any component I'm tempted to hand-code, ask whether the agent could instead learn-and-verify it itself; if it can, don't hard-code it. The only carve-outs are scaffolding — explicitly flagged as unverified, and never allowed to settle silently into "fact."

**Core idea:** Import the engineering/Popperian discipline that a claim counts as knowledge only if it comes with a test that could refute it — but retarget the tester onto the agent itself: an AI system may create and keep knowledge only to the extent that it can verify that knowledge from its own measurable experience.

**Non-trivial crux:** The analogy transfers the demand for a falsifying check but must break on WHO runs it — external human/community verification would just reinstate the designer; the verifier and the success signal must be things the agent can invoke and measure itself, otherwise the principle collapses into the very hand-supplied, untestable knowledge it forbids.
```

## fact-check changes

```plaintext
- (c) Marked the 'only search and learning scale with computation' claim as an intuition/conjecture rather than an established framing (the crisp 'bitter lesson' formulation postdates 2001 and was not the decisive contemporaneous argument).
- (a) Softened 'and test it' to 'and how I'd test it' to avoid implying experiments were already run at that moment.
```