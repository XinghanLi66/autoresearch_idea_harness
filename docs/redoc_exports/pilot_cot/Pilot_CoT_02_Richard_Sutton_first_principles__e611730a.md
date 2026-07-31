# Pilot CoT 02 — Richard Sutton · first_principles

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例1：RL 在 80 年代被学界抛弃 → 从动物心理学"翻译"出 TD Learning

- route: first_principles  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **160** · assistant(cot) **1028** · whole **1245**

- fact-check changes: 5

## user (setup)

```plaintext
It is the mid-1980s. You are working on how an agent can learn from experience to predict and act, but you keep hitting the temporal credit assignment problem: reward arrives only at the end of a long sequence of actions, and it is unclear how to attribute that delayed reward back to each earlier prediction. The dominant supervised-learning framing assumes an external teacher who supplies, for each input, the correct target. That framing sits badly with you: in a real stream of experience there is no teacher standing by with labels at every step. Reinforcement learning is out of fashion; symbolic AI and supervised methods rule. Reason from first principles about what a learning rule for prediction MUST look like if it is to run purely inside an agent's experience stream, and propose a genuinely novel construction.
```

## assistant (cot)

```plaintext
Let me not start from what's fashionable. Let me start from what must be true, and push until the rule is almost forced.

Take the agent's own point of view. All it has is a stream: sensations, actions, and occasionally a scalar signal. Suppose it wants to predict some quantity of future interest — call it the value of a state, the amount of reward it will eventually accumulate from here. I, Richard Sutton, want a rule that updates that prediction. The question is: what is the update allowed to depend on?

Here is the intuition I keep returning to: an agent can create and maintain knowledge to the extent that it can itself check that knowledge against its own experience. So the target of a prediction ought, ideally, to be something the agent can eventually verify on its own. The supervised framing sits badly with this — it assumes an external teacher hands over the true value at each step. But the true long-run value is exactly what the agent does not know; if it knew it, it wouldn't need to learn it. So a rule that waits for the final, fully-realized return has two defects for my purposes: it requires waiting to the end of the sequence, and it treats the target as an externally given label. Neither is intrinsic to the experience stream. (I should be careful: the full-return method — essentially a Monte Carlo estimate — is a perfectly valid and unbiased learning rule; my objection is that it is not incremental and not self-contained.)

So let me ask the deductive question: what can the agent measure at time t, before the episode ends? It can measure (a) whatever immediate signal just arrived, and (b) its own prediction at the next step. That's it. That's the whole ledger of quantities available in a single step.

Now force consistency. If my prediction at t is meant to estimate the same future quantity that my prediction at t+1 estimates — minus whatever reward was collected in between — then these two adjacent predictions must obey a relation. The prediction now should equal the reward just received plus the prediction next. If it doesn't, the discrepancy between them is not noise; it is information the agent generated itself. That discrepancy — the difference between temporally adjacent predictions — is a self-produced error signal. No teacher required. The prediction is its own evolving target.

This is the leap: use the difference between successive predictions to drive learning. The learning signal is endogenous to experience. Each prediction is both a guess and, one step later, part of the check on the previous guess. Every step carries its own correction; the surprise is literally the temporal difference, and the agent adjusts based on what actually happens next.

Why does this cut against the consensus? Because the field assumes learning needs supervised targets, and here I am claiming a prediction can be trained against another of the agent's own predictions — bootstrapping on itself — and that this is a natural construction once you take the agent's-experience view. It looks circular to the supervised mindset. It need not be: the immediate reward at each step injects real ground truth incrementally, and the chain of adjacent checks is anchored on actual outcomes. Credit can propagate backward one adjacency at a time, without storing the whole trajectory or waiting for the end.

I trust this partly because it echoes animal-learning psychology, where classical conditioning models describe an organism updating on the change between successive expectations — the Rescorla-Wagner idea taken across time. That the field's fashion abandoned reinforcement learning is not an argument against it.

How do I keep myself honest? I will not assert it wins in advance. The decision rule: compare learning that waits for the full realized return (Monte Carlo) against learning driven by the adjacent-prediction difference, on the same experience, and see how they trade off — the temporal-difference rule may learn faster from less data because it updates every step and exploits the consistency relation, but I should verify this empirically rather than assume it. If the endogenous, self-correcting rule does not clearly hurt prediction quality, I prefer it — on the principle that it is incremental and needs no external teacher. And I must probe the honest caveat: bootstrapping on your own estimates introduces bias, and with function approximation can in principle chase a biased or unstable fixed point; I need to check whether the immediate-reward anchoring is enough to keep the predictions consistent, and be explicit where it might not be.

**Core idea:** Train each prediction against the reward-plus-next-prediction rather than against an externally supplied target, so the learning signal is the difference between temporally adjacent predictions — an error the agent generates within its own experience stream, propagating delayed credit one step at a time.

**Non-trivial crux:** The target is the agent's own later prediction (bootstrapping), which is workable only because each step injects real immediate reward to anchor the chain; without that anchoring the self-referential update introduces bias and could converge to an unreliable fixed point, so the immediate-signal grounding is not optional.
```

## fact-check changes

```plaintext
- (a) Removed the framed claim that the self-produced signal is 'self-verifiable' and the assertion that TD is 'the only construction consistent with' an invariant; softened the verifiability principle to intuition per (c).
- (a) Reworded the honesty section so the comparison against Monte Carlo returns is a genuine test with a decision rule rather than an implied outcome; added the plausible-but-to-be-verified rationale (faster/incremental) without asserting a numeric winner.
- (d) Restored caveat that the full-return (Monte Carlo) method is a valid, unbiased rule and that bootstrapping introduces bias (not merely 'in principle chase'), and flagged instability under function approximation.
- (c) Marked the 'agent can only maintain knowledge it can verify' framing as intuition rather than a decisive contemporaneous invariant.
- Grounded the psychology link concretely in classical conditioning / Rescorla-Wagner rather than vague 'obvious to ordinary people'.
```