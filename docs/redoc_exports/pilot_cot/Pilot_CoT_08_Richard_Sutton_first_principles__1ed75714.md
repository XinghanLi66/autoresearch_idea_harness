# Pilot CoT 08 — Richard Sutton · first_principles

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例3：70 年 AI 史反复出现同一种失败 → 提炼出 "The Bitter Lesson"

- route: first_principles  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **204** · assistant(cot) **988** · whole **1249**

- fact-check changes: 3

## user (setup)

```plaintext
It is early 2019. You have spent decades watching the same pattern repeat across the history of AI: chess, Go, speech recognition, computer vision. In each domain, researchers first build systems by hand-encoding human domain knowledge — opening books, phonetic features, hand-crafted vision pipelines — and those systems look elegant and win in the short term. Then, again and again, they get overtaken, by large margins, by comparatively 'dumb' general methods that simply scale with more computation: massive search, massive learning from data. And the community keeps failing to internalize this; every generation re-learns it painfully. You want to state something more durable than another benchmark result or another architecture. You want to reason, almost from first principles, about what MUST be true over the long run for any method that competes on a substrate whose only exponentially growing resource is computation — and to turn that into an operational rule researchers can actually use when choosing directions. Produce the core idea.
```

## assistant (cot)

```plaintext
Let me not start from any experiment. Let me start from what MUST be true, and reason until the conclusion is nearly forced.

Take the agent's first-person view, but abstract it to the level of the whole enterprise. Over the relevant horizon, what is actually growing without bound? Not human insight — a human researcher's ability to hand-craft knowledge is roughly a constant per generation, maybe growing linearly with the number of people. What grows exponentially is computation. That is the one asymmetry I can rely on with near-certainty, because it is Moore's-law-like and holds across decades regardless of any particular AI fashion. So any comparison between 'method A leverages human knowledge' and 'method B leverages computation' is, in the limit, a comparison between a bounded resource and an unbounded one.

Now the invariant. Ask of any method: as I multiply available computation by 100, by 10,000, does the method automatically get better without me doing more work? For a hand-engineered knowledge system, the answer is essentially no — extra flops don't turn my hand-written heuristics into better heuristics; the ceiling is set by what I personally thought of. For a method built on search or learning, the answer is yes — more computation directly buys deeper search or more capacity trained on more experience. That single property partitions all methods into two classes, and it partitions them cleanly, on paper, before I run anything.

Which general methods actually possess this scaling property? I want to be careful and near-exhaustive here rather than hand-wavy. Scan seventy years. The things that kept winning by large margins all reduce to two primitives: search (spend computation exploring possibilities at decision time) and learning (spend computation extracting structure from experience). I cannot construct a third general primitive that turns raw computation into competence without human labor in the loop. So the class of 'methods that ride the one unbounded resource' is essentially {search, learning}. That's not an observation about a dataset; it's almost a closure argument about what computation alone can be spent on.

So the default move — inject human domain knowledge to win the current benchmark — is a bet that the bounded resource beats the unbounded one over the long run. In the short run it can and does win; that exception is real and I must keep it. But structurally, the historical pattern says it tends to lose asymptotically, and by a large margin, because the opponent's budget grows without limit and mine does not.

Now the leap, and it cuts against the consensus, because the consensus prizes exactly the elegant knowledge-engineering that wins papers today. The leap is to stop treating this as a collection of war stories and state it as a meta-level design intuition: build in only the meta-methods that can find and capture arbitrary complexity — search and learning — and refuse to hard-code the content of what has been discovered. Because encoding my own discoveries doesn't just cap the ceiling; it actively obscures how the discovering process itself could work, which is the only thing that scales. The goal is agents that can discover as we do, not agents that contain what we have discovered.

One more constraint to keep myself honest, in my own idiom: approximate the solution, not the problem. The tempting cheat is to simplify the task until some pet algorithm fits it — but then you've bet on human knowledge again, smuggled in as a distorted problem.

How do I keep this falsifiable rather than a slogan? It is not proven by a single run; it is a claim about limits. The decision rule I can actually apply: for any proposed component, ask whether 100x computation makes it better with no further human authoring. If yes, keep it; if it leads only because I hand-tuned it, treat that lead as temporary scaffolding, label it as such, and do not mistake present superiority for correct direction. The honest test over time is trajectory under scaling, not the current leaderboard — and if a scalable general method ever fails to eventually match a knowledge-heavy one under vastly more compute, that would count against the claim.

**Core idea:** Over the long run only general methods that turn the one unbounded resource — computation — into competence, namely search and learning, tend to win, and by a large margin; therefore build in the meta-methods that discover and capture complexity, never the discovered content itself.

**Non-trivial crux:** The argument is a limiting/asymptotic one about a bounded resource (human hand-crafting) versus an unbounded one (computation) — so hand-injected knowledge genuinely can and does win in the short term or under hard compute limits, and that exception must be kept; the lesson is about direction and the limit, not about any single present-day benchmark.
```

## fact-check changes

```plaintext
- Removed over-absolutized 'provably a bet' and 'must lose asymptotically' — softened to the historical/empirical tendency the essay actually rests on, since it is a pattern-based argument, not a proof.
- Changed 'meta-level design law' and 'law' to 'meta-level design intuition'/'lesson' to avoid framing it as a proven law rather than an inductive generalization.
- Adjusted core-idea wording ('can win' -> 'tend to win') to match the inductive, non-proof character of the claim.
```