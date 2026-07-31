# Pilot CoT 10 — Richard Sutton · empirical_anomaly

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例4：怎么判断一个系统"真在学" → 立下 Verification Principle 铁律

- route: empirical_anomaly  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **209** · assistant(cot) **982** · whole **1248**

- fact-check changes: 3

## user (setup)

```plaintext
You are working on AI systems in the early 2000s, at a moment when large hand-engineered knowledge bases and expert systems remain influential, and even learning systems are routinely seeded with human-supplied knowledge, priors, and hand-labeled representations. A practical, recurring irritation: when such a system misbehaves, nobody can say WHICH piece of its 'knowledge' was wrong, because most of it was never something the system could test for itself. Meanwhile a separate empirical pattern keeps showing up across chess, checkers, speech, and vision benchmarks: carefully hand-crafted domain systems win for a while, then get overtaken by cruder methods that simply learn or search more as compute grows. The field mostly explains this away case by case ('we just needed better features', 'that domain was special'). You are convinced there is a single, sharper question hiding underneath about what it even means for a system to KNOW something. Propose a genuinely novel organizing principle for how to build AI systems that keep improving from experience.
```

## assistant (cot)

```plaintext
Let me start where I always start: not with a theory, but with something I can actually watch happen, repeatedly, that the standard story keeps papering over.

Here is the observation that won't fit. Take any of these hand-built knowledge systems and feed them a situation the designers didn't anticipate. They fail — fine, everyone fails somewhere. But when I ask the concrete diagnostic question, 'which piece of the system's knowledge was wrong, and how would the system itself have found out?', I get nothing. The system can't tell. The designers can't tell without opening it up and re-deriving by hand. The 'knowledge' just sits there, unfalsifiable from the inside. And yet everyone calls it knowledge and builds on top of it.

And there's a second, quieter observation running alongside it. Across chess, checkers, speech, vision — I keep watching the same movie. Meticulous domain-knowledge systems lead, then get overtaken, often by a wide margin, by methods that just learn or search more as compute increases. The community explains each case away locally: better features here, a special domain there. But that's a lot of independent excuses for one very stubborn regularity. When I have to invent a new excuse every time the same thing happens, I should suspect the regularity is the real thing and the excuses are the illusion.

So let me refuse the excuses and let the contradiction force the issue. Put both observations on the table together. The knowledge that we hand in is exactly the knowledge the system cannot verify for itself. And the systems that tend to win long-term are the ones whose competence comes from what compute can amplify — learning and search — not from what we wrote down. Those two aren't separate complaints. They may be the same edge seen twice.

Now the default move here, the comfortable move, is to say: the fix is better knowledge engineering, cleaner ontologies, more careful priors. I don't buy it, and I think the pattern is why. Piling in more human-supplied knowledge can make the system look smarter today while making it harder to correct later, because none of that added knowledge carries its own test. You've increased the pile of things the agent can't check.

So take the agent's own point of view — seriously, all the way down. What does the system actually possess? Only what it can measure from its own experience. Everything else is something I, the designer, imagined and stuffed in. From that seat, the question 'is this knowledge true?' has no meaning unless the agent has a procedure IT can run to decide. That's the leap: don't treat knowledge as content to be inserted; treat it as a claim that must come bundled with a test the agent itself can execute against its own experience. If it can't run the test, it doesn't get to count as knowledge — not in the code, not in my conclusions.

This cuts against the prevailing instinct that 'more knowledge is better.' I'm saying knowledge you can't verify from the inside is dangerous, because it can't be corrected and you can't even locate the fault. It also reframes goals. Don't ask the agent to achieve what it cannot measure — a goal defined only by an outside judge the agent can't perceive produces no learning signal at all.

How do I keep myself honest, and how would I attack this rather than protect it? The sharp test of whether a system genuinely learns from experience: does it get surprised when what happens contradicts its prediction, and does it adjust based on what happened? A system that only reproduces a data distribution won't be surprised by anything outside it and won't self-correct. So the discipline is: get the root predict-and-verify loop right in the smallest reproducible setting where I can watch surprise-and-adjustment actually occur, before I claim anything about scale. And the decision rule when adding any hand-supplied structure: if the agent cannot run a test that would catch that structure being wrong, prefer to leave it out and let the agent learn it — even at a short-term cost — because unverifiable additions are exactly the kind I expect to lose over time.

**Core idea:** An AI system can create and maintain knowledge only to the extent that it can verify that knowledge itself — so every claim, representation, and goal must come with a test the agent can run against its own measurable experience, and anything unverifiable should be neither trusted nor inserted.

**Non-trivial crux:** Verifiability is judged strictly from the agent's own first-person, measurable experience, not by an external human judge; the exception is that during early prototyping or with reliable external supervision you may hang unverified scaffolding temporarily, but it must be flagged as unverified and never allowed to settle silently into 'fact.'
```

## fact-check changes

```plaintext
- Removed asserted empirical claim that learning/search systems win 'by a large margin' as if a settled result, softened to 'often by a wide margin' as an observed tendency.
- Softened over-absolutized claims ('makes it dumber forever', 'worse than nothing', 'the same edge', 'the data is why') into contemporaneous intuition rather than proven fact.
- No anachronistic competitors present; era references (chess, checkers, speech, vision) preserved.
```