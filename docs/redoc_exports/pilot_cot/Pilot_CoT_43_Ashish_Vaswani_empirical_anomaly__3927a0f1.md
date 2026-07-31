# Pilot CoT 43 — Ashish Vaswani · empirical_anomaly

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例3：消融即减法 —— "把部件一件件扔掉看会不会更差，结果反而更好"

- route: empirical_anomaly  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **177** · assistant(cot) **964** · whole **1198**

- fact-check changes: 5

## user (setup)

```plaintext
It's early 2017. We have a working sequence-to-sequence prototype for machine translation that leans entirely on attention instead of recurrence, and it already trains. But the model is a pile of parts: several normalization placements, auxiliary connections, position handling, per-layer structural choices — all inherited from what 'seemed reasonable' or from the recurrent models we grew up with. The dominant way people report architecture progress right now is additive: 'we added module X and got +0.3.' The prevailing intuition is that more machinery, carefully regularized, gets you more. You have fast enough infrastructure to run many train-and-compare cycles quickly. Question: given a complex model that runs and shows promise, how would you actually determine which components are load-bearing versus decorative — and what discipline would you adopt to reach the essential structure? Propose a genuinely novel approach.
```

## assistant (cot)

```plaintext
Let me start from something concrete that keeps nagging me. We were tidying the prototype, and to shave compute we stripped out one of the auxiliary pieces we'd assumed was necessary — and on some runs the quality didn't drop, in a few cases it even edged up. That is not the story I was told to expect. The standard narrative says every one of these components earns its keep; regularizers, extra connections, they're supposed to help or at worst be neutral. But the measurement in front of me contradicts that. A part I 'knew' was helping looks, empirically, like it may be dead weight or even hurting.

Most people would explain this away — noise, a bad seed, 'run it again, it'll regress.' I don't want to explain it away. I want to take the contradiction seriously and let it force the design. If throwing a piece away doesn't make the model worse, then that piece may not be load-bearing — it may be manufacturing overfitting or interference. That's not an anomaly to smooth over; that's a signal telling me where the essential structure might be hiding.

So here's the inversion I want to commit to. The reflexive research move is additive: 'what can I add to go up?' I'll flip the default. My daily action becomes subtractive: throw one component away and ask 'does it actually get worse?' Deletion is the sharpest probe I have, because addition can always fake progress — you can bolt on almost anything, tune around it, and claim a fraction of a point. But when you *remove* something and quality holds or improves, you've learned something concrete: that part was not load-bearing. Subtraction is harder to game than accumulation.

Why does this cut against the 2017 consensus? Because the whole aesthetic of the field is 'I stacked another clever block.' Reporting a win by *removing* recurrence, removing convolution, removing auxiliary machinery — that reads as a confession of having done less, not more. I think that reading is backwards. If I can compress the result into a claim of the form 'you only need X,' that minimalism is more plausibly evidence I found the actual skeleton than evidence I got lazy.

Now, the honesty part — the part that keeps this from being a vibe. Subtractive empiricism only works if the throw-away-and-measure loop is fast and trustworthy. If a single experiment takes a week, I'll never dare delete anything, and I'll rationalize keeping every part 'just in case.' So the real precondition is infrastructure: I invest first in the shortest possible change-run-compare cycle, so that removing a piece and getting a credible comparison is cheap. Only then does deletion become a routine design move rather than a nervous one-off.

And the decision rule has to be stated in advance so I don't fool myself. For each component: remove it, run the comparison under matched conditions. If quality clearly worsens, it's load-bearing — keep it, and now I understand *what* it was carrying. If quality is unchanged or improves, prefer the simpler model — remove it on parsimony grounds rather than because it 'won' by some margin. The one caveat I must respect: before I delete, I ask what side-duty this part quietly performs (word order, numerical stability, gradient scale). If deleting it silently kills a capability, the fix isn't 'add back a big module' — it's the cheapest explicit patch that restores exactly that capability. Otherwise 'simplification' degrades into 'missing function,' and I'll wrongly bury a good direction as 'this approach doesn't work.'

The thing I refuse to do is only ablate additively — 'this trick gave +0.2, keep it' — until nobody can say which parts are structural and which are ornamental. I'd rather run the whole thing as a demolition survey: knock on every wall, keep the ones that hold up the roof, and let the load-bearing minimum reveal the architecture.

**Core idea:** Make ablation the design act, not a post-hoc appendix: default to *removing* each component and measuring whether the model gets worse, treat 'it didn't get worse when I threw it away' as the signal to chase toward a simpler essential structure — with the whole method resting on an infrastructure investment that makes the throw-away-and-measure loop fast.

**Non-trivial crux:** Deletion is sharper than addition only under a fast, trustworthy compare loop AND a pre-committed rule; and before deleting you must identify any side-duty the component secretly performs and restore it with the cheapest explicit patch, or you'll mistake a lost capability for a failed idea.
```

## fact-check changes

```plaintext
- Removed fabricated confirmed outcome ('the BLEU didn't drop... it went up' as an established result); softened to what was observed on some runs and framed as a signal to test, per (a).
- Replaced 'that piece wasn't load-bearing — it was manufacturing overfitting' phrased as certainty with tentative framing consistent with a not-yet-fully-run investigation.
- Translated Chinese words '承重' and '承重墙' to English ('load-bearing', 'load-bearing wall/wall that holds up the roof') per (e).
- Toned down 'Subtraction can't be gamed' to 'harder to game' to avoid over-absolutized claim per (d).
- Adjusted decision-rule language so parsimony (not a numeric win) is the reason to remove, aligning with (a)/(c).
```