# Pilot CoT 09 — Richard Sutton · analogy_transfer

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例3：70 年 AI 史反复出现同一种失败 → 提炼出 "The Bitter Lesson"

- route: analogy_transfer  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **153** · assistant(cot) **914** · whole **1124**

- fact-check changes: 3

## user (setup)

```plaintext
You've spent decades watching the same movie replay across AI subfields: chess, Go, speech recognition, computer vision. Each time, researchers pour human domain knowledge into a system, get a satisfying short-term lead, and each time a more general method that just consumes more computation eventually overtakes it — by a wide margin. The field keeps forgetting this. It's early 2019. You want to write down, as sharply as possible, a general principle that predicts which research bets pay off over the long run. But a slogan isn't enough; you need a transferable mechanism that tells a working researcher what to build and what NOT to build. Propose the core idea, drawing on a mechanism you understand from outside the current deep-learning fashion.
```

## assistant (cot)

```plaintext
Let me not start from the deep learning excitement of the moment. Let me start from something I understand cold from another vantage: technology history — Moore's law, and the way an exponentially cheapening resource reorganizes every discipline that touches it. That is my source mechanism. When one input gets dramatically and reliably cheaper year after year, the winning designs are the ones that let you *pour more of that cheap input in* and automatically get more output. Designs that instead lean on a fixed, expensive input — human labor, hand-tuned craft — plateau, because the thing they depend on doesn't scale.

Now map that over onto AI, explicitly. The cheap-and-exponentially-growing input is computation. So the question I want to ask of any method is essentially the same one in disguise: does this method *convert additional computation into additional capability, automatically, without a human in the loop?* That is the analogy. Let me be precise about what transfers and what breaks.

What transfers cleanly: the asymmetry of scaling. Over 70 years, only two families of methods have shown they can absorb arbitrarily more computation and keep getting better — search and learning. Everything else, at bottom, is a human deciding in advance what the system should know. Human insight does not have a Moore's law. So the historical pattern isn't luck; it's the same dynamic — the input that scales wins, by a large margin, eventually.

What breaks in the analogy — I have to be careful here. The short-term picture is genuinely inverted. Hand-built domain knowledge really does win *now*, on this quarter's benchmark, in the low-compute regime. So the analogy fails if you read it as 'human knowledge is useless.' It isn't useless; it's a loan against the future. The correct reading is about the *long-run limit* and the *direction of the bet*, not about who leads today.

Why this cuts against the consensus of the moment: nearly every research program earns its keep by injecting cleverness — features, priors, ontologies, architecture tricks that encode what we already figured out. It's satisfying, it's publishable, it 'makes sense.' My claim is that this satisfaction is exactly the trap. Building in what we've already discovered *makes it harder to see how the discovering process itself works* — and the discovering process is the only thing that scales. So the goal shifts: we don't want systems that *contain* what we have found; we want systems that can *find* things the way we can. Build only the meta-methods that can capture arbitrary complexity — not the complexity itself.

The move, then, is to promote the observation from a war story into a *decision rule* a researcher can apply before running anything: every time you reach to hand-code a rule, stop and ask 'can the system learn this instead, given more compute?' And approximate the *solution*, not the *problem* — don't quietly simplify the task until it fits your favorite algorithm; that solves a distorted fantasy.

How do I keep myself honest, given I can't run '70 more years' to confirm? This is a claim about limits, and limits are always at risk of being unfalsifiable hindsight. So I hold the necessary caveats openly: in a genuinely compute-limited niche, hand-priors are legitimate triage — just never mistake the crutch for the destination. And I state the decision rule so it can be *attacked*: for any concrete method pair, the test is whether the more general, compute-leveraging variant closes the gap and passes the hand-built one as compute grows. If a hand-built system keeps its lead as you scale compute by orders of magnitude, my principle is wrong for that domain. That's the experiment I'd invite people to run against me, not for me.

**Core idea:** Ask of every method 'if I give it 100x compute, does it automatically get better?' — bet on the general methods (search and learning) that convert cheap, exponentially growing computation into capability, and build only meta-methods that discover, rather than hard-coding the knowledge we've already discovered.

**Non-trivial crux:** The claim is about the long-run scaling limit and the direction of the bet, NOT about who wins today — hand-built knowledge genuinely leads in the low-compute regime, so it's legitimate short-term triage; the error is mistaking that temporary lead for the final architecture.
```

## fact-check changes

```plaintext
- Translated the Chinese phrase in the case label context; setup/cot were already in English.
- Softened the 'economics' framing: Sutton's actual argument grounds the lesson in the cost trend of computation (Moore's law) and AI history, not an explicit economics-of-input-substitution model — recast as technology history to avoid hindsight-as-original over-framing.
- Removed 'the creative leap' self-aggrandizing phrasing, replaced with neutral 'the move' (voice-preserving, integrity-neutral).
```