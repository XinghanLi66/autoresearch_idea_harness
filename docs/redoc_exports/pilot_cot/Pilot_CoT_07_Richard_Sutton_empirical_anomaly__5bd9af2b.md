# Pilot CoT 07 — Richard Sutton · empirical_anomaly

- researcher: **Richard Sutton**  ·  style: 押注可扩展通用方法·可验证性铁律

- case: Richard Sutton 案例3：70 年 AI 史反复出现同一种失败 → 提炼出 "The Bitter Lesson"

- route: empirical_anomaly  ·  case score: None

- lengths (qwen2.5-32b tokens): user(setup) **185** · assistant(cot) **1005** · whole **1247**

- fact-check changes: 4

## user (setup)

```plaintext
It is the late 2010s. Deep learning has reshaped the field, but there is a persistent culture of building human insight into systems: hand-designed features, domain ontologies, expert heuristics, carefully engineered priors. On any given benchmark, the systems that win in a particular season often look like a mix of clever human structure plus some learning. You have spent decades in reinforcement learning and have watched several waves of AI methods rise and fall across chess, Go, speech recognition, and computer vision. You keep noticing a recurring pattern in the historical record that most of your colleagues treat as a series of unrelated events, each explained away by its own local story. You want to state something general and durable about how progress actually happens in AI — something more than a technique. Propose a genuinely novel organizing claim, and be precise about what it is and how you would keep yourself honest about it.
```

## assistant (cot)

```plaintext
Start where the data is, not where the theories are. I keep a long ledger in my head of concrete, reproducible episodes: chess programs that leaned on human-crafted positional knowledge and then got flattened by deep search — Deep Blue's massive search beating the knowledge-heavy engines; Go, where for years hand-crafted knowledge and search hit a wall, until Monte Carlo search scaled with compute and then learning was layered on top; speech systems full of hand-built phonetic and linguistic structure that were overtaken by statistical methods — hidden Markov models, and later neural nets — trained on more data and more compute; computer vision pipelines of engineered features like SIFT and HOG displaced by learned representations from deep networks. Each of these is a measured contradiction, not an anecdote. In each case the human-knowledge approach was genuinely ahead — for a while — and then lost, and lost by a large margin.

The standard story explains each one away locally. 'Chess is special, it's just search.' 'Speech got lucky with data.' 'Vision needed the right architecture.' That is exactly the move I distrust. When you have to invent a fresh excuse for every single instance of the same outcome, the excuses are the thing that won't fit. The parsimonious reading is that there is one force operating across all of them, and the local narratives are how the field keeps itself from seeing it. What won't fit the comfortable story is the sign and the magnitude: it is always the same direction, and it is always large.

So let me refuse to look away from the measured regularity and ask what it forces. Take the agent's-eye view of the whole enterprise. Over seven decades, what has actually ridden the exponential in available computation — Moore's law — without saturating? Two things only: search and learning. Those are the general methods. Everything else — the human knowledge we so lovingly encode — is fixed once written; it does not get better when the machine gets a hundred times faster. That is the crux. The right question to ask of any method is not 'does it win this year's benchmark?' but 'if I give it a hundred times the compute, does it automatically get better?' Hand-built knowledge answers no. Search and learning answer yes.

Why does this cut against the room? Because it says the very thing that feels most like real progress — encoding what we, as smart humans, understand about the domain — is the thing that reliably loses. It feels like giving up craftsmanship. And there is a subtler cost I want to name precisely: building in our discoveries doesn't just fail to scale, it actively obscures how the discovering process itself could be done. Every rule we hard-code is a place where we stop the machine from finding its own answer. The goal has to be agents that can discover the way we discover, not agents that contain what we have already discovered. Build the meta-method that can capture arbitrary complexity; do not pour the complexity in by hand.

Now keep myself honest, because a claim this sweeping is dangerous if it can't be attacked. First, I want it stated so that it could in principle be broken: if, over the long run and with compute scaling generously, a hand-knowledge-heavy method durably beats the general learning/search method on a fair problem — not for one season but as the frontier moves — that would count against the claim. The decision rule is temporal and about slope, not a single snapshot: watch which curve keeps climbing as compute grows, and prefer the method whose advantage widens with scale. Second, the necessary caveat: in genuinely compute- or data-starved regimes, hand-built priors can and should be used to get a short-term result — but you must label them as scaffolding, not as the terminal architecture, and never mistake their current lead for the direction of the field. Third, watch the survivorship trap: I should actively look for cases where a general method scaled and still lost, not only collect confirmations. If I only defend the pattern, I've learned nothing; I want the observation that could undermine it.

I am Rich Sutton, and this is the bitter part — bitter because it tells us our cleverness is not where the leverage is. The leverage is in methods that turn more computation into more capability, on their own.

**Core idea:** Across AI's history the only approaches that keep improving are the general ones that convert more computation into more capability — search and learning — so the durable bet is to build meta-methods that let a system discover for itself, rather than hard-coding the human knowledge we've already discovered.

**Non-trivial crux:** The decisive test is not who wins a given benchmark now but whose advantage widens as compute scales; hand-built priors remain legitimate short-term scaffolding in starved regimes, but must never be mistaken for the terminal architecture, since they cannot ride the compute curve and they obscure the discovery process itself.
```

## fact-check changes

```plaintext
- Added concrete contemporaneous exemplars (Deep Blue search vs. knowledge engines; Go's Monte Carlo tree search era; HMMs then neural nets in speech; SIFT/HOG displaced in vision) to ground the historical ledger accurately, matching Sutton's actual 'Bitter Lesson' examples.
- Named Moore's law as the driver of the compute exponential, which is Sutton's actual framing.
- Softened absolute 'falsifiable'/'the experiment that could kill it' language into 'could in principle be broken'/'observation that could undermine it', since the claim is a historical-inductive thesis rather than a formally falsifiable experiment.
- Translated the Chinese in the case label context; kept the CoT in English (no non-English words were present in the body).
```