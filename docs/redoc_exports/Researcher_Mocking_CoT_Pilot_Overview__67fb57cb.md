# Researcher-Mocking CoT — Pilot Overview

**69 chain-of-thought samples** distilled from **5 researchers** (pilot). Each sample teaches a model to reconstruct, in first person, how a top researcher reasoned to a real innovation — the creative idea and its non-trivial core stated clearly, and how they got there, with NO implementation details.

## Workflow — how each researcher-mocking prompt is constructed

1. **Pool crawl.** From the RedDoc 蒸馏计划 roster, for each researcher we pull their 第一步 case subdocs (①problem → ②idea → ③ablation → ④reflection, with quality scores) and their 第二步 Skills-v1 fingerprint.

2. **Synthesis (opus-4.8, call 1).** For each case we generate several CoTs, one per _reasoning route_ (empirical-anomaly / first-principles / analogy-transfer). The model writes a first-person trace in the researcher's voice (adaptive shape — no fixed template), plus a leak-free setup problem statement. It must name the researcher, omit implementation details, and end with two anchors: **Core idea** and **Non-trivial crux**.

3. **Fact-check (opus-4.8, call 2).** A separate pass revises the draft for factual/historical integrity — removing fabricated outcomes, anachronistic method names, hindsight-as-original, and restoring necessary caveats — using the model's own knowledge (not the source notes). Every edit is logged per record.

4. **SFT triple.** Each record becomes [system (researcher-reasoning instruction), user (setup), assistant (cot)].

See the **Raw construction prompts** subpage for the exact system/user prompts.

## Length stats (qwen2.5-32b tokens)

| segment | avg | min | max |
| --- | --- | --- | --- |
| user (setup) | 186 | 145 | 257 |
| assistant (cot) | 1100 | 914 | 1363 |
| whole (system+user+assistant) | 1343 | 1124 | 1633 |

All well within the Qwen2.5-32B context (32k native; SFT seq-len 8–16k).

## Coverage

| researcher | CoTs |
| --- | --- |
| Richard Sutton | 12 |
| Andrej Karpathy | 9 |
| 何恺明 Kaiming He | 15 |
| Ashish Vaswani | 15 |
| 苏剑林 Jianlin Su | 18 |

## Pages

_Subpage links are appended below after creation._

## Pages (links)

- [Raw construction prompts (synthesis + fact-check)](https://docs.xiaohongshu.com/doc/abe1b9a2f6cd920e8777137626c60358)

- [Pilot CoT 01 — Richard Sutton · empirical_anomaly](https://docs.xiaohongshu.com/doc/1da243254d31de85b228bb5f69f7df8a)

- [Pilot CoT 02 — Richard Sutton · first_principles](https://docs.xiaohongshu.com/doc/e611730ad421733f050266dc3bd0a81b)

- [Pilot CoT 03 — Richard Sutton · analogy_transfer](https://docs.xiaohongshu.com/doc/2d6dfdfc6aad3050cb88fa5a432ab31c)

- [Pilot CoT 04 — Richard Sutton · empirical_anomaly](https://docs.xiaohongshu.com/doc/11355e2b59a72702cf8662a2880078ab)

- [Pilot CoT 05 — Richard Sutton · first_principles](https://docs.xiaohongshu.com/doc/d37bbc6b359e3ff557aff93075a26bf3)

- [Pilot CoT 06 — Richard Sutton · analogy_transfer](https://docs.xiaohongshu.com/doc/ae60a28d8f82635361b9ddc1f433e8b3)

- [Pilot CoT 07 — Richard Sutton · empirical_anomaly](https://docs.xiaohongshu.com/doc/5bd9af2bbfe1f78a999991fef6b6ac11)

- [Pilot CoT 08 — Richard Sutton · first_principles](https://docs.xiaohongshu.com/doc/1ed75714482eed120078a78cca54745b)

- [Pilot CoT 09 — Richard Sutton · analogy_transfer](https://docs.xiaohongshu.com/doc/53ba782fc8e8ba35b208d40064ca41b7)

- [Pilot CoT 10 — Richard Sutton · empirical_anomaly](https://docs.xiaohongshu.com/doc/d0635dfac65e56fd7add423dfa2e51fe)

- [Pilot CoT 11 — Richard Sutton · first_principles](https://docs.xiaohongshu.com/doc/083b0c94a096eca3702396c9780ea3c8)

- [Pilot CoT 12 — Richard Sutton · analogy_transfer](https://docs.xiaohongshu.com/doc/3030ae8be7de2203220195a6f0839dda)

- [Pilot CoT 13 — Andrej Karpathy · empirical_anomaly](https://docs.xiaohongshu.com/doc/6ec35e1d58693d99331bd7680a90845a)

- [Pilot CoT 14 — Andrej Karpathy · first_principles](https://docs.xiaohongshu.com/doc/0366afc901ef0975c71eb73de218b6b6)

- [Pilot CoT 15 — Andrej Karpathy · analogy_transfer](https://docs.xiaohongshu.com/doc/6f4b09ee1ec2586bf296ff301c3568ab)

- [Pilot CoT 16 — Andrej Karpathy · empirical_anomaly](https://docs.xiaohongshu.com/doc/9a050e8eb871560d60ddd4f8d012356e)

- [Pilot CoT 17 — Andrej Karpathy · first_principles](https://docs.xiaohongshu.com/doc/17f592202dc4cff087b098912b23b3a9)

- [Pilot CoT 18 — Andrej Karpathy · analogy_transfer](https://docs.xiaohongshu.com/doc/7f1bb2bf29555f3aa9faa5243bc60b0d)

- [Pilot CoT 19 — Andrej Karpathy · empirical_anomaly](https://docs.xiaohongshu.com/doc/15b537c08c1d3f704f4345501be3e55d)

- [Pilot CoT 20 — Andrej Karpathy · first_principles](https://docs.xiaohongshu.com/doc/324c6755cc49933e55603a108d2f2bb8)

- [Pilot CoT 21 — Andrej Karpathy · analogy_transfer](https://docs.xiaohongshu.com/doc/0079b7fa5377ca3c59b2e1bb3ce0ed75)

- [Pilot CoT 22 — 何恺明 Kaiming He · empirical_anomaly](https://docs.xiaohongshu.com/doc/3da081ac995dcfebc05ae18251175292)

- [Pilot CoT 23 — 何恺明 Kaiming He · first_principles](https://docs.xiaohongshu.com/doc/a70425f0866b1906c6e63d14b26abab3)

- [Pilot CoT 24 — 何恺明 Kaiming He · analogy_transfer](https://docs.xiaohongshu.com/doc/3e77afd3513bcb77cf4f2c9eff697171)

- [Pilot CoT 25 — 何恺明 Kaiming He · empirical_anomaly](https://docs.xiaohongshu.com/doc/48f112df8dcbb410844e704a1ecb189e)

- [Pilot CoT 26 — 何恺明 Kaiming He · first_principles](https://docs.xiaohongshu.com/doc/c53f3ad0730019c162ab0b56451aa06d)

- [Pilot CoT 27 — 何恺明 Kaiming He · analogy_transfer](https://docs.xiaohongshu.com/doc/d56437e4f65a1387ea1cbfbf0af29371)

- [Pilot CoT 28 — 何恺明 Kaiming He · empirical_anomaly](https://docs.xiaohongshu.com/doc/5ab1795474c01899e811cb661f15e45c)

- [Pilot CoT 29 — 何恺明 Kaiming He · first_principles](https://docs.xiaohongshu.com/doc/1506f11c0b52622b857abebdd2b60f19)

- [Pilot CoT 30 — 何恺明 Kaiming He · analogy_transfer](https://docs.xiaohongshu.com/doc/8090df3d30d1c1470ec9162aefed7fdc)

- [Pilot CoT 31 — 何恺明 Kaiming He · empirical_anomaly](https://docs.xiaohongshu.com/doc/b47bd20b5d83b1ea792b5cb70693c35f)

- [Pilot CoT 32 — 何恺明 Kaiming He · first_principles](https://docs.xiaohongshu.com/doc/9608dd626d75626ea17fe1e875475f92)

- [Pilot CoT 33 — 何恺明 Kaiming He · analogy_transfer](https://docs.xiaohongshu.com/doc/f09fcea96ddb88e1e862bc089156794e)

- [Pilot CoT 34 — 何恺明 Kaiming He · empirical_anomaly](https://docs.xiaohongshu.com/doc/21dbb08dbc421b9dbe3a4e27c0a4bf5f)

- [Pilot CoT 35 — 何恺明 Kaiming He · first_principles](https://docs.xiaohongshu.com/doc/434f03702806a7f291ac78f93856b146)

- [Pilot CoT 36 — 何恺明 Kaiming He · analogy_transfer](https://docs.xiaohongshu.com/doc/de495c506459df1ef9196e17c016c10b)

- [Pilot CoT 37 — Ashish Vaswani · empirical_anomaly](https://docs.xiaohongshu.com/doc/7a581bef5fea829d965bdf7ea23e1802)

- [Pilot CoT 38 — Ashish Vaswani · first_principles](https://docs.xiaohongshu.com/doc/0b1399cf2b1937c8e3cd9eb5a7de07fe)

- [Pilot CoT 39 — Ashish Vaswani · analogy_transfer](https://docs.xiaohongshu.com/doc/baee979395cc1eeaa8a3b5494a7b146d)

- [Pilot CoT 40 — Ashish Vaswani · empirical_anomaly](https://docs.xiaohongshu.com/doc/34fcb6cd247ec704315449be87025c89)

- [Pilot CoT 41 — Ashish Vaswani · first_principles](https://docs.xiaohongshu.com/doc/ac873834a1623f28753c832747b36987)

- [Pilot CoT 42 — Ashish Vaswani · analogy_transfer](https://docs.xiaohongshu.com/doc/f3bcbe8d2b12cec75b9e64ad04c6f085)

- [Pilot CoT 43 — Ashish Vaswani · empirical_anomaly](https://docs.xiaohongshu.com/doc/3927a0f13f2f7062aeaebd6f39adb5c4)

- [Pilot CoT 44 — Ashish Vaswani · first_principles](https://docs.xiaohongshu.com/doc/103435f528a667d00cc785d3c4db8870)

- [Pilot CoT 45 — Ashish Vaswani · analogy_transfer](https://docs.xiaohongshu.com/doc/e53d6bbe7b979aae0dc4f5e27f701714)

- [Pilot CoT 46 — Ashish Vaswani · empirical_anomaly](https://docs.xiaohongshu.com/doc/d78e5fbab07798db0962d5896777bc74)

- [Pilot CoT 47 — Ashish Vaswani · first_principles](https://docs.xiaohongshu.com/doc/862a16623b3f4f07700be2a4ad7f5435)

- [Pilot CoT 48 — Ashish Vaswani · analogy_transfer](https://docs.xiaohongshu.com/doc/1cf41f43f9bddbeb829af8e70667b5ed)

- [Pilot CoT 49 — Ashish Vaswani · empirical_anomaly](https://docs.xiaohongshu.com/doc/61f8864e00a99eea58f31fc2c3640b7f)

- [Pilot CoT 50 — Ashish Vaswani · first_principles](https://docs.xiaohongshu.com/doc/292474fbf6e62eca1df0b041219f5a45)

- [Pilot CoT 51 — Ashish Vaswani · analogy_transfer](https://docs.xiaohongshu.com/doc/dbf3eb4fce7c04e13afb3df5e079289d)

- [Pilot CoT 52 — 苏剑林 Jianlin Su · empirical_anomaly](https://docs.xiaohongshu.com/doc/67b818321ecf7343cdc542a5a09099b9)

- [Pilot CoT 53 — 苏剑林 Jianlin Su · first_principles](https://docs.xiaohongshu.com/doc/a0958ce4372120c63e87431a347ca676)

- [Pilot CoT 54 — 苏剑林 Jianlin Su · analogy_transfer](https://docs.xiaohongshu.com/doc/a8695118e68c72e4b68f9fe47bef8550)

- [Pilot CoT 55 — 苏剑林 Jianlin Su · empirical_anomaly](https://docs.xiaohongshu.com/doc/e9ac4bb0630a8a98e028e1fbe12f437f)

- [Pilot CoT 56 — 苏剑林 Jianlin Su · first_principles](https://docs.xiaohongshu.com/doc/01cd2ab204ea4772c003e5a1e4bd7975)

- [Pilot CoT 57 — 苏剑林 Jianlin Su · analogy_transfer](https://docs.xiaohongshu.com/doc/7c15fe515c50aa7c2a837867cb59021d)

- [Pilot CoT 58 — 苏剑林 Jianlin Su · empirical_anomaly](https://docs.xiaohongshu.com/doc/c67a26e2e24938bf6bd5a8cc79a801f6)

- [Pilot CoT 59 — 苏剑林 Jianlin Su · first_principles](https://docs.xiaohongshu.com/doc/efa5d23d3ec90363611f17f265729c9f)

- [Pilot CoT 60 — 苏剑林 Jianlin Su · analogy_transfer](https://docs.xiaohongshu.com/doc/6df39aab479af0157f2503f73edff29d)

- [Pilot CoT 61 — 苏剑林 Jianlin Su · empirical_anomaly](https://docs.xiaohongshu.com/doc/715c945740808ece2713e81fdc8922aa)

- [Pilot CoT 62 — 苏剑林 Jianlin Su · first_principles](https://docs.xiaohongshu.com/doc/60855269060e70a0c15ee0eb750adbbc)

- [Pilot CoT 63 — 苏剑林 Jianlin Su · analogy_transfer](https://docs.xiaohongshu.com/doc/77e7623f03a5ddcc07b6fd0c05a16d91)

- [Pilot CoT 64 — 苏剑林 Jianlin Su · empirical_anomaly](https://docs.xiaohongshu.com/doc/28b795c7cb007a869cbab545569b0d64)

- [Pilot CoT 65 — 苏剑林 Jianlin Su · first_principles](https://docs.xiaohongshu.com/doc/522fdf4fc1c3f88033f90e49f6dfb324)

- [Pilot CoT 66 — 苏剑林 Jianlin Su · analogy_transfer](https://docs.xiaohongshu.com/doc/3969efdeee7bfa0ce9b4eb2133517327)

- [Pilot CoT 67 — 苏剑林 Jianlin Su · empirical_anomaly](https://docs.xiaohongshu.com/doc/f42a6b0fa7730897228f0dfdd2eb6c8f)

- [Pilot CoT 68 — 苏剑林 Jianlin Su · first_principles](https://docs.xiaohongshu.com/doc/287626a219656e8ae9d736b1a8fc5ba6)

- [Pilot CoT 69 — 苏剑林 Jianlin Su · analogy_transfer](https://docs.xiaohongshu.com/doc/e4ac950fd7529e19bf64e947a8f6fe35)