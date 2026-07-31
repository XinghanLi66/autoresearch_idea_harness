## 一句话结论

你的想法并不是空白；它正好落在四条已经成形、但还没有被完全打通的研究线上：

1. **reference-grounded / related-work-grounded idea generation benchmark**
代表作：IdeaBench、AI Idea Bench 2025。

2. **time-sliced / future-aligned proposal learning**
代表作：Learning to Predict Future-Aligned Research Proposals with Language Models、MOOSE-Chem、MOOSE-Star。

3. **直接做 ideation post-training（SFT / RL / GRPO）**
代表作：LDC、MLR-Copilot、MoRI、EvoIdeator、AI Can Learn Scientific Taste。

4. **inference-time / agentic literature organization 与 ideation**
代表作：ResearchAgent、Chain-of-Ideas、SciMON、IdeaSynth、Scideator、AI co-scientist、AI Scientist。

**和你的设定最接近的公开工作**是


Learning to Predict Future-Aligned Research Proposals with Language Models

：它把 proposal generation 形式化成一个**时间切分的 scientific forecasting** 任务，输入是**leakage-controlled research question + 5 个 pre-cutoff inspiring papers**，输出是结构化 proposal，并用未来论文上的 **Future Alignment Score (FAS)** 做评测。
**但最大的空缺**仍然在于：**full related-work conditioning + strict temporal leakage control + 真正的 post-training + future-grounded evaluation** 这四件事目前还没有被系统地合成到一个工作里。

---

## 你可以怎么理解这条线

### 1. 最接近你设定的问题定义

核心问题不是“让模型写一个看起来新颖的点子”，而是：

给定某个 cutoff time 之前可见的 research context（research question、references、related work、citation graph），让模型生成一个 proposal / hypothesis，并要求它和 cutoff 之后真正出现的 paper / research direction 对齐。

这是 **future-aligned ideation** / **time-sliced proposal prediction**。

### 2. 目前已有工作的主要差异点

现有工作大致在下面这些维度上不同：

- **conditioning context**

  - 只给 research question

  - 给少量 inspiring papers

  - 给 seed paper 的 title/abstract/introduction/related work

  - 给检索得到的 literature review summary

  - 给完整 reference set / full related work（这一点最少）

- **训练方式**

  - prompting only

  - RAG / agent / multi-agent

  - SFT

  - RL / PPO / GRPO / judge-model reward

- **时间一致性**

  - 没有严格控制

  - benchmark 层面控制 leakage

  - training / evaluation 都做 strict temporal split

- **评测方式**

  - LLM-as-judge

  - human / expert review

  - target-paper alignment

  - future-paper impact / future outcome grounded evaluation

  - execution-grounded evaluation

---

## Matrix A：最贴近你想法、以及直接做 post-training 的工作

| Work | Year | Bucket | Task / Output | Conditioning context | Training signal | Post-train? | Strict temporal control? | 对你最相关的点 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Learning to Predict Future-Aligned Research Proposals with Language Models | 2026 | future-aligned proposal learning | 结构化 research proposal | **research question + 5 structured inspiring papers**（来自 pre-cutoff citations） | time-consistent SFT；合成 gap-analysis / inspiration-borrowing / stepwise reasoning；目标是提升 **FAS** | **Yes** | **Yes** | **最像你的 formulation**，但它不是 full related work，而是 5 个 inspiring papers |
| MOOSE-Chem | 2024 | unseen-paper rediscovery | chemistry hypothesis | background question / background survey + chemistry corpus，用来检索 inspirations | 主要是 multi-agent retrieval / ranking / generation framework | No（不是参数级 post-train） | **Yes**（target papers 2024 online；LLM cutoff 2023） | 证明“recent unseen papers + historical context + rediscovery”是可做的 |
| MOOSE-Star | 2026 | tractable training for discovery | hypothesis generation | research background + inspirations（来自 citations） | decomposed training：**Inspiration Retrieval + Hypothesis Composition**；还加入 hierarchical search / bounded composition / motivation planning | **Yes** | **Yes**（2020–2025 papers；Oct 2025 holdout） | 给你一个重要启发：**不要直接吃 full related work 原文，最好做分解训练 / 分层搜索** |
| LDC: Learning to Generate Research Ideas with Dynamic Control | 2024 | ideation post-training | research idea | **一个 supporting paper**（从 related works 里选“最重要的一个”） | **SFT + controllable RL**；reward 分 novelty / feasibility / effectiveness 三维 | **Yes** | No | 它直接做 ideation post-training，但 conditioning context 很弱：**不是 full related work，只取 1 篇** |
| MLR-Copilot | 2024 | autonomous ML research | idea + experiment plan + code/exec | seed paper 的 **title / abstract / introduction / related work**，再配 recent works / code retrieval | SFT + RL；OpenReview feedback（overall / novelty / feasibility / effectiveness） | **Yes** | Partial（没有严格的 future-paper setup） | 更像“related-work-aware ideation + downstream execution” |
| MoRI | 2026 | motivation-grounded ideation | methodology / idea generation | structured research context（context → motivation → methodology） | SFT + RL；reward = **entropy-aware information gain + contrastive semantic gain** | **Yes** | **Yes**（ICLR 2024–2025，late-2025 holdout） | 重点不是 refs，而是把“motivation→method”的 reasoning 学进参数 |
| EvoIdeator | 2026 | feedback-aligned ideation RL | refined research idea | **(query, literature_review)**；literature review 来自检索到的 papers | RL + judge-generated **lexicographic rewards + actionable language feedback** | **Yes** | No（至少不是核心卖点） | 它说明“full related work”可以先压缩成 **literature review summary** 再做 post-training |
| AI Can Learn Scientific Taste | 2026 | community-feedback RL for ideation | follow-up research idea with high impact | seed paper 的 **title + abstract** | **RLCF**：先训练 Scientific Judge（700K paired papers），再用它做 reward train Scientific Thinker | **Yes** | Partial（future-year holdout / unseen fields） | 不是“复现 target paper”，而是学习 **high-impact scientific taste** |
| IdeaBench | 2024 | benchmark | idea generation benchmark | **target paper references** 作为背景 | 无训练；评测为主，Insight Score | No | Partial（target papers 是 2024 biomedical） | 它明确把“用 references 作为 idea 生成上下文” benchmark 化了 |
| AI Idea Bench 2025 | 2025 | benchmark | AI idea generation benchmark | target papers + inspired works | 无训练；ground-truth alignment + general reference material judgment | No | Partial；明确讨论 leakage | 强调 **knowledge leakage** 与 **grounded truth** 问题 |

### 对 Matrix A 的快速解读

- 真正最像你的，不是 LDC，而是 **Future-Aligned Proposals**。

- 真正做“post-train ideation”的代表，是 **LDC / MLR-Copilot / MoRI / EvoIdeator / Scientific Taste**。

- 但这些 post-training work 几乎都**没有**直接把“**完整 related work**”作为核心 conditioning object：

  - LDC：只取 1 篇 supporting paper

  - Future-Aligned Proposals：只取 5 篇 inspiring papers

  - EvoIdeator：把相关论文压缩成 literature review

  - MLR-Copilot：是 seed paper + related work，而不是 target paper 的完整 historical reference graph

  - MoRI / Scientific Taste：更偏 reasoning 或 taste，而不是 full RW conditioning

---

## Matrix B：更偏 inference-time / agentic / human-AI ideation 的工作

| Work | Year | Bucket | 主要做什么 | Conditioning context | 是否 post-train | 对你有什么启发 |
| --- | --- | --- | --- | --- | --- | --- |
| ResearchAgent | 2024/2025 | iterative agentic ideation | 从 core paper 出发，沿 academic graph 和 concept store 扩展文献，再让多个 reviewing agents 迭代修 proposal | core paper + academic graph + concept store | No（核心是 agent workflow） | 说明“related work”可以图结构化，而不是平铺上下文 |
| Chain of Ideas | 2024 | literature organization for ideation | 将 relevant literature 组织成 chain structure，模仿研究脉络演化 | chain-structured literature | No | 很适合做 full related work 的 **ordering / compression** baseline |
| SciMON | 2023/2024 | novelty-optimized ideation | 从 background context 生成 grounded idea，并与 prior papers 反复比较以提升 novelty | background + retrieved inspirations | No | 适合做 novelty optimization baseline，但不是 temporal post-train |
| IdeaSynth | 2024 | interactive idea development | 把想法拆成 problem / solution / evaluation / contribution 等 facets，反复演化与组合 | literature-grounded facet feedback | No（更偏系统/接口） | 说明 full RW 更适合先抽成 **facets** 再建模 |
| Scideator | 2024/2025 | mixed-initiative ideation | 从用户 papers 和 analogous papers 中抽取 purpose / mechanism / evaluation facets，再做 recombination + novelty evaluation | input papers + related papers + facets | No | 给出一个很自然的“paper facets as latent compression”的方向 |
| FutureGen | 2025 | future work generation | 从单篇 paper + related papers 生成 future work suggestions | paper sections + related papers via RAG | No | 更像“future work section”生成，不是 strict future-paper prediction |
| Towards an AI co-scientist | 2025 | multi-agent discovery | generate / debate / evolve hypotheses and proposals | scientist-provided objective + prior evidence | No（主打 test-time scaling / multi-agent） | 更偏系统范式，不直接回答“post-train ideation” |
| The AI Scientist | 2024 | end-to-end AI research | 生成 idea、写代码、跑实验、写论文、自动审稿 | task description + code / experiment loop | No（主打 end-to-end automation） | 适合作为“execution-aware downstream system”而不是 ideation post-training baseline |

### 对 Matrix B 的快速解读

这类工作很重要，但它们回答的是：

“怎样在 inference time / workflow 层面帮助 LLM 提出 idea？”

而不是：

“怎样把 ideation policy 本身通过 post-training 学进参数？”

如果你的 paper 想突出 **post-training**，最好把这类工作归到 **agentic / inference-time baselines**，而不是主线 related work。

---

## Matrix C：benchmark / evaluation 工作

| Work | Year | 评什么 | Ground truth / signal | 是否时间感知 | 对你有什么启发 |
| --- | --- | --- | --- | --- | --- |
| IdeaBench | 2024 | idea generation | target paper abstract 与 LLM idea 的相对质量 / Insight Score | Partial | reference-grounded benchmark 很重要，但还不是未来对齐 |
| AI Idea Bench 2025 | 2025 | AI idea generation | original paper alignment + general reference judgment | Partial；强调 leakage | 评测时要认真处理 leakage / grounded truth |
| Can LLMs Generate Novel Research Ideas? | 2024 | ideation stage | 100+ NLP researchers blind review | No | LLM idea 在“看起来 novel”上可以不错，但 feasibility 稍弱 |
| The Ideation-Execution Gap | 2025 | ideation vs execution outcome | researchers 真正执行 idea 后的论文评审结果 | No，但更接近真实科研结果 | **只看 ideation 阶段会高估模型** |
| HindSight | 2026 | future impact of generated ideas | 与 future papers 匹配，再按 citation / venue impact scoring | **Yes** | 非常适合作为你的外部验证：生成的 idea 有没有在未来真正“出现” |
| Proof of Time (PoT) | 2026 | scientific idea judgment / forecasting | freeze pre-cutoff evidence，预测 post-cutoff outcomes | **Yes** | 给你一套更 general 的 time-split evaluation 哲学 |
| MOOSE-Chem | 2024 | unseen hypothesis rediscovery | 与 unseen 2024 chemistry papers 的 hypothesis 对齐 | **Yes** | “recent unseen papers” 的 benchmark 思路已被证明可行 |

### 对 Matrix C 的快速解读

如果你的方法强调“未来论文 idea 预测”，那评测最好不要只靠 LLM judge。更强的评测组合通常是：

1. **future alignment**（类似 FAS / HindSight）

2. **expert pairwise review**

3. **pre-cutoff novelty check**

4. 可选：**execution-lite** / downstream implementation 验证

---

## 和你的想法逐项对比

### 你的想法

针对最近发表、还未被 pretrain 覆盖到的文章，conditioned on 它全部的 related work，训练模型提出和这篇文章类似的 idea。

### 已有工作里最接近的部分

#### A. “recent paper, not in pretraining”

- **MOOSE-Chem** 已经非常接近：用 2024 才公开上网的 chemistry papers，要求 2023-cutoff LLM 在只看历史背景时 rediscover hypothesis。

- **Future-Aligned Proposals** 则更 general：用 2024 train / 2025 test，严格做 temporal split。

#### B. “condition on related work / citations”

- **IdeaBench** 明确用 target paper 的 references 作背景。

- **Future-Aligned Proposals** 从 references 里选 inspiring papers。

- **MOOSE-Star** 直接把 inspirations 绑定到 historical citations。

#### C. “train the model”

- **LDC / MLR-Copilot / MoRI / EvoIdeator / Scientific Taste** 都是明确的 post-training 方向。

- 但它们几乎都没有把“**完整 related work 集**”作为核心建模对象。

#### D. “propose an idea similar to target paper”

- **IdeaBench / AI Idea Bench 2025**：更偏 benchmark / alignment。

- **Future-Aligned Proposals**：直接把这个过程形式化成 proposal prediction。

- **MOOSE-Chem**：是 domain-specific hypothesis rediscovery。

---

## 这条线里真正还缺什么

### 1. full related work conditioning

这是你最明显的空间。公开工作多数只做：

- 单 supporting paper（LDC）

- 少量 inspiring papers（Future-Aligned Proposals）

- 检索后生成 literature review summary（EvoIdeator）

- seed paper + related work（MLR-Copilot）

**几乎没有人系统地研究：**

完整 related work / full citation neighborhood 到底该如何表示，才能用于 parameter-level ideation training？

### 2. full RW 不适合直接平铺进 prompt

现有证据基本都指向一个结论：
**“全部 related work 原文直接塞上下文”大概率不是最佳方案。**

更有前景的表示方式是：

- **hierarchical retrieval / graph traversal**（ResearchAgent, MOOSE-Star）

- **chain / chronology compression**（Chain-of-Ideas）

- **facet extraction / recombination**（IdeaSynth, Scideator）

- **literature review synthesis**（EvoIdeator）

所以你的工作如果想做 “all related work”，最好不是：

把 related work section 全文 + cited papers 全文直接拼到 prompt

而更像：

用 pre-cutoff citation graph 构造 **compressed related-work representation**，再做 post-training。

### 3. 未来对齐评测应成为主指标

只做 LLM-as-judge 会比较脆弱。
更稳的组合是：

- proposal-to-target alignment

- proposal-to-future-corpus alignment

- pre-cutoff novelty / overlap check

- expert review

- optional execution-lite validation

### 4. “像 target paper”与“真的新”之间有张力

如果你把任务表述成：

生成与 target paper 相似的 idea

那么模型可能学到的是 **retrospective imitation**。
更好的表述通常是：

在 pre-cutoff evidence 下，生成一个**future-aligned** proposal，使其与 post-cutoff human research direction 高度一致。

这样更像 forecasting，而不是纯粹的 answer reconstruction。

---

## 我建议你如何给自己的工作定位

### 推荐的 problem statement

你可以把问题写成：

**Future-aligned ideation under full related-work conditioning**:
Given a target paper (Y) published after a cutoff (t_c), we reveal only a leakage-controlled research question (q) and a pre-cutoff related-work evidence set (R(Y, t_c)), derived from the paper’s historical citations / citation neighborhood. The model must generate a structured proposal (P) that is future-aligned with the post-cutoff research outcome while remaining novel with respect to pre-cutoff literature.

### 推荐的 novelty claim

可以直接和现有工作做差异化：

1. **vs Future-Aligned Proposals**
他们用 5 inspiring papers；你做 **full related-work evidence**。

2. **vs LDC**
他们只用 1 supporting paper；你建模完整相关文献集。

3. **vs EvoIdeator / MLR-Copilot**
他们用 literature summary 或 seed-paper context；你是严格的 **target-conditioned, temporally valid, future-aligned** setting。

4. **vs ResearchAgent / CoI / SciMON**
他们主要靠 inference-time orchestration；你优化的是 **parameter-level ideation policy**。

5. **vs IdeaBench / AI Idea Bench 2025**
他们主要是 benchmark；你做的是 training + evaluation 一体化。

---

## 如果你真要做这个题，比较自然的技术路线

### 数据构造

对每个 target paper (Y)：

1. 设定 cutoff (t_c &lt; t_Y)

2. 构造 **leakage-controlled research question** (q)

3. 提取 pre-cutoff 可见的：

  - direct references

  - optional 1-hop citation neighborhood

  - related-work section text

  - paper facets / topic clusters / method families

4. 构造 structured target proposal (\tilde&#123;P&#125;)：

  - research question

  - hypothesis

  - proposed method

  - novelty claims

  - experiment plan

### related work 表示方式（最值得做 ablation）

建议至少比较四种输入表示：

1. **Top-k inspiring papers**
复现 Future-Aligned 风格 baseline

2. **full raw references**
最直接但最笨的 baseline

3. **hierarchical compressed citation graph**
你的主方法候选

4. **facetized literature summary**
借鉴 IdeaSynth / Scideator / EvoIdeator

### 训练方式

可以从易到难做三档：

#### 档 1：SFT

- 输入：(q + R(Y, t_c))

- 输出：structured proposal (\tilde&#123;P&#125;)

#### 档 2：SFT + preference / RL

reward 可以混合：

- future alignment reward

- pre-cutoff novelty reward

- feasibility / specificity reward

- style / structure reward

#### 档 3：decomposed training

借鉴 MOOSE-Star，把任务拆成：

- relevant evidence selection

- gap identification

- inspiration borrowing

- method synthesis

- experiment design

这通常会比端到端直接学“从 full RW 到 target proposal”更稳。

---

## 最推荐优先精读的论文清单（按优先级）

### 第一组：最接近你的工作设定

1. Learning to Predict Future-Aligned Research Proposals with Language Models

2. MOOSE-Chem

3. MOOSE-Star

4. IdeaBench

5. AI Idea Bench 2025

### 第二组：最直接的 ideation post-training

1. LDC

2. MLR-Copilot

3. MoRI

4. EvoIdeator

5. AI Can Learn Scientific Taste

### 第三组：你可以借鉴其表示 / workflow 设计，但不一定是主线 related work

1. ResearchAgent

2. Chain of Ideas

3. SciMON

4. IdeaSynth

5. Scideator

### 第四组：评测视角一定要读

1. Can LLMs Generate Novel Research Ideas?

2. The Ideation-Execution Gap

3. HindSight

4. Proof of Time (PoT)

---

## 一个可以直接放进论文 related work 的写法（中文版草稿）

### Related Work

LLM-driven scientific ideation 的现有工作大致可以分为四类。第一类是 **reference-grounded / literature-grounded benchmark**。IdeaBench 使用 target paper 的 references 作为背景，评测模型在 biomedical 领域生成 research ideas 的能力；AI Idea Bench 2025 进一步强调 idea generation benchmark 中的 knowledge leakage 与 grounded truth 问题，并构建了 3,495 篇 AI papers 及其 inspired works 的评测框架。第二类是 **time-sliced / future-aligned ideation**。Learning to Predict Future-Aligned Research Proposals with Language Models 将 proposal generation 重构为一个时间切分的 forecasting 问题：给定 research question 与 pre-cutoff inspiring papers，模型生成 proposal，并通过与未来论文语义对齐的 FAS 进行评估。MOOSE-Chem 则在 chemistry 领域测试模型是否能基于历史背景与 inspirations 重新发现 2024 年 newly published papers 的 hypotheses；MOOSE-Star 进一步提出将 scientific discovery 分解为 inspiration retrieval 与 hypothesis composition 两个可训练子任务，并在 temporally held-out papers 上验证 tractable training 的有效性。

第三类工作尝试将 ideation 能力直接通过 **post-training** 学入模型参数。LDC 提出 SFT + controllable RL 的两阶段框架，但其输入只保留从 related works 中选出的单篇 supporting paper。MLR-Copilot 使用 seed paper 的 title、abstract、introduction 与 related work 生成 methodology 和 experiment plan，并利用 OpenReview 反馈进行 RL 优化。MoRI 从 accepted ICLR papers 中构建 motivation-grounded dataset，通过 SFT 和 RL 学习从研究动机到方法设计的推理过程。EvoIdeator 使用 query 与 literature-review 对作为输入，将 checklist-grounded language feedback 与 lexicographic rewards 统一到 RL 训练中。AI Can Learn Scientific Taste 则通过大规模 community feedback 学习 reward model，并进一步训练能提出高 potential-impact ideas 的 policy model。总体来看，现有 post-training work 已经覆盖了 novelty、feasibility、effectiveness、impact 等不同优化目标，但对于 **如何在严格时间一致性的前提下建模完整 related-work evidence** 仍缺乏系统研究。

第四类工作主要在 **inference-time / agentic workflow** 层面提升 ideation 质量。ResearchAgent 沿 academic graph 扩展文献并通过 reviewing agents 迭代修正 proposal；Chain-of-Ideas 将文献组织成链式结构以显式建模研究脉络；SciMON 通过检索 inspirations 并反复与 prior work 比较来优化 novelty；IdeaSynth 与 Scideator 则分别采用 facet-based 的 externalization、recombination 与 novelty evaluation。这些方法表明，related work 不必以原始全文形式直接输入模型，而更适合作为 graph、chain、facet 或 literature summary 等压缩表示。然而，这类方法大多依赖 inference-time orchestration，而非 parameter-level ideation training。

与上述工作相比，我们关注的是 **future-aligned ideation under full related-work conditioning**：在严格 temporal split 下，仅暴露 target paper 的 pre-cutoff related-work evidence 与 leakage-controlled research question，训练模型生成与 post-cutoff human research outcome 对齐的 structured proposal。该设定同时结合了 strict temporal validity、rich related-work conditioning、parameter-level training 与 future-grounded evaluation。

---

## 英文版 positioning paragraph（可直接改）

Recent work on LLM-based scientific ideation can be grouped into four lines: reference-grounded benchmarking, future-aligned proposal prediction, parameter-level post-training for ideation, and inference-time agentic literature organization. Benchmark efforts such as IdeaBench and AI Idea Bench 2025 evaluate idea generation conditioned on cited or motivating papers, but do not train models for temporally grounded ideation. Future-aligned proposal learning frames proposal generation as a time-sliced forecasting task, yet typically conditions on a small subset of inspiring papers rather than the full related-work evidence. In parallel, post-training approaches such as LDC, MLR-Copilot, MoRI, EvoIdeator, and AI Can Learn Scientific Taste optimize idea quality through SFT, RL, or preference learning, but they either use a single supporting paper, a seed-paper context, or a compressed literature review instead of a complete historical related-work set. Agentic systems such as ResearchAgent, Chain-of-Ideas, and SciMON improve ideation through retrieval, debate, or structured literature organization, but primarily at inference time. Our setting instead focuses on future-aligned ideation under full related-work conditioning: given only pre-cutoff related-work evidence and a leakage-controlled research question, the model is trained to generate a structured proposal aligned with a post-cutoff research outcome.

---

## References

### Surveys

- Large Language Models for Scientific Idea Generation: A Creativity-Centered Survey

- A Survey on Large Language Models in Scientific Discovery

### Core / closest works

- Learning to Predict Future-Aligned Research Proposals with Language Models

- MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific Hypotheses

- MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier

- IdeaBench: Benchmarking Large Language Models for Research Idea Generation

- AI Idea Bench 2025: AI Research Idea Generation Benchmark

### Post-training for ideation

- LDC: Learning to Generate Research Ideas with Dynamic Control

- MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents

- MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models

- EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning

- AI Can Learn Scientific Taste

### Inference-time / agentic / interfaces

- ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models

- Chain of Ideas: Revolutionizing Research Via Novel Idea Development with LLM Agents

- SciMON: Scientific Inspiration Machines Optimized for Novelty

- IdeaSynth: Iterative Research Idea Development Through Evolving and Composing Idea Facets with Literature-Grounded Feedback

- Scideator: Human-LLM Scientific Idea Generation Grounded in Research-Paper Facet Recombination

- FutureGen: A RAG-based Approach to Generate the Future Work of Scientific Articles

- Towards an AI co-scientist

- The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery

### Evaluation

- Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers

- The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas

- HindSight: Evaluating Research Idea Generation via Future Impact

- Proof of Time: A Benchmark for Evaluating Scientific Idea Judgments

<br/>