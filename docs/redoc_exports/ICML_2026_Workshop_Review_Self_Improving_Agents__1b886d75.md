<redoc-highlight emoji="dengpao" fillColor="green">
<font color="#16833A">核心结论</font>：四个 workshop 没有推翻原来的判断：公开证据仍支持“有边界的系统自优化”，而不是完整 full-stack recursive intelligence。但它们显著补强了三个关键环节：可信 evaluator / verifier、可回放 planning 或 forecasting environment、以及对 surrogate metric / self-report / CoT trace 的 epistemic 警惕。
</redoc-highlight>

<redoc-highlight emoji="gantanhao" fillColor="red">
<font color="#C01818">重要证据边界</font>：ICML virtual talk videos / slides / transcripts 目前没有无需登录的公开访问路径。本报告的 talk 笔记只基于 project page 的 schedule、speaker bio、公开 title / abstract；不把 speaker 既有工作强行当成本次 talk 内容。
</redoc-highlight>

## Why This Report

这份报告用之前的 Self-Improving Agent / Recursive Intelligence 系统调研作为参照，问题不是“这些论文是否都在做 self-improvement”，而是：它们是否给出了能推进 self-improving agent 的关键模块，尤其是 E_t evaluator、verifier、memory、planning harness、reward signal 和 anti-Goodhart governance。

## Coverage

| Workshop | Public talks | Reviewed paper-like items | Total child pages |
| --- | --- | --- | --- |
| Forecasting | 6 | 15 | 21 |
| AI4Math | 3 | 10 | 13 |
| Philosophy Meets ML | 6 | 10 | 16 |
| LM4Plan | 6 | 10 | 16 |

## Cross-Workshop Map

| Bottleneck in the original framework | Strongest workshop signals | Takeaway |
| --- | --- | --- |
| 谁来判断改进真的更好 | HorizonMath, AXLE, SEVerA, Reward Prediction with Factorized World States, VeryTrace, FutureSim | <font color="#16833A">Verifier-first</font> 是最清晰的前进路线：把 agent proposal 变成可运行、可证明、可回放或可结算的对象。 |
| Reward hacking / Goodhart | PhilML surrogate metric talk, approach-level diversity paper, prediction-market profit papers | <font color="#C01818">不能把分数当能力本身</font>；需要 causal / market / trace-level audit。 |
| Memory 是否真的帮助 self-improvement | ForecastCompass, Online Boundary-Aware Memory, Operative Contexts | <font color="#C76500">Memory 要有边界、适用条件和 revision protocol</font>，不是越多越好。 |
| Planning / test-time compute 如何评估 | HiPER, VeryTrace, LM4Plan invited talks, Allocation Not Volume, Noam Brown talk | 必须报告 compute allocation 和 trace verification，避免把 scaffold compute 误读为 base model 能力。 |
| AI 能否做真正科研 | MLS-Bench, QED, HorizonMath | 最有希望的路径仍是“强 verifier + constrained open-ended search + expert audit”，而不是纯模型内在反思。 |

## Cross-Workshop Key Works: Per-Paper Takeaway

这张表只放最值得回头看的代表性工作。每行都回答三个问题：它想解决什么问题、用了什么机制、对我们的 self-improving agent 框架有什么直接启发。

| Theme | Work | 更细 summary | Takeaway for our framework |
| --- | --- | --- | --- |
| Forecasting evaluator | [FutureSim](https://docs.xiaohongshu.com/doc/4ba300dc04cce3bd6051bb5eb20ab113) | 问题是普通 forecasting benchmark 太静态，无法测试 agent 随时间吸收新闻和修正 belief 的能力。它把真实历史事件按时间线 replay，让 agent 每天看到 date-gated news，再决定是否更新 forecast。 | 可以作为 adaptive-agent evaluator，专门测 memory、retrieval、uncertainty updating 和 long-horizon feedback，而不是只测一次性答题。 |
| Forecasting memory | [ForecastCompass](https://docs.xiaohongshu.com/doc/015cb6aceae700d34eee30434150a9e4) | 问题是 agent memory 往往只堆事实和反思，缺少“哪些 predictive factors 可复用”的结构。它维护 factor memory 和 reasoning memory，并用 retrospective analysis 更新哪些因素该保留、改写或降权。 | 这是最像 L1 self-improvement 的 forecasting work，提示我们 memory 要有 revision protocol、适用边界和 audit，而不是简单追加。 |
| Training signal | [Future-as-Label](https://docs.xiaohongshu.com/doc/06f58e0eb8c912ac601b41a9d734578b) | 问题是高质量监督很贵，而 forecast 的真实 outcome 会在未来自然揭晓。它把 resolved future events 变成 proper-scoring reward，用 causally masked information 训练 probabilistic forecaster。 | 给 L5 weight-level improvement 一个干净外部信号：reward 来自现实结果，不来自模型 self-judge。关键风险是 leakage、延迟反馈和问题选择偏差。 |
| Math evaluator | [HorizonMath](https://docs.xiaohongshu.com/doc/a4361892bafc2d5b581c64178e57529d) | 问题是开放数学发现难以评估，人工 review 慢且 contamination 难控。它挑选“发现难、验证相对便宜”的 computational 或 applied math problems，并配自动 verification。 | 这是科学发现里非常宝贵的 evaluator 形态：让 agent 的候选发现能被自动或半自动验证，而不是只靠语言自评。 |
| Proof tooling | [AXLE](https://docs.xiaohongshu.com/doc/26a961f04bcce504d89149405947c3db) | 问题是 Lean agent 要频繁验证、抽取、修复 proof，直接本地调用成本高且环境脆弱。AXLE 把 Lean 4 utilities 做成 cloud API，支持 verification、metadata extraction、proof repair 和隔离执行。 | 它更像 trusted infrastructure，不是新 agent。本质启发是：可演化 agent 外围需要稳定、隔离、可审计的 verifier/tool core。 |
| Dense supervision | [Self-Distillation Zero](https://docs.xiaohongshu.com/doc/000a3524a9ae7f242bfd356dae2fe1b1) | <redoc-comment commentGid="7661951830286836308" blockId="4155511709bb66fec47528807877ba03">问题是 math/code 任务常只有 binary reward，训练信号太稀疏。它让模型先 revision 自己的失败答案，再把 revision 分布蒸馏回 generator，把稀疏 reward 变成 token-level dense supervision。</redoc-comment> | 对 PRS/RL 很有启发：如果有可靠 verifier，可以把“对错”变成更细粒度学习信号。但 verifier bias 仍会被放大。 |
| AI-for-AI benchmark | [MLS-Bench](https://docs.xiaohongshu.com/doc/72acf17f3e2f79f16db8be044478dd75) | 问题是很多 agent 只会调参或套 baseline，不能真正提出 generalizable ML method。MLS-Bench 用 140 个任务评估 agent 是否能改进 ML system component，并跨 setting 和 scale 验证。 | 它直接对齐我们的 autoresearch 目标：衡量 proposal 是否产生可迁移 method，而不是局部工程 trick。也说明当前 agent research taste 仍弱。 |
| Safe self-evolution | [SEVerA](https://docs.xiaohongshu.com/doc/b3c565bd6f9abf85b5326158d054954a) | 问题是 self-evolving agent 可能越改越不安全，甚至绕过约束。SEVerA 让 agentic generation 在 first-order logic contracts 下运行，用 verified fallback 保证 hard constraints 不被突破。 | 这是“可演化外围 + trusted core”的直接技术版本。它告诉我们未来 harness 不能只评估 pass rate，还要有不可被 agent 修改的 formal guardrail。 |
| Reward model | [Reward Prediction with Factorized World States](https://docs.xiaohongshu.com/doc/bada6645d7f6a66eecb38a9107115b62) | 问题是 LLM-as-judge reward 对 long-horizon planning 不够稳。它先把 observation 抽成 factorized world state，再比较 current state 和 goal state，用结构化语义距离预测 reward。 | 这是一条改进 E_t 的实用路线：让 evaluator 看世界状态结构，而不是直接读一段轨迹后拍脑袋打分。 |
| Planning RL | [HiPER](https://docs.xiaohongshu.com/doc/0af1c5ac92bf565349dc3c16b329bf46) | 问题是 flat LLM policy 很难在 sparse reward 的多轮任务中分配 credit。HiPER 把 agent 分成 high-level planner 和 low-level executor，并用 hierarchical advantage estimation 做 credit assignment。 | 对 master-worker 结构很有参考价值：planning 可以是显式层级策略，而不只是长 CoT 或 prompt convention。 |
| Trace verification | [VeryTrace](https://docs.xiaohongshu.com/doc/4b548e499cc1d6b5de853c5bf89c7495) | 问题是 CoT trace 可读但不一定可信，错误还会在多步推理里传播。VeryTrace 把 reasoning trace 转成 DSL 和 structured verification object，再做 dependency、computation 和 semantic audit。 | 它把 trace 从“解释文本”变成“可检查对象”，适合作为 worker log、proposal rationale 和 planning trace 的 verifier。 |
| Surrogate metrics | [Surrogate Metric Evaluation](https://docs.xiaohongshu.com/doc/9393f4a3cfff6bd2c75c6910decd3f59) | 问题是 surrogate metric 历史上和目标相关，不代表被优化后仍会带来目标改进。该 talk 把 surrogate 可信性改写为 causal inference 问题，强调优化压力下的因果匹配。 | 这是 Goodhart 问题的理论核心：我们不能只看 benchmark score，还要问优化这个 score 是否仍 causal-imply 真实研究质量。 |
| Self-report risk | [Self-Reports Do Not Identify Self-Models](https://docs.xiaohongshu.com/doc/8772ca9b8bf9888fd3b44360c7714126) | 问题是模型说“我为什么失败”并不证明它真的有对应 self-model。该工作强调需要 counterfactual reports 或 identifiability tests 来区分真实 internal model 和漂亮叙述。 | 对 reflection / critique / master self-diagnosis 很关键：反思文本只能作为 hypothesis，不能直接当 evaluator evidence。 |
| Formal interface | [Formal Sidecars](https://docs.xiaohongshu.com/doc/9bd91b05c1c9efd4510c0a05aa064e1a) | 问题是 prompt 只是自然语言愿望，不足以承载可信约束。该方向主张把 prompt 周围加 formal sidecars，把要求变成 proof obligations 或可检查规格。 | 对未来 proposal/worker harness 很实用：让每个 idea 不只是一段描述，还附带可验证 assumptions、constraints 和 success criteria。 |

## Workshop-Level Takeaways

### Forecasting

<font color="#C76500">方法重点</font>：Forecasting workshop 把真实或模拟未来 outcome 变成可结算 reward，用 Brier/log score、prediction markets、chronological replay、memory revision 和 RL post-training 支撑 agent 改进。

<font color="#16833A">进展</font>：FutureSim、Future-as-Label、ForecastCompass、WALLA、Proper Betting 等工作提供了比普通 benchmark 更接近真实决策的 feedback loop。

<font color="#C01818">缺陷</font>：market profit 和 forecast score 会引入流动性、费用、时间窗、leakage 和 strategic behavior；如果 evaluator 不稳，agent 很容易学到交易或 benchmark shortcut。

### AI4Math

<font color="#C76500">方法重点</font>：AI4Math 是四个 workshop 中最直接服务 self-evolving agent 的一组：Lean / formal verification、proof agents、self-distillation、verified constrained synthesis 和 AI-for-AI benchmark。

<font color="#16833A">进展</font>：SEVerA 给出“可演化外围 + formal constraints”的直接版本；HorizonMath 和 MLS-Bench 把 open mathematical / ML discovery 转为可验证评测；AXLE/TorchLean 提供 trusted tooling。

<font color="#C01818">缺陷</font>：仍依赖人设定任务、verifier 和 specs；proof agent 的 research value 常需 expert review；FIPO-Prover 目前没有可公开核验 source。

### Philosophy Meets ML

<font color="#C76500">方法重点</font>：PhilML 不直接构建 self-improving agent，而是提供 epistemic governance：Bayesian evaluation、causal surrogate metrics、self-report identifiability、accountability、representation 和 explanation。

<font color="#16833A">进展</font>：它明确指出 evaluator 本身必须被审计，尤其是 self-report、CoT、surrogate metric 与 benchmark score 的证据地位。

<font color="#C01818">缺陷</font>：许多 accepted paper 在公开环境中只有 title/OpenReview candidate，不足以写强技术结论；该 workshop 的贡献更多是 framing 和标准，而不是可直接跑的系统。

### LM4Plan

<font color="#C76500">方法重点</font>：LM4Plan 聚焦 long-horizon planning 的 hierarchy、verifier、simulation、reward prediction、trace formalization 和 test-time compute。

<font color="#16833A">进展</font>：HiPER 是明确的 hierarchical RL agent training；VeryTrace 把 reasoning trace 变成可检查对象；Reward Prediction with Factorized World States 改善了 evaluator/reward generalization。

<font color="#C01818">缺陷</font>：不少 oral 只有 listing；planning benchmark 成绩容易混合 base model、prompt、tool、compute 和 environment-specific tricks，需要更严格 attribution。

## Recommendations For Agentic Training

1. 把 evaluator 设计作为主线，而不是只训练 proposal generator。优先引入 formal verifier、chronological replay、simulation-verified search、market-style scoring 和 expert audit。

2. 训练 master 时保留“提出 proposal + 回答 worker 问题 + 识别 evaluator 风险”的综合能力，不要只优化 proposal 文本。

3. 在 V3/V4 harness 中记录 compute budget、trace verification、memory revision、source leakage 和 expert disagreement，避免把表面 pass 率当作科学发现能力。

4. 用 Forecasting / AI4Math / LM4Plan 的可验证任务做小闭环；用 PhilML 的 epistemic standards 做报告和 evaluator 审计模板。

## Child Pages

每个 child page 都包含 sources、core claim、method、evidence、framework relation、caveats 和 takeaway。

## Forecasting as a New Frontier of Intelligence

- invited_talk [Invited Talk by Atlas Wang](https://docs.xiaohongshu.com/doc/e169d7b522fd1c0a8e515cbf0133a3ba)

- invited_talk [Invited Talk by Nicole Kagan](https://docs.xiaohongshu.com/doc/51b26bfb4063ea510bf7f8925c520982)

- invited_talk [Invited Talk by Philip E. Tetlock](https://docs.xiaohongshu.com/doc/19b7af9d55a10415577637f325c0ddda)

- invited_talk [Invited Talk by Scott Jeen](https://docs.xiaohongshu.com/doc/3ef032de6d708b5fbb906a1f39611652)

- invited_talk [Invited Talk by Seth Blumberg](https://docs.xiaohongshu.com/doc/cffa5be675632fb1eecaac6090f44eb1)

- invited_talk [Invited Talk by Simon Du](https://docs.xiaohongshu.com/doc/761de7d8a9d8770d8b23c516e7d2751e)

- oral [Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs](https://docs.xiaohongshu.com/doc/4ae21698148b6ee529310d7edf11bf9d)

- oral [Allocation, Not Volume: Test-Time Compute for Agentic Forecasting](https://docs.xiaohongshu.com/doc/6afae48ebf646d24af5d36e1d95b1463)

- oral [FutureSim: Replaying World Events to Evaluate Adaptive Agents](https://docs.xiaohongshu.com/doc/4ba300dc04cce3bd6051bb5eb20ab113)

- oral [Forecasting Emerges from Auto-Regressive Pretraining: Latent Predictive Structure in Language Models](https://docs.xiaohongshu.com/doc/045116a6f3f49fb3f084ceff29e6a8c8)

- oral [Forecasting Motion in the Wild](https://docs.xiaohongshu.com/doc/f14abf34ee8085345e1de0866e7f8abb)

- spotlight [ForecastCompass: Guiding Agentic Forecasting with Adaptive Factor Memory](https://docs.xiaohongshu.com/doc/015cb6aceae700d34eee30434150a9e4)

- spotlight [ForecastBench-Sim: A Simulated-World Forecasting Benchmark](https://docs.xiaohongshu.com/doc/6519a49de00eb39eb24c58eee56f2fe4)

- spotlight [Approximate Recall, Approximate Forecasts: Recall as a Diagnostic for LLM Forecasting Errors](https://docs.xiaohongshu.com/doc/1e9dbf12e1454e04c9e118e699666ac5)

- spotlight [Beyond Accuracy: Can LLM Forecasters Profit on Prediction Markets?](https://docs.xiaohongshu.com/doc/231a035c0e0ef24f6a8c4deb8fd320e7)

- spotlight [Curating the Future: A Scalable Recipe for Training Open-Ended Forecasters](https://docs.xiaohongshu.com/doc/8b905e974f892923c3ba3b9940c73f4c)

- spotlight [Forecast-to-Trade: Hierarchical Reinforcement Learning for Decision-Aware Financial Forecasting](https://docs.xiaohongshu.com/doc/f0136fd0c6c7eff5dd445d3c1b92356a)

- spotlight [Reaching the frontier of AI forecasting with reinforcement learning](https://docs.xiaohongshu.com/doc/6535137dd83543c28cad077a19560f75)

- spotlight [Future-as-Label: Scalable Supervision from Real-World Outcomes](https://docs.xiaohongshu.com/doc/06f58e0eb8c912ac601b41a9d734578b)

- spotlight [Decentralized Aggregation of LLM Predictions via Wagering Mechanisms](https://docs.xiaohongshu.com/doc/3c57fafcce71525c9aa7a4faa5bd4fbb)

- spotlight [When do prophets profit in prediction markets?](https://docs.xiaohongshu.com/doc/4fa7755bdb244d613fadecabef9039a8)

## AI4Math: Toward Self-Evolving Scientific Agents

- invited_talk [Numina-Lean-Agent: From Single Model to Agent](https://docs.xiaohongshu.com/doc/57b4afc999cff9a15beb5c881ce3d253)

- invited_talk [Machine-Checked Mathematics in the Age of AI](https://docs.xiaohongshu.com/doc/8d90ea1dc93a940c537f9934ca51d69d)

- invited_talk [Physics of Learning: The Least Action Principle of Modern Machine Learning](https://docs.xiaohongshu.com/doc/e297dc75f1ebaaf03467d5d5bd376ab4)

- oral [Measuring Progress in Reasoning Toward Mathematical Discovery with Automatic Verification](https://docs.xiaohongshu.com/doc/a4361892bafc2d5b581c64178e57529d)

- oral [AXLE: A Cloud Infrastructure for Lean 4 Theorem Proving Utilities](https://docs.xiaohongshu.com/doc/26a961f04bcce504d89149405947c3db)

- spotlight [SEVerA: Verified Self-Evolving Agents with Specification Guidance](https://docs.xiaohongshu.com/doc/b3c565bd6f9abf85b5326158d054954a)

- spotlight [Lean Refactor: Multi-Objective Controllable Proof Optimization via Agentic Strategy Search](https://docs.xiaohongshu.com/doc/42a5868b1030d19d963085c3006d84ab)

- spotlight [Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision](https://docs.xiaohongshu.com/doc/000a3524a9ae7f242bfd356dae2fe1b1)

- spotlight [QED: An Open-Source Multi-Agent System for Generating Mathematical Proofs on Open Problems](https://docs.xiaohongshu.com/doc/e09476fa9cccb659c9a10866085317b0)

- spotlight [FIPO-Prover: Formalization-Oriented Informal Proof Optimization for Efficient Formal Theorem Proving](https://docs.xiaohongshu.com/doc/ac3e1f9fde76901ca07fbd7a6d0811cd)

- spotlight [Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning](https://docs.xiaohongshu.com/doc/6b3d135e397a572cc80151313180c7aa)

- spotlight [TorchLean: Formalizing Neural Networks in Lean](https://docs.xiaohongshu.com/doc/8a96db0cfed184539d67f9c40af4c985)

- spotlight [MLS-Bench: A Holistic and Rigorous Assessment of AI Systems on Building Better AI](https://docs.xiaohongshu.com/doc/72acf17f3e2f79f16db8be044478dd75)

## Philosophy Meets ML: What Counts as Trustworthy?

- invited_talk [“Reasoning” models, philosophy of inference, and artificial epistemic agency](https://docs.xiaohongshu.com/doc/4b6428198964abaff7acbac1efa8fedc)

- invited_talk [A Bayesian epistemology for LLM evaluation](https://docs.xiaohongshu.com/doc/40e595ff555ed774b0a2597d8f1314ce)

- invited_talk [Surrogate Metric Evaluation is a Causal Inference Problem](https://docs.xiaohongshu.com/doc/9393f4a3cfff6bd2c75c6910decd3f59)

- invited_talk [(How) Does Accountability Require Understanding ML Models?](https://docs.xiaohongshu.com/doc/96f3d1e6525404747ac2308596c6764f)

- invited_talk [75 years of Turing's AGI](https://docs.xiaohongshu.com/doc/ae86bfbeeb0ae4f1168f7618f37aa9f7)

- invited_talk [Thinking about humans in the era of AI](https://docs.xiaohongshu.com/doc/5a6d02a905cbdc0005f6cc0558bb79a3)

- oral [Self-Reports Do Not Identify Self-Models: An Identifiability Test for Counterfactual Reports](https://docs.xiaohongshu.com/doc/8772ca9b8bf9888fd3b44360c7714126)

- oral [Before Normative and Moral Alignment: Causal Contract Faithfulness as a Precondition for Trustworthy AI](https://docs.xiaohongshu.com/doc/6987085ca1843062dda8b049f2c0e75e)

- oral [A Definition of Good Explanations and the Challenges Explaining LLM Outputs](https://docs.xiaohongshu.com/doc/16702a8b971580a858a95820e37eff31)

- oral [Getting Monosemantic About Monosemanticity](https://docs.xiaohongshu.com/doc/bc2c3451da74d963d85ce92528c5a661)

- oral [The Concept of Representation in ML: Beyond Plato and Aristotle](https://docs.xiaohongshu.com/doc/976240ae4af6d109be071a8f36a25699)

- oral [Why Sampling Is Not Choosing: Intentionality, Agency, and Moral Responsibility in Large Language Models](https://docs.xiaohongshu.com/doc/cf4569bb08827c54373c392b78f01d95)

- paper [Online Boundary-Aware Memory for Case-Based Reasoning Agents](https://docs.xiaohongshu.com/doc/cd11eec8de853e77a11cf5557fdb16b6)

- paper [From Prompts to Proof Obligations: Formal Sidecars as an Epistemic Interface for Trustworthy ML](https://docs.xiaohongshu.com/doc/9bd91b05c1c9efd4510c0a05aa064e1a)

- paper [Do LLMs Really Represent the World? A Teleosemantic Assessment of Pre-Trained, Fine-Tuned, and Agentic LLMs](https://docs.xiaohongshu.com/doc/8060e1080d4851a312a26d5be853e813)

- paper [Operative Contexts: Belief Revision and Memory in Agentic AI](https://docs.xiaohongshu.com/doc/1cfc2ef91ea747d7dbb90185a6dc6e72)

## LM4Plan: Planning in the Era of Language Models

- invited_talk [LLM Planning Success](https://docs.xiaohongshu.com/doc/3a35227a2a9289b1aabfe52052918009)

- invited_talk [Model Collapse and its Implications on Planning with LLMs](https://docs.xiaohongshu.com/doc/c10f21fc81d2f4c77066f23c23b9cfb0)

- invited_talk [On the Role of Verifiers and Thinking Traces in Reasoning Models](https://docs.xiaohongshu.com/doc/81464b9cdec3121866d27930867594f2)

- invited_talk [Reasoning with LLMs: Challenges and Opportunities](https://docs.xiaohongshu.com/doc/066f239f95cea7f8c6f79f2e3042385b)

- invited_talk [Implications of Large-Scale Test-Time Compute](https://docs.xiaohongshu.com/doc/ef5447304bdcd18627ac5db39d3466b1)

- invited_talk [Towards Causal Artificial Intelligence](https://docs.xiaohongshu.com/doc/1bcb90ebebd62f588ff694de9f4cb955)

- oral [AVATAR-AGENT: A Multi-Agent LLM Planning System for 3D Avatar Generation](https://docs.xiaohongshu.com/doc/a934176551ff6572ec75abdef68bda2b)

- oral [LM-Landmarks: Language Model Guided Landmark Generation for Classical Planning with Formal Soundness Guarantees](https://docs.xiaohongshu.com/doc/19fb4ee8775f6723904ccf472a996d10)

- oral [Reward Prediction with Factorized World States](https://docs.xiaohongshu.com/doc/bada6645d7f6a66eecb38a9107115b62)

- oral [HiPER: Hierarchical Plan–Execute RL for Multi-Turn LLM Agents](https://docs.xiaohongshu.com/doc/0af1c5ac92bf565349dc3c16b329bf46)

- oral [ICPRL: Acquiring Physical Intuition from Interactive Control](https://docs.xiaohongshu.com/doc/b0c2b0b67b24a5bec426275487360740)

- oral [OrigamiBench: An Interactive Environment to Synthesize Flat-Foldable Origamis](https://docs.xiaohongshu.com/doc/b964ff9164b53b0d3cee31ff1454f518)

- paper [Specialized LM Agents with Simulation-Verified Search for Service Workforce Planning](https://docs.xiaohongshu.com/doc/9b25abb58931ff0bf890f8c6aa81c06c)

- paper [End-to-End LLM Flight Planning with RAG-based Memory and Multi-modal Coach Agent](https://docs.xiaohongshu.com/doc/8540be9aaf8fef3d6707c334763755bd)

- paper [VeryTrace: Verifying Reasoning Traces through Compilable Formalism and Structured Verification](https://docs.xiaohongshu.com/doc/4b548e499cc1d6b5de853c5bf89c7495)

- paper [AVATAR-AGENT: A Multi-Agent LLM Planning System for 3D Avatar Generation](https://docs.xiaohongshu.com/doc/5fd4012fbeb9e5b7eafd9bb16cdb2ff4)