# <redoc-comment commentGid="7648962517743652277" blockId="fb13de55228dd1f06a37bb55c865b5af">Idea Proposal Training Ver 2.4 Plan</redoc-comment>

## Plan Summary

V2.4 的核心目标不是只训练一个会写 proposal 的窄模型，而是训练一个能承担 research master 角色的模块。这个 master 需要先提出 idea，也需要在 worker 实现过程中回答问题、判断失败原因、给出 revision，并保持通用 reasoning 和 coding guidance 能力。

V2.3 formal sweep 的输出会作为第一批训练原料。每个 sample 会保留 task_packet、proposal、expert forecast、worker log、eval result、settlement 和 continuous score label。后续训练优先使用连续分数和相对 improvement，不把 pass/fail 当成唯一标签，因为 pass 会非常稀疏。

## Training Data Collate

第一步应从 V2.3 sweep collate 一个统一的 training_table.jsonl。一行对应一次 proposal execution，包含：

- task、subtask、baseline、pass threshold、worker-only best 或 median。

- proposal module、proposal text、是否 empty worker-only control。

- expert probability、feasibility、novelty、expected delta、risk、confidence。

- worker metric、delta to pass、delta over worker-only、pass/fail、infra error 标记。

- settlement 里的 Brier score、log score 和 expert reliability 字段。

主要 label 应该是 continuous score，例如 raw metric、delta to pass、delta over worker-only、task-normalized percentile。pass/fail 只作为 auxiliary label。

## Training Forms

可以从三类训练开始：

1. Reward 或 value model：输入 task_packet + proposal，预测 continuous score、expected improvement 和 risk。它用于筛 proposal，也用于衡量 expert market 是否有真实信号。

2. <font color="#F06A1D">**DPO 或 preference training**</font>：输入是 task_packet，模型输出是 proposal。chosen 是同一 task 下真实分数更高的 proposal，rejected 是分数更低的 proposal。worker-only control 只作为 anchor，不作为 proposal target。

3. Expert calibration：输入 expert forecast 和 proposal/task 特征，标签是真实 outcome 和 continuous score。它不一定训练大模型，也可以先做 reliability weighting、calibration bucket 或简单 regression。

Worker edit trace 暂时不作为主训练目标。worker 失败可能来自实现能力、环境、timeout 或 benchmark 波动，不一定代表 idea 不好，因此更适合诊断，不适合第一阶段训练 master。

## Master Capability Preservation <font color="#F06A1D">（存疑？可以先不搞）</font>

训练 master 时不能只优化 task_packet -&gt; proposal。否则模型会变成窄 proposal generator，损失后续回答 worker 问题和 debug 的能力。

V2.4 应该采用 multi-role format，让同一个 master 学会多种模式：

- PROPOSE：根据 task 和 evidence 提出 idea。

- CRITIQUE：分析 proposal 的 feasibility、novelty、risk 和 expected delta。

- ADVISE_WORKER：worker 实现遇到 ambiguity、failure 或低分时，给出具体实现建议。

- REVISE：根据 eval failure 修改 idea 或缩小实现范围。

- SUMMARIZE_RESULT：读 worker log 和 result，解释成功或失败原因。

训练时需要混入 replay data，避免 catastrophic forgetting。一个保守的初始比例是：general reasoning/coding replay 约 40%，proposal generation 约 25%，critique/value reasoning 约 20%，worker-advice/revision dialogue 约 15%。比例后续根据能力退化和 V2.3 数据质量调整。

## Model Size And Architecture

7B 可以继续作为 cheap proposer 或 ablation baseline，但不应作为完整 master 的主力。完整 master 需要理解 ML task、读 evidence、生成 idea、指导 implementation、解释失败 log，7B 容量大概率不够稳定。

更合理的路线是：

- 7B：保留为 proposal module 和对照组。

- **32B：作为第一档可训练 master，优先尝试 LoRA 或 QLoRA multi-role training。**

- 强 API model 或更大 open model：作为 teacher、fallback master 和高质量 trace 生产者。

短期架构不要求一个模型包办所有。更稳的形式是 trainable proposer 负责批量产生 candidate ideas，value 或 expert model 负责预测收益，strong master 负责 worker dialogue 和复杂 debug。等 proposer/value 有稳定收益后，再考虑把它们蒸馏回一个 32B master。

## Proposal Granularity（还需讨论）

V2.4 需要明确 master 生成的 proposal 应该是什么粒度。我们不希望它只是 paper abstract 级别的愿景，也不希望它直接变成完整 code patch。理想 proposal 是一个介于 research idea 和 implementation contract 之间的对象：<font color="#F06A1D">**它要足够具体，让 worker 能独立开始实现；但也要保留研究空间，让 worker 可以根据代码结构做局部取舍。**</font>

一个合格 proposal 至少应该包含五类信息：

- Core mechanism：一句话说明真正的新机制是什么，而不是泛泛地说 improve training 或 use better regularization。

- Editable surface：指出大致应该改哪个 function、module 或 training component，让 worker 不需要重新理解整个任务。

- Implementation sketch：给出关键公式、状态变量、schedule、loss term 或 algorithm steps，但不要求写完整代码。

- Expected effect：说明为什么它可能超过 worker-only baseline，以及预期改善的是 optimization、generalization、calibration 还是 robustness。

- Risk and fallback：说明最可能失败在哪里，以及一个较小的 fallback version。

这个粒度的目标是让 proposal 具备可执行性和可评估性。太粗的 proposal 会把创造性压力全部转移给 worker，最后评估不到 master 的贡献；太细的 proposal 会退化成 code generation，而且会伤害 master 后续回答 worker 问题、debug 和 revise 的通用能力。

## How To Control Granularity

接下来我们会把 proposal 粒度作为 harness 中的显式控制变量，而不是只靠 prompt 口头要求。

第一，proposal prompt 会采用固定 schema，<font color="#F06A1D">**强制 master 输出 problem diagnosis、core mechanism、implementation sketch、expected effect、risk 和 fallback。**</font>每一段都有长度上限，避免变成长篇 survey 或完整代码。

第二，worker prompt 会把 proposal 解释为 implementation contract，而不是必须逐字照做的代码方案。worker 可以做工程上的局部调整，但如果偏离 core mechanism，需要在 worker log 里说明原因。这样可以区分 idea failure 和 implementation adaptation。

第三，expert market 会<font color="#F06A1D">**单独评估 proposal granularity**</font>。除了 success probability，还要看 proposal 是否 actionable、是否过度 vague、是否过度 prescriptive、是否相对 worker-only 有清晰增量。这些字段会进入 training_table，作为后续 filtering 或 reward model 的辅助特征。

第四，collate 训练数据时会按 granularity 分层。过粗的 proposal 不直接作为 chosen target；过细且像 patch 的 proposal 也不优先用于 master SFT。优先保留那些 worker 能执行、专家能判断、结果能归因的 proposal。

第五，V2.4 可以做 controlled ablation：同一 task packet 下让 master 生成 coarse、medium、detailed 三种 proposal，然后比较 worker success、expert calibration 和最终 metric。若 medium 粒度稳定最好，就把它作为后续 DPO 和 SFT 的主格式。

最终目标是训练一个 master，它提出的不是一句 abstract idea，也不是一段代码，而是一个清晰、可执行、可辩护、可被 worker 追问和修正的 research proposal。

## Raw And Canonical Proposal Interface

<font color="#F06A1D">**V2.4 不应该要求所有 proposal model 都先学会同一种高度格式化输出**</font>。我们希望 harness 能接入 API model、local checkpoint、future trained master、甚至弱模型 ablation。如果每个 proposal model 都必须单独做格式训练，系统会变得很脆，也不利于快速比较不同模型的真实 idea quality。

更稳的接口是两层：

- Raw proposal：完整保存 proposal model 的原始输出。它可以是自然语言、半结构化 markdown、旧 checkpoint 的自由格式输出，或者 API model 的较完整方案。raw proposal 是审计依据，不能被 normalizer 覆盖掉。

- Canonical proposal：由 harness adapter 或 proposal normalizer 把 raw proposal 转成统一内部结构，例如 core mechanism、editable surface、implementation sketch、expected effect、risk、fallback。worker、expert market 和 value model 优先消费 canonical proposal。

这个设计让<font color="#F06A1D">**格式约束主要落在 harness 层**</font>，而不是强迫每个 proposal model 本身都接受格式训练。**API model 只需要 prompt 尽量贴近 schema；旧 checkpoint 可以自由输出，再由 normalizer 整理；未来 trainable master 则可以逐步学习更稳定地产生 medium-granularity canonical proposal。**

normalizer 本身也必须可审计。每次转换都要保存 raw text、canonical fields、missing fields、parse success、granularity score、normalizer confidence，以及是否出现明显改写原意的风险。若 normalizer 信心不足，expert 和 worker 应该能同时看到 raw proposal 和 canonical proposal，避免因为格式化步骤损坏 idea。

训练时也要区分这两层。SFT 或 DPO 的 target 可以优先使用 canonical proposal，但必须能追溯到 raw proposal 和真实 outcome。value model 可以输入 canonical proposal 来获得更稳定的特征；但做错误分析时要回看 raw proposal，判断问题来自原始 idea、格式化损失，还是 worker 实现偏差。

因此，proposal 不需要一开始就是高度格式化的；真正需要稳定的是 harness 内部的 canonical interface，以及 raw 到 canonical 的可追溯转换链路。