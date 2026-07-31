## 背景

本报告分析当前 benchmark 评估体系的设计缺陷，核心问题是：**20 次 worker rollout 共享同一个 checkpoint 输出，导致 pass@k 指标测量的不是 checkpoint 的生成多样性，而是 worker 的实现能力**。

---

## 一、Checkpoint 生成：确定性输出

Checkpoint 在生成 proposal 时使用了 **greedy decoding**（do_sample=False），无论 temperature 参数设置为何值，输出均完全确定。

同时，所有 20 个 sample 共享同一个缓存的 prompt（相同论文、相同策略），由 prompt_cache.py 中的 get_or_build() 统一返回。

**结论：相同输入 + greedy decoding = 20 份完全一致的 proposal.txt。**

经过对 dl_lr_schedule_resnet20-cifar10_b1702e64 run 的全部 20 份 proposal.txt 进行 diff 验证，结果确认所有文件内容完全相同。

**Run b1702e64 的完整配置：**

- **Checkpoint**：exp12_research_q_sft_rl_20260510_193056/rl/final（exp12，with_research_question prompting 策略，full-FT，PRS reward）

- **Prompting 策略**：with_research_question — 在 prompt 中附加 research question 字段

- **Task / Subtask**：dl_lr_schedule / resnet20-cifar10（ResNet-20 on CIFAR-10，baseline accuracy = 92.71%，pass 阈值 ≥ 93.01%）

- **n_samples**：20，所有 sample 使用同一 prompt cache（相同参考论文集）

- **Worker**：Claude Code CLI（claude-sonnet-4-6），max_turns=20（已修复为 30），单 GPU 串行执行

---

## 二、Worker 忠诚度：名义上忠诚，实际上各自创作

### Worker prompt 的约束

worker_prompt.txt 明确要求：

> "只实现 proposal 中描述的内容，不添加其他改进、优化或修复。"

### 约束为何失效

该 run 的 proposal 核心思路是：**通过监控梯度幅度和 loss 变化速率来识别训练阶段，并自适应调整学习率**。

以下是 exp12 checkpoint 实际生成的 proposal 原文（节选关键段落）：

> **&lt;problem&gt;** Training deep neural networks requires careful manual tuning of learning rates ... A principled, automated approach to adapting learning rates throughout the training process could substantially improve both efficiency and accuracy.
> 
> **&lt;gap&gt;** Existing learning rate scheduling methods (e.g., SGDR, cyclical learning rates) rely on fixed schedules or require manual tuning of boundary values, and they do not account for the changing optimization landscape across training stages ... There is thus a clear opportunity to design a training framework that automatically adapts learning rates based on observable training signals rather than relying on fixed schedules.
> 
> **&lt;key_insight&gt;** The key hypothesis is that training deep networks involves distinct optimization stages (e.g., warm-up, fine-tuning, optimization) that require different learning rate regimes, and that these stages can be detected and handled automatically by monitoring training signals such as gradient magnitudes and loss changes.
> 
> **&lt;approach&gt;** The proposed framework would continuously monitor training signals—such as the magnitude of gradient updates and the rate of loss change—to identify which stage of training the model is currently in. Based on these signals, the framework would automatically adjust the learning rate to match the appropriate regime for that stage. For example, during the warm-up phase, the learning rate would be increased rapidly to encourage exploration of the loss landscape, while during the optimization phase, the learning rate would be reduced to stabilize convergence. The framework would also incorporate a dynamic scheduling component that further modulates the learning rate based on training progress ... The entire system would be designed to be architecture-agnostic and applicable across diverse tasks.

但 get_lr(epoch, total_epochs, base_lr, config) 函数签名中**根本不包含梯度或 loss 信息**，所有 worker 都必须用 epoch 分数来近似替代训练阶段检测。

<redoc-comment commentGid="7641893748899087758" blockId="0418061da82f9b705e4c12e36f8014b4">Proposal 同时未指定任何具体超参数：没有 warmup 长度、没有 cosine 衰减起点、没有 min_lr 数值、没有分几个阶段。</redoc-comment>

**结论：约束在形式上存在，但由于 proposal 本身是概念性描述而非工程配方，20 个 worker 实际上独立创作了 20 个不同的 LR schedule 实现。**

---

## 三、20 次 Rollout 的方差来源

以下是各 sample 的实现差异对比（dl_lr_schedule_resnet20-cifar10_b1702e64，baseline = 92.71%，pass 阈值 ≥ 93.01%）：

| Sample | val_metric | Warmup 结束 | 主衰减起点 | 阶段数 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 08 | **93.37%** ✅ | epoch 10（5%） | epoch 20（**10%**） | 2 | 最激进的早期衰减，唯一通过 |
| 15 | 93.06% | epoch 10 | epoch 80（40%） | 3 | — |
| 16 | 92.99% | epoch 5~8 | epoch 60（30%） | 4 | 唯一使用 config 做架构感知 |
| 02 | 92.97% | epoch 10 | epoch 10 | 3 | 最后 20 epoch 固定 LR |
| 01 | 92.83% | epoch 5（固定） | epoch 5 | 2 | min_lr 用绝对值，非相对值 |
| 00 | 92.78% | epoch 10 | epoch 10 | 4 | 探索期到 epoch 100，过长 |
| 09 | null ❌ | — | — | 4 | max_turns 耗尽（训练已完成，结果未读取） |
| 10 | null ❌ | — | — | 3 | bash 管道 bug，估计 ~93.53%，本应通过 |
| 13 | null ❌ | — | — | — | /tmp 磁盘满，OS 错误 |

**方差的决定性变量**是主 cosine 衰减的**起点**。Sample_08 在 epoch 20（10% 处）开始衰减，此后 180 个 epoch 持续下降至 min_lr，是最激进的策略，也是 CIFAR-10/ResNet-20 上表现最好的策略。

相比之下，设计了 4 阶段"探索期"的 sample（如 00、16）将高 LR 维持到 epoch 60~100，反而导致网络在平坦区停滞。

---

## 四、3 个 Error Sample 的真实原因

这 3 次失败均与实现质量无关：

- **sample_09**：训练正常完成（约 63 分钟），但 worker 用 20 轮 turn 反复轮询后台任务，在训练完成前耗尽了 max_turns=20，导致 result.json 未被读取。

- **sample_10**：worker 执行 bash run.sh result.json 2&gt;&1 &#124; tail -30，管道导致 result.json 写入目录错误，无法被 pipeline 发现。估计实际 val_metric 约为 93.53%，本应通过。

- **sample_13**：操作系统报错 No space left on device: /tmp/pymp-xxx，纯基础设施故障。

---

## 五、pass@k 指标的有效性分析

### 当前计算方式

pipeline.py 使用 Codex/HumanEval 的无偏估计公式：

```plaintext
pass@k = 1 - C(n_fail, k) / C(n_done, k)
```

该公式的设计前提是：**k 次采样来自同一模型的 k 次独立生成**，用于衡量模型能否生成多样化的解法。

### 当前设计下的实际含义

由于所有 20 个 proposal 完全相同，当前 pass@k 的实际含义是：

> "给定这一份固定的 proposal，k 个独立的 Claude Code worker 中至少有 1 个能成功实现它的概率。"

这测量的是 **proposal 的可实现性 × worker 的实现能力**，而非 **checkpoint 的生成多样性**。

### 对比：正确的 pass@k 应该测量什么

| 维度 | 当前设计 | 正确设计 |
| --- | --- | --- |
| Checkpoint 调用次数 | 1 次（greedy） | 20 次（sampled） |
| 20 份 proposal | 完全相同 | 各自不同 |
| Worker 数量 | 20 个并行 | 1 个/proposal |
| 方差来源 | Worker 实现差异 | Checkpoint 生成多样性 |
| 指标含义 | 单 proposal 可实现率 | Checkpoint 提案质量 |

---

## 六、改进建议

<redoc-highlight emoji="dengpao" fillColor="yellow">
**核心修复**：在 pipeline.py 的 _generate_one() 中开启采样解码，让每个 sample 生成不同的 proposal。

修改前：do_sample=False, temperature=1.0

修改后：do_sample=True, temperature=0.8, top_p=0.95
</redoc-highlight>

其他建议：

1. **更具体的 proposal 格式**：在 prompt 中要求模型在 &lt;approach&gt; 段输出可执行的超参数（如 warmup_epochs=10、eta_min_ratio=0.001），减少 worker 的自由裁量空间。

2. **修复 max_turns 耗尽问题**：将 --max-turns 从 20 调高至 30，或改用前台运行方式代替后台轮询。

3. **修复 bash 管道 bug**：run.sh 应直接运行，不通过管道传递，避免 result.json 写入路径错误。

4. **重命名现有指标**：在修复采样前，将 pass@k 改名为 worker_pass_rate，明确其实际含义。

---

## 七、结论

当前 benchmark 的方差完全来自 **worker（Claude Code）的随机性**，而非 checkpoint 的生成多样性。Pass@k 指标在当前设计下不能用于比较不同 checkpoint 的 proposal 质量。核心修复是将 greedy decoding 切换为温度采样，使每个 sample 获得不同的 proposal。

---

## 八、Proposal 抽象性问题：训练改进方向

### 问题描述

Benchmarker 报告指出，checkpoint 生成的 proposal 停留在概念层面，缺乏可操作的核心机制。以本次 run 为例：

- Proposal 描述"通过监控梯度信号识别训练阶段"，但 get_lr(epoch, total_epochs, base_lr, config) 函数签名中根本不包含梯度数据

- 20 个 worker 因此各自创作了完全不同的 LR schedule 实现

- 根本原因：proposal 描述的是**调查方向**（"我们应当研究 X"），而非**调查结论**（"X 有效，因为 Y 机制"）

**注意**：问题不在于缺少具体超参数（如 learning rate 数值）。标准的、无特殊意义的超参无需列出。真正的缺失是**核心机制的具体性**——如果某个设计选择是方法成功的决定性因素（例如异常大的学习率、特殊的阶段切换时机），则必须明确体现在 proposal 中。

### 根因分析

**1. Reward 目标本身是抽象的**

PRS reward = cosine similarity（proposal embedding, abstract embedding）。Embedding 捕捉的是主题层面的语义，而非表述精确度。"使用 transformer 层"与"使用 24 层 transformer，d_model=512，8 个 attention head"在 embedding 空间中几乎无法区分，两者获得相同的 reward 信号。

**2. SFT 训练数据在设计上回避具体性**

CoT 合成 prompt（synthesize_cot.py）明确要求：

&gt; "Do NOT mention the specific method names, algorithm names, or numerical results from the actual paper."
&gt; "The proposal should describe a DIRECTION plausible from the references, not the paper's actual solution."

这一规则是为了防止 leakage，但副作用是模型学会了用模糊语言写 proposal。7,358 条训练样本全部遵循这一模式，模型将抽象描述内化为正确风格。

**3. Reward 不区分机制陈述与方向陈述**

"可能可以通过监控梯度来检测训练阶段"与"通过计算梯度 L2 范数的滑动平均，当连续 5 步下降超过 20% 时触发阶段切换"在当前 reward 下得分相同。

### 改进方向

**方向 A（低成本，优先推荐）：修改 proposal 格式，要求包含核心机制**

修改 train/prompt_builder.py 中 _PROPOSAL_FORMAT 的 &lt;approach&gt; 描述，将"high level"改为要求模型陈述**最关键的设计决策及其原因**。同步修改 data/synthesize_cot.py 的合成 prompt，允许在 &lt;approach&gt; 中描述方法的**具体机制**，但仍禁止直接引用论文结果数据。

改前：

&gt; "How might the proposed method work at a high level? (3-5 sentences, no specific names)"

改后：

&gt; "What is the concrete mechanism of the proposed method? State: (a) the key algorithmic decision that makes it work, (b) what signal or criterion drives the key step, (c) one or two non-obvious design choices — skip standard choices like Adam or cosine LR unless they are the crux of the contribution. (3-5 sentences)"

**方向 B（低成本）：在 RL reward 中加入 hedging 惩罚**

在 train/verl_reward.py 的 compute_score() 中加入对模糊陈述语言的惩罚项：检测 proposal 中的 hedging 词（"may"、"could"、"might"、"would potentially"、"we propose to investigate"），以出现频率作为负向信号，权重 0.1。

**方向 C（中成本）：重新合成 SFT 数据，放开机制描述限制**

将 SYNTHESIS_USER prompt 中"禁止描述具体方法"的规则精化为：禁止引用论文名称、实验结果数值、模型具体名称，但**允许描述驱动方法的核心机制和关键设计选择**。重新运行 data/synthesize_cot.py（约 7 小时），以更具体的 SFT 样本重训模型。

**方向 D（高成本）：引入机制可验证性 reward**

构建评分规则，测量 proposal 的"可操作性"：是否包含可以直接映射到代码的具体条件、阈值、流程描述。适合作为后续专项实验。

### 推荐执行顺序

1. **立即执行**：方向 A（修改 proposal 格式和合成 prompt）+ 方向 B（hedging 惩罚）。实现成本低，可在当前实验轮次结束后直接部署，重合成 SFT 数据后验证效果。

2. **验证后执行**：如果 A+B 不足以改善，执行方向 C，以 7 小时重合成换取更根本的数据质量提升。

3. **长期研究**：方向 D 作为后续专项研究。