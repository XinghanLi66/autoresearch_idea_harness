## 总览

- 目标问题：当前 proposal training 到底学到了什么、没学到什么

- **最重要的结论：**<font color="#00A14B">**fine-tuning 是有效的，**</font><font color="#CC3030">**但当前 reward 和 target 主要让模型学会“从 references 写出像研究方向的 proposal”，还没有稳定学会“给出可执行 implementation detail”**</font>**。** Benchmark 的高分有相当一部分来自 worker model 对 LR schedule 的默认先验，而不是 proposer 自己给出了足够具体的参数化设计。

- 下一步：TeX-grounded target，及原因。

### 结论速查

| 结论 | 关键证据 | 对下一步的含义 |
| --- | --- | --- |
| Fine-tuning 真正有用 | Qwen base 在 full_refs、related_work、top_k_refs 上 p@1 都是 0；fine-tuned checkpoints 在这些无显式任务条件下可稳定通过 | 微调学到了instruction following以外的内容：<font color="#00A14B">**在没有显式question给出的情况下，模型依然可以自己提出方向提出idea**</font> |
| related_work / top_k_related_work 是更好的 inference context | related_work 跨实验均值 p@1 = 0.168 最高；top_k_related_work 均值 Δ = +0.060 最高 | 推理时应<font color="#00A14B">**优先使用叙述式 context，而不是 raw refs list**</font> |
| <font color="#CC3030">**PRS**</font> reward 冲顶能力强，<font color="#CC3030">**但容易 abstract overfit**</font> | exp09 PRS 最高 p@1 = 0.30，p@5 = 0.871；但 proposal 多停留在 agenda 层 | PRS 可以保留作 baseline，但不能单独解决 detail 缺失 |
| <font color="#00A14B">**FAS reward 分布更稳**</font> | exp15 FAS 最高 p@1 不如 exp09，但均值 Δ = +0.082，为所有实验最高 | FAS 更像质量分布 reward，适合作为后续 auxiliary objective |
| PPL reward 没有明显胜出 | exp17 PPL 最高 p@1 = 0.25，均值 p@1 = 0.150，均值 Δ = -0.010，并出现较低 diversity | PPL 有方向性，但当前没有明显优势 |
| <font color="#CC3030">**LoRA 在这个任务上失败**</font> | exp14 均值 p@1 = 0.040，均值 Δ = -0.178，并出现 template leakage | proposal RL 更需要 full fine-tuning 或更强 adaptation |
| <font color="#CC3030">**任务本身有 worker-prior bias**</font> | 多数通过实现收敛到 warmup + cosine + small floor；约 85% 即使 proposal 提到其他 schedule，worker 仍实现 warmup+cosine | 下阶段 benchmark 要增加更需要 proposal-specific detail 的任务 |

## 核心 raw data 速查

### Benchmark 规模

| 项目 | 数值 |
| --- | --- |
| 训练实验 | exp09 到 exp17，共 9 个 checkpoints |
| Evaluation strategies | full_refs、top_k_refs、related_work、with_research_question、top_k_related_work |
| 计划样本量 | 9 × 5 × 20，约 900 次 worker evaluation |
| 任务 | dl_lr_schedule / resnet20-cifar10 |
| 通过阈值 | test acc ≥ 93.01% |
| 总通过数 | 155 samples |
| 整体通过率 | 约 17% |
| 异常样本 | 7 个，来自一次 worker 卡死事件 |
| 估算计算量 | 200 到 300 GPU-hours |

### 最值得记住的 run

| Run | 为什么重要 | 关键数字 |
| --- | --- | --- |
| exp09 × top_k_related_work | 全场最高峰值，说明 PRS + full-FT 可以冲 benchmark | p@1 = 0.30，p@5 = 0.871，p@10 = 0.995 |
| exp16 full_refs 20×800 | 平均 p@1 最高，但 Δ 接近 0，显示更多 RL 有 reward hacking 风险 | mean p@1 = 0.180，mean Δ = -0.008 |
| exp15 FAS | 平均质量最好，说明 FAS 更稳 | mean Δ = +0.082 |
| exp17 PPL | reward 有方向性但没有胜出，且 diversity 较低 | best p@1 = 0.25，mean p@1 = 0.150 |
| exp14 LoRA | 负例，说明 LoRA adaptation 不够 | mean p@1 = 0.040，mean Δ = -0.178 |
| Qwen base + with_research_question | 重要对照，说明显式 task question 会绕过 proposal-from-refs 难题 | p@1 = 38.9% |
| Qwen base + raw refs strategies | 证明 fine-tuning 的真实增益来自无显式任务条件 | full_refs、related_work、top_k_refs 均为 0% |

### Reward 对比

| Reward | 代表实验 | 观察到的行为 | 当前判断 |
| --- | --- | --- | --- |
| PRS | exp09、exp10、exp11、exp12、exp13、exp16 | 峰值最高，容易产出能过阈值的 proposal | 强 baseline，但会鼓励 abstract-level recovery |
| FAS | exp15 | 最高通过率不一定最高，但均值 Δ 最好 | 更像质量分布 reward，值得继续用 |
| PPL | exp17 | p@1 中等，输出分布更集中 | 概念上有方向性，但当前没有明显优势 |
| Format-only component | 所有 RL reward 都包含 | 保证 XML structure 和 worker 可读性 | 必要但不区分 proposal quality |

### Strategy 对比

| Strategy | 跨实验均值 p@1 | 均值 Δ | 解释 |
| --- | --- | --- | --- |
| related_work | 0.168 | +0.052 | 最稳定，叙述式 literature context 最利于 proposer |
| with_research_question | 0.136 | +0.054 | 对 base model 极强，但可能泄露显式任务方向 |
| top_k_related_work | 0.128 | +0.060 | 平均质量最好，top-k filtering + narrative synthesis 有价值 |
| full_refs | 0.133 | -0.106 | 偶尔能过，但 raw full list 噪声大 |
| top_k_refs | 0.096 | -0.107 | 最弱，raw top-k refs 不如叙述化 context |

## 如何阅读后面的详细结果

后面的表格和代码分析应当按三层来看。

第一层是 **benchmark score**：p@1、p@5、p@10 反映 proposal 能否让 worker 产出过阈值实现。但这个 task 的默认强解是 warmup + cosine，所以分数不能完全等同于 proposer 的真实创造力。

第二层是 **proposal quality**：更重要的是 proposal 是否把 reasoning 落到具体 algorithm choice、parameter range、training recipe 和 evaluation plan。当前模型在这一层还明显不足。

第三层是 **training signal diagnosis**：PRS、FAS、PPL 分别对应 abstract recovery、future alignment 和 conditional predictability。当前结果显示 reward 换法有帮助，但不如 target 和 condition 的 detail level 重要。

因此，最直接的下一步不是继续堆更多 abstract-level reward，而是进入 TeX-Grounded Training Target Plan：用 full TeX source 合成 implementation-ready target，再用 reference evidence 和 paper classification 让 condition 与 target detail level 匹配。

## 原始数据与详细分析

### 1. 评测规模

本次 benchmark 覆盖 **9 个训练实验**（exp09–exp17），每个实验在 **5 种 evaluation strategy** 下分别评测，构成 9 × 5 = 45 个独立的（实验, 策略）运行配置。每次运行名义上抽取 **20 个 sample**，共计 **约 900 次评测**。每个 sample 是一次独立的 200-epoch ResNet-20/CIFAR-10 训练（H800 上约 15–20 分钟），总计算量约 **200–300 GPU-hours**。

- 155 个 sample 通过阈值（test acc ≥ 93.01%），整体通过率约 **17%**

- 7 个 sample 报错（ENOSPC 或 MLS-Bench eval 失败），均源于一次 worker 卡死事件

- 6 个 run 因孤儿 worker 进程仅完成 18–19 个 sample（而非 20 个）；mini-sweep 已补跑完毕

5 种 **evaluation strategy** 的区别在于推理时向模型提供的参考文献上下文形式：

| Strategy | 说明 |
| --- | --- |
| full_refs | 完整参考文献列表（原文） |
| top_k_refs | 仅保留 top-k 最相关文献 |
| related_work | 由 LLM 将文献提炼为一段叙述性 related work |
| with_research_question | top-k 文献 + LLM 生成的 research question |
| top_k_related_work | 先筛选 top-k 文献，再合成为 related work 叙述 |

---

## 2. 各实验与各策略性能

### 按实验（展示最优策略）

| Exp | 训练配置 | 最优 strategy | 最高 p@1 | 各策略均值 p@1 | 均值 Δ |
| --- | --- | --- | --- | --- | --- |
| exp09 | top_k_refs, SFT+RL (PRS) | top_k_related_work | **0.30** | 0.171 | +0.016 |
| exp16 | full_refs 20×800, SFT+RL (PRS) | related_work / with_rq / full_refs | 0.25 | **0.180** | −0.008 |
| exp15 | FAS reward，从 exp09 SFT 初始化 | full_refs | 0.20 | 0.164 | **+0.082** |
| exp13 | full_refs, SFT+RL (PRS) | related_work | 0.25 | 0.121 | +0.042 |
| exp10 | related_work, SFT+RL (PRS) | top_k_refs | 0.25 | 0.130 | +0.040 |
| exp17 | top_k_refs, PPL reward | with_research_question | 0.25 | 0.150 | −0.010 |
| exp12 | research_q, SFT+RL (PRS) | with_research_question | 0.37* | 0.115 | +0.002 |
| exp11 | topk+rw, SFT+RL (PRS) | related_work | 0.20 | 0.120 | −0.068 |
| exp14 | full_refs, **LoRA** SFT+RL (PRS) | top_k_related_work | 0.15 | **0.040** | **−0.178** |

*exp12 with_rq：n=19 的 run 未跑完；n=20 的 run（含 4 个错误）p@1 = 0.125。

**各实验核心发现：**

- **exp09** 取得全场最高 p@1（0.30）和 p@5（0.871）。训练时使用 top_k_refs 上下文，却在 top_k_related_work 策略下表现最好——说明叙述式合成（narrative synthesis）是推理阶段普遍更优的上下文格式。

- **exp16**（训练步数为标准 run 的 2 倍）均值 p@1 最高（0.180），但均值 Δ 接近零（−0.008）。更多 RL 训练能让更多 proposal 越过阈值，却无法提升整体 proposal 平均质量——这是 reward hacking 的典型特征。

- **exp15**（FAS reward，从 exp09 SFT checkpoint 初始化）均值 Δ 最高（+0.082）。FAS 优化的是与研究领域实际演进方向的对齐度，而非二元通过率。其生成的 proposal 在整个分布上都更可靠，而不只是在阈值边界附近偶尔取胜。

- **exp14**（LoRA）是绝对的末位。均值 p@1 = 0.040，而全参数模型均 ≥ 0.144。四个 strategy 中三个 p@1 = 0.000。均值 Δ = −0.178，最差单格为 −0.63。LoRA rank 不足以在 GRPO fine-tuning 下将模型表示改造到足以可靠生成结构化 proposal 的程度。

### 按 Evaluation Strategy（跨所有实验汇总）

| Strategy | 总通过数 | 均值 p@1 | 均值 p@5 | 均值 Δ |
| --- | --- | --- | --- | --- |
| related_work | 30 | **0.168** | **0.610** | +0.052 |
| with_research_question | 24 | 0.136 | 0.517 | +0.054 |
| top_k_related_work | 23 | 0.128 | 0.478 | **+0.060** |
| full_refs | 24 | 0.133 | 0.510 | −0.106 |
| top_k_refs | 17 | 0.096 | 0.396 | −0.107 |

related_work 是最稳定有效的策略——在 9 个实验中的 5 个里排名第一或并列第一。将文献预处理为连贯叙述，在通过率和均值 Δ 两个维度上都稳定优于原始列表。

top_k_refs 是整体最弱的策略。值得注意的是，exp09 本身就是用 top_k_refs 训练的，但用 top_k_refs 评测时却得到最差结果（p@1 = 0.053，Δ = −0.41）。跨格式迁移是真实存在的——模型可以泛化到不同上下文格式——但推理时的上下文格式仍有显著影响。

full_refs 呈现双峰分布：尽管通过数不少，均值 Δ = −0.106，主要拖累来自 exp11×full_refs（3 次通过，但均值 Δ = −0.51）。完整文献列表引入了噪声，拉低了整体 proposal 平均质量，尽管偶尔能产出越过阈值的 proposal。

### Pass@k 结构

p@1 → p@5 → p@10 的递进关系揭示了一个高多样性的输出分布。最优 run（exp09×top_k_related_work）：p@1 = 0.30，p@5 = 0.87，p@10 = 0.995。这一比例（p@5 ≈ 3× p@1）符合独立 Bernoulli 采样的理论曲线——通过与否不存在聚类，每个 sample 独立探索解空间。**所有非 LoRA 格子的均值 p@10 ≥ 0.84**，意味着 best-of-10 采样策略几乎对任何训练 checkpoint 都能可靠找到一个通过的 proposal。

---

## 3. 生成 Proposal 的多样性分析

### 表面多样性 vs. 结构同质性

从 5 个实验中抽样阅读 12 份 proposal.txt 后，呈现出一致的规律：**表面多样性高，结构多样性低**。

Proposal 的研究框架跨度极大——loss landscape 几何、PAC-Bayes 理论、生物可塑性类比、super-convergence 理论、IoT 生命周期评估框架、character-level language modeling 等。&lt;thinking&gt; 块在文献综合和 gap 识别上差异显著。这种表面多样性是真实的，也符合 temperature sampling 生成的预期。

然而，无论研究框架如何包装，proposal 最终都收敛到相同的元层研究议程结构：（1）识别 gap，（2）提出大规模实验研究或统一框架，（3）描述一个提及「warmup」「cosine annealing」或「cyclical learning rates」却不给出具体参数的 approach。

### 各实验系统性差异

**exp09 / exp15 / exp16**（全参数模型）：proposal 结构良好，&lt;thinking&gt; 推理链条连贯、多步骤展开，引用具体文献编号，写作风格接近学术论文。

**exp12**（research_q SFT+RL）：proposal 最为聚焦——模型明显更倾向于针对 LR scheduling 提出具体假设（如「我们假设不同 schedule 的渐近收敛速率因 batch size 不同而存在本质差异」）。这说明训练时在上下文中加入 research question，确实引导了模型生成更具体的 proposal。

**exp17**（PPL reward）：同一 run 内多样性较低。同一 run 中多个 sample 收敛于几乎相同的研究框架（「LR schedule 在 character-level language modeling 上的对比研究尚未充分」），说明 PPL reward 在输出分布中形成了更强的众数。

**exp14**（LoRA）：质量明显退化。多个 proposal 出现模板泄漏——XML 标签的问题文本出现在开闭标签之间而非被答案替换。&lt;thinking&gt; 部分更短且程式化。完全没有针对 LR scheduling 的具体推理。

### 与参考文献内容的结合程度

所有模型都能正确识别文献的主题簇（LR scheduling、residual architectures、loss landscape、distributed training）。然而，模型对文献的使用几乎停留在关键词提取层面，而非分析性综合。没有任何 proposal 将某篇论文的定量发现转化为具体可操作的洞察（例如「super-convergence 在 CIFAR-10 上用 1-cycle、peak LR ≈ 0.5 可减少 10 倍训练迭代次数——应直接测试这一 regime」）。

最典型的例外是 exp12/sample_09（93.25%），其 &lt;key_insight&gt; 写道：_「LR schedule 的有效性不仅由其数学形式决定，更根本地取决于它如何在训练过程中——尤其是在高维参数空间中——调节梯度噪声和曲率。」_ 这是一个有理论依据的论断，实质性地超越了对文献的简单引用。

---

## 4. 最优实现的代码分析

### 主流策略

top-10 通过率最高的实现（93.27–93.39%）中，9/10 属于同一结构家族：

**线性 warmup（5 epochs）→ cosine annealing 到较小的 floor 值**

```python
warmup_epochs = 5
min_lr = 1e-4          # 更简单的变体中为 0.0
if epoch < warmup_epochs:
    return min_lr + (base_lr - min_lr) * (epoch + 1) / warmup_epochs
progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

两种子变体的区别仅在于 cosine 尾部是衰减到 0 还是衰减到 min_lr ≈ 1e-4。两者性能在统计上无法区分——±0.05% 范围内的差异由 SGD 随机性主导，而非 schedule 设计本身。

多个实现中出现了基于架构的条件分支，但实际上都是虚设的：无论执行哪个分支，min_lr 或 warmup_epochs 的值都相同。这种分支是结构性的姿态，并非真正的功能差异化。

### 异类：1-Cycle Schedule（93.37%）

得分第二高的实现（sample_11，run 93af286d，exp15）是唯一真正与众不同的——一个三阶段 1-cycle policy：

```python
min_lr = base_lr / 25.0          # 0.004
final_lr = base_lr / 1000.0      # 0.0001
peak_epoch = int(total_epochs * 0.3)     # epoch 60：线性爬升阶段
anneal_epoch = int(total_epochs * 0.85)  # epoch 170：cosine 衰减至 min_lr
# 最后 15%：cosine 衰减 min_lr -> final_lr

if epoch <= peak_epoch:
    progress = epoch / peak_epoch
    return min_lr + (base_lr - min_lr) * progress
elif epoch <= anneal_epoch:
    progress = (epoch - peak_epoch) / (anneal_epoch - peak_epoch)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
else:
    progress = (epoch - anneal_epoch) / (total_epochs - anneal_epoch)
    return final_lr + 0.5 * (min_lr - final_lr) * (1 + math.cos(math.pi * progress))
```

参数选择（peak_epoch = 0.3 × total，min_lr = base_lr / 25）与 Smith 发表的 1-cycle 建议一致。这是整个 benchmark 中唯一一个模型明确调取并正确应用了特定命名算法的实现。

### 失败实现：哪里出了问题

**擦线未过（92.8–92.95%）：** 实现结构与通过实现完全相同。未过阈值的差距在 SGD 随机方差范围内，并非设计失误。

**SGDR / cyclic restarts（92.42–92.60%）：** 当模型实现 warm restart（T₀ = 50 epochs，T_mult = 2 等）时，性能比同类非重启 schedule 低 0.3–0.8%。在 200-epoch 固定预算训练的后期，LR 周期性重启会破坏已良好收敛的状态。模型知道 SGDR 算法，但不知道它什么时候会有害。

**exp14（LoRA）灾难性失败（84.96–90.13%）：** 有 sample 实现了 max_lr = base_lr × 5.0 的 super-convergence one-cycle——这种 regime 适用于极短的训练 run，而非 200-epoch ResNet-20。这证实了 LoRA 模型只是从 proposal 中提取了 scheduling 关键词，而没有理解其在当前任务下的合理参数范围。

### 代码质量

跨所有实验，代码质量整体较高：PyTorch 用法正确，cosine 公式有界，除法安全，参数量级合理。import math 写在函数体内略显非惯用，但无害。没有任何 sample 直接修改 optimizer state 或错误使用 PyTorch scheduler 对象。函数形式始终是纯函数（epoch → float），符合任务要求。

---

## 5. 模型是否真正掌握了核心逻辑？

**简短结论：部分掌握，但深度不足以泛化到 warmup+cosine 模板之外。**

### 学到了什么

模型已<font color="#00A14B">**正确内化了以下认知**</font>：

- Warmup 阶段通过避免随机初始化后的大梯度步长来稳定早期训练

- Cosine annealing 平滑地向 flat minima 衰减，而不是震荡或骤停

- 保留一个小的正 floor（约 1e-4）可防止 LR 降至零、在训练后期失去所有自适应性

- SGDR 等 cyclic restart 策略并非普遍有益，在固定预算训练上可能有害

这通过 9 个独立训练 checkpoint 在 5 种 evaluation strategy 下一致收敛到 warmup+cosine 家族来印证。一个完全不理解的模型会生成随机 schedule；而这里的模型以较高概率生成了正确类别的 schedule。

### 没学好什么

核心失败模式是<font color="#CC3030">**有抽象、无落地**</font>。Proposal 对 warmup 和 cosine 有效原因（曲率、loss landscape 几何、梯度噪声）的论述相当精密，但这些推理从未以具体参数建议的形式进入 &lt;approach&gt; 部分。<font color="#CC3030">**Worker 模型（Claude Sonnet 4.6）随后用自己的先验填补了参数空白**</font>，默认生成 warmup=5 / cosine / floor=1e-4。

关键证据：在 &lt;thinking&gt; 块中讨论了 SGDR、super-convergence、polynomial decay 和 step schedule 的 proposal，仍有约 85% 的情况产出 warmup+cosine 实现。<font color="#CC3030">**模型知道这些替代方案_是什么_，但不知道_何时_该用**</font>。唯一的例外——1-cycle 实现——得分与默认策略持平，说明模型如果能可靠地选择替代方案，完全有能力有效使用它。

**整个 pipeline 存在层层抽象问题**：proposer（fine-tuned Qwen 7B）在研究议程层面写作；worker（Claude Sonnet）提取可操作关键词；benchmark 衡量最终实现结果。<font color="#CC3030">**目前 benchmark 的好成绩，在相当程度上反映的是 worker 模型关于好 LR schedule 的先验**</font>，而非 proposer 模型 proposal 本身的质量。

### 对下一阶段迭代的启示

当前 benchmark 通过率（各配置 p@1 在 6–37% 之间）是有意义的 proposal 质量信号，但存在较大噪声，原因如下：

1. 接近阈值的 run（92.85–93.00%）只能靠 SGD 随机性区分，而非 proposal 质量差异

1. Worker 模型的默认 schedule（warmup+cosine）自身即可达到约 93.0–93.2%——该任务不需要_特定的_ proposal，只需要一个_说得过去的_ proposal 即可

3. 产出真正新颖 schedule（1-cycle、改进版 SGDR）的 proposal 可以与默认策略持平甚至更好，但训练信号没有稳定地奖励这种具体性

TeX-Grounded Training Target Plan（见独立文档）通过将 proposal 锚定到具体论文的实际实验轨迹来解决这一问题，应能促使 proposer 模型给出更具操作意义的具体建议。

---

_分析生成时间：2026-05-25。数据：9 checkpoints × 5 strategies × 20 samples，任务 dl_lr_schedule/resnet20-cifar10，runs 目录：/newcpfs/lxh/agentic-training/proposal_rl/runs/benchmark/。5 个不完整 combo 的 mini-sweep 已于同日完成补跑。_

---

## 附录：Qwen2.5-7B-Instruct 基座模型 Baseline（2026-05-26）

为了解 SFT+RL 微调带来的实际增益，我们用原始基座模型（未经任何微调）对 dl_lr_schedule/resnet20-cifar10 进行了全量评测：5 种 prompt 策略 × 20 样本，共 100 次评测。

### 结果汇总

| 策略 | 通过/总数 | p@1 | p@3 | p@5 | p@10 | 均值偏差 |
| --- | --- | --- | --- | --- | --- | --- |
| with_research_question | <font color="#00A14B">**7/18**</font> | <font color="#00A14B">**38.9%**</font> | 79.8% | 94.6% | 100.0% | +0.16% |
| top_k_related_work | 1/20 | 5.0% | 15.0% | 25.0% | 50.0% | −0.02% |
| full_refs | 0/20 | 0.0% | 0.0% | 0.0% | 0.0% | −0.19% |
| related_work | 0/20 | 0.0% | 0.0% | 0.0% | 0.0% | −0.12% |
| top_k_refs | 0/20 | 0.0% | 0.0% | 0.0% | 0.0% | −0.44% |

（均值偏差 = 样本平均 val_metric − 参考基线 92.71%；with_research_question 仅完成 18/20 样本）

### 关键观察

**1. with_research_question 异常突出（p@1 = 38.9%）**

基座模型在该策略下的 p@1 甚至高于所有微调 checkpoint 的最高记录（exp09 p@1 = 30.0%）。原因推测：with_research_question 在 prompt 中显式给出了研究问题（"improve the learning rate schedule"），与 LR scheduling 的任务形式完全对齐——**这是基座模型通过指令理解直接解决的，不需要任何 proposal 泛化能力。**

**2. 其余四种策略几乎归零**

full_refs、related_work、top_k_refs 的 p@1 = 0.0%，说明基座模型在给定纯参考文献上下文但没有显式任务说明的情况下，不能可靠地生成有效 LR schedule proposal。

**3. SFT+RL 微调的核心价值**

微调的实际贡献在于让模型在 top_k_refs、full_refs、related_work 这些<font color="#00A14B">**只有文献上下文、没有明确任务说明的策略下也能稳定通过**</font>。最佳微调配置（exp09 top_k_related_work，p@1 = 30.0%）在这些策略上超过基座的幅度，才是真正的微调增益。

**4. Pipeline 配置确认**

实际评测使用 pipeline.py（sweep 正式流程）：每样本最多 30 轮对话、7200s 超时，与 MLS-Bench 官方配置（2h / 30 turns）一致。

_基座模型评测完成时间：2026-05-26 17:06 CST。数据位于 runs/benchmark/dl_lr_schedule_resnet20-cifar10_&#123;bc9c60e3,77ad7003,5fcd7442,1ffedc32,229eba63&#125;。_