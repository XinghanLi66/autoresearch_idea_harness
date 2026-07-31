## 概述

本报告记录 worker-only 基准实验的完整结果（2026-06-05）。实验使用 worker_only 策略：worker（claude-opus-4-6）**仅接收任务描述，不接收任何 proposal**，且被沙箱限制为只能修改 editable_region.py。

**实验目的**：测量 worker 在无 proposal 指导下的能力上限，以此作为新的通过阈值。任何 proposal-guided 模型超越此阈值，才视为 proposal 带来了增量价值。

**关键发现**：

- Worker-only 运行的最佳成绩即为新阈值（ceiling + 0.01pp），worker 自身通过率定义为 0/20

- <font color="#F06A1D">Jigsaw 历史数据存在 </font><font color="#F06A1D">**grader bug**</font>：旧版 grader 记录的是 norm_score 而非真实 leaderboard 百分位，旧结果（0.82–0.91）无效，需要重新评估

- cv_data_augmentation 旧阈值（93.87%）超出 worker 上限，历史上该任务 0 通过率并非模型能力不足，而是<font color="#F06A1D">阈值设置过高</font>

---

## 一、Worker-Only 运行结果

以下为 4 个任务的 worker-only 沙箱实验结果（n=19–20，2026-06-03）。**通过阈值 = worker 最佳成绩 + 0.01pp**，因此 worker 自身通过率定义为 0。

### 1.1 dl_lr_schedule（ResNet-20/CIFAR-10）

- 参考基线：warmup_cosine，92.71%

- Worker 最佳：93.18%（sample_12）→ 新阈值 **≥ 93.19%**

- Worker 通过率：**0/19（0%）**，均值 92.871%，最差 92.44%（sample_07）

| Sample | test_acc (%) | Sample | test_acc (%) |
| --- | --- | --- | --- |
| sample_00 | 93.03 | sample_10 | 93.03 |
| sample_01 | 92.58 | sample_11 | 93.15 |
| sample_02 | 93.11 | sample_12 | **93.18** ← best |
| sample_03 | 93.05 | sample_13 | 92.50 |
| sample_04 | 92.74 | sample_14 | 93.03 |
| sample_05 | — (missing) | sample_15 | 92.63 |
| sample_06 | 92.50 | sample_16 | 92.60 |
| sample_07 | 92.44 | sample_17 | 93.09 |
| sample_08 | 92.91 | sample_18 | 93.10 |
| sample_09 | 92.84 | sample_19 | 93.03 |

### 1.2 dl_activation_function（ResNet-20/CIFAR-10）

- 参考基线：GELU，92.97%

- Worker 最佳：93.34%（sample_07）→ 新阈值 **≥ 93.35%**

- Worker 通过率：**0/20（0%）**，均值 88.66%（含 sample_06 异常值 10.00%），有效均值 92.69%，最差有效值 91.96%（sample_08）

| Sample | test_acc (%) | Sample | test_acc (%) |
| --- | --- | --- | --- |
| sample_00 | 92.77 | sample_10 | 92.88 |
| sample_01 | 92.64 | sample_11 | 93.28 |
| sample_02 | 92.82 | sample_12 | 92.65 |
| sample_03 | 92.91 | sample_13 | 92.69 |
| sample_04 | 92.86 | sample_14 | 92.67 |
| sample_05 | 93.05 | sample_15 | 93.29 |
| sample_06 | **10.00** ← broken activation | sample_16 | 92.94 |
| sample_07 | **93.34** ← best | sample_17 | 92.45 |
| sample_08 | 91.96 | sample_18 | 92.83 |
| sample_09 | 92.80 | sample_19 | 92.34 |

### 1.3 cv_data_augmentation（ResNet-20/CIFAR-10）

- 参考基线：Cutout，93.67%

- Worker 最佳：93.75%（sample_15）→ 新阈值 **≥ 93.76%**

- Worker 通过率：**0/19（0%）**，均值 93.271%，最差 92.75%（sample_11）；sample_19 结果缺失（仅有 workspace/result.json = 93.65%，未计入）

| Sample | test_acc (%) | Sample | test_acc (%) |
| --- | --- | --- | --- |
| sample_00 | 93.17 | sample_10 | 93.18 |
| sample_01 | 93.32 | sample_11 | 92.75 |
| sample_02 | 93.38 | sample_12 | 93.04 |
| sample_03 | 92.80 | sample_13 | 93.52 |
| sample_04 | 93.69 | sample_14 | 93.12 |
| sample_05 | 93.01 | sample_15 | **93.75** ← best |
| sample_06 | 93.41 | sample_16 | 92.94 |
| sample_07 | 93.41 | sample_17 | 93.47 |
| sample_08 | 93.36 | sample_18 | 93.25 |
| sample_09 | 93.57 | sample_19 | — (missing) |

### 1.4 jigsaw-unintended-bias（Kaggle MLE）

- 评测指标：leaderboard 百分位排名（0 = 最差，1 = 第一名）

- Leaderboard：n=2634，中位数=0.934，均值=0.900

- Worker 完成率：**9/20**（5 超时，6 未写 result.json）

- Worker 通过率：**不适用**（无预设通过阈值）

- Worker 均值：0.0758（7.58th percentile），最佳 0.0850（sample_03，raw AUC=0.840）

| Sample | Percentile | Raw AUC | Status | Sample | Percentile | Raw AUC | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sample_00 | 0.0729 | 0.7588 | done | sample_10 | — | — | timeout |
| sample_01 | 0.0737 | 0.7675 | timeout+ws | sample_11 | 0.0676 | 0.7274 | error+ws |
| sample_02 | — | — | error | sample_12 | 0.0771 | 0.7920 | timeout+ws |
| sample_03 | **0.0850** | **0.8398** | done | sample_13 | — | — | error |
| sample_04 | 0.0740 | 0.7801 | done | sample_14 | — | — | error |
| sample_05 | 0.0737 | 0.7711 | done | sample_15 | 0.0805 | 0.8257 | timeout+ws |
| sample_06 | 0.0778 | 0.7996 | done | sample_16 | — | — | error |
| sample_07 | — | — | timeout | sample_17 | — | — | error |
| sample_08 | — | — | error | sample_18 | — | — | error |
| sample_09 | — | — | error | sample_19 | — | — | error |

> **注意（Jigsaw grader bug）**：2026-06-03 之前的所有 Jigsaw 评测结果（val_metric 0.82–0.91）均无效。旧版 grader 将 norm_score 错误存储为 val_metric，而非真实 leaderboard 百分位。当前 worker-only 结果（0.067–0.085）为正确值。旧数据需重新评估后方可用于比较。

---

## 二、新基线与阈值

| Task | 参考方法 | 参考基线 | Worker 上限 | 新阈值 | 旧阈值 | 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| dl_lr_schedule | warmup_cosine | 92.71% | 93.18%（sample_12） | **93.19%** | 93.01% | +0.18pp 更难 |
| dl_activation_function | GELU | 92.97% | 93.34%（sample_07） | **93.35%** | 93.27% | +0.08pp 更难 |
| cv_data_augmentation | Cutout | 93.67% | 93.75%（sample_15） | **93.76%** | 93.87% | −0.11pp 更容易 |

**阈值设计原则**：threshold = worker ceiling + 0.01pp，使 worker 自身 pass rate = 0/20（0%）。超越阈值 = proposal 带来了超出无约束 worker 自身能力的增量。

---

## 三、Worker 最优实现

每个任务中 worker 的最高分实现（code cell 格式展示）：

### dl_lr_schedule — 93.18%（sample_12，沙箱 v2）

实现：5 epoch linear warmup + cosine decay to floor（warmup_cosine 变体）

```python
def get_lr(epoch, total_epochs, base_lr, config):
    import math
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

跨所有 run（含 proposal-guided）的最高单次成绩：93.39%（run 354b9904，同一结构）。

### dl_activation_function — 93.34%（sample_07，沙箱）

实现：Learnable Swish / SiLU，带可训练标量参数 β：

```python
class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
```

跨所有 run 最高：93.48%（run a4865567，同一结构）。

### cv_data_augmentation — 93.75%（sample_15，沙箱）

实现：RandAugment（2 ops, magnitude 9）+ RandomErasing（p=0.5）：

```python
return transforms.Compose([
    transforms.RandomCrop(config['img_size'], padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(config['mean'], config['std']),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])
```

跨所有 run 最高：93.81%（run 18163080，同一组合）。

### jigsaw — 8.50th percentile（sample_03，raw AUC=0.840）

Worker 在 4 小时预算内完成数据预处理 + RoBERTa/BERT 训练 + submission 提交。成功率约 45%（9/20）；失败原因包括超时、内存不足、submission 格式错误。

---

## 四、历史 Proposal-Guided 运行重算（新阈值）

以下三张大表列出所有已完成的 proposal-guided sweep 结果，均按**新阈值**重算通过率。每行取该 experiment × task 下表现最佳的策略。

**说明**：

- pass/n：通过样本数 / 有效样本数

- p@1：通过率 = pass/n（越高越好）

- mean：所有有效样本的均值（%）

- best：单次最高成绩（%）

- worker-only 行：**0/n（0%）**，定义性结果，不与 proposal-guided 比较

### 4.1 dl_lr_schedule — 新阈值 93.19%

| Model / Checkpoint | Best Strategy | pass/n | p@1 | mean (%) | best (%) |
| --- | --- | --- | --- | --- | --- |
| **worker-only (v2 sandboxed)** | worker_only | **0/19** | **0%** | 92.87 | 93.18 |
| **exp09 top_k_refs SFT+RL** | top_k_related_work | 2/20 | 10% | 92.89 | 93.28 |
| exp10 related_work SFT+RL | top_k_refs | 1/20 | 5% | 92.87 | 93.21 |
| exp11 topk_rw SFT+RL | top_k_related_work | 1/15 | 7% | 92.83 | 93.25 |
| **exp12 research_q SFT+RL** | with_research_question | 1/16 | 6% | 92.87 | 93.37 |
| exp13 full_refs SFT+RL | with_research_question | 2/17 | 12% | 92.95 | 93.22 |
| exp14 full_refs LoRA | top_k_related_work | 1/20 | 5% | 92.75 | 93.22 |
| **exp15 FAS from exp09** | top_k_related_work | 2/20 | 10% | 92.83 | 93.27 |
| exp16 full_refs 20×800 | top_k_refs | 1/20 | 5% | 92.59 | 93.24 |
| **exp17 top_k_refs PPL** | top_k_related_work | 1/20 | 5% | 92.84 | 93.30 |
| claude-opus-4-6 | with_research_question | 1/20 | 5% | 92.57 | 93.23 |
| Qwen2.5-7B-Instruct base | with_research_question | 1/18 | 6% | 92.86 | 93.26 |

**结论**：新阈值下通过率普遍下降（原阈值 93.01% 下 exp12/exp13 可达 37%，现降至 6–12%）。exp09 和 exp13 表现最好（各 10%/12%）。

### 4.2 dl_activation_function — 新阈值 93.35%

| Model / Checkpoint | Best Strategy | pass/n | p@1 | mean (%) | best (%) |
| --- | --- | --- | --- | --- | --- |
| **worker-only (sandboxed)** | worker_only | **0/20** | **0%** | 88.66* | 93.34 |
| exp09 top_k_refs SFT+RL | related_work | 1/19 | 5% | 92.83 | 93.35 |
| exp10 related_work SFT+RL | with_research_question | 0/20 | 0% | 92.71 | 93.34 |
| exp11 topk_rw SFT+RL | top_k_related_work | 1/19 | 5% | 92.73 | 93.41 |
| **exp12 research_q SFT+RL** | top_k_related_work | 1/20 | 5% | 91.85 | 93.48 |
| **exp13 full_refs SFT+RL** | top_k_related_work | 1/20 | 5% | 92.14 | 93.45 |
| exp14 full_refs LoRA | with_research_question | 1/20 | 5% | 92.80 | 93.37 |
| exp15 FAS from exp09 | — | 0/20 | 0% | 92.76 | 93.30 |
| exp16 full_refs 20×800 | with_research_question | 1/20 | 5% | 92.84 | 93.42 |
| **exp17 top_k_refs PPL** | with_research_question | 1/20 | 5% | 92.75 | 93.45 |
| claude-opus-4-6 | with_research_question | 0/20 | 0% | 92.77 | 93.31 |
| Qwen2.5-7B-Instruct base | with_research_question | 0/20 | 0% | 92.87 | 93.20 |

*worker mean 偏低因为 sample_06 激活函数实现崩溃（10.00%）；有效均值约 92.69%。

**结论**：绝大多数 fine-tuned 模型通过率约 5%（1/20），claude-opus-4-6 和 qwen base 无法通过新阈值。ACT 任务难度高，最佳方案（Learnable Swish）在所有 run 中均属小概率事件。

### 4.3 cv_data_augmentation — 新阈值 93.76%

| Model / Checkpoint | Best Strategy | pass/n | p@1 | mean (%) | best (%) |
| --- | --- | --- | --- | --- | --- |
| **worker-only (sandboxed)** | worker_only | **0/19** | **0%** | 93.27 | 93.75 |
| **exp09 top_k_refs SFT+RL** | top_k_refs | 1/19 | 5% | 93.39 | 93.77 |
| exp10 related_work SFT+RL | with_research_question | 0/19 | 0% | 93.32 | 93.63 |
| exp11 topk_rw SFT+RL | top_k_related_work | 0/20 | 0% | 93.38 | 93.68 |
| exp12 research_q SFT+RL | with_research_question | 0/20 | 0% | 93.24 | 93.64 |
| **exp13 full_refs SFT+RL** | top_k_refs | 1/20 | 5% | 93.35 | 93.77 |
| exp14 full_refs LoRA | top_k_refs | 0/18 | 0% | 93.36 | 93.69 |
| exp15 FAS from exp09 | full_refs | 0/20 | 0% | 93.25 | 93.72 |
| **exp16 full_refs 20×800** | full_refs | 2/20 | 10% | 93.20 | 93.79 |
| exp17 top_k_refs PPL | top_k_related_work | 0/20 | 0% | 93.33 | 93.74 |
| claude-opus-4-6 | with_research_question | 0/20 | 0% | 93.06 | 93.55 |
| **Qwen2.5-7B-Instruct base** | with_research_question | 1/20 | 5% | 93.22 | 93.78 |

**结论**：旧阈值（93.87%）使该任务历史上 0% 通过率。**新阈值后（93.76%），exp16 成为最佳（10%），exp09/exp13/qwen_base 也有 5%**。这是首次在 AUG 任务上观察到非零通过率，证明 proposal 确实带来了增量（尽管信号稀疏）。

---

## 五、关于 Jigsaw 历史数据的说明

旧版评测（2026-05-29 至 2026-06-02）报告的 Jigsaw val_metric 值（0.78–0.91）**全部无效**。

**问题根源**：旧 grader 使用了不同的归一化方式（norm_score），而非正确的 leaderboard 百分位排名（mean(leaderboard_scores &lt;= raw_score)）。

**正确的 leaderboard 信息**：

- 共 2634 条提交，中位数 0.934，最高 0.947

- 我们的 raw AUC（0.63–0.85）仅处于 leaderboard **底部 6–9%**

- Worker-only 获得的百分位值（0.067–0.085）是**正确的**

- 旧的 proposal-guided 数据报告值 0.78–0.91 是**错误的**

所有旧 Jigsaw 结果需用修正后的 grader 重新评测，方可与 worker-only 基线进行比较。

## 六、Jigsaw 历史 Proposal-Guided 运行重评（修正 grader）

旧 Jigsaw 运行的 val_metric（0.82–0.91）已确认为 norm_score 而非真实百分位。以下为用修正 grader 重新计算后的结果，仅有 3 个 run 保存了实际 submission 数据（其余 57 个 run 仅运行了 proposal preview，未执行 worker 评测）。

### 数据来源

| Run ID | 模型 / Checkpoint | 策略 | 有效样本数 |
| --- | --- | --- | --- |
| 3cf9ac85 | worker-only（无 proposal） | worker_only | 5/20 |
| 6b19c320 | claude-opus-4-6 | with_research_question | 20/20 |
| dbfc3e7e | exp13_full_refs SFT+RL | with_research_question | 18/20 |
| f62e35c7 | exp16_full_refs_20×800 SFT+RL | with_research_question | 17/20 |

**修正百分位公式**：percentile = mean(leaderboard_scores &lt;= raw_AUC)，其中 leaderboard 共 2634 条提交，中位数=0.934，最高=0.947。

### 6.1 claude-opus-4-6 / with_research_question（run 6b19260605c320）

均值百分位：0.0671（6.71%），最佳：0.0866（sample_01 和 sample_11，raw AUC≈0.849）

| Sample | Raw AUC | Correct Percentile | Old val_metric（无效） |
| --- | --- | --- | --- |
| sample_00 | 0.7910 | 0.0763 | 0.8648 |
| sample_01 | 0.8487 | **0.0866** | 0.9102 |
| sample_02 | 0.7557 | 0.0729 | 0.8392 |
| sample_03 | 0.7130 | 0.0653 | 0.8102 |
| sample_04 | 0.7808 | 0.0740 | 0.8572 |
| sample_05 | 0.7380 | 0.0687 | 0.8269 |
| sample_06 | 0.7303 | 0.0680 | 0.8216 |
| sample_07 | 0.7218 | 0.0661 | 0.8160 |
| sample_08 | 0.8225 | 0.0805 | 0.8890 |
| sample_09 | 0.7366 | 0.0687 | 0.8259 |
| sample_10 | 0.7299 | 0.0680 | 0.8214 |
| sample_11 | 0.8468 | **0.0866** | 0.9087 |
| sample_12 | 0.6876 | 0.0634 | 0.7938 |
| sample_13 | 0.5000 | 0.0251 | 0.6909 |
| sample_14 | 0.7565 | 0.0729 | 0.8397 |
| sample_15 | 0.7188 | 0.0661 | 0.8139 |
| sample_16 | 0.7970 | 0.0774 | 0.8693 |
| sample_17 | 0.7119 | 0.0653 | 0.8094 |
| sample_18 | 0.7209 | 0.0661 | 0.8154 |
| sample_19 | 0.5000 | 0.0251 | 0.6909 |

### 6.2 exp13_full_refs SFT+RL / with_research_question（run dbfc3e7e）

均值百分位：0.0621（6.21%），最佳：0.0782（sample_13，raw AUC=0.801）；缺少 sample_06、sample_18 数据

| Sample | Raw AUC | Correct Percentile | Old val_metric（无效） |
| --- | --- | --- | --- |
| sample_00 | 0.6445 | 0.0585 | 0.7675 |
| sample_01 | 0.7007 | 0.0642 | 0.8022 |
| sample_02 | 0.6340 | 0.0566 | 0.7614 |
| sample_03 | 0.5001 | 0.0266 | 0.5001 |
| sample_04 | 0.6135 | 0.0543 | 0.7497 |
| sample_05 | 0.7138 | 0.0653 | 0.8107 |
| sample_06 | — | — | — |
| sample_07 | 0.7078 | 0.0645 | 0.8067 |
| sample_08 | 0.7571 | 0.0729 | 0.8402 |
| sample_09 | 0.7114 | 0.0653 | 0.8091 |
| sample_10 | 0.7114 | 0.0653 | 0.8091 |
| sample_11 | 0.6793 | 0.0626 | 0.7886 |
| sample_12 | 0.6126 | 0.0535 | 0.7492 |
| sample_13 | 0.8005 | **0.0782** | 0.8720 |
| sample_14 | 0.7322 | 0.0683 | 0.8230 |
| sample_15 | 0.7592 | 0.0733 | 0.8417 |
| sample_16 | 0.6969 | 0.0634 | 0.7997 |
| sample_17 | 0.7114 | 0.0653 | 0.8091 |
| sample_18 | — | — | — |
| sample_19 | 0.6554 | 0.0604 | 0.7740 |

### 6.3 exp16_full_refs_20×800 SFT+RL / with_research_question（run f62e35c7）

均值百分位：0.0657（6.57%），最佳：0.0756（sample_01，raw AUC=0.789）；缺少 sample_00、sample_02、sample_14 数据

| Sample | Raw AUC | Correct Percentile | Old val_metric（无效） |
| --- | --- | --- | --- |
| sample_00 | — | — | — |
| sample_01 | 0.7890 | **0.0756** | 0.8633 |
| sample_02 | — | — | — |
| sample_03 | 0.6357 | 0.0569 | 0.7624 |
| sample_04 | 0.6518 | 0.0600 | 0.7719 |
| sample_05 | 0.7452 | 0.0718 | 0.8318 |
| sample_06 | 0.6937 | 0.0634 | 0.7977 |
| sample_07 | 0.7732 | 0.0737 | 0.8517 |
| sample_08 | 0.6436 | 0.0585 | 0.7670 |
| sample_09 | 0.7068 | 0.0645 | 0.8061 |
| sample_10 | 0.6959 | 0.0634 | 0.7991 |
| sample_11 | 0.7869 | 0.0748 | 0.8617 |
| sample_12 | 0.5239 | 0.0421 | 0.7025 |
| sample_13 | 0.7558 | 0.0729 | 0.8393 |
| sample_14 | — | — | — |
| sample_15 | 0.7149 | 0.0653 | 0.8114 |
| sample_16 | 0.7107 | 0.0649 | 0.8086 |
| sample_17 | 0.7412 | 0.0695 | 0.8291 |
| sample_18 | 0.7164 | 0.0657 | 0.8124 |
| sample_19 | 0.7698 | 0.0737 | 0.8493 |

### 6.4 汇总对比

| Run | 模型 | 策略 | 有效样本 | 均值百分位 | 最佳百分位 | 最佳 Raw AUC |
| --- | --- | --- | --- | --- | --- | --- |
| worker-only（3cf9ac85） | claude-opus-4-6 | worker_only | 5/20 | 0.0767 | 0.0850 | 0.840 |
| 6b19c320 | claude-opus-4-6 | with_research_question | 20/20 | 0.0671 | 0.0866 | 0.849 |
| dbfc3e7e | exp13 full_refs SFT+RL | with_research_question | 18/20 | 0.0621 | 0.0782 | 0.801 |
| f62e35c7 | exp16 full_refs 20×800 | with_research_question | 17/20 | 0.0657 | 0.0756 | 0.789 |

**结论**：所有 run 的正确百分位均在 leaderboard 底部 6–9%，远低于 leaderboard 中位数（0.934）。claude-opus-4-6 proposal-guided 运行（6b19c320）的最佳单次成绩（0.0866）略高于 worker-only 最佳（0.0850），但样本量小（worker-only 仅完成 5/20），差异无统计意义。fine-tuned 模型（exp13、exp16）表现略低于 claude-opus-4-6 基线。