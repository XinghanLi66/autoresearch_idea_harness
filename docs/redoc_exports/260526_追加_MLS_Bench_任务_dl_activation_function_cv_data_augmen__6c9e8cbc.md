## 1. 任务选取背景

本项目已在 MLS-Bench 框架下完成 dl_lr_schedule/resnet20-cifar10 基准测试（9 个 checkpoint × 5 种评测策略 × 20 样本，详见《260525 Benchmark Analysis》）。现新增两个任务，要求：

- **快速可验证**：单样本 wall time ≤ 30 min（H800 GPU）

- **原子化输出**：模型只需生成一个小型函数/类，worker 直接运行验证

- **覆盖不同维度**：与 LR scheduling 无概念重叠

- **无已发表方法可轻易通过**：pass threshold 须高于所有已知 published baseline 的本地实测值

经对 MLS-Bench 全部 task 目录、官网及论文（arXiv 2605.08678v1）的系统分析，最终选取：

**dl-activation-function** + **cv-data-augmentation**

---

## 2. 全量 Baseline 本地复现（H800，seed=42，ResNet-20/CIFAR-10，200 epoch）

所有数值均为本地实测，非 MLS-Bench leaderboard 数值（两者因硬件差异有 0.05–0.15pp 偏差）。

### 2.1 dl-activation-function — 激活函数

| Baseline | 本地实测 | Leaderboard | 差值 |
| --- | --- | --- | --- |
| GELU（参考基线，run 1） | 92.97% | 93.11% | −0.14pp |
| GELU（参考基线，run 2） | 92.84% | 93.11% | −0.27pp |
| SiLU | 92.50% | 92.72% | −0.22pp |
| Mish | 92.65% | 92.78% | −0.13pp |

**已发表方法最高分（本地）：GELU 92.97%**

Pass threshold：**≥ 93.27%**（GELU 92.97% + 0.30pp）

- 0.30pp buffer 确保即使 GELU 在同硬件 run-to-run 方差（~0.13pp）内波动，也不能通过

- 同时高于 leaderboard GELU（93.11%）0.16pp

### 2.2 cv-data-augmentation — 数据增广

| Baseline | 本地实测 | Leaderboard | 差值 |
| --- | --- | --- | --- |
| Standard（RandomCrop+HFlip，模板默认） | 92.54% | — (未列出) | — |
| RandAugment | 93.20% | 93.51% | −0.31pp |
| TrivialAugment | 93.57% | 93.36% | +0.21pp |
| **Cutout（参考基线）** | **93.67%** | 93.72% | −0.05pp |

**已发表方法最高分（本地）：Cutout 93.67%**

Pass threshold：**≥ 93.87%**（Cutout 93.67% + 0.20pp）

- Cutout、TrivialAugment、RandAugment 三者均低于 93.87%，均不能通过

- 模型需提出真正超越现有方法的增广方案

---

## 3. 任务配置汇总

| 字段 | dl_activation_function | cv_data_augmentation |
| --- | --- | --- |
| 模型输出 | CustomActivation(nn.Module)，实现 forward(x) | build_train_transform(config) → transforms.Compose |
| 可编辑行范围 | lines 32–49 | lines 246–275 |
| 参考基线 | gelu | cutout |
| 参考基线本地值 | 92.97% | 93.67% |
| Pass threshold | +0.30pp → **≥ 93.27%** | +0.20pp → **≥ 93.87%** |
| active_subtasks | resnet20-cifar10 | resnet20-cifar10 |
| 单样本 eval time | ~13 min | ~13–20 min |
| Frontline papers | GELU, SiLU, Mish | Cutout, RandAugment, TrivialAugment |

---

## 4. Pass Threshold 设计原则

三个任务的 threshold 统一遵循：**pass = 本地复现参考基线 + buffer，且 buffer 足以确保所有已发表方法均不能通过**。

| 任务 | 参考基线（本地） | 已发表最高（本地） | threshold | 距已发表最高的 buffer |
| --- | --- | --- | --- | --- |
| dl_lr_schedule | warmup_cosine 92.71% | warmup_cosine 92.71%（即参考基线） | +0.30 → 93.01% | +0.30pp |
| dl_activation_function | GELU 92.97% | GELU 92.97% | +0.30 → 93.27% | +0.30pp |
| cv_data_augmentation | Cutout 93.67% | Cutout 93.67% | +0.20 → 93.87% | +0.20pp |

注：cv_data_augmentation 使用 +0.20 而非 +0.30，因为 Cutout 本身已是强基线（93.67%），+0.30 将门槛提至 93.97%，对 7B 模型过于严苛。

---

## 5. 实现细节

两个任务均已集成到 proposal_rl/benchmark/tasks/mls_bench.py（DlActivationFunctionTask，CvDataAugmentationTask），注册于 benchmark/tasks/__init__.py 的 REGISTRY 和 SUBTASKS，frontline papers 写入对应 mls_tasks/ 目录。run_mls_eval.py 无需修改。

代码变更详见 agent-memory/coder/code.md。

---

_分析生成于 2026-05-26。全量 baseline 复现于本地 H800（M1，seed=42，200 epoch ResNet-20/CIFAR-10）。代码：proposal_rl/benchmark/tasks/mls_bench.py。_