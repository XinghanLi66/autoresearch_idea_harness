## 概述

本报告记录了 **End-to-End Benchmarking Pipeline** 的设计与实验结果。该 Pipeline 将 proposal model 生成的研究建议与 loyal worker（Claude Code CLI）结合，验证建议的可实施性并量化其对下游任务的改进效果。

---

## 一、任务选择（Task Selection）

### 为什么选择 dl_lr_schedule？

评测任务需满足以下条件：

- **自包含**：不依赖外部 API 或复杂基础设施，单机可运行

- **快速**：单次训练在 1 块 GPU 上可在约 20 分钟内完成

- **可量化**：输出单一数值指标，便于客观比较

- **改进空间丰富**：baseline 较简单，有大量已知改进方法可供 proposal model 参考

- **与 proposal model 训练数据相关**：任务涉及 learning rate scheduling，是深度学习领域经典研究课题

dl_lr_schedule 来自 **MLS-Bench**（Machine Learning Science Benchmark），恰好满足上述所有条件。

---

## 二、任务细节（Task Details）

### 2.1 任务定义

| 字段 | 值 |
| --- | --- |
| 任务名称 | dl_lr_schedule |
| 模型架构 | ResNet-20（默认）/ ResNet-56 / VGG-16BN / MobileNetV2 |
| 数据集 | CIFAR-10（默认）/ CIFAR-100 / FMNIST |
| 训练轮数 | 200 epochs |
| Reference baseline | warmup_cosine（warmup + cosine decay） |
| Baseline val_metric | 92.71%（ResNet-20 on CIFAR-10） |
| Pass 阈值 | baseline + 0.30%（即 ≥ 93.01%） |
| 评估指标 | test accuracy（%） |

### 2.2 可编辑区域（Editable Region）

Worker 只允许修改 editable_region.py 中的 get_lr() 函数：

```python
def get_lr(epoch, total_epochs, base_lr, config):
    """Compute learning rate for the given epoch."""
    # 在此实现 LR schedule
    ...
```

### 2.3 Prompt Strategies（评估策略）

同一 checkpoint 分别以 5 种不同 prompt strategy 测试，覆盖不同的参考文献组合：

| Strategy | 描述 |
| --- | --- |
| full_refs | 全部 frontline papers（<redoc-comment commentGid="7641164974552848664" blockId="46dc47f08d625f5379903e761c213b65">约 40 篇</redoc-comment>） |
| top_k_refs | 按相关度排序的 top-k 论文 |
| related_work | 仅包含 related work 部分 |
| with_research_question | 含显式 research question 引导 |
| top_k_related_work | top-k + related work 组合 |

---

## 三、Pipeline 架构（v2）

### 3.1 整体流程

```plaintext
[Frontline Papers]
       |
       v
[Build Prompt Cache]     ← benchmark/prompt_cache.py
       |
       v
[Generate Proposal]      ← probe.py（checkpoint model，GPU 0）
       |
       v
  ★ 立即派发（pipelined）
       |
       v
[SlotPool 动态分配 GPU]  ← benchmark/slot_pool.py
       |
       v
[Worker (Claude Code)]   ← SSH 到目标机器，CUDA_VISIBLE_DEVICES=slot.gpu
       |
       v
[run.sh → run_mls_eval.py → 200 epoch 训练]
       |
       v
[Verify HMAC + Score]    ← benchmark/pipeline.py
```

### 3.2 关键设计：SlotPool（v2 新增）

**问题**：v1 中每个 pipeline run 固定分配一台机器 + 一块 GPU，20 个 sample 串行共享同一 GPU，导致首个结果需等待约 2.5 小时。

**解决**：引入 SlotPool，全局共享所有可用 GPU。每个 sample 生成 proposal 后立即派发 worker 线程，线程阻塞等待从池中获取一个空闲 (machine, gpu) slot。

```python
# benchmark/slot_pool.py
class SlotPool:
    def acquire(self, caller="") -> Slot:  # 阻塞直到有空闲 slot
    def release(self, slot, caller="")     # 释放 slot，触发等待线程
    def slot(self, caller="")              # context manager

# 构建示例
pool = SlotPool.from_machines(
    ["lxh_agent_1", "lxh_agent_2", "lxh_agent_3"],
    gpus=range(8),
    log_file="runs/slot_pool.log",  # 记录每次 acquire/release
)
```

实时日志（tail -f runs/slot_pool.log）：

```plaintext
[2026-05-18 16:04:12] INIT total=24 slots: ['lxh_agent_1:GPU0', ...]
[2026-05-18 16:07:33] ACQUIRE slot=lxh_agent_1:GPU0 caller='sample_00' waited=False free=23 busy=1/24
[2026-05-18 16:07:35] ACQUIRE slot=lxh_agent_1:GPU1 caller='sample_01' waited=False free=22 busy=2/24
...
[2026-05-18 17:23:41] RELEASE slot=lxh_agent_1:GPU0 caller='sample_00' free=23 busy=1/24
```

**GPU 注入机制**：Claude Code 的 shell-snapshot 机制会重置环境变量，因此 CUDA_VISIBLE_DEVICES 无法通过 SSH env 传递给 run_mls_eval.py。解决方案：在 setup_workspace() 之后自动将 GPU 编号写入 run.sh 头部：

```bash
#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=3   # ← pipeline 自动注入
RESULT="${1:-result.json}"
python run_mls_eval.py ...
```

### 3.3 BenchmarkPipeline（pipeline.py）

| 组件 | 说明 |
| --- | --- |
| 状态机 | PENDING → BUILDING_PROMPT → GENERATING → RUNNING_WORKER → DONE/ERROR |
| 并行模式 | 生成串行（单 GPU），worker 并行（由 SlotPool 控制） |
| 持久化 | state.json、proposal.txt、worker.log、result.json 写入磁盘 |
| HMAC 防伪 | result.json 含 _sig 字段，防止 worker 伪造评估结果 |
| WORKER_TIMEOUT | 7200s（2 小时），覆盖 200-epoch 训练（约 60–90 分钟） |

### 3.4 Sweep 启动器（sweep.py）

```bash
# 单 checkpoint × 2 strategies × 20 samples，使用 M1/M2/M3 全部 GPU
python benchmark/sweep.py \
  --checkpoints "exp12 research_q" \
  --strategies with_research_question \
  --machines lxh_agent_1,lxh_agent_2,lxh_agent_3 \
  --gpus 0-7 \
  --n-samples 20

# 查看 slot 分配日志
tail -f runs/slot_pool.log
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| --checkpoints | all | 逗号分隔的 exp 前缀或 all |
| --strategies | all | 逗号分隔的策略名或 all |
| --n-samples | 20 | 每个 (checkpoint × strategy) 的采样数 |
| --machines | 4 台 DSW 机器 | SSH alias 列表 |
| --gpus | 0-7 | GPU 范围，如 0-3 或 0,2,4 |
| --max-parallel | 4 | 最大并发生成 pipeline 数（CPU bound） |

### 3.5 Dashboard TUI（dashboard_tui.py）

实时监控所有 sweep 进度：

```bash
python scripts/dashboard_tui.py
```

矩阵视图：行 = (checkpoint, strategy) 对，列 = Progress / Pass/Done / Mean Δ

| 列 | 含义 |
| --- | --- |
| Progress | fin/total ⏳Nw ↻Ng（Nw=活跃 worker，Ng=生成中） |
| Pass/Done | passed/done  N✗（N✗=error 数） |
| Mean Δ | 已完成 sample 的平均 improvement（%） |

---

## 四、当前实验结果（截至 2026-05-18）

### 4.1 正在运行的 Sweep

| Sweep | Checkpoint | Strategies | n_samples | 机器 |
| --- | --- | --- | --- | --- |
| exp13 | exp13_full_refs_sft_rl | top_k_refs, with_research_question | 20 | M0 (固定 GPU 0-1) |
| exp12 | exp12_research_q_sft_rl | with_research_question | 20 | M1/M2/M3 GPU 0-7（SlotPool） |

### 4.2 exp13 初步结果（部分完成）

**Checkpoint**：exp13_full_refs_sft_rl（1839 steps SFT+RL）
**Task**：dl_lr_schedule，ResNet-20 / CIFAR-10，baseline = 92.71%，pass ≥ 93.01%

#### Strategy: t<redoc-comment commentGid="7641170648204811288" blockId="2f53fc8b0bb93f466e875f94b4e2a055">op_k_refs</redoc-comment>（4/20 完成）

| Sample | val_metric | Improvement | Passed | 耗时 |
| --- | --- | --- | --- | --- |
| sample_00 | 92.92% | +0.21% | ❌ | 55 min |
| sample_01 | 92.61% | -0.10% | ❌ | 52 min |
| sample_02 | 92.86% | +0.15% | ❌ | 48 min |
| sample_03 | 92.96% | +0.25% | ❌ | 47 min |

#### Strategy: with_research_question（2/20 完成）

| Sample | val_metric | Improvement | Passed | 耗时 |
| --- | --- | --- | --- | --- |
| sample_00 | 92.63% | -0.08% | ❌ | 66 min |
| sample_01 | 92.84% | +0.13% | ❌ | 75 min |

<redoc-highlight emoji="tuding" fillColor="yellow">
**初步观察**：exp13 在 top_k_refs 策略下，4/4 个样本均在 baseline 附近（92.6–93.0%），差距在 0.3% 以内但均未超过 pass 阈值（93.01%）。with_research_question 表现略弱，第一个样本甚至低于 baseline。需更多样本才能得出统计显著结论。
</redoc-highlight>

### 4.3 早期 API 验证结果（参考）

| Sample | Proposal Model | val_metric | Improvement | Passed |
| --- | --- | --- | --- | --- |
| sample_00 | claude-opus-4-6 (API) | 92.76% | +0.05% | ❌ |
| sample_01 | claude-opus-4-6 (API) | 92.47% | -0.24% | ❌ |
| sample_02 | claude-opus-4-6 (API) | N/A | N/A | ❌ |

---

## 五、已知问题与修复

| 问题 | 状态 | 解决方案 |
| --- | --- | --- |
| CUDA_VISIBLE_DEVICES 不传递到 run_mls_eval | ✅ 已修复 | run.sh 头部硬编码 GPU 编号 |
| 20 samples 串行等待同一 GPU（首结果 2.5h） | ✅ 已修复 | SlotPool：生成后立即派发 worker |
| 孤儿进程占用 GPU 显存 | ✅ 已清理 | 按 CWD 识别并 kill |
| WORKER_TIMEOUT 太短（900s / 1800s） | ✅ 已修复 | 调整为 7200s（2 小时） |
| slot pool 无日志，分配情况不透明 | ✅ 已修复 | SlotPool 写入 runs/slot_pool.log |

---

## 六、下一步计划

- 等待 exp12 / exp13 sweep 完成（预计完成时间：2026-05-18 晚）

- 汇总 pass@1、mean Δ，与 claude-opus-4-6 API 对比

- 扩展 sweep：exp09、exp11、exp14、exp16（不同训练配置）

- 分析 proposal 文本质量与 val_metric 的相关性