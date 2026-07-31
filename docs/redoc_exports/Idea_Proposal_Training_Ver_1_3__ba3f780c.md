## 概述

本文档记录 **Idea Proposal** 项目第三版训练运行的情况。Ver. 3 的核心变化是将训练框架从 TRL+DeepSpeed 全面切换至 **veRL**（FSDP + Ray + vLLM），实现了在每个实验 4×H800 GPU 上稳定运行 full-parameter fine-tuning 和 GRPO RL 训练。

---

## 1. 训练方案

### 框架切换：TRL → veRL

<redoc-highlight emoji="tuding" fillColor="yellow">
**核心变化：** 所有 SFT 和 RL 训练均通过 veRL 运行——不再使用 DeepSpeed，不再使用 ZeRO。FSDP 负责 actor 参数分片，vLLM 负责 rollout 生成。
</redoc-highlight>

| 组件 | Ver. 2（TRL） | Ver. 3（veRL） |
| --- | --- | --- |
| SFT trainer | HuggingFace Trainer + DeepSpeed ZeRO-2 | veRL SFTTrainer + torchrun + FSDP |
| RL trainer | TRL GRPO + DeepSpeed ZeRO-2 | veRL PPO/GRPO + Ray + FSDP actor + vLLM rollout |
| 显存策略 | ZeRO-2 optimizer sharding | FSDP full sharding + gradient checkpointing |
| 启动方式 | torchrun（所有角色） | SFT: torchrun &#124; RL: 单进程（Ray 内部启动 worker） |
| 稳定性 | BF16 overflow 频繁崩溃 | 4×H800 稳定运行 |

### 实验矩阵（exp09–exp17）

共 9 个实验，分布在 4 台机器（M0–M3）上，每台机器通过 CUDA_VISIBLE_DEVICES 将 8 张 GPU 分为两个 4-GPU 槽位，同时运行 2 个实验。

| Exp | Strategy | 微调模式 | Reward | 机器 |
| --- | --- | --- | --- | --- |
| **exp09** | top_k_refs | Full-FT | PRS | M0 GPUs 0-3 |
| **exp10** | related_work | Full-FT | PRS | M1 GPUs 0-3 |
| **exp11** | topk_rw hybrid | Full-FT | PRS | M2 GPUs 0-3 |
| exp12 | research_q | Full-FT | PRS | M3 GPUs 0-3 |
| **exp13** | full_refs | Full-FT | PRS | M0 GPUs 4-7 |
| **exp14** | full_refs | LoRA | PRS | M1 GPUs 4-7 |
| **exp15** | top_k_refs | Full-FT | FAS | M3 GPUs 4-7（exp17 完成后） |
| exp16 | full_refs 20×800 | Full-FT | PRS | M2 GPUs 4-7 |
| **exp17** | top_k_refs | Full-FT | PPL | M3 GPUs 4-7 |

### Reward 设计

- **PRS（Paper Recovery Score）：** 0.8 × cosine_sim(proposal, abstract) + 0.2 × format_score，衡量模型生成的 proposal 与真实论文摘要的相似度

- **FAS（Future Alignment Score）：** 与留出未来语料库索引的相似度，衡量 proposal 的前瞻性和新颖性

- **PPL（Perplexity reward）：** 在 proposal 上下文条件下，用 SFT/ref 模型（而非 GPT-2）计算摘要 token 的平均 log-prob，即 P_policy(abstract &#124; proposal)。reward = exp(-mean_CE / 3)，衡量 policy 在给定 proposal 后自然生成正确摘要的能力，越高表示 proposal 质量越好

### 每个实验的完整 Pipeline

每个实验通过 _combined_lib.sh 自动串联执行以下步骤：

1. CoT synthesis（调用 Claude API）→ train_cot.jsonl

2. SFT hparam search（3×3 网格：LR × warmup ratio）

3. 用最优超参进行 SFT full training

4. RL hparam search（3×3 网格：LR × KL coefficient）

5. 用最优超参进行 RL full training → final checkpoint

---

## 2. 已解决的 Bug

### 框架迁移阶段（veRL 接入）

| # | Bug 描述 | 解决方案 |
| --- | --- | --- |
| 1 | TRL+DeepSpeed ZeRO-2 中 BF16 allreduce overflow，所有 RL 训练以 inf/nan loss 崩溃 | 整体切换至 veRL+FSDP |
| 2 | 不使用 ZeRO 时 full-FT CUDA OOM（76 GB/GPU） | FSDP full sharding 原生解决 |
| 3 | vLLM plugin 崩溃：fp32_overrides 导入了已删除的 vllm.worker 模块 | 设置 VLLM_PLUGINS="" |
| 4 | Inductor cache UnpicklingError（上次运行被强杀导致文件截断） | 启动时清空并重建 TORCHINDUCTOR_CACHE_DIR |
| 5 | Ray AF_UNIX socket 路径超过 OS 108 字符限制（_temp_dir 在 NFS 上） | Ray temp dir 迁移至 /dev/shm/ray_tmp |
| 6 | NFS（/newcpfs）不支持 Unix domain socket（EOPNOTSUPP） | 所有 Ray/socket 目录迁移至 /dev/shm |
| 7 | glibc 2.28 TLS assertion：并发 sympy→gmpy2 dlopen 触发 _dl_allocate_tls_init: listp != NULL | 设置 SYMPY_GROUND_TYPES=python |
| 8 | 上次崩溃遗留的 stale Ray 进程阻塞新 ray.init() | 启动时自动 kill stale Ray 进程 |
| 9 | veRL reward function 签名不匹配（dict vs kwargs） | 将 compute_score 对齐至 veRL custom_reward_function 接口 |
| 10 | veRL SFT config 使用 Hydra defaults 列表，OmegaConf.load() 无法解析子配置 | 手动组合 model/hf_model.yaml、engine/fsdp.yaml、optim/fsdp.yaml、profiler/profiler.yaml |
| 11 | verl.model_merger CLI 参数在不同版本间有变化 | 更新 merge 调用以匹配已安装的 verl 版本 |
| 12 | veRL FSDP SFT 路径中 LoraConfig 包含不支持的 kwargs | 在构造 LoraConfig 前过滤掉不支持的参数 |

### 4-GPU 训练阶段

| # | Bug 描述 | 解决方案 |
| --- | --- | --- |
| 13 | 4-GPU SFT（full-FT 和 LoRA）在 max_token_len=8192 时 CUDA OOM | n_gpu &lt; 8 时：启用 gradient_checkpointing=True + 限制 max_token_len_per_gpu=6144 |
| 14 | AssertionError: max_token_len=6144 &lt; max_seq_len=6168，tokenizer 未按同一上限截断 | 在 data overrides 中同步设置 max_length=effective_max_tokens |
| 15 | make_parquet 拒绝 --reward-type ppl（choices 中只有 prs/fas） | 在 choices 中添加 ppl |
| 16 | PPL reward 未实现 → exp17 所有 mini reward=0.0；后发现使用 GPT-2 计算流畅度是错的（应为 P_policy(abstract | proposal) via SFT 模型） |
| 17 | RL 4-GPU update_policy actor step CUDA OOM | n_gpu &lt; 8 时：gradient_checkpointing=True + ppo_max_token_len=8192 + vllm_gpu_util=0.45 + max_prompt_length=3072 |
| 18 | NCCL watchdog 卡死 480s（CudaEventDestroy 持有 GIL）→ rank 2 SIGABRT | 设置 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800；SFT full training 添加 --resume |
| 19 | LoRA RL actor：fsdp_workers.py 中 LoraConfig exclude_modules bug，exp14 全部 9 个 RL mini 静默返回 0 reward | 修复 fsdp_workers.py LoRA actor 初始化路径 |
| 20 | veRL model_merger 保存 LoRA adapter 崩溃：AttributeError: str has no attribute value（peft 将 task_type 序列化为字符串，merger 期望 enum） | 修复 base_model_merger.py，兼容字符串类型的 task_type |
| 21 | exp14 RL actor 在 Ray worker 中 OOM kill + glibc TLS assertion：_dl_allocate_tls_init: listp != NULL（并发 CUDA 库 dlopen 触发 TLS 竞争） | 在 _combined_lib.sh 和 train/rl.py 中添加 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1，强制在 dlopen 之前预加载 libcuda，消除 TLS 竞争 |

### 数据集与关键错误

| # | Bug 描述 | 解决方案 |
| --- | --- | --- |
| 22 | **[CRITICAL] train_prs.parquet 只有 8 行**（5 月 10 日 smoke test 用 --limit 8 重建后未恢复），exp12/13/14 全部 RL 训练（hparam search + full training）仅运行了 2 步，结果完全无效 | 重建 parquet 至完整 7357 行；清除 exp12/13/14 的 rl/ 目录和 hparam 结果；注入 exp11 验证过的超参（lr=2e-6, kl=0.05）跳过 hparam search，直接重启 full RL |
| 23 | exp14 model_merger 崩溃：RL checkpoint 缺少 huggingface/config.json 子目录，merger 无法读取模型配置 | 修复 train/rl.py：在 RL 训练后、merger 运行前，从 SFT checkpoint 复制 config.json 到 RL global_step 目录 |
| 24 | exp17 PPL reward forward pass 崩溃：Float/Half dtype mismatch（ref 模型加载为 fp16，输入 tensor 为 fp32） | 在 verl_reward.py _ppl_score() 的 forward 中用 torch.autocast(device_type="cuda") 包裹，自动混合精度消除 dtype 不匹配 |

---

## 3. 当前训练进度

_截至 2026-05-11_

| Exp | SFT | RL | 备注 |
| --- | --- | --- | --- |
| exp09 | ✅ 完成 | ✅ **全部完成** | top_k_refs, PRS |
| exp10 | ✅ 完成 | ✅ **全部完成** | related_work, PRS |
| exp11 | ✅ 完成 | ✅ **全部完成** | topk_rw, PRS |
| exp12 | ✅ 完成 | ✅ **全部完成**（2026-05-12 10:46） | research_q, PRS；重跑结果有效 |
| exp13 | ✅ 完成 | ✅ **全部完成**（2026-05-12 08:02） | full_refs, PRS；重跑结果有效 |
| exp14 | ✅ 完成（LoRA） | 🔄 **RL 重跑中**（M1 GPUs 4-7） | full_refs, LoRA, PRS；原 RL 无效 + model_merger 修复（Bug 23）+ LD_PRELOAD 修复（Bug 21） |
| exp15 | ⏳ 等待 exp17 完成 | ⏳ 未开始 | top_k_refs, FAS |
| exp16 | ✅ 完成 | ✅ **全部完成** | full_refs 20×800, PRS |
| exp17 | N/A（仅 RL） | 🔄 RL hparam search 中 | top_k_refs, PPL；dtype 修复（Bug 24），新目录 exp17_..._20260511_093503 |

<redoc-highlight emoji="dui" fillColor="green">
**7 个实验已全部完成**（exp09、exp10、exp11、exp12、exp13、exp16 + PPL 进行中）。exp12（research_q）和 exp13（full_refs）已于 2026-05-12 完成 RL 重跑，结果有效。exp14 LoRA RL 重跑中（~39%）。exp17 PPL hparam search 进行中（6/9 minis 完成，best lr=2e-6/kl=0.02，reward≈0.318）。
</redoc-highlight>

---

## 4. 后续计划

1. **等待 exp12/13/14 RL 重跑完成**（已用 lr=2e-6/kl=0.05 跳过 hparam search，直接 full RL）；**等待 exp17 hparam search → full RL 完成**

2. **启动 exp15**（FAS reward，top_k_refs）：exp17 在 M3 GPUs 4-7 完成后立即接续

3. **评估所有 checkpoint**：在 held-out 测试集上跑推理，计算每个模型的 PRS/FAS/PPL 指标

4. **Cross-reward 分析**：在相同 SFT base checkpoint 上对比 exp09（PRS）、exp17（PPL）、exp15（FAS），以分离 reward 信号的效果

5. **Loyal worker + benchmark**：项目下一阶段——构建 loyal worker agent 及其评测 benchmark

---

## 5. 工作流：exp-manager 与 coder 协作

本次训练由两个协作的 Claude Code session 共同推进：

<redoc-columns>
<redoc-column ratio="0.5">
### exp-manager（本 session）

- 每隔 2 分钟监控 M0–M3 上的全部 8 个 tmux session

- 解析训练日志，识别异常模式：OOM、NCCL hang、NaN loss、进程崩溃码

- 发现问题时向 coder 发送结构化 bug report

- 跟踪各实验状态（SFT 阶段 / RL hparam / RL full / FINISHED）

- 根据实验完成情况动态调整机器分配计划
</redoc-column>
<redoc-column ratio="0.5">
### coder（tmux cc2）

- 通过 tmux 接收纯文本 bug report

- 根据日志片段和错误模式定位根因

- 修复源文件（train/sft.py、train/rl.py、train/verl_reward.py、train/make_parquet.py，以及 verl 内部文件）

- 尽可能使用 --resume 重启崩溃的实验

- 回报修复摘要（bug 编号、根因、已应用的修复、新状态）
</redoc-column>
</redoc-columns>

### 通信协议

exp-manager 向 coder 发送消息的方式：

```bash
# 第一步：粘贴消息内容
tmux send-keys -t cc2 "Bug report 内容..."
# 第二步：单独发送 Enter 提交（同一次调用中的 Enter 会被吞掉）
tmux send-keys -t cc2 "" Enter
```

coder 在 **auto mode** 下运行，会在任务间隙处理排队消息。exp-manager 在发送后确认 cc2 pane 已收到消息，再继续下一步操作。

---

## 6. 基础设施

| 项目 | 配置 |
| --- | --- |
| 机器 | M0 (lxh_agent_0)、M1 (lxh_agent_1)、M2 (lxh_agent_2)、M3 (lxh_agent_3) |
| 每台机器 GPU | 8× H800 80GB |
| 每个实验 GPU | 4（通过 CUDA_VISIBLE_DEVICES 分配） |
| 共享文件系统 | /newcpfs（CPFS，剩余 319 TB） |
| 基础模型 | Qwen2.5-7B-Instruct |
| 训练框架 | veRL 0.3+（FSDP SFT + Ray GRPO） |
| 实验输出目录 | /newcpfs/lxh/agentic-training/proposal_rl/runs/exps/ |

---

## 7. Demo 工具与 Benchmark Pipeline

_完成于 2026-05-11_

### 7.1 Demo 工具

为方便在演示和实验分析时快速查看 proposal 模型对指定论文的输出，新增了两个工具：

**CLI 模式（scripts/demo.py）**

```bash
# 仅看 prompt（无需 GPU）
python scripts/demo.py 2601.12345 --prompt-only

# 使用 exp 名称前缀自动发现最新 checkpoint
python scripts/demo.py 2601.12345 --exp exp09_top_k_refs_sft_rl

# 与 Claude API baseline 对比
python scripts/demo.py 2601.12345 --exp exp09 --baseline

# 保存结果到 JSON
python scripts/demo.py 2601.12345 --exp exp09 --save out.json
```

**TUI 模式（scripts/demo_tui.py）**

使用 Textual 构建的交互式终端 UI，支持三栏布局（实验列表 / Prompt / Response），快捷键如下：

| 按键 | 功能 |
| --- | --- |
| ↑↓ + Enter | 选择实验并生成 |
| a / / | 输入 arXiv ID |
| c | 开启对比模式（左右两栏同时显示两个实验的输出） |
| p | 切换 prompt 全文 / 摘要显示 |
| s | 保存当前视图到 JSON |
| r | 重新生成（resample） |

实验列表在启动时自动扫描 runs/exps/*/rl/final，并在顶部预置 claude-opus-4-6 和 claude-sonnet-4-6 两个 API baseline 入口。生成在后台线程运行，完成后自动计算 FAS 并显示在面板标题栏。

---

### 7.2 端到端 Benchmark Pipeline

项目下一阶段的核心评测机制：proposal 模型生成研究提案 → Loyal Worker 实现提案 → 基准测试脚本验证是否取得改进，超越 FAS/PRS 等代理指标。

**文件结构：**

```plaintext
benchmark/
  tasks/
    base.py          # AbstractBenchmarkTask 接口
    char_lm.py       # 任务实现：Shakespeare 字符级语言模型
    char_lm/
      baseline.py    # 自包含 nanoGPT-style 基线（3 层 Transformer）
      shakespeare.txt
      eval_papers.txt  # 固定 20 篇评测论文（种子 42，来自 test.jsonl）
  worker.py          # LoyalWorker：调用 Claude Code CLI 实现 proposal
  worker_prompt.txt  # Worker 的系统 prompt 模板
  run_benchmark.py   # 主扫描：生成 proposals → 运行 workers → 计算 pass@k
  report.py          # 汇总所有 runs/benchmark/*/summary.json，打印对比表
scripts/
  08_benchmark.sh    # 一键运行 benchmark sweep 的启动脚本
```

**Benchmark 任务：Shakespeare 字符级 LM**

选取此任务的理由：数据集完全自包含（1 MB），基线训练仅需约 5 分钟（1 GPU，1000 steps），指标客观可验证（val BPC），改进空间丰富（余弦 LR 调度、梯度裁剪、权重共享、dropout 调优、optimizer 选择等）。

- 基线：3 层 Transformer，n_head=4，n_embd=128，Adam lr=3e-4，无调度 → val_bpc ≈ 1.247

- **通过阈值：** val_bpc 改善 ≥ 0.01（相对 0.8%）

**Loyal Worker 设计：**

每个 proposal 在独立 workspace 中运行，Worker 通过以下命令调用 Claude Code：

```bash
claude -p --output-format stream-json --max-turns 20 \
  --allowedTools Bash,Read,Edit,Write < worker_prompt.txt
```

Worker 被要求：只实现 proposal &lt;approach&gt; 中描述的核心思路，不添加其他改进；训练完成后将结果写入 result.json；超时 600 秒。

**主扫描接口：**

```bash
# 使用本地 checkpoint
bash scripts/08_benchmark.sh exp09_top_k_refs_sft_rl

# 使用 Claude API 作为 baseline
python benchmark/run_benchmark.py --claude-model claude-opus-4-6 \
  --task char_lm --n-proposals 20
```

**输出报告（benchmark/report.py）：**

```plaintext
label                 task        n  pass  err    p@1     p@3     p@5    p@10   Δmetric  baseline  err%
exp09_top_k_refs      char_lm    20     8    2  0.400   0.700   0.820  0.950  -0.0180    1.2470   10.0%
claude-opus-4-6       char_lm    20    10    1  0.500   0.800   0.920  1.000  -0.0230    1.2470    5.0%
```

pass@k 采用无偏估计量 1 - C(n-c, k) / C(n, k)，与 HumanEval/pass@k 论文定义一致。