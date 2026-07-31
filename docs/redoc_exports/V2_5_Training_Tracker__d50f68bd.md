## 计划

- 输入：预处理后的 with_research_question task packet，包含 task context、frontline refs、research question 和 worker constraint。

- 输出：一个粒度合适的 proposal，应该足够具体，让 worker 能直接实现和评测；不能只是 broad survey direction，也不能是 trivial baseline rename。

- 训练安全性：不允许 target leakage。训练 prompt 不能包含 full TeX target、TeX excerpt、COT target 或未来论文信息。

- 训练顺序：<font color="#F06A1D">**按时间从早到晚训练**</font>，避免模型先看到未来论文后再评估较早任务。

- 目标效果：10 个 MLS tasks 上 pass@1 要显著提升，<font color="#F06A1D">**长期希望接近 80% 级别**</font>；至少要看到多个 task 上真实 worker result 明显超过 worker-only baseline，并提出明显nontrivial的改进（我可以来验证）

## 当前已完成

- strict1000 target synthesis、audit、collate 完成：1024 targets 中 929 条进入训练。

- Qwen2.5-32B strict1000 SFT 完成，DLC job dlc19eoxl6dp93di succeeded。

- 10-task checkpoint proposal batch 完成：10/10 生成成功，本地 proposal quality mean 75.0，2/10 rule-based pass。

- 真实 worker eval 已跑通两个 task：dl_lr_schedule 和 dl_activation_function。

- V3 dashboard 已能查看训练状态、proposal batch、V2.3 snapshot 和 worker eval result。

- GitHub 已同步 V2 repo，最新相关 commit：4224d84。

## Prompt Audit

已检查 strict1000 训练 parquet：

| 检查项 | 结果 |
| --- | --- |
| rows | 929 |
| columns | messages, sample_id, arxiv_id, created, split, quality_score |
| user prompt 是否含 TeX excerpt | 否 |
| user prompt 是否含 target fields | 否 |
| user prompt 是否含 COT 或 &lt;thinking&gt; | 否 |
| assistant target | 全部为 &lt;proposal&gt; XML |
| include_cot | false |
| condition strategy | 全部为 with_research_question |

注意：已经训练出的 strict1000 parquet 里仍使用旧措辞 Prefer this XML schema when possible。源码已经改成更强约束 Output one proposal using this XML schema，但需要重新 collate 和训练才会进入模型。

## 真实 Worker Eval

| Task | Baseline | Pass line | V3 result | Improvement | Pass |
| --- | --- | --- | --- | --- | --- |
| dl_lr_schedule / resnet20-cifar10 | 92.71 | 93.19 | 92.99 | +0.28pp | no |
| dl_activation_function / resnet20-cifar10 | 92.97 | 93.35 | 93.24 | +0.27pp | no |

解读：这是第一个真实训练信号。模型 proposal 让 worker 实现后确实超过 baseline，但还没有超过 worker-only-best reset threshold，所以只能算 promising near miss，不能算正式成功。

## 主要问题

1. Proposal quality pass 只有 2/10。失败主要来自 **code_level_plan 不够具体，worker 难以直接照着实现。**

2. 当前**训练样本只覆盖 2025-04 accepted rows，规模和时间范围都太窄。**

3. 已训练 prompt 的 **XML 约束偏软**，需要用新硬约束重新 collate。

4. worker 执行会偏离 proposal，需要单独记录 proposal-to-worker compliance。

5. 当前只做 SFT，还没有 reward / DPO / score-prediction / RL 闭环。

## Dashboard 和关键路径

本地 dashboard：

```bash
cd /newcpfs/lxh/agentic-training/autoresearch_idea_harness
python scripts/dashboard_v3_tui.py --config configs/default.yaml
```

关键 artifact：

- SFT checkpoint: runs/training/v3_sft_qwen25_32b/v3_sft_strict_batch1000_auto/checkpoints/phase_000_2025-04/final

- Proposal batch: runs/v3_checkpoint_proposal_batch/dlc_mls10_strict1000/output

- Worker eval: runs/v3_precomputed_worker_eval/dlc_strict1000_lr_act_realbin

- Local preliminary report: runs/reports/v3_preliminary_training_report.md

Dashboard result 分类：

- real: 真实 worker eval result。

- error: 环境或执行失败，不算模型结果。

- fixture: 测试 fixture，不算模型结果。

- skipped: 显式跳过，不算模型结果。

- incomplete: artifact 不完整，需要排查。

## 下一步执行队列

### P0：把 promising signal 放大

- 对 strict1000 checkpoint 在 LR 和 activation 上做多 sample proposal generation。

- 用 local proposal scorer 先过滤，优先送 worker 的是 high-quality proposal。

- 目标：确认 near miss 是否能通过 sampling 找到 true pass。

### P1：重做更硬 prompt 的训练数据

- 用已经更新的 hard XML prompt 重新 collate。

- 强化 target 中的 code-level plan、editable file constraint、ablation 和 failure mode。

- 训练后重新跑 10-task proposal batch。

### P2：扩展训练集

- 扩大到 2025-04 到 2025-05，仍保持 chronological training。

- 优先 method_algorithm、system_tooling、vision / ML category。

- 过滤掉 TeX evidence 不干净、target 不够 nontrivial、reference detail 不足的样本。

### P3：加入评分预测而不只预测 pass/fail

- pass 很稀疏，不能只做 binary pass prediction。

- 需要同时预测 expected score、expected delta、risk 和 confidence。

- 后续可用于 proposal reranking、DPO pair construction 或 reward model。

### P4：训练 master 而不是只训练 proposal generator

- master 需要后续回答 worker 问题，所以不能只优化 proposal 格式。

- 训练数据应包含 proposal、clarification、debug advice、worker dialogue 四类能力。

- 7B 可能不足以保住这些综合能力；32B 更合理，后续可比较 API master、32B SFT master、checkpoint proposal modules。

## 当前判断

V3 现在不是失败，也不是完成。最重要的事实是：训练出的 32B 模型已经能在两个真实 worker eval 上给出 baseline improvement，但还没过强 threshold。接下来应该优先验证“多采样 + 过滤”能否把 near miss 变成 pass；如果可以，说明训练方向有效，再投入更大规模 hard-prompt 数据和 ranking / reward 训练。