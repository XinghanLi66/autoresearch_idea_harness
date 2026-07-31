## 概述

<redoc-highlight emoji="dengpao" fillColor="orange">
**V3 = human-researcher CoT distillation。** 用 opus-4.8 合成 125 位顶尖 researcher 的第一人称 CoT（重建他们如何得到关键 innovation：idea + non-trivial core + 如何想到，不含 implementation 细节），用来 SFT 一个 32B model，再用 creativity eval 度量。本页记录 **Part B3（judged 数据集精炼 + 统计）** 与 **Part C（creativity 评测 harness）**。代码在分支 V3。
</redoc-highlight>

## Part B3 — 数据集精炼 pipeline

opus-4.8 合成 CoT（每个 case 3 条 reasoning route）+ 一个独立的 opus-4.8 fact-check pass；再经如下 funnel 得到 SFT 数据集：

| 阶段 stage | 数量 | 说明 |
| --- | --- | --- |
| synthesized CoTs | 1776 | opus-4.8，596 个 source case（综合≥8）× 3 route，均已 fact-check |
| rule gate 通过 | 1751 | 丢弃格式坏的：缺 Core-idea / Non-trivial-crux anchor、含 code-fence、长度越界、非英文 |
| opus-4.8 judge | 1751 | 逐条 rubric 打分（faithfulness / clarity / non-triviality / creativity / fingerprint + impl-leak） |
| kept（质量） | 1711 | keep = verdict keep、四项子分≥4、fingerprint≥3、无 impl-leak、anchor 齐全 |
| **train / val** | **1626 / 85** | dedup 后；researcher 通过 conditioned system prompt 命名 |

<redoc-highlight emoji="dui" fillColor="green">
**净流失仅约 3.7%**（synthesized 1776 → kept 1711）。fact-check 过的合成数据本就高质量，judge 主要是在 **确认** 质量而非筛除 → 数据质量 **不是** 瓶颈。
</redoc-highlight>

<redoc-highlight emoji="dengpao" fillColor="orange">
**关键精炼决策：** 每条 SFT 样本用一个带 researcher 名字的 **system prompt**（You are [researcher]…）。好处：(1) model 可被 steer 去 mock 指定 researcher；(2) cot 不必自报名字 → 不会因为没自报名字而被误删（否则会错删约 950 条好样本）。
</redoc-highlight>

## Part B3 — 数据集属性与统计

### Judge 质量（1–5，共 1,751 条）

| 维度 dimension | mean | median |
| --- | --- | --- |
| faithfulness | 4.95 | 5 |
| clarity_of_core | 5.0 | 5 |
| non_triviality | 4.24 | 4 |
| creativity | 4.22 | 4 |
| fingerprint | 4.72 | 5 |

### 覆盖 coverage

- **researcher**：全部 **125 / 125** 覆盖；每人 6–18 条（mean 13.7）。

- **reasoning route（均衡）**：analogy_transfer 575 · empirical_anomaly 568 · first_principles 568。

### Token 长度（Qwen2.5-32B tokenizer）

| 段 segment | mean | min | max | p90 |
| --- | --- | --- | --- | --- |
| system（researcher-conditioned） | 67 | 65 | 73 | — |
| user（setup） | 191 | 116 | 303 | — |
| assistant（cot） | 1071 | 861 | 1371 | 1158 |
| 整条 whole | 1330 | 1089 | 1646 | 1432 |

所有样本都远在训练 seq-len 之内（7B 用 2048；8–16k 可用）。

## Part C — Creativity 评测 harness

<redoc-highlight emoji="dengpao" fillColor="orange">
**这是评测工具，不是训练。** 没有任何 RL reward 或 gradient 用到它。分两部分。
</redoc-highlight>

### (i) Intrinsic scorer — scripts/score_creativity.py

opus rubric 对生成的 proposal/CoT 打分：novelty / non_triviality / clarity_of_core / how_arrived / feasibility（各 1–5）。已在 7B v1 输出上运行：

| 维度 | v1 IID（val） | v1 OOD（recent papers） |
| --- | --- | --- |
| novelty | 2.2 | 2.6 |
| non_triviality | 2.6 | 2.5 |
| clarity_of_core | 2.8 | 2.2 |
| how_arrived | 3.2 | 3.5 |
| feasibility | 2.4 | 2.4 |
| **overall key-creativity** | **2.53** | **2.43** |

<redoc-highlight emoji="cuo" fillColor="red">
**v1 结论（negative）：** 最强项是 **how_arrived**（路径追踪 ~3.2–3.5），最弱是 **novelty / clarity_of_core**（~2.2–2.8）。model 学到了 creative reasoning 的 **形式**，还没学到 **实质**。这正是后续 RL / 更强信号要攻的点。
</redoc-highlight>

### (ii) Freer worker（extrinsic eval）— 已实现

--free-hparams 让 worker prompt 的 scope rule 允许 worker 自己调 hyperparameter（lr / batch / epochs / …），但不得改 metric / split / reporting；--max-turns / --worker-timeout 已有，加大即可。已接入 prompts.py / end_to_end.py / formal_sweep.py + 3 个 run 脚本。

<redoc-highlight emoji="dui" fillColor="green">
**Part C 是训练还是评测？→ 评测。** intrinsic scorer 度量生成结果；freer worker 是评测 harness。两者都不训练 model。creativity 维度未来 **可以** 变成 RL reward，但当前没接。
</redoc-highlight>