## 一条主线

最初的问题很简单：我们用 abstract 做 alignment target，<font color="#F06A1D" backgroundColor="#FFFF33">**但 abstract 本身太抽象**</font>。它通常只说明论文贡献和结论，不会告诉模型一个 idea 应该如何实现、需要哪些 algorithm steps、怎样构造 training recipe、怎样设计 evaluation plan。模型如果只对齐 abstract，就会学到同样的毛病：输出看起来像研究方向，但不够可执行。

于是我们收集 full TeX source，想用论文正文来合成更 detailed 的 training target。这个方向是对的：TeX 里的 method、experiment、appendix 往往包含 abstract 没有的 implementation details。

但做 TeX synthesis 之后会暴露第二个问题：<font color="#F06A1D" backgroundColor="#FFFF33">**target 变具体以后，condition 也必须有足够细节**</font>。如果 input 仍然只是 reference abstracts，却要求模型输出很具体的 method design 和 experiment recipe，模型就只能学会编细节，而不是基于 reading list 推断细节。

所以当前计划自然变成三步：

1. 先用 target paper 的 full TeX 合成 detailed target。

2. 再判断这篇 paper 的 references 是否真的提供了足够 detail，可以支撑模型生成 detailed target。

3. 最后按文章类型和 evidence support 做分类，不同类别使用不同 conditioning 和 target schema。

## 当前核心判断

TeX synthesis 只解决 target 太抽象的问题；它不能单独解决 input evidence 不足的问题。

因此，训练 detailed proposal model 需要三件东西同时匹配：

- detailed target：来自 target paper 的 full TeX synthesis。

- detailed condition：来自 references 的 abstracts、related work、research question，以及必要时的 reference TeX snippets。

- correct paper route：不同 paper type 使用不同 conditioning 和 target schema。

target paper 的 TeX 只能用于 target synthesis 和 reward reference，不能进入该 paper 的 model input。reference papers 的 TeX snippets 可以进入 input，因为它们属于 researcher reading list 的可用信息。

## 当前先做什么

第一阶段先 cache building blocks，不急着直接训练。

每篇 target paper 要缓存：

- metadata：title、abstract、categories、created date。

- prompt variants：full_refs、top_k_refs、related_work、top_k_related_work、with_research_question。

- target-paper tex_synthesis：从 target paper TeX 合成的 implementation-ready target。

- paper_classification：文章类型、detail support、推荐 conditioning、推荐 target schema、训练 route。

同时，对重要 reference papers 也要 cache selected TeX snippets，尤其 method、implementation、dataset、evaluation sections。这样后面可以自由拼 condition，而不是每次训练都重新调用 LLM。

## 文章分类

分类器应该在 dataset assembly 之前运行。它的任务不是“描述这篇论文是什么”，而是回答训练问题：这篇 paper 应该怎样被训练？

每篇 paper 至少输出这些字段：

- paper_type：empirical_method、system、benchmark_or_survey、theory_or_analysis、dataset_or_resource、other。

- detail_support_level：high、medium、low，表示 references 和可用 reference snippets 是否足以支撑 detailed target。

- reference_evidence_need：abstracts_only、top_k_ref_tex、related_work_context、research_question_context、holdout。

- recommended_condition：使用哪种 prompt variant，是否加入 reference TeX snippets。

- recommended_target_schema：implementation、system_design、benchmark_protocol、theory_analysis、dataset_resource、high_level_proposal。

- training_route：target_only_ablation、matched_detailed、matched_high_level、category_specific、skip_or_holdout。

第一版分类器可以 lightweight。输入 title、abstract、categories、section headings、selected target TeX snippets、refs metadata。之后再升级成更强的 LLM classifier。

## 每类文章如何设计 conditioning

### Empirical ML method paper

这是最适合 detailed implementation target 的类型，但前提是 references 里能看到足够技术线索。

如果 detail_support_level 是 high，conditioning 应该使用 top-k references 加 selected reference TeX snippets。snippets 重点来自 reference methods、training setup、loss、dataset、evaluation sections。目标是让模型看到足够多的 prior method details，然后学会组合出新的 method proposal。

推荐 condition：top_k_refs 或 top_k_related_work 作为骨架，再加入 top-k reference TeX snippets。

推荐 target schema：implementation。字段包括 implementation_plan、algorithm_or_system、training_or_data_recipe、evaluation_plan。

如果 support 是 low，不要强行训练 detailed target。可以改成 high_level_proposal，或者先补 reference snippets 再进入 detailed route。

### System paper

System paper 的 detail 通常不是 loss 或 dataset，而是 architecture、component interface、workflow、latency/resource tradeoff、failure mode。

conditioning 应该给 related work context，加上 references 中 system architecture 或 implementation sections 的 snippets。只给 abstracts 往往不够，因为 abstracts 很少包含 component-level design。

推荐 condition：related_work 或 top_k_related_work，加 system/component reference snippets。

推荐 target schema：system_design。字段应强调 system goal、components、interfaces、workflow、resource constraints、deployment/evaluation protocol、failure modes。

### Benchmark or survey paper

这类 paper 不应该被训练成“提出一个 algorithm”。它们的核心是 taxonomy、coverage、benchmark construction、evaluation protocol、metrics、baselines 和 analysis。

conditioning 应该偏 related work context，必要时加入 reference evaluation sections 或 existing benchmark protocol snippets。

推荐 condition：related_work，加 benchmark/evaluation reference snippets。

推荐 target schema：benchmark_protocol 或 survey_plan。字段应强调 scope、taxonomy、dataset/task selection、metrics、baselines、evaluation protocol、expected analysis。

### Theory or analysis paper

Theory/analysis paper 经常没有 training recipe。强行要求 implementation_plan 会产生错误监督。

conditioning 应该给 definitions、assumptions、problem setup、prior theorem/analysis 的 reference snippets。如果没有这些细节，就只训练 high-level formal proposal。

推荐 condition：full_refs 或 related_work，加 theory setup snippets。

推荐 target schema：theory_analysis。字段应强调 formal problem、assumptions、claim/hypothesis、derivation or proof plan、validation experiment。

### Dataset or resource paper

这类 paper 的 detail 在 data pipeline，而不是 model architecture。

conditioning 应该给 similar dataset/resource papers 的 collection、annotation、filtering、quality control、benchmark task snippets。

推荐 condition：top_k_related_work，加 dataset construction reference snippets。

推荐 target schema：dataset_resource。字段应强调 data source、collection protocol、annotation schema、filtering, quality checks、release format、downstream evaluation tasks。

### Reference-evidence-poor paper

如果 references 只是 broad background，不能支撑 target paper 的 concrete method，那这类样本不应该进入 matched_detailed training。

可选路线：

- 用 high_level_proposal target，训练模型提出合理方向但不要求具体 recipe。

- 加强 condition，比如加入 more related work 或 reference TeX snippets 后再判断。

- 暂时 holdout，不用于第一版 detailed training。

## Dataset assembly 的主次

第一组实验可以做 target-only ablation：condition 不变，只把 old abstract target 换成 TeX synthesis target。这组实验的作用是验证 target 是否变好了，但它不是最终方案。

真正重要的是第二组：condition-target matched dataset。这里要根据 classification route 决定 input 里是否加入 reference snippets，以及 target schema 是否需要切换。

第三组再做 category-specific schema，让 system、benchmark、theory、dataset/resource paper 各自使用合适的输出结构，而不是全部挤进 empirical method schema。

## <redoc-comment commentGid="7643765431222595653" blockId="abde1aebe01681889c946e2f10e3f36a">Implementation Plan</redoc-comment>

第一步，完成 target-paper TeX synthesis cache。当前 proposal_rl/data/synthesize_tex_targets.py 已经实现第一层：读取本地 TeX、抽取 useful sections、合成 tex_implementation target，并在 scripts/demo_tui.py 里预览 original abstract target 和 TeX synthesis target。

第二步，实现 paper classification cache。这是独立 stage，不是 synthesis script 的附属统计。它负责输出 paper_type、detail_support_level、reference_evidence_need、recommended_condition、recommended_target_schema、training_route。

第三步，实现 reference TeX snippet cache。对 top-k references 或策略选择出的 important references，抽取 method、implementation、dataset、evaluation snippets。cache 按 reference arxiv_id 独立保存，避免重复处理。

第四步，实现 dataset assembler。assembler 不重新调用 LLM，而是读取 cached building blocks，根据 classification routing 组合 condition prompt 和 target。

第五步，再改 SFT/RL parquet builders。make_parquet.py 支持 target field 和 cot field 选择。RL reward 可以新增 tis，即 TeX Implementation Score，把 generated proposal 与 target_impl_proposal 比较，而不是只和 abstract 比较。

## Quality Gates

每个样本进入训练前要检查两个层面。

target quality：是否包含 required fields，是否有 concrete implementation/evaluation content，是否只是 abstract paraphrase，是否太短或 generic。

condition-target match：condition evidence 是否足以支持 target 的 detail，classification route 是否和 target schema 一致，reference snippets 是否真的提供了 method/evaluation detail。

对于 first pilot，每个类别都要抽样人工检查，而不是只看总体通过率。

## Experiment Plan

先做小规模 pilot，不要直接全量训练。

1. 抽取 20 到 50 篇有 usable TeX 的 paper，跑 target-paper TeX synthesis。

2. 对同一批 paper 跑 classification，人工检查 paper_type 和 detail_support_level。

3. 为 high-support empirical/system/benchmark/dataset paper 缓存 reference TeX snippets。

4. 组装三套小数据：target-only replacement、condition-target matched、category-specific schema。

5. 跑 tiny SFT smoke，检查模型输出是否更 concrete，是否开始 hallucinate unsupported details。

6. 通过 benchmark sweep 和人工 review 比较 worker success rate、concreteness、faithfulness。

## Agent 分工

coder 负责 TeX extraction、target synthesis、paper classifier、reference snippet cache、dataset assembler、parquet target switch 和 reward path。

benchmarker 负责 target audit、classification audit、condition-target match audit、concreteness metrics 和 pilot benchmark。

exp-manager 负责在 audit 通过后启动 synthesis、SFT、RL 和 benchmark jobs，并记录 job status。

meta-coordinator 负责维护 checkins 和云端 plan，确保关键设计不只留在对话 context 里。

## Assumptions

- Target paper TeX 不进入该 paper 的 model input。

- Reference paper TeX snippets 可以作为 input building block。

- Paper classification 是 dataset assembly 前的独立 stage。

- 第一版优先选择有 usable TeX 且 method/evaluation sections 足够丰富的 paper。

- 新 target 的质量比规模更重要。

- 如果 target 很 detailed 但 condition evidence 很弱，这个样本应降级或 holdout，而不是强行训练。