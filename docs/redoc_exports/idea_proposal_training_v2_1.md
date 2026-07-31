# <redoc-comment commentGid="7646738553612800235" blockId="5fd95cf6006ed2a0d09cf7d7a50f4116">Idea Proposal Training Ver 2.1</redoc-comment>

## 一页摘要

上一版 training target 的核心问题是 abstract 太 abstract。abstract 通常会描述 paper 的贡献和结论，但很少给出足够清楚的 implementation plan、algorithm detail、training recipe、evaluation protocol 和 failure modes。模型如果只对齐 abstract target，很容易学到“会写研究方向”，但不会学到“如何把 idea 说到可执行”。

Ver 2.1 的方向是把 full TeX source 引入 target synthesis，但这一步不能只增强 target。更关键的是让 condition 和 target 的 detail level 匹配：如果 target 里要求具体实现细节，那么 condition 侧也要提供能支持这些细节的 evidence，例如 selected reference TeX snippets。否则模型只是在学习写细节，而不是学习从 reading list 推断细节。

因此当前新增了一个 paper classification layer。它先判断每篇 paper 属于什么类型、TeX 细节是否足够、是否需要 reference-side detail，再决定后续 synthesis 应该使用什么 conditiet schema。

## 当前数据产物

本轮分类输出目录：

/newcpfs/lxh/agentic-training/data/arxiv_tex_classified/v1

主要文件：

- papers.jsonl：TeX-ok 的主分类表。

- splits/train.jsonl、splits/val.jsonl、splits/test.jsonl：按原 split 保留的分类结果。

- indices/by_paper_type/：按 paper type 分组。

- indices/by_training_route/：按后续 training route 分组。

- indices/needs_ref_tex.jsonl：后续需要 reference TeX detail 的样本。

- indices/skipped.jsonl：无 TeX 或 TeX 不可用的 audit 记录。

- manifest.json：保存分类版本、核心 routing rules 和 summary。

本轮统计：

- 输入 records：7,717

- TeX-ok accepted records：7,441

- skipped records：276

- implementation-candidate records：5,384

- 需要 reference TeX detail 的 records：4,004

无 TeX paper 不进入主训练表，只进入 skipped audit。其中 missing_tex_files 为 261，missing_tex_dir 为 15。

## 分类逻辑

第一层使用 arXiv categories。这些标签不是最终 paper type，但它们是很有用的 domain prior 和 stratification key。例如 cs.CV、cs.CL、cs.IR、cs.LG、stat.ML 会帮助我们控制训练数据的领域比例，避免某几个方向过度主导。

第二层解析 full TeX source。脚本会读取本地 tex/*.tex，按 section heading 拆分正文，并统计 method、implementation、evaluation、results 等 section 的数量和字符量。这个步骤决定 detail_support_level，也就是这篇 paper 是否真的有足够细节支撑 detailed target。

第三层判断 paper_type。当前分为：

- method_algorithm：新 model、algorithm、loss、training recipe。

- system_tooling：system、framework、toolkit、pipeline。

- dataset_benchmark：dataset、benchmark、leaderboard、evaluation suite。

- empirical_analysis：empirical study、analysis、ablation-heavy paper。

- survey_position：survey、review、taxonomy、position。

- theory_math：theorem、proof、bound、convergence-heavy paper。

- application_domain：领域应用主导，需要单独 review。

这个分类的目的不是做 taxonomy 本身，而是决定后续 training target 应该怎么合成。

## 当前分类结果

Paper type 分布：

- method_algorithm：4,604

- dataset_benchmark：1,042

- system_tooling：872

- empirical_analysis：452

- survey_position：287

- theory_math：182

- application_domain：2

TeX detail support 分布：

- tex_rich：2,849

- tex_good：2,457

- tex_light：1,985

- tex_weak：150

Training route 分布：

- tex_target_with_ref_detail：4,004

- tex_target_light：1,380

- dataset_benchmark_target：1,033

- experiment_target：430

- separate_theory_target：182

- exclude_from_impl_sft：411

- domain_target_review：1

## Routing 决策

最重要的 route 是 tex_target_with_ref_detail。这类样本通常是 method_algorithm 或 system_tooling，并且目标 paper 的 TeX 里有足够 method、implementation、evaluation detail。它们后续应该使用 topk_refs_plus_ref_tex 或 related_work_plus_system_ref_tex 作为 condition，而不是只用 reference abstract。

tex_target_light 可以作为轻量实现训练候选，但需要谨慎抽查。它们的 target 可以更短，不应该强行要求过细的 implementation。

dataset_benchmark_target 不应该被塞进 implementation proposal schema。这类 paper 更适合生成 dataset motivation、resource construction、benchmark design 和 evaluation protocol。

experiment_target 更适合生成 experiment plan and findings，而不是伪装成新 algorithm。

separate_theory_target 应该单独设计 theory/analysis schema。把 theorem-heavy paper 强行转成 implementation recipe 会污染训练信号。

survey_position 和 tex_weak 默认不进入 implementation SFT。

## 对训练的含义

Ver 2.1 的主线不是“把所有 abstract target 换成 detailed target”。更准确的说法是：先把 paper 分流，再为每一类 paper 选择匹配的 condition 和 target。

推荐下一步优先做三件事：

1. 从 tex_target_with_ref_detail 中采样，构造第一版 reference-TeX-conditioned synthesis dataset。

2. 为 dataset_benchmark_target 和 experiment_target 单独设计 target schema，不和 implementation target 混训。

3. 在 sampling 时按 category_family、paper_type 和 detail_support_level 做 stratification，避免 cs.CV 或 cs.CL 占比过高。

这一步完成后，我们训练的模型才有机会学到“从 references 和 evidence 推出可执行 proposal”，而不是只学会把 target 写得更长。