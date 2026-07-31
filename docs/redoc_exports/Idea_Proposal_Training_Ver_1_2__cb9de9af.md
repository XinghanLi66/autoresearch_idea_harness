## Exp

## <redoc-comment commentGid="7633374904605724792" blockId="0650d1fe416d9351c7875a74abf4279a">8 ablation scripts (scripts/ablations/)</redoc-comment>

| Script | Strategy | Finetune | Reward | Ablation |
| --- | --- | --- | --- | --- |
| exp01_baseline.sh | top_k_refs | full | embed-PRS | <font color="#F06A1D">**baseline**</font> |
| exp02_full_refs.sh | <font color="#F06A1D">**full_refs (40×400)**</font> | full | embed-PRS | top-k → full refs |
| exp03_related_work.sh | <font color="#F06A1D">**related_work**</font> | full | embed-PRS | top-k → LLM-synthesized narrative |
| exp04_topk_related_work.sh | <font color="#F06A1D">**top_k_related_work**</font> | full | embed-PRS | top-k → select-then-synthesize |
| exp05_topk_research_question.sh | <font color="#F06A1D">**with_research_question**</font> | full | embed-PRS | top-k → top-k + research question |
| exp06_lora.sh | `top_k_refs` | <font color="#F06A1D">**lora**</font> | embed-PRS | full-ft → LoRA |
| exp07_fas.sh | `top_k_refs` | full | <font color="#F06A1D">**embed-FAS**</font> | embed-PRS → embed-FAS |
| exp08_llm_prs.sh | `top_k_refs` | full | <font color="#F06A1D">**llm-judge-FAS**</font> | embed-PRS → LLM-judge |

<redoc-comment commentGid="7633378447953340920" blockId="6f8ad527607051761479a038d080d582">Each script:</redoc-comment>

generates a config,

registers a dashboard tab,

runs a 2×2 hparam mini-search from** LR ∈ &#123;2e-6, 5e-6&#125; × KL ∈ &#123;0.02, 0.05&#125;, 512 examples**,

1. picks the best combo by mean reward over last 10 logged steps,

2. runs full training.

**Runtime estimate:**

- A full RL epoch over ~50K examples on 8 GPUs (1 per-device batch + 8 accumulation steps = effective batch 64)  

- at 2048 completion length takes roughly **10–14 hours per experiment **on 8*L20Z

- The 4-point hparam search, each with 512 samples, adds ~1–1.5h.  

- Total per experiment: ~11–15h. Currently 2 exp per machine -&gt; **30h on each machine.**

## Training Results @ 260427

Finished: 1, 2, 5, 6, 7

Unfinished due to speed: 8

Unfinished due to bugs: 3, 4

**Hparam search — all converged to the same best config: lr=5e-6, kl=0.02**

<redoc-columns>
<redoc-column ratio="0.474239">
![](https://xhs-doc.xhscdn.com/104004dg31vf6h4s24c040bf9lc?redoc-w=435&redoc-h=158)
</redoc-column>
<redoc-column ratio="0.525761">
![](https://xhs-doc.xhscdn.com/104004dg31vf6hfsu1m1emm9skk?redoc-w=468&redoc-h=152)
</redoc-column>
</redoc-columns>

<redoc-columns>
<redoc-column ratio="0.473219">
![](https://xhs-doc.xhscdn.com/104004dg31vf6i8fr48053l26p0?redoc-w=434&redoc-h=138)
</redoc-column>
<redoc-column ratio="0.526781">
![](https://xhs-doc.xhscdn.com/104004dg31vf6ijph4c0f1qdnt4?redoc-w=462&redoc-h=145)
</redoc-column>
</redoc-columns>

![](https://xhs-doc.xhscdn.com/104004dg31vf6jf2o480dk1uqus?redoc-w=446&redoc-h=142)

**Final training metrics — end of epoch 1:**

| Exp | Reward Strategy | Reward | eval PRS | eval FAS | eval Format | eval KL | Avg length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 (baseline) | 0.8PRS + 0.2fmt | 0.753 | 0.691 | – | 0.998 | 0.018 | 1126 |
| exp02 (full_refs) | 0.8PRS + 0.2fmt | 0.774 | 0.718 | – | 0.998 | 0.017 | 1214 |
| exp05 (top_k+RQ) | 0.8PRS + 0.2fmt | 0.775 | 0.719 | – | 0.998 | 0.017 | 1296 |
| exp06 (LoRA) | 0.8PRS + 0.2fmt | 0.751 | 0.689 | – | 1.000 | 0.018 | 1088 |
| exp07 (FAS reward) | 0.6FAS + 0.2fmt + 0.2antileak | 0.868 | – | 0.781 | 0.994 | **0.089** | 1167 |

- *Antileak is 1.0 for all examples, inflating the combined reward. FAS-based reward is not directly comparable to PRS-based reward.

**Key takeaways at end of training:**

- **Prompt strategy helps:** exp02 and exp05, which use more context-rich prompts, both push PRS ~0.027 higher than the `exp01` baseline, despite using the same reward and model. <font color="#F06A1D">**full_refs and RQ are best so far.**</font>

- **LoRA underperforms** full fine-tuning by ~0.002 PRS at similar KL.

- <font color="#F06A1D">**FAS reward causes much higher KL**</font>: 0.089 vs 0.017–0.018. The model drifts further from the base in pursuit of FAS. Whether this translates to better FAS on the test set is what the eval run will tell us.

- exp05 has longest completion length -&gt; <font color="#F06A1D">**research-question prompt elicit more thorough proposals.**</font>

## Test Results

M0 finished both exp05 and exp06. The table already printed by M0's `compare.py` at the end shows all completed results. Here's the full picture (exp01, exp05, exp06, exp07, Claude — exp02 still running on M3):

| Model | FAS | recall@50 | PRS | fmt | leakage (&lt;0.85) | gen time |
| --- | --- | --- | --- | --- | --- | --- |
| **exp05** top_k+RQ / full-ft / PRS | **0.657** | **0.735** | 0.596 | 0.938 | 0.522 | ~20min |
| **exp06** LoRA / PRS | 0.650 | 0.712 | **0.609** | 0.938 | 0.527 | ~20min |
| **exp01** baseline top_k / full-ft / PRS | 0.643 | 0.700 | 0.604 | 0.912 | 0.518 | ~33min |
| **claude-sonnet-4-6** | 0.623 | 0.688 | 0.570 | 0.005† | 0.428 | 4min |
| **exp07** FAS-trained | 0.365 | 0.088 | 0.414 | 0.945 | 0.367 | ~20min |

**Analysis:**

**LoRA vs full fine-tuning (exp06 vs exp01, same top_k_refs/PRS setup):**

<font color="#F06A1D">**LoRA is surprisingly competitive**</font> — FAS 0.650 vs 0.643 and actually edges ahead on PRS (0.609 vs 0.604). At a fraction of the compute cost, this is a strong result for the ablation. The slightly shorter completions during training (1088 vs 1126 avg tokens) don't hurt quality.

**Prompt strategy (exp05 vs exp01):**

<font color="#F06A1D">**Adding a research question improves FAS by +0.014 and recall by +3.5pp. Interestingly, PRS drops slightly (0.596 vs 0.604)**</font> — the research question augmentation makes proposals more forward-looking and less tightly coupled to the specific source paper, which helps FAS but slightly hurts PRS.

**Claude sonnet-4-6 vs trained models:**

Claude without fine-tuning is a strong baseline — its FAS (0.623) and recall (0.688) sit just below exp01. The fine-tuned models add ~2–3.5pp FAS on top of Claude zero-shot, which is the signal that RL training is providing genuine value.

†Claude's format score is near zero — it generates semantically valid proposals but doesn't follow the `&lt;problem&gt;`/`&lt;gap&gt;`/`&lt;key_insight&gt;`/`&lt;approach&gt;`/`&lt;expected_contributions&gt;` XML structure. Likely hitting the 1024-token limit before completing the template (Claude first writes a `&lt;thinking&gt;` section). FAS and PRS are still meaningful.

<font color="#F06A1D">**The most striking finding so far: exp07 (FAS-trained) has dramatically worse FAS**</font>** and recall than the PRS-trained exp01, despite being explicitly trained to maximise FAS.** A few reasons likely combine:

1. **Val→test shift**: exp07 was optimised against the `val_index` (Nov–Dec 2025 papers). The eval uses `test_index` (Jan–Mar 2026). The model may have overfit to val-paper directions.

2. **Reward shape mismatch**: the FAS training reward was `mean_sim(top-50)` — it learns to generate broadly plausible proposals. PRS training forces proposals to specifically match the _source_ abstract, which turns out to be correlated with landing in the correct paper's neighborhood at eval time.

3. **Higher KL (0.089 vs 0.018)**: exp07 drifted further from the base model, producing more generic-sounding proposals.

exp02 and exp06 results will fill in the LoRA vs full-FT and full_refs vs top_k comparisons.

<br/>

<br/>

**Full results table (all 6 completed evals, sorted by FAS):**

| exp | prompt strategy | reward | FAS | recall@50 | PRS | fmt | gen_time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp02 | full_refs | PRS | **0.677** | **0.753** | **0.622** | **0.952** | 2022s |
| exp05 | topk_rq | PRS | 0.657 | 0.735 | 0.596 | 0.938 | 1191s |
| exp06 | top_k_refs + LoRA | PRS | 0.650 | 0.712 | 0.609 | 0.938 | 1185s |
| exp01 | top_k_refs (baseline) | PRS | 0.643 | 0.700 | 0.604 | 0.912 | 1969s |
| claude-sonnet-4-6 | — | — | 0.623 | 0.688 | 0.570 | 0.005 | 236s |
| exp07 | top_k_refs | FAS | 0.365 | 0.088 | 0.414 | 0.945 | 1171s |

---

**Analysis:**

**exp02 is the clear winner so far.** `full_refs` beats `top_k_refs` (exp01) by +3.4 FAS points (+0.053 recall@50). The LLM-based top-K selection in exp01 was supposed to filter signal from noise, but simply feeding all references lets the model do its own implicit attention over more context — and it works better.

**exp05 (topk + research question) beats exp01 but loses to exp02.** The appended LLM-generated research question adds some signal (+1.4 FAS), but it can't compensate for losing the full reference list.

**exp06 (LoRA) is competitive with exp01 full-FT** at 0.650 vs 0.643, while using far fewer trainable parameters. PRS is slightly higher for LoRA (0.609 vs 0.604). Given the parameter efficiency, LoRA is a strong choice for deployment.

**Claude API baseline** (0.623) is beaten by all 4 trained checkpoints. It also has near-zero format score because it doesn't follow the XML sub-tag format without fine-tuning.

**exp07 collapse** is confirmed — FAS reward during training causes catastrophic failure at test-time FAS evaluation (0.365), despite producing well-formatted outputs (fmt=0.945). The val→test distribution shift is the primary cause.

**Still pending:** exp03 (related_work), exp04 (topk_related_work) training on M1.

<br/>

<br/>

<br/>

Results are in. Updated leaderboard:

| Rank | Exp | Description | FAS | recall@50 | PRS |
| --- | --- | --- | --- | --- | --- |
| 1 | **exp02** | full_refs | **0.6767** | 0.7529 | **0.6222** |
| 2 | exp03 | related_work | 0.6615 | 0.7353 | 0.6093 |
| 3 | exp05 | topk_rq | 0.6572 | 0.7353 | 0.5961 |
| 4 | exp06 | lora | 0.6502 | 0.7118 | 0.6090 |
| 5 | exp04 | topk_related_work | 0.6401 | 0.7000 | 0.6010 |
| 6 | exp01 | baseline | 0.6433 | 0.7094 | 0.6037 |
| <br/> | Opus 4.6 | API | 0.6400 | 0.7235 | 0.5703 |
| — | _Sonnet 4.6_ | _API_ | _0.6228_ | _0.6882_ | _0.5700_ |
| — | exp07 | FAS reward | 0.3648 | 0.0882 | 0.4140 |

exp03 (related_work prompt) is a solid #2. exp04 (topk_related_work) is slightly below the no-prompt baseline (exp01), suggesting that truncating related-work references hurts more than it helps. exp08 still pending.

<br/>

<br/>

#### BTW

- We badly need a scheduler agent when routing on multiple-machines :)

- If idea proposal is not verifiable, it's full of opportunities. We can extract some verifiable parts (such as in one topic, the model has 10 directions that potentially makes sense, we can evaluate the model by looking at the recall rate). Or we can force the model to give code or initial results...

- If training 7B already surpasses claude, how about training large models?

<br/>