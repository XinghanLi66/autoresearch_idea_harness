"""Score spec for llm-pretrain-sparse-attention (auto-generated)."""
from mlsbench.scoring.dsl import *

term("val_loss_gpt_345m",
    col("val_loss_gpt-345m").lower().id()
    .bounded_power(bound=0.0))

term("wikitext2_ppl_gpt_345m",
    col("wikitext2_ppl_gpt-345m").lower().id()
    .bounded_power(bound=1.0))

term("lambada_ppl_gpt_345m",
    col("lambada_ppl_gpt-345m").lower().id()
    .bounded_power(bound=1.0))

term("arc_easy_lm_eval_345m",
    col("arc_easy_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("hellaswag_lm_eval_345m",
    col("hellaswag_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("piqa_lm_eval_345m",
    col("piqa_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("winogrande_lm_eval_345m",
    col("winogrande_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

setting("gpt-345m", weighted_mean(("val_loss_gpt_345m", 1.0), ("wikitext2_ppl_gpt_345m", 1.0), ("lambada_ppl_gpt_345m", 1.0)))
setting("lm-eval-345m", weighted_mean(("arc_easy_lm_eval_345m", 1.0), ("hellaswag_lm_eval_345m", 1.0), ("piqa_lm_eval_345m", 1.0), ("winogrande_lm_eval_345m", 1.0)))

task(gmean("gpt-345m", "lm-eval-345m"))
