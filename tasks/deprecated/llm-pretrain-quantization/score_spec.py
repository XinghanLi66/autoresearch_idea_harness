"""Score spec for llm-pretrain-quantization.

Reference baseline: ptq_int4 (seed=mean)
  val_loss=2.383, quant_val_loss=2.3832, quant_degradation=0.0001,
  wikitext2_ppl=50.41, lambada_ppl=77.62,
  arc_easy=54.5, hellaswag=32.99
"""
from mlsbench.scoring.dsl import *

# gpt-345m quality + quantization degradation metrics
# val_loss and quant_val_loss: lower is better, bound=0
# quant_degradation: gap between quantized and full precision -- lower is better, bound=0
# *_ppl: lower is better, bound=1
# lm-eval benchmarks: 0-100 scale, higher is better, bound=25 (random baseline)

term("val_loss",
    col("val_loss_gpt-345m").lower().id()
    .bounded_power(bound=0.0))

term("quant_degradation",
    col("quant_degradation_gpt-345m").lower().id()
    .bounded_power(bound=0.0))

term("wikitext2_ppl",
    col("wikitext2_ppl_gpt-345m").lower().id()
    .bounded_power(bound=1.0))

term("lambada_ppl",
    col("lambada_ppl_gpt-345m").lower().id()
    .bounded_power(bound=1.0))

term("arc_easy",
    col("arc_easy_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("hellaswag",
    col("hellaswag_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("piqa",
    col("piqa_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("winogrande",
    col("winogrande_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

setting("gpt-345m", weighted_mean(
    ("val_loss", 1.0),
    ("quant_degradation", 1.0),
    ("wikitext2_ppl", 1.0),
    ("lambada_ppl", 1.0),
))

setting("lm-eval-345m", weighted_mean(
    ("arc_easy", 1.0),
    ("hellaswag", 1.0),
    ("piqa", 1.0),
    ("winogrande", 1.0),
))

task(gmean("gpt-345m", "lm-eval-345m"))
