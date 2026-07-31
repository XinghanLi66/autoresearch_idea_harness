## Model

- **Base:** Qwen2.5-7B-Instruct  

- **Path:** runs/model_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

- **Fine-tuning method:** **LoRA** for both SFT and GRPO

- **LoRA setup:** `r=64`, `α=128`, `dropout=0.05`

- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

---

## Data

- **Source:** arXiv papers filtered to `cs.LG`, `cs.AI`, `cs.CL`, `cs.CV`, `cs.IR`, `cs.NE`, `stat.ML`

- **Temporal split (strict, no leakage, Qwen2.5-7B-Instruct released @ 2025-03-07):**

- **Train:** 2025-04 to 2025-10 (7 months), 7,357 examples

- **Val:** 2025-11 to 2025-12, 190 examples, used as the GRPO reward index

- **Test:** 2026-01 to 2026-03, 170 examples

- **Filtering:** at least 8 resolved references per paper, with up to 40 references per example

- **Prompt format:** system prompt with a researcher persona, plus a user prompt listing up to 40 references (title + truncated abstract up to 400 characters each), requesting a structured output in the form `&lt;thinking&gt;...&lt;/thinking&gt;&lt;proposal&gt;...&lt;/proposal&gt;`

- **SFT target:** chain-of-thought proposals synthesized by Claude Sonnet 4.6, stored in `train_cot.jsonl`

---

## SFT Hyperparameters

| Parameter | Value |
| --- | --- |
| Epochs | 2 |
| Per-device batch size | 2 |
| Gradient accumulation | 8 steps |
| Effective batch size | 2 × 8 × 8 GPUs = 128 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.05 |
| Max sequence length | 8192 |
| Precision | bfloat16 |
| DeepSpeed | ZeRO-2 |
| Total steps | 48 (completed ~40 before saving final) |

---

## GRPO Hyperparameters

| Parameter | Value |
| --- | --- |
| Init from | SFT LoRA checkpoint (`runs/sft/final`) |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 steps |
| Learning rate | `5e-6` |
| LR scheduler | cosine |
| Warmup ratio | 0.05 |
| Max completion length | **2048 (increased from 1024 and restarted after observing 93% clipping)** |
| Generations per prompt (group size) | 8 |
| KL coefficient (β) | 0.05 |
| Total steps | 1839 (1 full epoch over 7,357 train examples) |
| Precision | bfloat16 |
| DeepSpeed | ZeRO-2 |

---

## GRPO Reward Functions (3 components)

| Reward | Weight | Description |
| --- | --- | --- |
| `reward_fas` | 0.6 | Cosine similarity of proposal embedding vs. validation corpus index (`sentence-transformers/all-MiniLM-L6-v2`, top-50 retrieval) |
| `reward_format` | 0.2 | Fraction of required XML tags present: `&lt;problem&gt;`, `&lt;gap&gt;`, `&lt;key_insight&gt;`, `&lt;approach&gt;`, `&lt;expected_contributions&gt;` |
| `reward_antileak` | 0.2 | Penalizes proposals with word-cosine similarity &gt; 0.80 to the source abstract (anti-copying) |

Rewards are scaled within each group of 8 generations (`scale_rewards=True`).

---

## Final Results

| Checkpoint | FAS | Recall@50 | Mean Sim | Format |
| --- | --- | --- | --- | --- |
| SFT | 0.6232 | 0.6647 | 0.5816 | 0.9424 |
| GRPO | 0.6062 | 0.6000 | 0.6123 | 0.9612 |

GRPO **improved mean similarity and format, but hurt Recall@50**, resulting in a net FAS regression. This is consistent with over-optimization against the validation corpus index, suggesting a train/validation-to-test distribution mismatch.

---

## Training Process

![](https://xhs-doc.xhscdn.com/104004dg31v66qj1ois08ve2g4c?redoc-w=2890&redoc-h=1482)

## Example

- paper id: 2601.18346. Abstract:

```
Recent advances in multimodal large language models (MLLMs) have demonstrated impressive performance on existing low-level vision benchmarks, which primarily focus on generic images. However, their capabilities to perceive and assess portrait images, a domain characterized by distinct structural and perceptual properties, remain largely underexplored. To this end, we introduceQ-Bench-Portrait, the first holistic benchmark specifically designed for portrait image quality perception, comprising 2,765 image–question–answer triplets and featuring (1) diverse portrait image sources, including natural, synthetic distortion, AI-generated, artistic, and computer graphics images; (2) comprehensive quality dimensions, covering technical distortions, AIGC-specific distortions, and aesthetics; and (3) a range of question formats, including single-choice, multiplechoice, true/false, and open-ended questions, at both global and local levels. Based on Q-BenchPortrait, we evaluate 20 open-source and 5 closedsource MLLMs, revealing that although current models demonstrate some competence in portrait image perception, their performance remains limited and imprecise, with a clear gap relative to human judgments. We hope that the proposed benchmark will foster further research into enhancing the portrait image perception capabilities of both general-purpose and domain-specific MLLMs.
```

- baseline: opus 4.6 with 89.9s, **Approach didn't finish, still get FAS = 0.7803**

```
<thinking>
Let me analyze the themes, methods, and open problems across these 35 papers:

**Core Themes:**

1. **Visual Quality Assessment (IQA/VQA)**: Papers [1], [2], [4], [11], [14], [17], [26], [33], [34] focus on image/video quality assessment, with increasing use of LMMs/VLMs.

2. **Human-Centric Visual Understanding**: Papers [4], [13], [15], [18], [23], [28], [31] focus specifically on human faces, bodies, portraits, and 3D human generation quality.

3. **Multimodal Large Language Models (MLLMs/LMMs)**: Papers [1], [5], [6], [7], [8], [12], [16], [19], [20], [22], [24], [25], [27], [30], [32], [35] cover a range of frontier VLMs with increasing capability.

4. **Aesthetic and Artistic Assessment**: Papers [3], [9], [21], [29] focus on aesthetics, artistic images, and affective responses to visual content.

5. **AI-Generated Content Evaluation**: Papers [2], [7], [13], [15] specifically target AIGC quality evaluation.

6. **Benchmarking LMMs for specialized domains**: Papers [3], [6], [11], [13], [18], [26], [28], [34] introduce specialized benchmarks for evaluating LMM capabilities.

7. **3D/4D Content Quality**: Papers [10], [13] address quality assessment for 3D/4D digital humans.

8. **Reinforcement Learning for Quality Understanding**: Paper [2] uses progressive visual reinforcement learning for video quality.

**Key Gaps I Identify:**

1. **Holistic Human-Centric Quality + Aesthetics + Affect**: Most existing work handles IQA (technical quality), aesthetics, and affective/emotional response as separate tasks. There's no unified framework that simultaneously evaluates technical quality, aesthetic quality, and the emotional/affective impact of human-centric images/videos — particularly for AI-generated human content.

2. **Temporal + Spatial Quality for Human-Centric AIGC Video**: While [2] addresses AIGC video quality and [4] addresses face video quality, there's no unified framework for evaluating human-centric AI-generated videos that incorporates temporal coherence, facial quality, body quality, aesthetic quality, AND affective quality in a single model.

3. **Fine-grained, multi-dimensional human-centric quality assessment with reasoning**: Paper [15] notes that existing IQA methods only provide global quality scores. Paper [34] introduces a "perception-reasoning paradigm" for medical IQA. There's a gap in applying this reasoning-based approach to AI-generated human content (faces, bodies, full scenes) with multi-dimensional assessment (technical quality, aesthetics, affect, alignment).

4. **Cross-domain generalization of quality assessment**: Most datasets and methods are domain-specific (faces, portraits, general images, CGI, etc.). There's no unified model that generalizes across domains of human visual content.

5. **Reinforcement Learning for Quality-Aligned Assessment**: Paper [2] uses progressive visual reinforcement learning for AIGC video quality. This approach could be extended to human-centric content with multi-dimensional rewards.

**The Most Compelling Gap:**

Looking at the constellation of papers, I see a very strong convergence around:
- Human-centric content (faces, bodies, 3D humans, portraits) — papers [4], [13], [15], [18], [23], [28], [31], [10]
- AIGC quality evaluation — papers [2], [7], [13], [15]
- LMM-based quality understanding with reasoning — papers [1], [2], [7], [11], [26], [34]
- Aesthetic and affective dimensions — papers [3], [9], [29]
- Multi-dimensional, fine-grained assessment — papers [15], [34]

**Proposed Research Direction:**

A unified LMM-based framework for **multi-dimensional quality assessment of AI-generated human-centric content** (images and videos) that simultaneously evaluates: (1) technical/perceptual quality, (2) aesthetic quality, (3) text-image/video alignment for human attributes, and (4) affective/emotional resonance — all with natural language reasoning explanations.

The key insight is that AI-generated human content (people, faces, bodies in images and videos) requires a fundamentally different quality evaluation framework than general content because:
1. Humans are hypersentitive to artifacts in human faces/bodies (uncanny valley)
2. Human-centric content has unique aesthetic dimensions (expressiveness, naturalness)
3. Human-centric AIGC often has specific prompts with human attributes that need alignment checking
4. The emotional/affective response to human imagery is particularly important for applications

**Approach:**
- Construct a comprehensive multi-dimensional annotation dataset covering technical quality, aesthetics, alignment, and affective dimensions for AI-generated human images and videos
- Fine-tune/train a large VLM using a combination of supervised learning and reinforcement learning with multi-dimensional reward signals
- The model should produce both scalar scores per dimension AND natural language reasoning explanations
- Benchmark against existing methods on all dimensions

**Key Contributions:**
1. First multi-dimensional quality assessment framework for AI-generated human-centric content
2. A large-scale dataset with multi-dimensional human annotations
3. An LMM trained with reinforcement learning for multi-dimensional human-centric quality reasoning
4. Comprehensive benchmark establishing new SOTA

Let me now write the formal proposal.
</thinking>

<proposal>

<problem>
Current quality assessment frameworks for AI-generated human-centric visual content (faces, bodies, portraits, and full-scene human videos) are fragmented across siloed dimensions — technical perceptual quality, aesthetic appeal, text-semantic alignment, and affective/emotional resonance — and lack a unified, reasoning-capable model. As generative AI floods platforms with human imagery and video, evaluating such content requires simultaneous, multi-dimensional understanding that mirrors how real human observers actually perceive and judge human-centric visuals, not just a single scalar score.
</problem>

<gap>
Existing literature reveals several compounding gaps: (1) IQA/VQA methods (e.g., MANIQA, Q-Align, LMM-VQA) target technical distortion or overall quality but are domain-agnostic and cannot handle the heightened human perceptual sensitivity to facial/body artifacts (uncanny valley effects, identity coherence, expression naturalness); (2) human-centric quality datasets (AGHI-QA, FVQ, DSL-FIQA, portrait IQA) each address isolated aspects — faces only, bodies only, static images only — with no dataset bridging technical quality, aesthetics, alignment, and affect for the same content; (3) aesthetic and affective understanding pipelines (AesBench, ArtEmis) are designed for artworks and photography but are not adapted to the unique properties of AIGC human content; (4) emerging reasoning-based evaluation frameworks (MedQ-Bench, VQ-Insight's reinforcement learning) demonstrate the value of language-grounded, multi-step quality reasoning, but this paradigm has not been applied to the human-centric AIGC domain; and (5) 3D/4D human avatar quality (DHQA-4D, LMME3DHF) is treated entirely separately from 2D image/video quality, preventing any cross-modality unification. No work simultaneously addresses all four dimensions — perceptual quality, aesthetics, semantic alignment, and affective resonance — for both generated human images and videos within a single framework.
</gap>

<key_insight>
Human observers evaluate AI-generated human content through an intertwined, hierarchically organized perceptual process: first assessing basic technical fidelity (absence of artifacts, sharpness, coherence), then aesthetic composition and naturalness, then semantic alignment with a prompt or intended depiction, and finally the affective/emotional resonance the content evokes. We hypothesize that a large multimodal model explicitly trained to perform this hierarchical multi-dimensional reasoning — via structured supervision on dimension-aware annotations and reinforced through multi-dimensional reward signals — will produce significantly better-aligned, more interpretable, and more generalizable quality assessments than any single-dimension model. The key insight is that these four dimensions are not independent: an LMM that understands their causal relationships (e.g., a subtle facial identity artifact degrades both technical quality and affective resonance) will produce superior assessments with richer, actionable natural language explanations.
</key_insight>

<approach>
**Phase 1 — Dataset Construction (HumanQA-4D Dataset):** Curate a
```

- cot generated by sonnet 4.6 with 29.3s, **FAS = 0.8224**

```
<thinking>
Step 1: What themes and methods appear across these references?

Looking at the references, I can identify several key themes:
- Large Multimodal Models (LMMs/MLLMs) for visual understanding and quality assessment
- Benchmarks for evaluating MLLMs on various visual tasks (Q-Bench, AesBench, MMBench, etc.)
- Image/Video Quality Assessment (IQA/VQA) datasets and methods
- Domain-specific quality assessment (face images, portraits, AI-generated content, medical images, artistic images, computer graphics)
- Human-centric understanding and evaluation
- Low-level vision perception and understanding

Specific relevant references:
- Q-Bench (ref 26): benchmark for low-level vision in MLLMs
- Q-Bench-Video (ref 11): extending Q-Bench to video
- Q-Align (ref 1): teaching LMMs for visual scoring
- AesBench (ref 3): aesthetics perception benchmark for MLLMs
- Portrait IQA dataset (ref 31): image quality for portraits
- Face-Human-Bench (ref 28): face and human understanding benchmark
- Human-MME (ref 18): human-centric evaluation
- AGHI-QA (ref 15): AI-generated human images quality
- DSL-FIQA (ref 23): face image quality assessment
- FVQ (ref 4): face video quality assessment
- LMME3DHF (ref 13): 3D human face generation evaluation
- Various MLLMs: LLaVA, InternVL, Qwen2.5-VL, DeepSeek-VL, etc.

Step 2: What open problems or limitations do they collectively suggest?

1. Existing benchmarks for MLLMs focus on generic images or general low-level vision - portrait/face-specific quality perception hasn't been systematically benchmarked
2. Portrait images have unique characteristics (skin texture, facial structure, aesthetic composition, identity preservation) that aren't captured in generic benchmarks
3. The diversity of portrait image types (natural photos, AI-generated, artistic, CG) hasn't been covered comprehensively
4. Current evaluation doesn't cover the full range of quality dimensions relevant to portraits (technical distortions, AIGC artifacts, aesthetic quality)
5. MLLMs' capabilities for portrait-specific perception are unknown

Step 3: What would be a natural next research direction?

Given the gap between general-purpose MLLM benchmarks and the specific domain of portrait image quality assessment, a natural direction would be to create a comprehensive benchmark specifically for portrait image quality perception. This would:
- Cover diverse portrait image sources (natural, synthetic, AI-generated, artistic)
- Include various quality dimensions relevant to portraits
- Feature multiple question formats to probe different aspects of perception
- Evaluate existing MLLMs to reveal their capabilities and limitations in this domain

Step 4: What approach might address this gap?

The approach would involve:
1. Constructing a large-scale benchmark dataset with diverse portrait images from multiple sources
2. Defining quality dimensions specific to portrait perception (technical quality, AIGC artifacts, aesthetics)
3. Creating diverse question formats (single-choice, multiple-choice, yes/no, open-ended) at different granularities
4. Evaluating a comprehensive set of open-source and closed-source MLLMs
5. Comparing MLLM performance to human judgments to quantify the gap
</thinking>

<proposal>
<problem>Multimodal large language models (MLLMs) have shown promise in general-purpose low-level vision tasks, yet their ability to perceive and assess portrait images—a visually and semantically distinct domain—remains systematically unexplored. Portrait images exhibit unique structural, aesthetic, and perceptual properties (e.g., facial geometry, skin texture fidelity, identity coherence) that demand specialized evaluation frameworks beyond what existing generic benchmarks provide.</problem>

<gap>Current benchmarks for evaluating MLLMs on low-level vision (e.g., Q-Bench, AesBench) focus primarily on generic photography and do not address the specific quality dimensions relevant to portrait imagery, such as AIGC-specific facial distortions, artistic stylization artifacts, or fine-grained local region quality. Existing portrait or face quality assessment datasets (e.g., GFIQA datasets, portrait IQA datasets) are designed for traditional regression models and do not probe the open-ended reasoning and multi-dimensional perception capabilities of modern MLLMs. Meanwhile, human-centric benchmarks largely address semantic understanding rather than perceptual quality. This leaves a significant gap in our understanding of how well MLLMs can function as assessors of portrait image quality aligned with human perception.</gap>

<key_insight>Portrait images constitute a high-stakes, high-frequency category of visual content with distinct quality criteria that span technical fidelity, generative artifact detection, and aesthetic composition—dimensions that interact in ways not captured by generic quality benchmarks. We hypothesize that systematically probing MLLMs across these dimensions, using diverse portrait image sources and varied question formats, will reveal structured capability gaps and point toward targeted improvements in portrait-aware visual intelligence.</key_insight>

<approach>We propose constructing a comprehensive benchmark dataset specifically targeting portrait image quality perception, drawing images from heterogeneous sources including natural photographs, synthetically distorted images, AI-generated portraits, artistic works, and computer graphics renderings. Quality dimensions would be carefully defined to span technical distortions (blur, noise, compression artifacts), AIGC-specific issues (anatomical inconsistencies, texture artifacts), and aesthetic attributes (composition, lighting, style coherence). The benchmark would include multiple question formats—from closed-form single- and multiple-choice to open-ended descriptive questions—at both global image-level and local region-level granularities to probe perception at different scales. A broad suite of state-of-the-art open-source and closed-source MLLMs would be evaluated, with human performance collected as a reference ceiling to quantify the human-machine gap.</approach>

<expected_contributions>
- A large-scale, carefully curated benchmark for portrait image quality perception that covers diverse image sources, comprehensive quality dimensions, and varied question formats, filling a critical gap in MLLM evaluation.
- A systematic empirical analysis revealing the current capabilities and limitations of leading MLLMs in portrait-specific quality perception, including fine-grained breakdowns across distortion types, image sources, and question formats.
- A human performance baseline that quantifies the gap between current MLLMs and human-level portrait quality understanding, providing a concrete target for future model development.
- Identification of key failure modes and underexplored challenges in portrait perception that can guide the development of more capable domain-aware or fine-tuned MLLMs.
</expected_contributions>
</proposal>
```

- checkpoint response, **FAS = 0.8519**

```plaintext
The proposed work would begin by constructing a large-scale, diverse dataset that captures the complexity of real-world visual content across multiple domains, including faces, humans, medical images, art, and AI-generated content. Each sample would be paired with human annotations reflecting fine-grained perceptual judgments, enabling the training of LMMs to produce human-aligned evaluations. A reinforcement learning framework would be employed to iteratively refine model predictions against ground-truth human annotations, allowing the model to learn to reason about visual quality and aesthetic properties in a manner consistent with human judgment. A standardized benchmark would then be defined, encompassing multiple evaluation tasks spanning quality assessment, aesthetic perception, and human-centric understanding, providing a unified framework for comparing LMMs across these domains. The entire pipeline—from data collection and annotation to model training and benchmarking—would be designed to be reproducible and accessible to the broader research community.
```

- Full output, reproducible by `python scripts/[probe.py](probe.py) --checkpoint runs/grpo/final --paper 2601.18346 --cot --baseline`

```plaintext
Looking up arxiv_id=2601.18346 in dataset...
  [found in test split]

======================================================================
  Paper  : 2601.18346
  Title  : Q-Bench-Portrait: Benchmarking Multimodal Large Language Models on Portrait Imag
  Created: 2026-01-26
  Refs   : 35
======================================================================

── SYSTEM PROMPT ──────────────────────────────────────────────────────
You are a research scientist with deep expertise in machine learning and AI. You will be given a list of papers (title and abstract each) that a researcher has been reading. Your task is to generate a structured research proposal for a novel research direction that these papers collectively suggest — as if you were the researcher proposing new work inspired by this body of literature.

── USER PROMPT ────────────────────────────────────────────────────────
Below are 35 papers from a researcher's reading list. Based on these references, propose a novel research direction.

[1] Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels (2023)
Abstract: The explosion of visual content available online underscores the requirement for an accurate machine assessor to robustly evaluate scores across diverse types of visual contents. While recent studies have demonstrated the exceptional potentials of large multi-modality models (LMMs) on a wide range of related fields, in this work, we explore how to teach them for visual rating aligned with human op...

[2] VQ-Insight: Teaching VLMs for AI-Generated Video Quality Understanding via Progressive Visual Reinforcement Learning (2025)
Abstract: Recent advances in AI-generated content (AIGC) have led to the emergence of powerful text-to-video generation models. Despite these successes, evaluating the quality of AIGC-generated videos remains challenging due to limited generalization, lack of temporal awareness, heavy reliance on large-scale annotated datasets, and the lack of effective interaction with generation models. Most current appro...

[3] AesBench: An Expert Benchmark for Multimodal Large Language Models on Image Aesthetics Perception (2024)
Abstract: With collective endeavors, multimodal large language models (MLLMs) are undergoing a flourishing development. However, their performances on image aesthetics perception remain indeterminate, which is highly desired in real-world applications. An obvious obstacle lies in the absence of a specific benchmark to evaluate the effectiveness of MLLMs on aesthetic perception. This blind groping may impede...

[4] FVQ: A Large-Scale Dataset and an LMM-based Method for Face Video Quality Assessment (2025)
Abstract: Face video quality assessment (FVQA) deserves to be explored in addition to general video quality assessment (VQA), as face videos are the primary content on social media platforms and human visual system (HVS) is particularly sensitive to human faces. However, FVQA is rarely explored due to the lack of large-scale FVQA datasets. To fill this gap, we present the first large-scale in-the-wild FVQA ...

[5] Pixtral 12B (2024)
Abstract: We introduce Pixtral-12B, a 12--billion-parameter multimodal language model. Pixtral-12B is trained to understand both natural images and documents, achieving leading performance on various multimodal benchmarks, surpassing a number of larger models. Unlike many open-source models, Pixtral is also a cutting-edge text model for its size, and does not compromise on natural language performance to ex...

[6] MMBench: Is Your Multi-modal Model an All-around Player? (2023)
Abstract: Large vision-language models (VLMs) have recently achieved remarkable progress, exhibiting impressive multimodal perception and reasoning abilities. However, effectively evaluating these large VLMs remains a major challenge, hindering future development in this domain. Traditional benchmarks like VQAv2 or COCO Caption provide quantitative performance measurements but lack fine-grained ability asse...

[7] LMM4LMM: Benchmarking and Evaluating Large-multimodal Image Generation with LMMs (2025)
Abstract: Recent breakthroughs in large multimodal models (LMMs) have significantly advanced both text-to-image (T2I) generation and image-to-text (I2T) interpretation. However, many generated images still suffer from issues related to perceptual quality and text-image alignment. Given the high cost and inefficiency of manual evaluation, an automatic metric that aligns with human preferences is desirable. T...

[8] Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context (2024)
Abstract: In this report, we introduce the Gemini 1.5 family of models, representing the next generation of highly compute-efficient multimodal models capable of recalling and reasoning over fine-grained information from millions of tokens of context, including multiple long documents and hours of video and audio. The family includes two new models: (1) an updated Gemini 1.5 Pro, which exceeds the February ...

[9] Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method (2023)
Abstract: Image aesthetics assessment (IAA) is a challenging task due to its highly subjective nature. Most of the current studies rely on large-scale datasets (e.g., AVA and AADB) to learn a general model for all kinds of photography images. However, little light has been shed on measuring the aesthetic quality of artistic images, and the existing datasets only contain relatively few artworks. Such a defec...

[10] DHQA-4D: Perceptual Quality Assessment of Dynamic 4D Digital Human (2025)
Abstract: With the rapid development of 3D scanning and reconstruction technologies, dynamic digital human avatars based on 4D meshes have become increasingly popular. A high-precision dynamic digital human avatar can be applied to various fields such as game production, animation generation, and remote immersive communication. However, these 4D human avatar meshes are prone to being degraded by various typ...

[11] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs (2024)
Abstract: With the rising interest in research on Large Multi-modal Models (LMMs) for video understanding, many studies have emphasized general video comprehension capabilities, neglecting the systematic exploration into video quality understanding. To address this oversight, we introduce Q-Bench-Video in this paper, a new benchmark specifically designed to evaluate LMMs' proficiency in discerning video qua...

[12] Qwen2.5-VL Technical Report (2025)
Abstract: We introduce Qwen2.5-VL, the latest flagship model of Qwen vision-language series, which demonstrates significant advancements in both foundational capabilities and innovative functionalities. Qwen2.5-VL achieves a major leap forward in understanding and interacting with the world through enhanced visual recognition, precise object localization, robust document parsing, and long-video comprehensio...

[13] LMME3DHF: Benchmarking and Evaluating Multimodal 3D Human Face Generation with LMMs (2025)
Abstract: The rapid advancement in generative artificial intelligence have enabled the creation of 3D human faces (HFs) for applications including media production, virtual reality, security, healthcare, and game development, etc. However, assessing the quality and realism of these AI-generated 3D human faces remains a significant challenge due to the subjective nature of human perception and innate percept...

[14] MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment (2022)
Abstract: No-Reference Image Quality Assessment (NR-IQA) aims to assess the perceptual quality of images in accordance with human subjective perception. Unfortunately, existing NR-IQA methods are far from meeting the needs of predicting accurate quality scores on GAN-based distortion images. To this end, we propose Multi-dimension Attention Network for no-reference Image Quality Assessment (MANIQA) to impro...

[15] AGHI-QA: A Subjective-Aligned Dataset and Metric for AI-Generated Human Images (2025)
Abstract: The rapid development of text-to-image (T2I) generation approaches has attracted extensive interest in evaluating the quality of generated images, leading to the development of various quality assessment methods for general-purpose T2I outputs. However, existing image quality assessment (IQA) methods are limited to providing global quality scores, failing to deliver fine-grained perceptual evaluat...

[16] LLaVA-OneVision: Easy Visual Task Transfer (2024)
Abstract: We present LLaVA-OneVision, a family of open large multimodal models (LMMs) developed by consolidating our insights into data, models, and visual representations in the LLaVA-NeXT blog series. Our experimental results demonstrate that LLaVA-OneVision is the first single model that can simultaneously push the performance boundaries of open LMMs in three important computer vision scenarios: single-i...

[17] LMM-VQA: Advancing Video Quality Assessment With Large Multimodal Models (2024)
Abstract: The explosive growth of videos on streaming media platforms has underscored the urgent need for effective video quality assessment (VQA) algorithms to monitor and perceptually optimize the quality of streaming videos. However, VQA remains an extremely challenging task due to the diverse video content and the complex spatial and temporal distortions, thus necessitating more advanced methods to addr...

[18] Human-MME: A Holistic Evaluation Benchmark for Human-Centric Multimodal Large Language Models (2025)
Abstract: Multimodal Large Language Models (MLLMs) have demonstrated significant advances in visual understanding tasks. However, their capacity to comprehend human-centric scenes has rarely been explored, primarily due to the absence of comprehensive evaluation benchmarks that take into account both the human-oriented granular level and higher-dimensional causal reasoning ability. Such high-quality evaluat...

[19] OpenAI GPT-5 System Card (2025)
Abstract: This is the system card published alongside the OpenAI GPT-5 launch, August 2025.
  GPT-5 is a unified system with a smart and fast model that answers most questions, a deeper reasoning model for harder problems, and a real-time router that quickly decides which model to use based on conversation type, complexity, tool needs, and explicit intent (for example, if you say 'think hard about this' in ...

[20] Visual Instruction Tuning (2023)
Abstract: Instruction tuning large language models (LLMs) using machine-generated instruction-following data has improved zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. In this paper, we present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce ...

[21] Subjective and Objective Quality Assessment for in-the-Wild Computer Graphics Images (2023)
Abstract: Computer graphics images (CGIs) are artificially generated by means of computer programs and are widely perceived under various scenarios, such as games, streaming media, etc. In practice, the quality of CGIs consistently suffers from poor rendering during production, inevitable compression artifacts during the transmission of multimedia applications, and low aesthetic quality resulting from poor ...

[22] DeepSeek-VL: Towards Real-World Vision-Language Understanding (2024)
Abstract: We present DeepSeek-VL, an open-source Vision-Language (VL) Model designed for real-world vision and language understanding applications. Our approach is structured around three key dimensions: We strive to ensure our data is diverse, scalable, and extensively covers real-world scenarios including web screenshots, PDFs, OCR, charts, and knowledge-based content, aiming for a comprehensive represent...

[23] DSL-FIQA: Assessing Facial Image Quality via Dual-Set Degradation Learning and Landmark-Guided Transformer (2024)
Abstract: Generic Face Image Quality Assessment (GFIQA) evalu-ates the perceptual quality of facial images, which is crucial in improving image restoration algorithms and selecting high-quality face images for downstream tasks. We present a novel transformer-based method for GFIQA, which is aided by two unique mechanisms. First, a “Dual-Set Degradation Representation Learning” (DSL) mechanism uses facial im...

[24] InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models (2025)
Abstract: We introduce InternVL3, a significant advancement in the InternVL series featuring a native multimodal pre-training paradigm. Rather than adapting a text-only large language model (LLM) into a multimodal large language model (MLLM) that supports visual inputs, InternVL3 jointly acquires multimodal and linguistic capabilities from both diverse multimodal data and pure-text corpora during a single p...

[25] Qwen3-VL Technical Report (2025)
Abstract: We introduce Qwen3-VL, the most capable vision-language model in the Qwen series to date, achieving superior performance across a broad range of multimodal benchmarks. It natively supports interleaved contexts of up to 256K tokens, seamlessly integrating text, images, and video. The model family includes both dense (2B/4B/8B/32B) and mixture-of-experts (30B-A3B/235B-A22B) variants to accommodate d...

[26] Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision (2023)
Abstract: The rapid evolution of Multi-modality Large Language Models (MLLMs) has catalyzed a shift in computer vision from specialized models to general-purpose foundation models. Nevertheless, there is still an inadequacy in assessing the abilities of MLLMs on low-level visual perception and understanding. To address this gap, we present Q-Bench, a holistic benchmark crafted to systematically evaluate pot...

[27] mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models (2024)
Abstract: Multi-modal Large Language Models (MLLMs) have demonstrated remarkable capabilities in executing instructions for a variety of single-image tasks. Despite this progress, significant challenges remain in modeling long image sequences. In this work, we introduce the versatile multi-modal large language model, mPLUG-Owl3, which enhances the capability for long image-sequence understanding in scenario...

[28] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants (2025)
Abstract: Faces and humans are crucial elements in social interaction and are widely included in everyday photos and videos. Therefore, a deep understanding of faces and humans will enable multi-modal assistants to achieve improved response quality and broadened application scope. Currently, the multi-modal assistant community lacks a comprehensive and scientific evaluation of face and human understanding a...

[29] ArtEmis: Affective Language for Visual Art (2021)
Abstract: We present a novel large-scale dataset and accompanying machine learning models aimed at providing a detailed understanding of the interplay between visual content, its emotional effect, and explanations for the latter in language. In contrast to most existing annotation datasets in computer vision, we focus on the affective experience triggered by visual artworks and ask the annotators to indicat...

[30] UI-TARS: Pioneering Automated GUI Interaction with Native Agents (2025)
Abstract: This paper introduces UI-TARS, a native GUI agent model that solely perceives the screenshots as input and performs human-like interactions (e.g., keyboard and mouse operations). Unlike prevailing agent frameworks that depend on heavily wrapped commercial models (e.g., GPT-4o) with expert-crafted prompts and workflows, UI-TARS is an end-to-end model that outperforms these sophisticated frameworks....

[31] An Image Quality Assessment Dataset for Portraits (2023)
Abstract: Year after year, the demand for ever-better smartphone photos continues to grow, in particular in the domain of portrait photography. Manufacturers thus use perceptual quality criteria throughout the development of smartphone cameras. This costly procedure can be partially replaced by automated learning-based methods for image quality assessment (IQA). Due to its subjective nature, it is necessary...

[32] InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency (2025)
Abstract: We introduce InternVL 3.5, a new family of open-source multimodal models that significantly advances versatility, reasoning capability, and inference efficiency along the InternVL series. A key innovation is the Cascade Reinforcement Learning (Cascade RL) framework, which enhances reasoning through a two-stage process: offline RL for stable convergence and online RL for refined alignment. This coa...

[33] PIPAL: a Large-Scale Image Quality Assessment Dataset for Perceptual Image Restoration (2020)
Abstract: Image quality assessment (IQA) is the key factor for the fast development of image restoration (IR) algorithms. The most recent IR methods based on Generative Adversarial Networks (GANs) have achieved significant improvement in visual performance, but also presented great challenges for quantitative evaluation. Notably, we observe an increasing inconsistency between perceptual quality and the eval...

[34] MedQ-Bench: Evaluating and Exploring Medical Image Quality Assessment Abilities in MLLMs (2025)
Abstract: Medical Image Quality Assessment (IQA) serves as the first-mile safety gate for clinical AI, yet existing approaches remain constrained by scalar, score-based metrics and fail to reflect the descriptive, human-like reasoning process central to expert evaluation. To address this gap, we introduce MedQ-Bench, a comprehensive benchmark that establishes a perception-reasoning paradigm for language-bas...

[35] DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding (2024)
Abstract: We present DeepSeek-VL2, an advanced series of large Mixture-of-Experts (MoE) Vision-Language Models that significantly improves upon its predecessor, DeepSeek-VL, through two key major upgrades. For the vision component, we incorporate a dynamic tiling vision encoding strategy designed for processing high-resolution images with different aspect ratios. For the language component, we leverage Deep...

Generate a structured research proposal using this exact format:
<thinking>
[Analyze what themes, methods, and open problems span these references. Identify the most compelling gap or opportunity. Think step-by-step before writing the proposal.]
</thinking>
<proposal>
<problem>What core research problem should be addressed?</problem>
<gap>What gap in the existing literature motivates this work?</gap>
<key_insight>What key insight or hypothesis drives the proposed approach?</key_insight>
<approach>How would the proposed method work at a high level?</approach>
<expected_contributions>What are the expected scientific contributions?</expected_contributions>
</proposal>
======================================================================

Loading FAS index (test)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████| 103/103 [00:00<00:00, 10282.36it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

Running baseline (claude-opus-4-6)...

── BASELINE (claude-opus-4-6)  [89.9s] ──────────────────────
<thinking>
Let me analyze the themes, methods, and open problems across these 35 papers:

**Core Themes:**

1. **Visual Quality Assessment (IQA/VQA)**: Papers [1], [2], [4], [11], [14], [17], [26], [33], [34] focus on image/video quality assessment, with increasing use of LMMs/VLMs.

2. **Human-Centric Visual Understanding**: Papers [4], [13], [15], [18], [23], [28], [31] focus specifically on human faces, bodies, portraits, and 3D human generation quality.

3. **Multimodal Large Language Models (MLLMs/LMMs)**: Papers [1], [5], [6], [7], [8], [12], [16], [19], [20], [22], [24], [25], [27], [30], [32], [35] cover a range of frontier VLMs with increasing capability.

4. **Aesthetic and Artistic Assessment**: Papers [3], [9], [21], [29] focus on aesthetics, artistic images, and affective responses to visual content.

5. **AI-Generated Content Evaluation**: Papers [2], [7], [13], [15] specifically target AIGC quality evaluation.

6. **Benchmarking LMMs for specialized domains**: Papers [3], [6], [11], [13], [18], [26], [28], [34] introduce specialized benchmarks for evaluating LMM capabilities.

7. **3D/4D Content Quality**: Papers [10], [13] address quality assessment for 3D/4D digital humans.

8. **Reinforcement Learning for Quality Understanding**: Paper [2] uses progressive visual reinforcement learning for video quality.

**Key Gaps I Identify:**

1. **Holistic Human-Centric Quality + Aesthetics + Affect**: Most existing work handles IQA (technical quality), aesthetics, and affective/emotional response as separate tasks. There's no unified framework that simultaneously evaluates technical quality, aesthetic quality, and the emotional/affective impact of human-centric images/videos — particularly for AI-generated human content.

2. **Temporal + Spatial Quality for Human-Centric AIGC Video**: While [2] addresses AIGC video quality and [4] addresses face video quality, there's no unified framework for evaluating human-centric AI-generated videos that incorporates temporal coherence, facial quality, body quality, aesthetic quality, AND affective quality in a single model.

3. **Fine-grained, multi-dimensional human-centric quality assessment with reasoning**: Paper [15] notes that existing IQA methods only provide global quality scores. Paper [34] introduces a "perception-reasoning paradigm" for medical IQA. There's a gap in applying this reasoning-based approach to AI-generated human content (faces, bodies, full scenes) with multi-dimensional assessment (technical quality, aesthetics, affect, alignment).

4. **Cross-domain generalization of quality assessment**: Most datasets and methods are domain-specific (faces, portraits, general images, CGI, etc.). There's no unified model that generalizes across domains of human visual content.

5. **Reinforcement Learning for Quality-Aligned Assessment**: Paper [2] uses progressive visual reinforcement learning for AIGC video quality. This approach could be extended to human-centric content with multi-dimensional rewards.

**The Most Compelling Gap:**

Looking at the constellation of papers, I see a very strong convergence around:
- Human-centric content (faces, bodies, 3D humans, portraits) — papers [4], [13], [15], [18], [23], [28], [31], [10]
- AIGC quality evaluation — papers [2], [7], [13], [15]
- LMM-based quality understanding with reasoning — papers [1], [2], [7], [11], [26], [34]
- Aesthetic and affective dimensions — papers [3], [9], [29]
- Multi-dimensional, fine-grained assessment — papers [15], [34]

**Proposed Research Direction:**

A unified LMM-based framework for **multi-dimensional quality assessment of AI-generated human-centric content** (images and videos) that simultaneously evaluates: (1) technical/perceptual quality, (2) aesthetic quality, (3) text-image/video alignment for human attributes, and (4) affective/emotional resonance — all with natural language reasoning explanations.

The key insight is that AI-generated human content (people, faces, bodies in images and videos) requires a fundamentally different quality evaluation framework than general content because:
1. Humans are hypersentitive to artifacts in human faces/bodies (uncanny valley)
2. Human-centric content has unique aesthetic dimensions (expressiveness, naturalness)
3. Human-centric AIGC often has specific prompts with human attributes that need alignment checking
4. The emotional/affective response to human imagery is particularly important for applications

**Approach:**
- Construct a comprehensive multi-dimensional annotation dataset covering technical quality, aesthetics, alignment, and affective dimensions for AI-generated human images and videos
- Fine-tune/train a large VLM using a combination of supervised learning and reinforcement learning with multi-dimensional reward signals
- The model should produce both scalar scores per dimension AND natural language reasoning explanations
- Benchmark against existing methods on all dimensions

**Key Contributions:**
1. First multi-dimensional quality assessment framework for AI-generated human-centric content
2. A large-scale dataset with multi-dimensional human annotations
3. An LMM trained with reinforcement learning for multi-dimensional human-centric quality reasoning
4. Comprehensive benchmark establishing new SOTA

Let me now write the formal proposal.
</thinking>

<proposal>

<problem>
Current quality assessment frameworks for AI-generated human-centric visual content (faces, bodies, portraits, and full-scene human videos) are fragmented across siloed dimensions — technical perceptual quality, aesthetic appeal, text-semantic alignment, and affective/emotional resonance — and lack a unified, reasoning-capable model. As generative AI floods platforms with human imagery and video, evaluating such content requires simultaneous, multi-dimensional understanding that mirrors how real human observers actually perceive and judge human-centric visuals, not just a single scalar score.
</problem>

<gap>
Existing literature reveals several compounding gaps: (1) IQA/VQA methods (e.g., MANIQA, Q-Align, LMM-VQA) target technical distortion or overall quality but are domain-agnostic and cannot handle the heightened human perceptual sensitivity to facial/body artifacts (uncanny valley effects, identity coherence, expression naturalness); (2) human-centric quality datasets (AGHI-QA, FVQ, DSL-FIQA, portrait IQA) each address isolated aspects — faces only, bodies only, static images only — with no dataset bridging technical quality, aesthetics, alignment, and affect for the same content; (3) aesthetic and affective understanding pipelines (AesBench, ArtEmis) are designed for artworks and photography but are not adapted to the unique properties of AIGC human content; (4) emerging reasoning-based evaluation frameworks (MedQ-Bench, VQ-Insight's reinforcement learning) demonstrate the value of language-grounded, multi-step quality reasoning, but this paradigm has not been applied to the human-centric AIGC domain; and (5) 3D/4D human avatar quality (DHQA-4D, LMME3DHF) is treated entirely separately from 2D image/video quality, preventing any cross-modality unification. No work simultaneously addresses all four dimensions — perceptual quality, aesthetics, semantic alignment, and affective resonance — for both generated human images and videos within a single framework.
</gap>

<key_insight>
Human observers evaluate AI-generated human content through an intertwined, hierarchically organized perceptual process: first assessing basic technical fidelity (absence of artifacts, sharpness, coherence), then aesthetic composition and naturalness, then semantic alignment with a prompt or intended depiction, and finally the affective/emotional resonance the content evokes. We hypothesize that a large multimodal model explicitly trained to perform this hierarchical multi-dimensional reasoning — via structured supervision on dimension-aware annotations and reinforced through multi-dimensional reward signals — will produce significantly better-aligned, more interpretable, and more generalizable quality assessments than any single-dimension model. The key insight is that these four dimensions are not independent: an LMM that understands their causal relationships (e.g., a subtle facial identity artifact degrades both technical quality and affective resonance) will produce superior assessments with richer, actionable natural language explanations.
</key_insight>

<approach>
**Phase 1 — Dataset Construction (HumanQA-4D Dataset):** Curate a
  FAS=0.7803  recall@50=✓ hit  mean_sim=0.5606  [baseline:claude-opus-4-6]
======================================================================

Synthesizing CoT via claude-sonnet-4-6...

── COT SYNTHESIS (claude-sonnet-4-6)  [29.3s] ────────────────────────
<thinking>
Step 1: What themes and methods appear across these references?

Looking at the references, I can identify several key themes:
- Large Multimodal Models (LMMs/MLLMs) for visual understanding and quality assessment
- Benchmarks for evaluating MLLMs on various visual tasks (Q-Bench, AesBench, MMBench, etc.)
- Image/Video Quality Assessment (IQA/VQA) datasets and methods
- Domain-specific quality assessment (face images, portraits, AI-generated content, medical images, artistic images, computer graphics)
- Human-centric understanding and evaluation
- Low-level vision perception and understanding

Specific relevant references:
- Q-Bench (ref 26): benchmark for low-level vision in MLLMs
- Q-Bench-Video (ref 11): extending Q-Bench to video
- Q-Align (ref 1): teaching LMMs for visual scoring
- AesBench (ref 3): aesthetics perception benchmark for MLLMs
- Portrait IQA dataset (ref 31): image quality for portraits
- Face-Human-Bench (ref 28): face and human understanding benchmark
- Human-MME (ref 18): human-centric evaluation
- AGHI-QA (ref 15): AI-generated human images quality
- DSL-FIQA (ref 23): face image quality assessment
- FVQ (ref 4): face video quality assessment
- LMME3DHF (ref 13): 3D human face generation evaluation
- Various MLLMs: LLaVA, InternVL, Qwen2.5-VL, DeepSeek-VL, etc.

Step 2: What open problems or limitations do they collectively suggest?

1. Existing benchmarks for MLLMs focus on generic images or general low-level vision - portrait/face-specific quality perception hasn't been systematically benchmarked
2. Portrait images have unique characteristics (skin texture, facial structure, aesthetic composition, identity preservation) that aren't captured in generic benchmarks
3. The diversity of portrait image types (natural photos, AI-generated, artistic, CG) hasn't been covered comprehensively
4. Current evaluation doesn't cover the full range of quality dimensions relevant to portraits (technical distortions, AIGC artifacts, aesthetic quality)
5. MLLMs' capabilities for portrait-specific perception are unknown

Step 3: What would be a natural next research direction?

Given the gap between general-purpose MLLM benchmarks and the specific domain of portrait image quality assessment, a natural direction would be to create a comprehensive benchmark specifically for portrait image quality perception. This would:
- Cover diverse portrait image sources (natural, synthetic, AI-generated, artistic)
- Include various quality dimensions relevant to portraits
- Feature multiple question formats to probe different aspects of perception
- Evaluate existing MLLMs to reveal their capabilities and limitations in this domain

Step 4: What approach might address this gap?

The approach would involve:
1. Constructing a large-scale benchmark dataset with diverse portrait images from multiple sources
2. Defining quality dimensions specific to portrait perception (technical quality, AIGC artifacts, aesthetics)
3. Creating diverse question formats (single-choice, multiple-choice, yes/no, open-ended) at different granularities
4. Evaluating a comprehensive set of open-source and closed-source MLLMs
5. Comparing MLLM performance to human judgments to quantify the gap
</thinking>

<proposal>
<problem>Multimodal large language models (MLLMs) have shown promise in general-purpose low-level vision tasks, yet their ability to perceive and assess portrait images—a visually and semantically distinct domain—remains systematically unexplored. Portrait images exhibit unique structural, aesthetic, and perceptual properties (e.g., facial geometry, skin texture fidelity, identity coherence) that demand specialized evaluation frameworks beyond what existing generic benchmarks provide.</problem>

<gap>Current benchmarks for evaluating MLLMs on low-level vision (e.g., Q-Bench, AesBench) focus primarily on generic photography and do not address the specific quality dimensions relevant to portrait imagery, such as AIGC-specific facial distortions, artistic stylization artifacts, or fine-grained local region quality. Existing portrait or face quality assessment datasets (e.g., GFIQA datasets, portrait IQA datasets) are designed for traditional regression models and do not probe the open-ended reasoning and multi-dimensional perception capabilities of modern MLLMs. Meanwhile, human-centric benchmarks largely address semantic understanding rather than perceptual quality. This leaves a significant gap in our understanding of how well MLLMs can function as assessors of portrait image quality aligned with human perception.</gap>

<key_insight>Portrait images constitute a high-stakes, high-frequency category of visual content with distinct quality criteria that span technical fidelity, generative artifact detection, and aesthetic composition—dimensions that interact in ways not captured by generic quality benchmarks. We hypothesize that systematically probing MLLMs across these dimensions, using diverse portrait image sources and varied question formats, will reveal structured capability gaps and point toward targeted improvements in portrait-aware visual intelligence.</key_insight>

<approach>We propose constructing a comprehensive benchmark dataset specifically targeting portrait image quality perception, drawing images from heterogeneous sources including natural photographs, synthetically distorted images, AI-generated portraits, artistic works, and computer graphics renderings. Quality dimensions would be carefully defined to span technical distortions (blur, noise, compression artifacts), AIGC-specific issues (anatomical inconsistencies, texture artifacts), and aesthetic attributes (composition, lighting, style coherence). The benchmark would include multiple question formats—from closed-form single- and multiple-choice to open-ended descriptive questions—at both global image-level and local region-level granularities to probe perception at different scales. A broad suite of state-of-the-art open-source and closed-source MLLMs would be evaluated, with human performance collected as a reference ceiling to quantify the human-machine gap.</approach>

<expected_contributions>
- A large-scale, carefully curated benchmark for portrait image quality perception that covers diverse image sources, comprehensive quality dimensions, and varied question formats, filling a critical gap in MLLM evaluation.
- A systematic empirical analysis revealing the current capabilities and limitations of leading MLLMs in portrait-specific quality perception, including fine-grained breakdowns across distortion types, image sources, and question formats.
- A human performance baseline that quantifies the gap between current MLLMs and human-level portrait quality understanding, providing a concrete target for future model development.
- Identification of key failure modes and underexplored challenges in portrait perception that can guide the development of more capable domain-aware or fine-tuned MLLMs.
</expected_contributions>
</proposal>

  leakage score: 0.7654
  FAS=0.8224  recall@50=✓ hit  mean_sim=0.6449  [cot:claude-sonnet-4-6]
======================================================================

Checkpoint: runs/grpo/final
Device: cuda
Loading base: /newcpfs/lxh/agentic-training/proposal_rl/runs/model_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████████████████████████████████████| 339/339 [00:02<00:00, 146.51it/s]
Loading LoRA adapter: runs/grpo/final
Loaded as LoRA checkpoint (merged)

Generating...
The following generation flags are not valid and may be ignored: ['top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

── MODEL OUTPUT (final) ──────────────────────────────────
<thinking>
[Step 1: What themes and methods appear across these references?]
The references cover several interconnected themes:
- Large multimodal models (LMMs) for various tasks: image/video quality assessment, aesthetic/perception understanding, video generation, face/human understanding, medical image QA, etc.
- Benchmarking LMMs: Q-Bench, Q-Bench-Video, MMBench, LMM4LMM, Qwen2.5-VL, Qwen3-VL, InternVL3, InternVL3.5, DeepSeek-VL, DeepSeek-VL2, LLaVA-OneVision, Gemini 1.5, Pixtral 12B
- Specific domains: face video quality (FVQA), human-centric understanding, medical image QA, artistic image aesthetics, AI-generated content quality
- Methods: reinforcement learning for teaching LMMs, progressive visual reinforcement learning, dual-set degradation learning, cascade reinforcement learning
- Challenges: subjective nature of quality assessment, generalization across diverse content, temporal awareness in video

[Step 2: What open problems or limitations do they collectively suggest?]
1. LMMs struggle with subjective, perceptual tasks where human judgment is required
2. Current benchmarks don't fully capture the complexity of real-world visual content
3. Temporal consistency and long-term reasoning in video understanding is underexplored
4. Fine-grained, human-aligned evaluations are lacking for LMMs
5. There's a gap between general multimodal understanding and specialized quality/perception tasks
6. Existing methods often rely on large-scale datasets or expensive training pipelines

[Step 3: What would be a natural next research direction?]
A natural direction would be to develop a benchmark and evaluation methodology that:
- Addresses the subjective, perceptual nature of quality assessment tasks
- Captures the complexity of real-world visual content (diverse domains, temporal dynamics)
- Uses reinforcement learning to teach LMMs for fine-grained, human-aligned evaluations
- Provides a standardized way to measure LMMs across multiple domains

[Step 4: What approach might address this gap?]
The approach could involve:
1. Creating a large-scale, diverse dataset covering multiple visual domains (faces, humans, medical images, art, etc.)
2. Using reinforcement learning to train LMMs to perform fine-grained, human-aligned evaluations
3. Developing a benchmark that evaluates LMMs across multiple tasks (quality assessment, aesthetic perception, human understanding)
4. Providing a standardized evaluation protocol for researchers to compare models
</thinking>

<proposal>
<problem>Large multimodal models (LMMs) have shown impressive general capabilities across diverse visual tasks, yet their performance on subjective, perceptual quality assessment and fine-grained aesthetic understanding remains poorly characterized. These tasks require models to reason about complex visual content in a manner that aligns closely with human judgment, making them uniquely challenging compared to standard classification or retrieval benchmarks.</problem>

<gap>Existing multimodal benchmarks predominantly focus on general visual understanding and reasoning, leaving open the question of whether LMMs can reliably perform fine-grained, human-aligned evaluations of visual content. While some recent work has explored teaching LMMs for specific quality assessment tasks, these efforts have been limited to narrow domains and have not provided a standardized, comprehensive evaluation framework. Furthermore, the subjective nature of quality assessment—where human judgment is inherently variable and difficult to quantify—remains a fundamental challenge that current benchmarks do not adequately address.</gap>

<key_insight>We hypothesize that reinforcement learning can serve as a powerful training signal for teaching LMMs to perform fine-grained, human-aligned evaluations of visual content, enabling models to internalize nuanced perceptual judgments without requiring large-scale manually labeled datasets. By designing a benchmark that spans multiple visual domains and tasks, we can systematically characterize the strengths and weaknesses of LMMs across diverse quality and perception challenges.</key_insight>

<approach>The proposed work would begin by constructing a large-scale, diverse dataset that captures the complexity of real-world visual content across multiple domains, including faces, humans, medical images, art, and AI-generated content. Each sample would be paired with human annotations reflecting fine-grained perceptual judgments, enabling the training of LMMs to produce human-aligned evaluations. A reinforcement learning framework would be employed to iteratively refine model predictions against ground-truth human annotations, allowing the model to learn to reason about visual quality and aesthetic properties in a manner consistent with human judgment. A standardized benchmark would then be defined, encompassing multiple evaluation tasks spanning quality assessment, aesthetic perception, and human-centric understanding, providing a unified framework for comparing LMMs across these domains. The entire pipeline—from data collection and annotation to model training and benchmarking—would be designed to be reproducible and accessible to the broader research community.</approach>

<expected_contributions>
- A large-scale, diverse dataset covering multiple visual domains and tasks, enabling systematic evaluation of LMMs on subjective, perceptual quality assessment and fine-grained aesthetic understanding.
- A reinforcement learning-based training methodology that teaches LMMs to produce human-aligned evaluations without requiring large-scale manually labeled datasets, addressing the scarcity of high-quality annotations for these tasks.
- A standardized benchmark that evaluates LMMs across multiple quality and perception tasks, providing a unified evaluation framework for the community to compare models.
- Insights into the strengths and limitations of LMMs when performing fine-grained, human-aligned evaluations, guiding future research directions in multimodal understanding.
</expected_contributions>
</proposal>
  FAS=0.8519  recall@50=✓ hit  mean_sim=0.7038  [final]
======================================================================
```

<br/>

## Future Work

- **Full-finetune**

- **Weakness / truncate & compress context ? 400?**

  - **question / 5 papers / sometimes 1 paper**

- **re-implement the FAS score**

- citation list -&gt; related work

- need to beat the strongest API I can get

- train data 

  - change to orals + many citations?

  - now data is same for SFT & GRPO. Consider difference?

  - training needn't be anti-leakage, only eval is enough.

## Next Steps

- Write a bunch of bash scripts:

  - baseline: full-finetune + top-k + embed-based PRS

  - topk -&gt; full reference (truncated to 40 * 400char)

  - topk -&gt; full reference + llm-based summarizing

  - topk -&gt; related work (synthesized)

  - topk -&gt; topk + research question.

  - full-finetune -&gt; lora.

  - embed-based PRS -&gt; embed-based FAS.

  - embed-based PRS -&gt; llm-based PRS.

- For each setting search locally for the best hparam.

- For each setting, when launching, automatically add a tab in the dashboard (dashboard containing multiple pages, each corresponding to one run, named by setting + launching time) and automatically refresh that page while training. The launching script should be purely rule-based.

## Large Picture

- optimizer proposal + critic, try to hack proposed optimizers within small scales

<br/>

<br/>