## 概述

用现有 32B checkpoint（v3_sft_v1sem_cot_strict929_16k，在 928 条 CoT 样本上做 LoRA SFT）在 **held-out 论文** 上跑的创造力 demo：每篇论文给模型一份 reading list（参考文献）+ 一个 research question，模型按训练时的 with_research_question 格式生成一条 XML research proposal。

- 5 篇均为 **2505 held-out** 论文（不在 929 训练集内），不存在目标论文泄漏；参考文献做了 leakage 过滤。

- 本地单卡 **L20Z (80GB)** 推理：模型加载约 118s，**5/5 schema 完整**（10 个 XML tag 齐全），每篇生成约 25–37s。

- 脚本：autoresearch_idea_harness/scripts/demo_held_out_proposals.py；产物目录：runs/demo_held_out/。

### 总览

| Held-out 论文 | 领域 | 模型提出的方向 |
| --- | --- | --- |
| 2505.00009 · Efficient Knowledge Transfer in Multi-Task Learning (task-adaptive low-rank) | cs.CL | Theoretical and Practical Framework for Multi-Task Compositional Parameter-Efficient Fine-Tuning with Causal Reasoning Fidelity |
| 2505.00044 · Learning to Borrow Features for small-object detection | cs.CV | Meta-Learning Framework for Multi-Scale MRI Reconstruction with Non-Local Context Aggregation |
| 2505.00049 · Humanizing LLMs: survey of psychological measurements | cs.CY | Evaluating Temporal Stability and Behavioral Consistency of LLM Personality Traits Across Tasks |
| 2505.00133 · Efficient & robust 3D blind harmonization | eess.IV | Developing Native 3D Generative Models for Medical Imaging with Trustworthy Outputs |
| 2505.00169 · GEOM-Drugs Revisited (3D molecule benchmarks) | cs.LG | Unified Framework for Generating Physically Valid, High-Binding Affinity, Synthesizable Molecules Using Hybrid 1D/3D Diffusion Models |

## 1. 2505.00009 — Efficient Knowledge Transfer in Multi-Task Learning (task-adaptive low-rank)

**领域：** cs.CL · **prompt 5127 tok · 生成 892 tok · schema 完整：是**

**提出方向：** Theoretical and Practical Framework for Multi-Task Compositional Parameter-Efficient Fine-Tuning with Causal Reasoning Fidelity

**亮点：** 把 prompt tuning / LoRA / prefix tuning 等 PEFT 方法与 causal reasoning、Pareto-manifold 统一到一个框架，并给出 shared hypernetworks + modular prompts 的具体组件与 GLUE/SuperGLUE 评测。

```xml
<proposal>
    <title>Theoretical and Practical Framework for Multi-Task Compositional Parameter-Efficient Fine-Tuning with Causal Reasoning Fidelity</title>
    <problem>
        Current parameter-efficient fine-tuning (PEFT) methods, such as prompt tuning, adapters, LoRA, and prefix tuning, show promise in reducing the computational cost and parameter burden of adapting large language models to multiple tasks. However, they lack a unified theoretical framework to explain their success or failure in preserving causal structures during multi-step reasoning tasks. Furthermore, these methods often require task-specific architectural choices and rely on scale-dependent emergent behaviors, making them less flexible and generalizable.
    </problem>
    <gap>
        The existing literature has explored PEFT methods individually but lacks a cohesive understanding of how they can collectively address multi-task compositionality, causal reasoning fidelity, and Pareto-efficient task trade-offs. There is a need to develop a theoretical foundation that guides the design of PEFT methods capable of optimizing these aspects without task-specific modifications.
    </gap>
    <core_idea>
        Develop a unified framework that integrates multi-task prompt tuning, low-rank adaptation, and causal reasoning enhancement to create a versatile PEFT method. This framework will leverage shared hypernetworks and modular prompts to ensure compositional generalization and causal fidelity across diverse tasks.
    </core_idea>
    <implementation_plan>
        1. **Develop a Unified PEFT Architecture**: Combine elements of prompt tuning, LoRA, and causal reasoning enhancement into a single framework. Use shared hypernetworks to generate task-specific adapters and modular prompts.
        2. **Integrate Causal Reasoning Mechanisms**: Incorporate causal relationship enhancement (CRE) mechanisms to ensure the validity of causal relationships during reasoning tasks.
        3. **Optimize for Pareto Efficiency**: Implement Pareto manifold learning with low-rank structure to manage trade-offs across multiple tasks.
    </implementation_plan>
    <algorithm_or_system>
        <name>UnifiedMultiTaskCausalPEFT (UMTC-PEFT)</name>
        <description>
            UMTC-PEFT integrates shared hypernetworks for generating task-specific adapters, modular prompts for task transfer, and causal reasoning mechanisms to ensure fidelity. The system also includes a Pareto manifold learning component to handle task trade-offs.
        </description>
    </algorithm_or_system>
    <training_or_data_recipe>
        1. **Pre-train Hypernetworks**: Train shared hypernetworks on a diverse set of NLP tasks to generate task-specific adapters.
        2. **Prompt Tuning Initialization**: Initialize modular prompts using a set of source tasks and refine them for target tasks.
        3. **Causal Reasoning Training**: Introduce CRE mechanisms during training to enforce causal consistency.
        4. **Pareto Front Optimization**: Apply low-rank structure to manage Pareto trade-offs.
    </training_or_data_recipe>
    <evaluation_plan>
        1. **Benchmark Performance**: Evaluate UMTC-PEFT on a suite of NLP benchmarks including GLUE, SuperGLUE, and custom long-range reasoning datasets.
        2. **Comparative Analysis**: Compare UMTC-PEFT with full fine-tuning, prompt tuning, adapter fusion, LoRA, and prefix tuning.
        3. **Ablation Studies**: Conduct ablation studies to isolate the contributions of shared hypernetworks, modular prompts, and causal reasoning mechanisms.
    </evaluation_plan>
    <expected_results>
        UMTC-PEFT is expected to outperform or match full fine-tuning on a variety of NLP tasks while maintaining a much smaller parameter footprint. The system should demonstrate robust causal reasoning and compositional generalization across tasks, and effectively manage Pareto-efficient task trade-offs.
    </expected_results>
    <risks_and_limitations>
        1. **Complexity of Integration**: Integrating multiple components (hypernetworks, modular prompts, causal reasoning) may introduce complexity and potential failures in training stability.
        2. **Scalability Issues**: Ensuring the system scales efficiently across a large number of tasks and diverse datasets could pose challenges.
        3. **Generalizability**: The performance might degrade if the system is applied to tasks or domains not covered in the pre-training phase.
    </risks_and_limitations>
</proposal>
```

## 2. 2505.00044 — Learning to Borrow Features for small-object detection

**领域：** cs.CV · **prompt 6252 tok · 生成 667 tok · schema 完整：是**

**提出方向：** Meta-Learning Framework for Multi-Scale MRI Reconstruction with Non-Local Context Aggregation

**亮点：** 跨领域迁移的典型例子：参考文献是 object detection，模型顺着该论文的 research question 把 multi-scale / non-local context 思想迁移到 MRI reconstruction，体现了对 RQ 的忠实跟随与领域桥接。

```xml
<proposal>
<title>Meta-Learning Framework for Multi-Scale MRI Reconstruction with Non-Local Context Aggregation</title>
<problem>
Current MRI reconstruction techniques often struggle with accurately reconstructing fine structural details, especially in small anatomical structures, due to varying spatial scales and lack of explicit coil sensitivity estimation.
</problem>
<gap>
There is a need for a unified reconstruction framework that can leverage multi-scale feature hierarchies and non-local context aggregation to handle scale variance in MRI anatomy, akin to advancements in object detection architectures.
</gap>
<core_idea>
Develop a meta-learning framework that incorporates multi-scale feature hierarchies and non-local context aggregation principles to enhance the reconstruction of MRI images, particularly focusing on fine details in small anatomical structures.
</core_idea>
<implementation_plan>
1. Design a multi-scale feature hierarchy using spatial pyramid pooling (SPP) to capture context at different resolutions.
2. Integrate non-local context aggregation to enable the model to capture long-range dependencies and improve feature extraction.
3. Apply domain-conditioned priors to guide the reconstruction process, ensuring consistency with the underlying MRI physics.
4. Utilize a meta-learning approach to optimize the reconstruction parameters for different scales and anatomical structures.
</implementation_plan>
<algorithm_or_system>
The proposed algorithm, named Multi-Scale Meta-Learning MRI Reconstruction (MSML-MRI), will consist of:
- A backbone network with SPP for multi-scale feature extraction.
- Non-local blocks for context aggregation.
- Domain-conditioned priors derived from MRI physics.
- Meta-learning optimization to adaptively tune the model parameters.
</algorithm_or_system>
<training_or_data_recipe>
1. Use a diverse dataset of undersampled MRI images across various anatomical structures and scales.
2. Split the dataset into training, validation, and test sets.
3. Preprocess the data to generate pairs of undersampled k-space data and fully sampled reference images.
4. Train the MSML-MRI model using a bilevel optimization framework to learn the common feature encoder and task-specific parameters.
5. Validate the model on the validation set to fine-tune hyperparameters.
</training_or_data_recipe>
<evaluation_plan>
1. Evaluate the model on the test set using metrics such as peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), and mean absolute error (MAE).
2. Compare the performance against state-of-the-art single-task learning methods.
3. Assess the reconstruction quality visually and quantitatively for small anatomical structures.
</evaluation_plan>
<expected_results>
1. Improved reconstruction accuracy, especially for small anatomical structures, with higher PSNR, SSIM, and lower MAE values.
2. Faster reconstruction times compared to traditional methods, thanks to the meta-learning framework.
3. Enhanced generalization ability across different undersampling patterns and anatomical structures.
</expected_results>
<risks_and_limitations>
1. Complexity in training the multi-scale and non-local components might lead to longer training times and increased computational requirements.
2. Ensuring the robustness of the model across different anatomical structures and scales could be challenging.
3. Potential overfitting to the training data if the meta-learning framework is not properly regularized.
</risks_and_limitations>
</proposal>
```

## 3. 2505.00049 — Humanizing LLMs: survey of psychological measurements

**领域：** cs.CY · **prompt 10162 tok · 生成 739 tok · schema 完整：是**

**提出方向：** Evaluating Temporal Stability and Behavioral Consistency of LLM Personality Traits Across Tasks

**亮点：** 从 humanizing LLMs 综述出发，提出评测 LLM personality traits 在不同 task 间的 temporal stability 与 behavioral consistency，问题定义清晰、可落地。

```xml
<proposal>
<title>Evaluating Temporal Stability and Behavioral Consistency of LLM Personality Traits Across Tasks</title>
<problem>The current research lacks a unified framework that can simultaneously validate the temporal stability, measurement-invariance, and genuine predictive power of personality representations in LLMs.</problem>
<gap>Previous studies have focused on either self-reporting of personality traits or specific behavior in isolated tasks, without a cohesive approach to ensure consistency and stability over time and across different elicitation methods.</gap>
<core_idea>Create a comprehensive framework to assess the temporal stability and behavioral consistency of personality traits in LLMs, ensuring that the traits are embedded in model weights rather than prompt-induced.</core_idea>
<implementation_plan>
<step>Develop a set of tasks that cover strategic, social, and affective behaviors, ensuring these tasks are varied and cover different domains (game-theory tasks, social interactions, and affective reasoning).</step>
<step>Design a battery of personality assessment tasks that vary in elicitation methods (self-report questionnaires, role-playing scenarios, and task-specific behavior).</step>
<step>Conduct repeated evaluations of the same LLMs over time to check for temporal stability in personality traits.</step>
<step>Implement a control experiment to differentiate between prompt-induced performance and intrinsic model behaviors.</step>
</implementation_plan>
<algorithm_or_system>
<personality_assessment_battery>
<task type="strategic">GTBench</task>
<task type="social">SUVA Framework</task>
<task type="affective">Affective Cognition Evaluation Framework</task>
</personality_assessment_battery>
<temporal_stability_check>
<evaluation_period>3 months</evaluation_period>
<model_versions>Initial version and post-training versions</model_versions>
</temporal_stability_check>
<control_experiment>
<control_type>Randomized prompts</control_type>
<comparison_method>Comparative behavior analysis</comparison_method>
</control_experiment>
</algorithm_or_system>
<training_or_data_recipe>
<dataset>
<type>Personality Assessment Dataset</type>
<description>Mixed-method dataset including self-reports, role-playing scenarios, and behavior-specific tasks.</description>
</dataset>
<method>Supervised Fine-Tuning and Direct Preference Optimization</method>
</training_or_data_recipe>
<evaluation_plan>
<metric type="temporal_stability">Pearson Correlation Coefficient</metric>
<metric type="measurement_invariance">Intra-class Correlation Coefficient (ICC)</metric>
<metric type="behavioral_prediction">Predictive Accuracy Scores</metric>
<validation_method>Cross-validation and external validation using human raters</validation_method>
</evaluation_plan>
<expected_results>
<result type="temporal_stability">High correlation coefficients indicating stable personality traits over time.</result>
<result type="measurement_invariance">High ICC scores confirming consistency across different elicitation methods.</result>
<result type="behavioral_prediction">Significant predictive accuracy in downstream tasks, validating genuine trait-behavior alignment.</result>
</expected_results>
<risks_and_limitations>
<risk type="prompt_sensitivity">Results may be influenced by the specific prompts used, necessitating careful prompt design.</risk>
<limitation type="model_capacity">Limited capacity models might not exhibit the same level of stability and consistency, requiring high-capacity models for accurate assessment.</limitation>
<limitation type="generalizability">Findings may not generalize across all LLMs, necessitating further validation across diverse model architectures.</limitation>
</risks_and_limitations>
</proposal>
```

## 4. 2505.00133 — Efficient & robust 3D blind harmonization

**领域：** eess.IV · **prompt 5914 tok · 生成 759 tok · schema 完整：是**

**提出方向：** Developing Native 3D Generative Models for Medical Imaging with Trustworthy Outputs

**亮点：** 从 3D blind harmonization 的参考出发，提出面向医学影像的 native 3D generative model，并强调 trustworthy / 可信输出。

```xml
<proposal>
<title>Developing Native 3D Generative Models for Medical Imaging with Trustworthy Outputs</title>
<problem>
The current 3D generative models for medical imaging often rely on extending 2D diffusion models through patching, slice-blending, or perpendicular fusion, which can lead to issues such as boundary artifacts, reduced anatomical fidelity, and hallucinations. These limitations hinder their applicability in downstream diagnostic tasks such as segmentation, harmonization, and reconstruction.
</problem>
<gap>
There is a need for a generative model that operates natively in 3D, maintains high anatomical fidelity, and ensures verifiably trustworthy outputs, thus avoiding hallucinations and improving the reliability of downstream tasks.
</gap>
<core_idea>
We propose a novel approach called **Native3DGen**, which integrates a 3D diffusion model with a conditional normalization flow (NF) module to ensure anatomical fidelity and prevent hallucinations. Native3DGen will be trained using a multi-scale feature alignment strategy, ensuring consistency across different scales of the 3D volume. Additionally, the model will incorporate a trustworthiness verification mechanism to identify and correct any potential hallucinations.
</core_idea>
<implementation_plan>
1. Design the architecture of Native3DGen, integrating a 3D diffusion model with a NF module.
2. Implement a multi-scale feature alignment strategy for training.
3. Develop a trustworthiness verification mechanism using post-processing techniques.
4. Train the model on a dataset of 3D medical images, such as brain MRI volumes.
5. Evaluate the model's performance on segmentation, harmonization, and reconstruction tasks.
</implementation_plan>
<algorithm_or_system>
Native3DGen will consist of two main components:
- **3D Diffusion Model**: Generates high-fidelity 3D medical images.
- **Conditional Normalizing Flow Module**: Ensures anatomical fidelity and prevents hallucinations.

The trustworthiness verification mechanism will use a combination of structural similarity measures and anomaly detection techniques to identify and correct any hallucinations.
</algorithm_or_system>
<training_or_data_recipe>
1. Collect a dataset of 3D medical images (e.g., brain MRI volumes).
2. Preprocess the data to ensure consistency and quality.
3. Train the 3D diffusion model on the full 3D volumes using a multi-scale feature alignment strategy.
4. Fine-tune the conditional NF module using the same dataset to enforce anatomical constraints.
5. Validate the model using a held-out validation set.
</training_or_data_recipe>
<evaluation_plan>
1. Assess the generated 3D images using metrics such as Structural Similarity Index Measure (SSIM) and Peak Signal-to-Noise Ratio (PSNR).
2. Evaluate the model's performance on downstream tasks such as segmentation, harmonization, and reconstruction using established benchmarks.
3. Conduct a comparison with existing 2D-based and 3D patch-based models.
4. Analyze the hallucinations and anatomical fidelity of the generated images.
</evaluation_plan>
<expected_results>
We expect Native3DGen to generate high-fidelity 3D medical images without boundary artifacts or hallucinations. The model should perform comparably or better than existing methods on segmentation, harmonization, and reconstruction tasks, while also maintaining anatomical fidelity.
</expected_results>
<risks_and_limitations>
1. Training a native 3D model requires significantly more computational resources and may be challenging for very high-resolution datasets.
2. The trustworthiness verification mechanism might not catch all hallucinations, especially subtle ones.
3. The multi-scale feature alignment strategy might introduce additional complexity and require careful tuning.
</risks_and_limitations>
</proposal>
```

## 5. 2505.00169 — GEOM-Drugs Revisited (3D molecule benchmarks)

**领域：** cs.LG · **prompt 4329 tok · 生成 615 tok · schema 完整：是**

**提出方向：** Unified Framework for Generating Physically Valid, High-Binding Affinity, Synthesizable Molecules Using Hybrid 1D/3D Diffusion Models

**亮点：** 混合 1D/3D diffusion 同时优化 physical validity、binding affinity、synthesizability，点名 PubChem/GEOM/PoseBusters/RDKit 及 Megalodon/PILOT 等真实数据集、工具与 baseline。

```xml
<proposal>
    <title>Unified Framework for Generating Physically Valid, High-Binding Affinity, Synthesizable Molecules Using Hybrid 1D/3D Diffusion Models</title>
    <problem>Current 3D molecular generative models struggle to balance physical validity, binding affinity, and synthesizability simultaneously. Existing models often prioritize one aspect over others, leading to generated molecules that are either geometrically incorrect, unstable, or not synthesizable.</problem>
    <gap>The gap lies in the need for a unified framework that integrates physical validity, binding affinity, and synthesizability into a single optimization objective, rather than treating them as separate concerns.</gap>
    <core_idea>Create a hybrid 1D/3D diffusion model that leverages 1D language models for ensuring chemical validity and a 3D diffusion component for accurate spatial configurations. Integrate multi-objective optimization techniques to ensure that the generated molecules meet all specified criteria.</core_idea>
    <implementation_plan>
        1. Develop a 1D language model trained on large-scale molecular datasets to ensure chemical validity.
        2. Create a 3D diffusion model using an equivariant architecture to handle spatial configurations.
        3. Combine the 1D and 3D models to generate molecules with both chemical and spatial correctness.
        4. Implement multi-objective optimization to balance physical validity, binding affinity, and synthesizability.
    </implementation_plan>
    <algorithm_or_system>Hybrid 1D/3D Diffusion Model (HDMD) with Multi-Objective Optimization (MOO)</algorithm_or_system>
    <training_or_data_recipe>
        1. Train the 1D language model on the PubChem database.
        2. Pre-train the 3D diffusion model on the GEOM dataset.
        3. Fine-tune the combined HDMD on a curated dataset of known drugs and their protein targets.
        4. Use PoseBusters to validate generated molecules against physical plausibility and binding affinity.
    </training_or_data_recipe>
    <evaluation_plan>
        1. Evaluate chemical validity using RDKit.
        2. Assess binding affinity using docking scores against known protein targets.
        3. Measure synthesizability using synthetic accessibility scores.
        4. Compare results with state-of-the-art models like Megalodon and PILOT.
    </evaluation_plan>
    <expected_results>Generate molecules that are chemically valid, geometrically correct, and have high binding affinity and synthesizability. Expect an improvement in the overall quality and diversity of generated molecules compared to current models.</expected_results>
    <risks_and_limitations>
        1. Difficulty in balancing multiple objectives without compromising any single aspect.
        2. Potential overfitting on the training set, resulting in poor generalization to unseen molecules.
        3. Computational cost of training and fine-tuning the hybrid model.
    </risks_and_limitations>
</proposal>
```