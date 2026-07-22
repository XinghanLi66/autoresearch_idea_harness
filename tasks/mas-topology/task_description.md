# Multi-Agent Collaboration Topology Design

## Objective
Design a novel multi-agent collaboration topology that maximizes the quality of LLM-generated code. You implement a `generate_topology(node_num)` function that returns directed edges forming a DAG. The agents are organized according to your topology: each agent receives a predecessor's code solution, reviews it, and produces an improved version. When a node has multiple predecessors, solutions are aggregated.

## Background
MacNet (Scaling Large-Language-Model-based Multi-Agent Collaboration) organizes LLM agents as nodes in a directed acyclic graph. The topology (graph structure) determines how agents collaborate and significantly impacts code generation quality. Simple chain topologies offer deep iterative refinement but no diversity; star topologies offer breadth but no depth; layered (MLP-like) topologies balance both.

## Editable Interface
Modify `custom_topology.py` which contains:

```python
def generate_topology(node_num: int) -> list[tuple[int, int]]:
    """Return directed edges (source, target) forming a DAG over nodes 0..node_num-1."""
```

### Constraints
- Must return a valid DAG (no cycles)
- All nodes 0 to node_num-1 must be reachable from the input sentinel
- Edges should go from lower-numbered nodes to higher-numbered nodes (or at least respect topological order)
- The system automatically adds input sentinel (-1) connecting to source nodes and output sentinel (-2) connecting from sink nodes

## Evaluation
Your topology is evaluated across **3 settings** (2 benchmarks × different MacNet backbone LLMs), using 4 agent nodes each:

| # | Benchmark | MacNet backbone | test_cmd label |
|---|-----------|----------------|----------------|
| 1 | HumanEval (33 problems) | deepseek-chat | `humaneval-4-deepseek` |
| 2 | HumanEval (33 problems) | qwen2.5-72b-instruct | `humaneval-4-qwen` |
| 3 | SRDD (20 prompts) | deepseek-chat | `srdd-4-deepseek` |

A good topology should generalize: improve over the chain baseline across all three settings, not just one.

### HumanEval (`humaneval-4-*`)
A 33-problem subset of HumanEval coding problems with unit tests. For each problem, the multi-agent system collaborates according to your topology to generate a Python function, which is then tested against the problem's unit tests.

- **Metric: `pass_at_1_deepseek` / `pass_at_1_qwen`** = fraction of problems where the generated code passes all unit tests on the first attempt.

### SRDD (`srdd-4-deepseek`)
20 curated software development prompts from the SRDD (Software Requirement Document Dataset) categories. For each prompt, agents collaborate to generate a complete software project. The generated project is tested for executability: whether its entry point (main.py) runs without crashing.

- **Metric: `srdd_exec_rate`** = fraction of generated projects that execute successfully (exit code 0 or still running at timeout, with no Traceback in stderr).

The default chain topology (0->1->2->...->N-1) is the baseline. Better topologies enable richer agent collaboration, producing more correct code across all three settings.

**Note on reproducibility**: The topology is deterministic (same node_num -> same edges). Variability across seeds comes from the LLM API responses, not the topology. Multiple seeds test robustness of a topology under different LLM sampling outcomes.

**Network requirement**: This task requires internet access at runtime to call LLM APIs. Set both `DEEPSEEK_API_KEY` (for humaneval-4-deepseek and srdd-4-deepseek) and `QWEN_API_KEY` (for humaneval-4-qwen via DashScope) environment variables before running. Not compatible with offline/air-gapped compute nodes.
