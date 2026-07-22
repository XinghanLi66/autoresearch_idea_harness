# LLM Agent Tool-Use Reasoning Strategy

## Objective
Design a better search/reasoning strategy for an LLM-based tool-use agent. Your code goes in the `search()` method of `custom_search.py`.

## Background
StableToolBench evaluates LLM agents on multi-step tool use tasks. Given a user query and a set of tool APIs, the agent must decide which tools to call, with what arguments, and in what order to arrive at a final answer.

The search strategy controls how the agent explores the action space:
- **Greedy chain (CoT)**: Call LLM, execute tool, repeat. No backtracking. Simple but gets stuck on errors.
- **DFS with ranking**: Generate multiple children, use LLM to rank them, expand best first. Backtracks on failure. More robust but costs extra LLM queries for ranking.
- **DFSDT**: Generate one child, immediately recurse depth-first. Backtrack a fixed number of steps on failure. Balance between exploration and cost.

## What you can modify
The `search(self, root_node)` method in `custom_search.py` (the editable region). You have access to:

- `self._step(node)` -- one LLM call + tool execution, returns new leaf nodes
- `self._add_diversity_prompt(node)` -- encourages different actions when re-expanding
- `self._rank_nodes(candidates)` -- LLM pairwise ranking (costs extra queries)
- Tree state: `self.query_count`, `self.max_query_count`, `self.terminal_node`, etc.
- Node properties: `node.is_terminal`, `node.pruned`, `node.observation_code`, `node.get_depth()`

## Evaluation metrics
- **pass_rate**: Fraction of queries where the agent produces a valid final answer (higher is better)
- **avg_queries**: Average number of LLM queries per task (lower is better for efficiency)
- **give_up_rate**: Fraction of queries where the agent gives up (lower is better)

