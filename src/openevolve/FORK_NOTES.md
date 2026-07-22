# OpenEvolve — in-tree fork for MLS-Bench

## Upstream
- Source: https://github.com/algorithmicsuperintelligence/openevolve
- Forked from commit `80945ed` ("Fix bugs (#442)")
- Local mirror at fork time: `/home/bohanlyu/exp2/openevolve/openevolve`

## Why we vendor it
MLS-Bench runs OpenEvolve as a second agent type (`--agent-type openevolve`) and needs tight integration with our workspace tools, leaderboard, and token accounting. Keeping the library in-tree under `src/openevolve/` (sibling to `src/mlsbench/`) lets the existing src-layout pick it up automatically — no extra pip step.

## Hard rule: do not modify the evolutionary core
The research point of pulling OpenEvolve in is its **methods**. The algorithmic core must stay faithful to upstream so comparisons are apples-to-apples:

- `controller.py` — main orchestration loop
- `database.py` — MAP-Elites grid + island model + selection
- `process_parallel.py` — parallel iteration execution
- `prompt/sampler.py`, `prompt/templates.py` — prompt construction
- `evaluator.py` — cascade evaluation logic
- `iteration.py`, `evolution_trace.py`, `novelty_judge.py`, `embedding.py`

Allowed changes: the LLM adapter (`llm/openai.py`), `config.py` field additions for adapter knobs, and pure add-ons (new files) that don't alter evolution semantics.

## Applied deltas vs upstream `80945ed`

All deltas must be enumerated here. Diff with `git diff 80945ed -- src/openevolve/` against an upstream checkout to audit.

1. **`llm/openai.py`** — added a module-level token-usage observer hook (`set_token_observer`, `_token_observer`, `_extract_usage`) and a call-site inside `_call_api` that reports per-call usage when either (a) an observer is installed or (b) `MLSBENCH_OE_TOKENS_LOG` env var points to a log file. The env-var fallback ensures `ProcessPoolExecutor` worker subprocesses — which re-import this module fresh and lose the in-process observer — still log usage to the same file. No changes to request construction, retry logic, or return values. Default behavior (observer unset and env var unset) is upstream-identical.
2. **`llm/openai.py`** — when `api_base` points at OpenRouter, inject `extra_body.provider.order` (pinned by model-name prefix via `_openrouter_provider_order`) + `allow_fallbacks: false`, plus `extra_body.usage.include=true`. This keeps the prompt cache warm across sequential calls (OpenRouter sticky-routes per upstream provider; without pinning, cache misses on every rotation). Matches the identical logic added to MLS-Bench's `src/mlsbench/agent/models.py` for the InteractiveAgent path.
3. **`llm/openai.py`** — skip the OpenAI-compatible `seed` request parameter for Gemini endpoints, detected either by the Google AI Studio OpenAI-compatible base URL or by `gemini` in the model name. Gemini rejects `seed`, so this preserves successful calls for direct and LiteLLM-proxied Gemini while leaving non-Gemini API calls unchanged.

## Deps added to MLS-Bench's `pyproject.toml`
OpenEvolve requires: `openai>=1.0.0`, `pyyaml>=6.0`, `numpy>=1.22.0`, `tqdm>=4.64.0`, `flask`, `dacite>=1.9.2`. Added under the `[openevolve]` optional-dependencies extra.
