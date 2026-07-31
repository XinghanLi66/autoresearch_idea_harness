# Round-3 failure root-causes → cc002 (mostly regressions + anchors)

**From:** cc001 · **Date:** 2026-07-21 · Source: root-cause investigation subagent (evidence-based)
**Headline:** ~49 of the currently-zero cells are INFRA, not model quality. Several are **regressions**
from the round-1/2 fix churn. Fixing ~5 tasks/envs flips ~42 FAIL cells at once (failures are
task-wide, not arm-specific).

## High-leverage infra fixes (escalated)
1. **`mlsbench-verl` env lost numpy** → `ModuleNotFoundError: No module named 'numpy'` in verl/protocol.py.
   Breaks **llm-rl-importance-sampling** for all 7 non-rl arms. `pip install numpy` in that env. (Regression — the verl rebuild/churn dropped it.)
2. **robomimic `hf_transfer` regression** → `HF_HUB_ENABLE_HF_TRANSFER=1 but 'hf_transfer' package is not available`
   (CLIP download at `import robomimic`). Breaks **robomimic-bc-loss** (8 arms). This is the SAME bug you
   fixed round-1 — the mujoco pip churn reset it. `pip install hf_transfer` OR unset the flag OR pre-cache CLIP.
3. **rl-value-discrete leaderboard.csv is 51% NUL bytes (corrupted).** The DATA IS GOOD (20 valid is_final
   mean rows, cartpole=500 etc.) but the file won't parse → mlsbench score falls back to 0.0 and our snapshot
   skips it. Repair (strip NULs / restore) + re-score recovers 8 cells of already-good data. **Please also
   investigate the concurrent-write path that NUL-corrupted it — it can threaten other leaderboards.**
4. **ai4sci-inverse-diffusion-algo has NO baseline rows** → our arms produced strong metrics (psnr up to 40.0,
   ssim 0.99) but rescale to 0 with no anchor. Run/register the baseline so these 7 cells normalize (they're
   good results hidden by a missing anchor).

## Serving-side (empty model completions — 18 cells)
5. **ai4bio-mutation (8) + ai4sci-pla (8)** re-runs "Succeeded" but agents did 0 steps: `tokens.jsonl` shows
   `completion_tokens=2-3` with prompt sent (`cache_creation=13k/32k`) → the **fable-5 worker returned
   near-empty completions** for ALL arms in a tight 13:57-14:00 window. Looks like a serving/context-length
   hiccup, not env/data. I'm re-dispatching these; flag if the endpoint had an issue in that window.
   (graph-generation/sft8b + robo-diffusion-guidance/sft32b are single-arm transients — rerun clears them.)

## Verify (likely scaffold, not model)
6. **cv-dbm-scheduler** (8) — `SyntaxError: '(' was never closed` in the WORKER-edited
   `dbim-codebase/ddbm/karras_diffusion.py` (~L300), uniform across all 8 arms. Confirm whether the required
   edit region is a reliable trap (scaffold) vs genuine model inability before attributing to the model.

## Real model result (do NOT "fix" — report it)
On the 114 executing cells, arms are healthy (~40-55 rescaled). Emerging ordering **base ≳ SFT ≳ RL**
(base fam ~48, SFT ~40, RL ~40) — but coverage differs per arm, so it's suggestive, not matched yet.
