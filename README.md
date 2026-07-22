# MLS-Bench

**Machine Learning Science Benchmark** — evaluating LLM agents on ML research tasks.

An LLM agent receives a task description and source code with marked editable regions, then iteratively edits code and runs containerized evaluations. Results are recorded to a per-task leaderboard with multi-seed aggregation.

---

## Quick Start

```bash
pip install -e .                               # install CLI
mlsbench fetch                                 # clone external packages
mlsbench build pytorch-examples                # build container image
mlsbench data pytorch-examples                 # prepare datasets (if any)
mlsbench baseline demo-task-1 --name default   # run a baseline
mlsbench agent demo-task-1 --model claude-sonnet-4-6  # run an LLM agent
```

If you already have local Claude Code MCP servers or skills on the machine, you can sync them into Codex with:

```bash
python scripts/sync_claude_to_codex.py
```

This imports `~/.mcp.json` into `~/.codex/config.toml`, links any `~/.claude/skills/*/SKILL.md` skills into Codex, and prints whether you need to restart Codex for tool discovery.

## Core Concepts

### Tasks and External Packages (Decoupled Design)

Tasks and external packages are **independent entities** with a many-to-many relationship:

- A **task** may use multiple external packages (e.g., `llm-offline-rl` uses `LLaMA-Factory`, `alpaca_eval`, and `MathRuler`)
- An **external package** may serve multiple tasks (e.g., `CORL` is used by `rl-offline-continuous`, `rl-offline-adroit`, `rl-offline-off2on`)

Each task's `config.json` declares which packages it needs via the `package` field in `test_cmds`. Package environment configs live in `vendor/pkg_configs/<PackageName>/config.json` and are shared across all tasks that use that package.

```
vendor/pkg_configs/CORL/config.json      # base image, install commands, env vars
vendor/pkg_configs/CORL/pre_edit.py      # package-level source patches

tasks/rl-offline-continuous/config.json  # test_cmds reference package "CORL"
tasks/rl-offline-adroit/config.json      # also references package "CORL"
```

### Full Pipeline: fetch → build → agent/baseline

```bash
mlsbench fetch                                    # 1. clone packages from packages.yaml
mlsbench build CORL --config configs/config.yaml  # 2. build container image + prepare data
mlsbench agent rl-offline-continuous --model claude-sonnet-4-6  # 3. run agent
```

**Auto-triggers**: The entire pipeline is automatic. When `agent`/`baseline` runs and an image is missing, `_ensure_image()` auto-fetches the package, builds the image, and prepares any data dependencies. `mlsbench build` also prepares data deps after building. You can still use `mlsbench data --list` to check data status or `mlsbench data <package>` to manually trigger preparation.

### Workspace Setup Pipeline

When an agent or baseline starts, the framework builds a workspace in three stages:

```
vendor/external_packages/CORL/  ──copy──>  vendor/workspace/<task>/CORL/
                                               │
                                          1. pre_edit    (package-level)
                                               │
                                          2. mid_edit    (task-level)
                                               │
                                          3. agent edits (or baseline edit_ops)
```

#### Stage 1: pre_edit (Package-Level Patches)

Defined in `vendor/pkg_configs/<Package>/pre_edit.py`. Applied regardless of which task uses the package. Typical use case: many packages log to wandb by default, but the benchmark needs text output for the parser to extract metrics. `pre_edit` injects `TRAIN_METRICS` / `TEST_METRICS` print statements so the agent gets structured text feedback.

Example — `vendor/pkg_configs/CORL/pre_edit.py` injects metric prints after `wandb.log()` in 9 algorithm files:

```python
OPS = [
    {"op": "insert", "file": "CORL/algorithms/offline/cql.py", "line": 946,
     "content": "        print(f'TRAIN_METRICS ' + ' '.join(f'{k}={v:.4f}' for k,v in log_dict.items()))"},
    ...
]
```

#### Stage 2: mid_edit (Task-Level Scaffolding)

Defined in `tasks/<task>/edits/mid_edit.py`. Creates task-specific template files in the workspace. Used when the task requires the agent to write a new algorithm rather than modifying existing code — for example, some RL repos are structured as one `.py` file per algorithm, so the task provides a scaffold template for the agent to fill in.

Example — `tasks/demo-task-2/edits/mid_edit.py` creates `main_custom.py` from a template:

```python
OPS = [
    {"op": "create", "file": "pytorch-examples/mnist/main_custom.py", "content": _CUSTOM_PY},
]
```

### Editable Regions and File Access Control

Each task's `config.json` specifies which files the agent can see and where it can edit:

```json
{
  "files": [
    {
      "filename": "CORL/algorithms/offline/custom.py",
      "read": [{"start": -1, "end": -1}],
      "edit": [{"start": 167, "end": 357}]
    },
    {
      "filename": "CORL/algorithms/offline/cql.py",
      "read": [{"start": -1, "end": -1}]
    }
  ]
}
```

- `read: [{"start": -1, "end": -1}]` — agent can see the entire file
- `edit: [{"start": 167, "end": 357}]` — agent can only edit lines 167-357
- No `edit` field — file is read-only (reference material for the agent)

### rigorous_codebase

When `mid_edit` creates a scaffold template, a natural question arises: is the template well-designed? If editable regions are too restrictive, even expert solutions can't be expressed; too permissive, and the task loses focus.

Setting `"rigorous_codebase": true` enforces a validation mechanism: **every baseline must also be implementable via `edit_ops` on the same template**. This guarantees the scaffold and editable regions are adequate for real algorithms.

- `demo-task-1`: `rigorous_codebase = false` — agent edits the original `main.py` directly, no scaffold needed
- `demo-task-2`: `rigorous_codebase = true` — agent edits `main_custom.py` (created by `mid_edit`), and both baselines (`cosine_annealing`, `warmup_cosine`) are implemented via `edit_ops` on this same file

When `rigorous_codebase = true` and baselines have `edit_ops`, the initial prompt shows code diffs so the agent can learn how existing algorithms are implemented on this codebase.

### Baselines: Two Modes of Operation

Baselines run through the same `WorkspaceTools.test()` pipeline as the agent, ensuring fair comparison. There are two modes depending on `rigorous_codebase`:

#### Mode 1: Command Replacement (`rigorous_codebase = false`)

The baseline replaces the evaluation script command. The `apply_baseline()` function swaps the `cmd` field in `test_cmds`:

```json
{
  "test_cmds": [
    {"cmd": "scripts/train.sh", "label": "train", "group": 1, "package": "pytorch-examples"}
  ],
  "baselines": {
    "default":   {"cmd": "scripts/default.sh"},
    "batchnorm": {"cmd": "scripts/batchnorm.sh", "edit_ops": "edits/batchnorm.edit.py", "labels": ["train"]}
  }
}
```

- `"default"` has no `labels` — replaces ALL `test_cmds` commands
- `"batchnorm"` has `"labels": ["train"]` — only replaces commands whose `label` matches `"train"`

This is useful for multi-stage pipelines. For instance, `llm-offline-rl` has train (group 1) and eval (group 2). The baselines `dpo` and `simpo` only replace the train command (`"labels": ["train"]`), keeping the same evaluation scripts.

If a baseline also has `edit_ops`, the code edits are applied to the workspace after command replacement.

#### Mode 2: Edit-Only (`rigorous_codebase = true`)

The `test_cmds` are NOT replaced — the baseline uses the exact same scripts as the agent. Instead, the baseline applies `edit_ops` to modify the workspace code, just like the agent would:

```json
{
  "baselines": {
    "cosine_annealing": {
      "cmd": "scripts/cosine_annealing.sh",
      "edit_ops": "edits/cosine_annealing.edit.py"
    }
  }
}
```

`edits/cosine_annealing.edit.py` replaces lines in the template file:

```python
OPS = [
    {"op": "replace", "file": "pytorch-examples/mnist/main_custom.py",
     "start_line": 131, "end_line": 131,
     "content": "    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)\n"},
    {"op": "replace", "file": "pytorch-examples/mnist/main_custom.py",
     "start_line": 7, "end_line": 7,
     "content": "from torch.optim.lr_scheduler import CosineAnnealingLR\n"},
]
```

Note: in this mode, the `cmd` field in baseline config is **not used for command replacement** — the original `test_cmds` scripts run unchanged.

### Test Execution: Groups and Parallelism

Commands in `test_cmds` use a `group` field to control execution order:

```json
{
  "test_cmds": [
    {"cmd": "scripts/train.sh",       "label": "train",    "group": 1, "compute": 4},
    {"cmd": "scripts/alpaca_eval.sh", "label": "evaluate", "group": 2, "compute": 1},
    {"cmd": "scripts/mathruler.sh",   "label": "mathruler", "group": 2, "compute": 1}
  ]
}
```

- **Same group** = run in parallel (e.g., `alpaca_eval` and `mathruler` both in group 2)
- **Different groups** = run sequentially (group 1 finishes before group 2 starts)
- **`compute`** = GPU allocation (4 = 4 GPUs, 0.33 = 1/3 of a GPU for co-locating 3 jobs)

### Container Execution

All training and evaluation run inside containers (Apptainer or Docker) for reproducibility.

`vendor/pkg_configs/<Package>/config.json` defines the environment:

```json
{
  "base_image": "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime",
  "install_cmds": ["pip install torchvision", "..."],
  "env": {"LD_LIBRARY_PATH": "/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}"},
  "use_cuda": true,
  "workdir": "/workspace"
}
```

Build images with:

```bash
mlsbench build pytorch-examples          # Apptainer: vendor/images/pytorch-examples.sif
mlsbench build CORL --config configs/config.gpublaze.yaml  # Docker (if config says docker)
mlsbench build LLaMA-Factory --dry-run   # generate .def/.Dockerfile only, skip build
```

### Data Dependencies

Some packages require large datasets that are bind-mounted into containers at runtime (not baked into images). These are declared via `data_deps` in `vendor/pkg_configs/<Package>/config.json`:

```json
{
  "data_deps": [
    {
      "name": "D4RL-medium",
      "host_path": "{data_root}/d4rl-medium",
      "container_path": "/data/d4rl-medium",
      "prepare": "scripts/prepare_d4rl.py",
      "description": "D4RL medium-quality datasets"
    }
  ]
}
```

Manage data with:

```bash
mlsbench data                  # prepare all datasets across all packages
mlsbench data CORL             # prepare datasets for one package
mlsbench data --list           # list all data deps and their status (READY/MISSING)
```

### Agent Loop

The agent operates with three tools: `edit`, `test`, `undo`.

```
┌──────────────────────────────────────────────────┐
│  Initial prompt:                                  │
│  - task_description.md                           │
│  - source code (with editable line annotations)  │
│  - baseline results table from leaderboard       │
└──────────────┬───────────────────────────────────┘
               │
               v
        ┌─────────────┐
        │  LLM Agent   │
        │  (loop)      │◄──────────────────┐
        └──────┬───────┘                   │
               │                           │
       ┌───────┼───────┐                   │
       v       v       v                   │
    edit()  test()  undo()                 │
       │       │       │                   │
       └───────┼───────┘                   │
               │ feedback (parsed metrics) │
               └───────────────────────────┘
```

- `edit(op, filename, content, ...)` — modify files within allowed editable regions
- `test(is_final=False)` — run evaluation scripts, get parsed metric feedback
- `undo(n)` — revert the last N edits
- Max `max_steps` loop iterations, max `max_tests` test calls before forced final submission
- Seed scheduling: first and final test = all seeds; intermediate tests = single seed (fast iteration)

### Output Parsing

Each task defines a `parser.py` that extracts metrics from raw container output:

```python
class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        # Extract metrics like accuracy, d4rl_score, win_rate from stdout
        return ParseResult(feedback="...", metrics={"test_accuracy": 0.993})
```

Parsed metrics are shown to the agent as feedback and recorded to `leaderboard.csv`.

### Multi-Seed Evaluation

Final submissions (`test(is_final=True)`) run across multiple seeds (default: `[42, 123, 456]`). The leaderboard records per-seed results and computes mean/std aggregation.

---

## Project Structure

```
MLS-Bench/
├── src/mlsbench/
│   ├── cli.py                  # 6 subcommands: agent, baseline, build, run, fetch, data
│   ├── agent/
│   │   ├── base.py             # BaseAgent: workspace setup + modify-test loop
│   │   ├── interactive.py      # InteractiveAgent: LLM-powered agent
│   │   ├── models.py           # Model clients (Anthropic / OpenAI / DeepSeek / Qwen / Gemini / Kimi)
│   │   ├── tools.py            # WorkspaceTools: edit, test, undo engine + container execution
│   │   ├── parsers.py          # Output parser base + task-specific loader
│   │   ├── leaderboard.py      # Per-task CSV leaderboard
│   │   ├── logger.py           # RunLogger: JSONL + file snapshots
│   │   ├── slurm.py            # SLURM job submission + GPU bin-packing
│   │   └── local_executor.py   # LocalSchedulerExecutor: test-time GPU allocation via scheduler
│   └── relay/
│       └── server.py           # API relay for compute nodes without internet
│
├── tasks/                       # Task definitions (one dir per task, 145+ tasks)
│   ├── <task>/
│   │   ├── config.json          # test_cmds, baselines, editable files, rigorous_codebase
│   │   ├── task_description.md  # Problem statement shown to agent
│   │   ├── parser.py            # Metric extraction from stdout
│   │   ├── scripts/             # All shell scripts (test + baseline)
│   │   ├── edits/               # Edit ops, templates, mid_edit
│   │   └── leaderboard.csv     # Results
│   └── ...
│
├── vendor/
│   ├── pkg_configs/             # Per-package container + pre_edit configs (72 packages)
│   │   └── <Package>/
│   │       ├── config.json      # base_image, install_cmds, env, use_cuda, data_deps
│   │       └── pre_edit.py      # (optional) Package-level source patches
│   ├── external_packages/       # Cloned repos (gitignored, via `mlsbench fetch`)
│   ├── images/                  # Built container images (gitignored)
│   ├── data/                    # Large datasets (from `mlsbench data`)
│   └── workspace/               # Runtime working copies (gitignored)
│
├── configs/
│   ├── config.yaml              # Runtime config (API keys, seeds, SLURM)
│   ├── config.gpublaze.yaml     # gpublaze server config (Docker, no SLURM, 8x H100)
│   ├── config.local.yaml        # Local config (Docker/Apptainer, single-node)
│   └── packages.yaml            # External package registry (git URLs + commits)
│
├── src/mlsbench/
│   ├── scheduler.py             # GPU-aware job scheduler (supports script/agent/baseline jobs)
│   └── slurm_compat.py          # squeue/sacct/sbatch/scancel shims for the local scheduler
│
├── scripts/
│   └── batch_agent.sh           # Batch-launch agents for a domain (nohup)
│
├── tests/                       # Tests
│   └── test_edit_ranges.py      # Validate baseline edit_ops fit within editable ranges
│
├── docs/                        # Design documentation
└── pyproject.toml               # Package definition
```

## Task Domains (145+ tasks)

| Domain | Count | Example Tasks |
|--------|-------|--------------|
| LLM | 24 | llm-pretrain-attention, llm-pretrain-loss, llm-rl-advantage, llm-scaling-law-discovery |
| CV | 11 | cv-diffusion-cfg, cv-diffusion-conditioning, cv-classification-loss, cv-data-augmentation |
| ML | 15 | ml-svm-kernel, ml-ensemble-boosting, ml-anomaly-detection, ml-continual-regularization |
| RL | 11 | rl-offline-continuous, rl-onpolicy-continuous, rl-value-atari, rl-reward-learning |
| Optimization | 15 | optimization-gradient-compression, optimization-dp-sgd, optimization-variance-reduction, optimization-bayesian-acquisition |
| AI4Sci | 8 | ai4sci-weather-forecast-aggregation, ai4sci-mol-property-prediction, ai4sci-sbdd-drug-design |
| Time Series | 6 | ts-long-term-forecast, ts-anomaly-detection, ts-classification, ts-imputation |
| Graph | 5 | graph-node-classification, graph-link-prediction, graph-generation, graph-signal-propagation |
| DL | 6 | dl-activation-function, dl-normalization, dl-residual-connection, dl-weight-initialization |
| AI4Bio | 6 | ai4bio-protein-inverse-folding, ai4bio-antibody-cdr-design, ai4bio-mutation-effect-prediction |
| Quant | 5 | quant-portfolio-opt, quant-stock-prediction, quant-order-execution |
| PDE | 5 | pde-autoregressive-solver, pde-foundation-icl, pde-steady-solver |
| Causal | 5 | causal-discovery-discrete, causal-treatment-effect, causal-observational-nonlinear |
| Speech | 4 | speech-asr-encoder, speech-vocoder, speech-enhancement, speech-speaker-embedding |
| Security | 4 | security-adversarial-attack-white-box-linf, security-adversarial-training |
| Other | 29 | meta-fewshot-classification, fed-aggregation-strategy, sr-symbolic-regression, mas-topology, ... |

## CLI Reference

```bash
# Full pipeline
mlsbench fetch [--name PACKAGE]                          # clone external packages
mlsbench build <package> [--force] [--dry-run] [--config PATH]  # build container
mlsbench data [<package>] [--list] [--config PATH]       # prepare datasets

# Evaluation
mlsbench agent <task> --model <model> [--config PATH] [--mode sci|eng] [-v]
mlsbench baseline <task> [--name NAME] [--config PATH] [--seed N] [--group G] [--label L]

# Utilities
mlsbench run <package> --run-cmd <script> [--task TASK] [--bind SRC:DST] [--dry-run]
```

## Configuration

### `configs/config.yaml`

```yaml
max_steps: 20              # Max agent loop iterations
max_tests: 5               # Max test() calls before forced final
save_path: /path/to/saves  # Root for model checkpoints
seeds: [42, 123, 456]      # Seeds for multi-seed evaluation
container_runtime: apptainer  # or "docker"

providers:                  # API keys per provider
  anthropic:
    api_key: "sk-ant-..."
  openai:
    api_key: "sk-..."
  deepseek:
    api_key: "sk-..."
    base_url: "https://api.deepseek.com/v1"

slurm:                      # Optional: SLURM config
  partition: pli-c
  account: pli
  constraint: gpu80
```

### Local Mode (No Docker/Apptainer)

For machines without container runtimes, MLS-Bench supports **local conda execution** with a lightweight GPU scheduler.

```bash
# 1. Build per-package conda environment
mlsbench build qlib --config configs/config.gpublaze.local.yaml

# 2. Start the GPU scheduler daemon (manages GPU allocation for compute jobs)
nohup python -m mlsbench.scheduler start \
  --gpus 0,1,2,3,4,5,6,7 --ignore-busy --daemon \
  --config configs/config.gpublaze.local.yaml \
  > .scheduler/scheduler.log 2>&1 &

# 3. Run agents/baselines with nohup (they submit compute to the scheduler)
nohup mlsbench agent quant-stock-prediction --model claude-sonnet-4-6 \
  --config configs/config.gpublaze.local.yaml \
  > agent.log 2>&1 &

nohup mlsbench baseline quant-stock-prediction --name lgbm \
  --config configs/config.gpublaze.local.yaml \
  > baseline.log 2>&1 &

# 4. Monitor GPU compute jobs
python -m mlsbench.scheduler status
```

**How it works**: Agent/baseline processes run without GPU. When `test()` is called, the `LocalSchedulerExecutor` submits compute scripts to the scheduler daemon, which assigns GPUs from the pool on demand. After compute finishes, GPUs are released. This mirrors the SLURM architecture — many agents can run concurrently, sharing GPUs only during actual compute.

Set `container_runtime: local` in your config to enable this mode. The scheduler auto-detection is transparent: if the daemon is running, test execution routes through it; otherwise it falls back to direct execution.

### Relay Server

For compute nodes without internet, the relay proxies LLM API calls:

```bash
# On login node (has internet)
python -m mlsbench.relay.server --host 0.0.0.0 --port 9123

# In configs/config.yaml
relay:
  base_url: "http://login-node:9123"
```

## Adding a New Task

1. Create `tasks/<task-name>/config.json` with `test_cmds`, `files`, and `baselines`
2. Write `tasks/<task-name>/task_description.md`
3. Implement `tasks/<task-name>/parser.py` to extract metrics from stdout
4. Add all shell scripts (test + baseline) to `tasks/<task-name>/scripts/`
5. Add baseline edit_ops and templates to `tasks/<task-name>/edits/`
6. (Optional) Add `edits/mid_edit.py` if the task needs a scaffold template
7. Ensure package configs exist in `vendor/pkg_configs/<Package>/`
8. Register packages in `configs/packages.yaml` if new

See [docs/evaluation-design.md](docs/evaluation-design.md) for detailed architecture documentation.
