"""Parameter budget check for pde-foundation-icl (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Imports each baseline, instantiates models, counts params, and
asserts the agent's model doesn't exceed 1.05x the largest baseline.
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch

TASK_DIR = os.environ.get("MLSBENCH_TASK_DIR", "/workspace/_task")
WORKSPACE_ROOT = os.environ.get("MLSBENCH_PKG_DIR", "/workspace/VICON")
WORKSPACE_FILE = os.path.join(WORKSPACE_ROOT, "src", "custom_model.py")

# Ensure the package root is on sys.path so that VICON imports work
sys.path.insert(0, WORKSPACE_ROOT)

# -- Model config (same across all eval datasets: NS2D, COMPRESSIBLE2D, EULER2D) --
# The model architecture is shared; only the eval dataset changes.
# Config matches the training script args (scripts/ns2d.sh etc.).
MODEL_CFG = {
    "transformer": {
        "dim_channel": 7,
        "dim_token": 1024,
        "nhead": 8,
        "dim_feedforward": 2048,
        "num_layers": 10,
        "dropout": 0.0,
    },
    "demo_num": 10,
    "patch_num_in": 8,
    "patch_num_out": 8,
    "patch_resolution": 16,  # 128 / 8
    "use_patch_pos_encoding": True,
    "use_func_pos_encoding": True,
}


def load_module(path, name=None):
    name = name or f"_mod_{hash(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_ops(lines, ops, filename):
    result = list(lines)
    sorted_ops = sorted(
        [o for o in ops if o.get("file") == filename],
        key=lambda o: -o.get("start_line", o.get("after_line", 0)),
    )
    for op in sorted_ops:
        if op["op"] == "replace":
            s, e = op["start_line"] - 1, op["end_line"]
            result[s:e] = op["content"].splitlines()
        elif op["op"] == "insert":
            after = op["after_line"]
            result[after:after] = op["content"].splitlines()
        elif op["op"] == "delete":
            s, e = op["start_line"] - 1, op["end_line"]
            del result[s:e]
    return result


def count_params(module_path):
    """Import module, instantiate CustomModel, return total param count."""
    mod = load_module(module_path, f"_check_{id(module_path)}")
    model = mod.CustomModel(MODEL_CFG)
    return sum(p.numel() for p in model.parameters())


# -- Get template content --
mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
editable_file = None
for f in config.get("files", []):
    if f.get("edit"):
        editable_file = f["filename"]
        break

template_content = None
for op in mid_edit.OPS:
    if op.get("op") == "create" and op.get("file") == editable_file:
        template_content = op["content"]
        break

assert template_content, f"No template found for {editable_file}"
template_lines = template_content.splitlines()

# -- Count params for each baseline --
baseline_params = {}
for bl_name, bl_cfg in config.get("baselines", {}).items():
    edit_path = os.path.join(TASK_DIR, bl_cfg["edit_ops"])
    if not os.path.exists(edit_path):
        continue
    bl_mod = load_module(edit_path, f"_bl_{bl_name}")
    ops = getattr(bl_mod, "OPS", [])
    modified_lines = apply_ops(template_lines, ops, editable_file)
    modified_code = "\n".join(modified_lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(modified_code)
        tmp_path = f.name
    try:
        params = count_params(tmp_path)
        baseline_params[bl_name] = params
        print(f"  baseline {bl_name}: {params} params")
    except Exception as e:
        print(f"  baseline {bl_name}: ERROR ({e})")
    finally:
        os.unlink(tmp_path)

if not baseline_params:
    print("WARNING: no baselines could be evaluated, skipping budget check")
    sys.exit(0)

max_baseline = max(baseline_params.values())
max_name = max(baseline_params, key=baseline_params.get)
budget = int(max_baseline * 1.05)

# -- Count params for agent's version --
agent_params = count_params(WORKSPACE_FILE)
print(f"\n  agent model: {agent_params} params")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline})")
env_label = os.environ.get("ENV", "NS2D")
print(f"  env={env_label}")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
