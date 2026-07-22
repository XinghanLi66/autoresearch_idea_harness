"""Parameter budget check for pde-steady-solver (standalone).

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

TASK_DIR = "/workspace/_task"
WORKSPACE_FILE = "/workspace/Neural-Solver-Library/models/Custom.py"

# Ensure the package root is on sys.path so that layer imports work
sys.path.insert(0, "/workspace/Neural-Solver-Library")

# -- Dataset-specific args (no data loading needed) --
# Grid dimensions: Airfoil=[221,51], Darcy=[85,85] (421 downsampled 5x),
# Pipe=[129,129].
DATASET_ARGS = {
    "Airfoil": dict(
        n_hidden=128, n_heads=8, n_layers=8, mlp_ratio=2, slice_num=64,
        modes=12, unified_pos=0, ref=8, space_dim=2, fun_dim=2, out_dim=1,
        geotype="structured_2D", shapelist=[221, 51], act="gelu", dropout=0.0,
        time_input=False,
    ),
    "Darcy": dict(
        n_hidden=128, n_heads=8, n_layers=8, mlp_ratio=2, slice_num=64,
        modes=12, unified_pos=1, ref=8, space_dim=2, fun_dim=1, out_dim=1,
        geotype="structured_2D", shapelist=[85, 85], act="gelu", dropout=0.0,
        time_input=False,
    ),
    "Pipe": dict(
        n_hidden=128, n_heads=8, n_layers=8, mlp_ratio=2, slice_num=64,
        modes=12, unified_pos=0, ref=8, space_dim=2, fun_dim=2, out_dim=1,
        geotype="structured_2D", shapelist=[129, 129], act="gelu", dropout=0.0,
        time_input=False,
    ),
}

env_label = os.environ.get("ENV", "Airfoil")
args_dict = DATASET_ARGS.get(env_label, DATASET_ARGS["Airfoil"])


class MockArgs:
    """Lightweight args object that mimics argparse.Namespace."""
    def __init__(self, d):
        self.__dict__.update(d)


mock_args = MockArgs(args_dict)


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
    """Import module, instantiate Model, return total param count."""
    mod = load_module(module_path, f"_check_{id(module_path)}")
    model = mod.Model(mock_args)
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
print(f"  env={env_label}, n_hidden={args_dict['n_hidden']}, geotype={args_dict['geotype']}")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
