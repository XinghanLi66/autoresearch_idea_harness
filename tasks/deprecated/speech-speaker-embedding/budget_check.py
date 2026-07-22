"""Parameter budget check for speech-speaker-embedding (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Extracts model class from template/agent file, applies baseline edits,
instantiates models, counts params, asserts agent <= 1.05x largest baseline.
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch

TASK_DIR = "/workspace/_task"

# Task-specific model class and default constructor args
MODEL_CLASS = "SpeakerEncoder"
MODEL_KWARGS = dict(
    n_mels=80, d_model=512, n_layers=4,
    embed_dim=192, dropout=0.1,
)


def load_module(path, name=None):
    name = name or f"_mod_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_ops(lines, ops, filename):
    result = list(lines)
    for op in sorted(
        [o for o in ops if o.get("file") == filename],
        key=lambda o: -o.get("start_line", o.get("after_line", 0)),
    ):
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


def extract_model_code(source):
    """Extract imports + class definitions before the FIXED training section."""
    lines = source.splitlines()
    cut = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "FIXED" in stripped and any(
            kw in stripped
            for kw in ["Data loading", "training", "Training", "evaluation",
                        "AAMSoftmax", "Loss"]
        ):
            while i > 0 and lines[i - 1].strip().startswith("#"):
                i -= 1
            cut = i
            break
    return "\n".join(lines[:cut])


def count_params(model_source, class_name, kwargs):
    """Write model source to temp file, import, instantiate, count params."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(model_source)
        tmp = f.name
    try:
        mod = load_module(tmp, f"_chk_{abs(hash(tmp))}")
        cls = getattr(mod, class_name)
        with torch.no_grad():
            model = cls(**kwargs)
        return sum(p.numel() for p in model.parameters())
    finally:
        os.unlink(tmp)


# -- Get template content --
config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
editable_file = next(f["filename"] for f in config["files"] if f.get("edit"))
workspace_file = os.path.join("/workspace", editable_file)

mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
template_content = next(
    op["content"] for op in mid_edit.OPS
    if op.get("op") == "create" and op.get("file") == editable_file
)
template_lines = template_content.splitlines()

# -- Count params for each baseline --
print("Baseline parameter counts:")
baseline_params = {}
for bl_name, bl_cfg in config.get("baselines", {}).items():
    edit_path = os.path.join(TASK_DIR, bl_cfg.get("edit_ops", ""))
    if not os.path.exists(edit_path):
        continue
    bl_mod = load_module(edit_path, f"_bl_{bl_name}")
    ops = getattr(bl_mod, "OPS", [])
    modified = apply_ops(template_lines, ops, editable_file)
    model_code = extract_model_code("\n".join(modified))
    try:
        p = count_params(model_code, MODEL_CLASS, MODEL_KWARGS)
        baseline_params[bl_name] = p
        print(f"  {bl_name}: {p:,} params")
    except Exception as e:
        print(f"  {bl_name}: ERROR ({e})")

if not baseline_params:
    print("WARNING: no baselines evaluated, skipping budget check")
    sys.exit(0)

max_name = max(baseline_params, key=baseline_params.get)
budget = int(baseline_params[max_name] * 1.05)

# -- Count params for agent's version --
with open(workspace_file) as f:
    agent_source = f.read()
agent_model_code = extract_model_code(agent_source)
try:
    agent_params = count_params(agent_model_code, MODEL_CLASS, MODEL_KWARGS)
except Exception as e:
    print(f"ERROR counting agent params: {e}", file=sys.stderr)
    sys.exit(1)

print(f"\n  agent model: {agent_params:,} params")
print(f"  budget: {budget:,} (1.05 x {max_name}={baseline_params[max_name]:,})")

if agent_params > budget:
    print(f"\nFAILED: {agent_params:,} > {budget:,}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
