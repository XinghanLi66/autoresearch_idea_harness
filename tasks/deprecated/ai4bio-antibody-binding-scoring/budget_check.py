"""Zero-shot contract check for ai4bio-antibody-binding-scoring (standalone).

Run by tools.py before evaluation. Verifies:
  1. The custom_abscore.py module imports.
  2. It defines a ScoringFunction class with a `score_batch` method.
  3. ScoringFunction does NOT subclass torch.nn.Module (the task is
     zero-shot; no trainable parameters, no supervised training on
     binding labels). If a contributor needs nn.Module as a container
     they can still use it, but the harness never calls backward and
     never passes binding labels into score_batch — this check is a
     soft reminder, printed as a warning rather than a hard failure.
"""
import importlib.util
import os
import sys

TASK_DIR = "/workspace/_task"
WORKSPACE_FILE = "/workspace/AbBiBench/custom_abscore.py"


def load_module(path, name=None):
    name = name or f"_mod_{hash(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if not os.path.exists(WORKSPACE_FILE):
    print(f"FAILED: {WORKSPACE_FILE} not found", file=sys.stderr)
    sys.exit(1)

try:
    mod = load_module(WORKSPACE_FILE, "_check_abscore")
except Exception as e:
    print(f"FAILED: import error: {e}", file=sys.stderr)
    sys.exit(1)

if not hasattr(mod, "ScoringFunction"):
    print("FAILED: custom_abscore.py does not define ScoringFunction",
          file=sys.stderr)
    sys.exit(1)

sf_cls = mod.ScoringFunction
if not hasattr(sf_cls, "score_batch"):
    print("FAILED: ScoringFunction has no score_batch method",
          file=sys.stderr)
    sys.exit(1)

try:
    import torch.nn
    if isinstance(sf_cls, type) and issubclass(sf_cls, torch.nn.Module):
        print("WARNING: ScoringFunction subclasses torch.nn.Module. "
              "The harness runs zero-shot only — no backward() will be "
              "called and no binding labels are ever passed into "
              "score_batch. Trainable parameters are ignored.")
except Exception:
    pass

print("PASSED")
