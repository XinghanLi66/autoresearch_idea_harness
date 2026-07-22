"""MEAN baseline -- subprocess wrapper for the upstream paper-faithful trainer.

Reference: Kong et al., "Conditional Antibody Design as 3D Equivariant Graph
Translation" (ICLR 2023)
Source:    vendor/external_packages/chimera-bench/baselines/mean/

This baseline does NOT reimplement MEAN inside the rigorous_codebase template.
Instead it delegates to the upstream `chimera_trainer.py` (which is the exact
script reported in the CHIMERA-Bench paper, Table 9) and forwards CLI args
plus stdout. The wrapper:

  1. Replaces the editable CustomCDRModel block (lines 225-519) with a
     sentinel `CustomCDRModel = None` -- budget_check.py understands this as
     "subprocess-mode baseline, no in-process model to size".
  2. Replaces main() (lines 613-699) with an argparse front end that calls
     `python chimera-bench/baselines/mean/chimera_trainer.py --split <split>
      --cdr_type 3 --gpu <gpu> --seed <seed> --max_epoch <epochs>
      --batch_size <bs> --lr <lr> --no-wandb`,
     streams stdout and rewrites `Epoch N | train_loss=... val_loss=...`
     into the `TRAIN_METRICS` lines required by parser.py.
  3. After training completes, reads
     `${CHIMERA_DATA_ROOT}/results/mean/mean_cdr3_<split>/` (the trainer's
     own output_dir, defined in chimera_trainer.py:115-116) for the saved
     `test_metrics.csv` (written by save_test_csv at chimera_trainer.py:265-276)
     and emits `TEST_METRICS aar=... rmsd=... tm_score=...` so parser.py and
     score_spec.py pick up the H3 metrics.

CDR-H3 only -- matches the CHIMERA-Bench Track 1 evaluation protocol, the
DiffAb/MEAN/dyMEAN papers, and `evaluate(..., cdr_filter='H3')` in the
custom template.
"""

_FILE = "chimera-bench/custom_cdr.py"

_MODEL_CONTENT = """\

# =====================================================================
# Subprocess-wrapper baseline: no in-process CustomCDRModel.
# main() (below) shells out to vendor/external_packages/chimera-bench/
# baselines/mean/chimera_trainer.py.  The sentinel `CustomCDRModel = None`
# tells budget_check.py to skip parameter counting (see budget_check.py).
# =====================================================================
CustomCDRModel = None
"""

_MAIN_CONTENT = '''\
def _stream_subprocess(cmd, env=None, cwd=None, train_log_re=None):
    """Run `cmd`, stream stdout line-by-line, optionally rewriting training
    log lines into TRAIN_METRICS so parser.py picks them up."""
    import subprocess as _sp
    import re as _re
    import sys as _sys
    p = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, cwd=cwd, env=env,
                  text=True, bufsize=1)
    last_metrics = {}
    for line in p.stdout:
        line = line.rstrip()
        print(line, flush=True)
        if train_log_re is not None:
            m = train_log_re.search(line)
            if m:
                gd = m.groupdict()
                last_metrics.update(gd)
                fields = " ".join(f"{k}={v}" for k, v in gd.items())
                print("TRAIN_METRICS " + fields, flush=True)
    rc = p.wait()
    return rc, last_metrics


def _emit_test_metrics_from_csv(csv_path, label, metric_keys=("aar", "rmsd", "tm_score"),
                                cdr_filter="H3"):
    """Read the chimera_trainer.py-written test_metrics.csv and emit the
    metrics for the requested CDR row as TEST_METRICS lines."""
    import csv as _csv
    import os as _os
    import sys as _sys
    if not _os.path.exists(csv_path):
        print(f"ERROR: expected test_metrics.csv at {csv_path} not found", file=_sys.stderr)
        return False
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    target = None
    for r in rows:
        if r.get("cdr_type") == cdr_filter:
            target = r
            break
    if target is None and rows:
        target = rows[0]  # MEAN single-CDR mode produces one row only
    if target is None:
        print(f"ERROR: empty {csv_path}", file=_sys.stderr)
        return False
    for k in metric_keys:
        v = target.get(k, "")
        # Cells may be "0.42\u00b10.05" (mean\u00b1std); take the mean part.
        if isinstance(v, str) and "\u00b1" in v:
            v = v.split("\u00b1")[0]
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        print(f"TEST_METRICS {k}={fv:.6f}", flush=True)
    return True


def main():
    import argparse as _argparse
    import os as _os
    import re as _re
    import sys as _sys
    parser = _argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True,
                        choices=['epitope_group', 'antigen_fold', 'temporal'])
    parser.add_argument('--data-root', type=str,
                        default=_os.environ.get('CHIMERA_DATA_ROOT', '/data/chimera-bench-v1.0'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # CONFIG_OVERRIDES kept for parity with the agent template.
    CONFIG_OVERRIDES = {}
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': args.lr = _v

    upstream_dir = '/workspace/chimera-bench/baselines/mean'
    upstream_script = _os.path.join(upstream_dir, 'chimera_trainer.py')
    if not _os.path.exists(upstream_script):
        print(f"ERROR: missing upstream trainer at {upstream_script}", file=_sys.stderr)
        _sys.exit(1)

    cmd = [_sys.executable, upstream_script,
           '--split', args.split,
           '--cdr_type', '3',
           '--gpu', str(args.gpu),
           '--seed', str(args.seed),
           '--max_epoch', str(args.epochs),
           '--batch_size', str(args.batch_size),
           '--lr', str(args.lr),
           '--no-wandb']

    env = _os.environ.copy()
    env['CHIMERA_DATA_ROOT'] = args.data_root

    # Match upstream log format:
    # `Epoch  3 | train_loss=2.7510 val_loss=2.7110 train_ppl=15.66 ...`
    train_re = _re.compile(
        r"Epoch\\s*(?P<epoch>\\d+)\\s*\\|\\s*train_loss=(?P<train_loss>[-\\d.eE+]+)\\s+"
        r"val_loss=(?P<val_loss>[-\\d.eE+]+)"
    )
    rc, _ = _stream_subprocess(cmd, env=env, cwd=upstream_dir,
                               train_log_re=train_re)
    if rc != 0:
        print(f"ERROR: upstream trainer exited with code {rc}", file=_sys.stderr)
        _sys.exit(rc)

    # chimera_trainer.py writes test_metrics.csv to its own _SCRIPT_DIR
    # (chimera_trainer.py:267).  We also accept the per-run output_dir copy
    # if a future upstream change moves it.
    csv_candidates = [
        _os.path.join(upstream_dir, 'test_metrics.csv'),
        _os.path.join(args.data_root, 'results', 'mean',
                      f'mean_cdr3_{args.split}', 'test_metrics.csv'),
    ]
    csv_path = next((p for p in csv_candidates if _os.path.exists(p)), csv_candidates[0])
    ok = _emit_test_metrics_from_csv(csv_path, args.split)
    if not ok:
        _sys.exit(1)


if __name__ == '__main__':
    main()
'''


OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 613,
        "end_line": 699,
        "content": _MAIN_CONTENT,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 225,
        "end_line": 519,
        "content": _MODEL_CONTENT,
    },
]
