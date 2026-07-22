"""dyMEAN baseline -- subprocess wrapper for the upstream paper-faithful trainer.

Reference: Kong et al., "End-to-End Full-Atom Antibody Design" (ICML 2023)
Source:    vendor/external_packages/chimera-bench/baselines/dymean/

Delegates training to the upstream `chimera_trainer.py` (the exact script
reported in the CHIMERA-Bench paper, Table 9). dyMEAN is a multi-CDR model
but per the CHIMERA Track 1 / paper protocol we restrict generation to CDR-H3
via `--cdr H3 --paratope H3`. test_metrics.csv (chimera_trainer.py:432-453)
is parsed and the H3 row's `aar / rmsd / tm_score` columns are surfaced as
TEST_METRICS lines for parser.py and score_spec.py.
"""

_FILE = "chimera-bench/custom_cdr.py"

_MODEL_CONTENT = """\

# =====================================================================
# Subprocess-wrapper baseline: no in-process CustomCDRModel.
# main() (below) shells out to vendor/external_packages/chimera-bench/
# baselines/dymean/chimera_trainer.py.  The sentinel `CustomCDRModel = None`
# tells budget_check.py to skip parameter counting (see budget_check.py).
# =====================================================================
CustomCDRModel = None
"""

_MAIN_CONTENT = '''\
def _stream_subprocess(cmd, env=None, cwd=None, train_log_re=None):
    """Run `cmd`, stream stdout, optionally rewriting upstream training log
    lines into TRAIN_METRICS for parser.py."""
    import subprocess as _sp
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
    """Read chimera_trainer.py's test_metrics.csv (one row per CDR; cells are
    "<mean>\u00b1<std>") and emit TEST_METRICS for the H3 row only."""
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
    if target is None:
        print(f"ERROR: no {cdr_filter} row in {csv_path}", file=_sys.stderr)
        return False
    for k in metric_keys:
        v = target.get(k, "")
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    CONFIG_OVERRIDES = {}
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': args.lr = _v

    upstream_dir = '/workspace/chimera-bench/baselines/dymean'
    upstream_script = _os.path.join(upstream_dir, 'chimera_trainer.py')
    if not _os.path.exists(upstream_script):
        print(f"ERROR: missing upstream trainer at {upstream_script}", file=_sys.stderr)
        _sys.exit(1)

    # Restrict to CDR-H3 (paper-protocol) by passing --cdr H3 --paratope H3.
    cmd = [_sys.executable, upstream_script,
           '--split', args.split,
           '--cdr', 'H3',
           '--paratope', 'H3',
           '--gpu', str(args.gpu),
           '--seed', str(args.seed),
           '--max_epoch', str(args.epochs),
           '--batch_size', str(args.batch_size),
           '--lr', str(args.lr),
           '--no-wandb']

    env = _os.environ.copy()
    env['CHIMERA_DATA_ROOT'] = args.data_root

    # dyMEAN per-epoch log:
    # `Epoch  3 | train_loss=2.7510 val_loss=2.7110 train_aar=... val_aar=...`
    train_re = _re.compile(
        r"Epoch\\s*(?P<epoch>\\d+)\\s*\\|\\s*train_loss=(?P<train_loss>[-\\d.eE+]+)\\s+"
        r"val_loss=(?P<val_loss>[-\\d.eE+]+)"
    )
    rc, _ = _stream_subprocess(cmd, env=env, cwd=upstream_dir,
                               train_log_re=train_re)
    if rc != 0:
        print(f"ERROR: upstream trainer exited with code {rc}", file=_sys.stderr)
        _sys.exit(rc)

    # chimera_trainer.py writes test_metrics.csv to its own directory
    # (chimera_trainer.py:433). We also accept the per-run output_dir copy.
    csv_candidates = [
        _os.path.join(upstream_dir, 'test_metrics.csv'),
        _os.path.join(args.data_root, 'results', 'dymean',
                      f'dymean_H3_{args.split}', 'test_metrics.csv'),
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
