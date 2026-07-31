# MLS-Bench harness modifications (eval fixes)

Our eval-harness changes on top of the MLS-Bench fork. Reproduce by cloning the fork
at BASE_REF.txt's commit, then `git apply harness_code.patch`. Only src/scripts/vendor
+ root *.py are captured (the 3000+ dirty log/task files are eval run artifacts, excluded).
NOTE: the QS-ported modified MLS-Bench also lives at /mnt/3fs/lxh/mlsbench (branch
mlsbench-qs-code) — cc000 to confirm/push that as the canonical QS-side copy.
