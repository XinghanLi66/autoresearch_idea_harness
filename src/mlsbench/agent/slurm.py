"""SlurmExecutor: submit and monitor SLURM jobs for MLS-Bench test execution."""

import fcntl
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Global squeue/sacct serialization lock
# ---------------------------------------------------------------------------
# PROBLEM:  Multiple Claude Code sessions run MLS-Bench concurrently on the
# same account.  Each session's wait_for_job() polls squeue every 15 seconds
# per job.  With 5+ sessions × 11+ jobs this creates ~55 concurrent squeue
# calls every 15 s.  When the SLURM controller is under load each call takes
# 1-5+ s, but new calls keep arriving before old ones finish, creating a
# pile-up that makes squeue take *minutes* — a feedback loop that degrades
# the entire cluster's scheduler.
#
# FIX (two parts):
#   1. poll_interval raised from 15 s → 60 s  (4× fewer calls)
#   2. All squeue/sacct calls are serialized behind a file lock so only ONE
#      runs at a time across all processes on this machine.
#
# Combined effect: load drops from ~55 concurrent calls / 15 s to ~11
# sequential calls / 60 s (~11 s total), well within budget.
#
# The lock is held ONLY during the subprocess call (≤30 s timeout), never
# during the sleep between polls, so no session is stalled.
# ---------------------------------------------------------------------------
_SLURM_LOCK_PATH = os.environ.get(
    "MLSBENCH_SLURM_LOCK_PATH",
    f"/tmp/mlsbench-slurm-query-{os.getuid()}.lock",
)


def _resolve_slurm_command(name: str) -> str:
    """Prefer the system SLURM binaries over any PATH shim.

    MLS-Bench ships scheduler compatibility shims that can be installed into a
    conda env as `sbatch`/`squeue`/`sacct`/`scancel`. Those wrappers are useful
    for the local scheduler, but they break real cluster submission if they
    shadow `/usr/bin/sbatch`. Prefer the system binaries when available.
    """
    override = os.environ.get(f"MLSBENCH_{name.upper()}_BIN")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override

    system_path = f"/usr/bin/{name}"
    if os.path.isfile(system_path) and os.access(system_path, os.X_OK):
        return system_path

    return shutil.which(name) or name


def _run_slurm_query(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a squeue/sacct command while holding a global file lock.

    Serializes all SLURM query calls across processes to prevent pile-up
    on the SLURM controller.  If the command hangs beyond *timeout* seconds
    the lock is released so other callers can proceed.
    """
    resolved_cmd = [_resolve_slurm_command(cmd[0]), *cmd[1:]]
    lock_fd = open(_SLURM_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return subprocess.run(resolved_cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(resolved_cmd, returncode=1, stdout="", stderr="timeout")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


class SlurmExecutor:
    """Submit grouped container commands as SLURM jobs and wait for completion."""

    def __init__(self, slurm_config: dict, project_root: Path):
        self.partition = slurm_config.get("partition", "")
        self.account = slurm_config.get("account", "chij")
        self.constraint = slurm_config.get("constraint", "gpu80")
        self.qos = slurm_config.get("qos", "")
        self.nodelist = slurm_config.get("nodelist", "")
        self.exclude = slurm_config.get("exclude", "")
        self.mail_user = slurm_config.get("mail_user", "")
        self.logs_dir = project_root / slurm_config.get("logs_dir", "results")
        self.project_root = project_root
        self.cpus_per_gpu = slurm_config.get("cpus_per_gpu", 12)
        # CPU-only overrides
        self.no_gpu = slurm_config.get("no_gpu", False)
        self.cpu_partition = slurm_config.get("cpu_partition", "")    # partition for CPU-only jobs
        self.cpus_per_task = slurm_config.get("cpus_per_task")       # e.g. 8
        self.mem_per_cpu = slurm_config.get("mem_per_cpu")           # e.g. "8G"

    # ------------------------------------------------------------------
    # GPU assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_gpus(entries: list[dict]) -> list[str]:
        """Return list of CUDA_VISIBLE_DEVICES strings, one per entry.

        Whole-GPU entries (compute >= 1) get dedicated GPUs and advance the cursor.
        Fractional entries (compute < 1) share the current GPU without advancing.

        Examples:
          [4]                 -> total=4 -> ["0,1,2,3"]
          [1, 1]              -> total=2 -> ["0", "1"]
          [0.33, 0.33, 0.33]  -> total=1 -> ["0", "0", "0"]
        """
        total_gpus = max(1, math.ceil(sum(e["compute"] for e in entries)))
        current = 0
        assignments = []
        for entry in entries:
            compute = entry["compute"]
            n = max(1, math.ceil(compute))
            n = min(n, max(1, total_gpus - current))
            start = min(current, total_gpus - 1)
            end = min(start + n, total_gpus)
            gpus = list(range(start, end))
            if not gpus:
                gpus = [total_gpus - 1]
            assignments.append(",".join(str(g) for g in gpus))
            # Only advance cursor for whole-GPU entries
            if compute >= 1:
                current = end
        return assignments

    # ------------------------------------------------------------------
    # SLURM script generation
    # ------------------------------------------------------------------

    def _build_slurm_script(
        self,
        group_cmds: list[dict],
        job_name: str,
        out_dir: Path,
    ) -> str:
        """Generate a SLURM batch script for a group of commands.

        Each group_cmd is: {"label": str, "apptainer_cmd": list[str], "compute": float, "time": str}
        The "apptainer_cmd" key holds the container command regardless of backend (apptainer or docker).
        """
        max_time = max(group_cmds, key=lambda c: c.get("time", "1:00:00"))["time"]

        # Determine CPU-only mode:
        # 1. Global no_gpu override forces CPU mode for all jobs
        # 2. If all entries in the group have use_cuda=False, auto-detect CPU mode
        cpu_only = self.no_gpu or all(
            not entry.get("use_cuda", True) for entry in group_cmds
        )

        if cpu_only:
            # CPU-only mode
            cpus = self.cpus_per_task or 8
            lines = [
                "#!/bin/bash",
                f"#SBATCH --nodes=1",
                f"#SBATCH --ntasks=1",
                f"#SBATCH --cpus-per-task={cpus}",
            ]
            if self.mem_per_cpu:
                lines.append(f"#SBATCH --mem-per-cpu={self.mem_per_cpu}")
            else:
                cpu_mem_override = max((e.get("mem", 0) for e in group_cmds), default=0)
                cpu_mem = cpu_mem_override if cpu_mem_override > 0 else 32
                lines.append(f"#SBATCH --mem={cpu_mem}g")
            gpu_assignments = [""] * len(group_cmds)
        else:
            # GPU mode (default)
            gpu_assignments = self._assign_gpus(group_cmds)
            all_gpu_ids = set()
            for assignment in gpu_assignments:
                for g in assignment.split(","):
                    all_gpu_ids.add(int(g))
            total_gpus = max(1, len(all_gpu_ids))
            # Allow per-group CPU override via "cpus" field in test_cmd.
            # Otherwise fall back to config-level override or cpus_per_gpu scaling.
            cpus_override = max((e.get("cpus", 0) for e in group_cmds), default=0)
            default_cpus = self.cpus_per_task or (total_gpus * self.cpus_per_gpu)
            cpus = cpus_override if cpus_override > 0 else default_cpus
            # Allow per-group memory override via "mem" field in test_cmd
            mem_override = max((e.get("mem", 0) for e in group_cmds), default=0)
            mem = mem_override if mem_override > 0 else total_gpus * 100
            lines = [
                "#!/bin/bash",
                f"#SBATCH --nodes=1",
                f"#SBATCH --ntasks=1",
                f"#SBATCH --ntasks-per-node=1",
                f"#SBATCH --cpus-per-task={cpus}",
                f"#SBATCH --gres=gpu:{total_gpus}",
                f"#SBATCH --mem={mem}g",
            ]

        # Use cpu_partition for CPU-only jobs if configured, otherwise fall back to default partition
        effective_partition = (self.cpu_partition if cpu_only and self.cpu_partition else self.partition)
        if effective_partition:
            lines.append(f"#SBATCH --partition={effective_partition}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.constraint and not cpu_only:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")
        if self.nodelist:
            lines.append(f"#SBATCH --nodelist={self.nodelist}")
        if self.exclude:
            lines.append(f"#SBATCH --exclude={self.exclude}")
        lines += [
            f"#SBATCH --time={max_time}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output=/dev/null",
            f"#SBATCH --mail-type=ALL",
            f"#SBATCH --mail-user={self.mail_user}",
            "",
            "source ~/.bashrc",
            f"cd {self.project_root}",
            "",
        ]

        if not cpu_only and len(group_cmds) > 1:
            # Multiple commands sharing a SLURM job: read SLURM-allocated GPUs
            # and remap _assign_gpus indices to actual device IDs.
            lines.append("# Read SLURM-allocated GPU list and remap relative indices")
            lines.append("IFS=',' read -ra _SLURM_GPUS <<< \"$CUDA_VISIBLE_DEVICES\"")
            lines.append("")

        if len(group_cmds) == 1:
            # Single command: SLURM already set CUDA_VISIBLE_DEVICES correctly
            label = group_cmds[0]["label"]
            safe_label = re.sub(r"\W", "_", label)
            cmd_str = " ".join(shlex.quote(s) for s in group_cmds[0]["apptainer_cmd"])
            lines.append(f"START_{safe_label}=$(date +%s)")
            if cpu_only:
                lines.append(f"# --- {label} (CPU) ---")
                lines.append(f"{cmd_str} > {out_dir}/{label}.out 2>&1")
            else:
                lines.append(f"# --- {label} (uses SLURM-allocated GPUs: $CUDA_VISIBLE_DEVICES) ---")
                lines.append(f"{cmd_str} > {out_dir}/{label}.out 2>&1")
            lines.append(f'echo "{label}:$?" >> {out_dir}/exit_codes.txt')
            lines.append(f'END_{safe_label}=$(date +%s)')
            lines.append(f'echo "{label}:$((END_{safe_label} - START_{safe_label}))" >> {out_dir}/elapsed.txt')
        else:
            # Multiple commands: remap relative GPU indices to SLURM-allocated devices
            for i, (cmd, rel_gpus) in enumerate(zip(group_cmds, gpu_assignments)):
                label = cmd["label"]
                cmd_str = " ".join(shlex.quote(s) for s in cmd["apptainer_cmd"])
                lines.append(f"START_{i}=$(date +%s)")
                if cpu_only:
                    lines.append(f"# --- {label} (CPU) ---")
                    lines.append(f"{cmd_str} > {out_dir}/{label}.out 2>&1 &")
                else:
                    # Convert relative indices (0,1) to actual SLURM GPU IDs
                    remap_parts = [f"${{_SLURM_GPUS[{g}]}}" for g in rel_gpus.split(",")]
                    remap_expr = ",".join(remap_parts)
                    lines.append(f"# --- {label} (relative GPU: {rel_gpus}) ---")
                    lines.append(f"CUDA_VISIBLE_DEVICES={remap_expr} NVIDIA_VISIBLE_DEVICES={remap_expr} {cmd_str} > {out_dir}/{label}.out 2>&1 &")
                lines.append(f"PID_{i}=$!")
                lines.append("")

            # Wait for each PID and record exit codes + elapsed time
            for i, cmd in enumerate(group_cmds):
                label = cmd["label"]
                lines.append(f"wait $PID_{i}; echo \"{label}:$?\" >> {out_dir}/exit_codes.txt")
                lines.append(f'END_{i}=$(date +%s)')
                lines.append(f'echo "{label}:$((END_{i} - START_{i}))" >> {out_dir}/elapsed.txt')

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def submit_group(
        self,
        group_cmds: list[dict],
        job_name: str,
        out_dir: Path,
    ) -> str:
        """Submit a SLURM job for a group of commands. Returns the job ID."""
        out_dir.mkdir(parents=True, exist_ok=True)
        script = self._build_slurm_script(group_cmds, job_name, out_dir)

        script_path = out_dir / "submit.sh"
        script_path.write_text(script)
        print(f"[slurm] Script written: {script_path}")
        print(f"[slurm] --- Script ---\n{script}[slurm] ----------------")

        result = subprocess.run(
            [_resolve_slurm_command("sbatch"), str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # Parse job ID from "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")

        job_id = match.group(1)
        # Save job ID for resume recovery
        (out_dir / "job_id.txt").write_text(job_id)
        print(f"[slurm] Submitted job {job_id}: {job_name}")
        return job_id

    # ------------------------------------------------------------------
    # Job monitoring
    # ------------------------------------------------------------------

    def wait_for_job(self, job_id: str, poll_interval: int = 60, timeout: int | None = None) -> str:
        """Poll squeue/sacct until the job reaches a terminal state. Returns the final state.

        Strategy: use squeue first (authoritative for active jobs). Only after the
        job disappears from squeue, fall back to sacct for the final state.  This
        avoids false 'COMPLETED' from sacct while the job is still PENDING.

        All squeue/sacct calls go through _run_slurm_query() which holds a
        global file lock — see module docstring for rationale.
        """
        terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
        start = time.time()
        consecutive_query_failures = 0

        while True:
            # --- Phase 1: squeue (reliable for PENDING / RUNNING) ---
            sq = _run_slurm_query(
                ["squeue", "-j", job_id, "--noheader", "-o", "%T"],
            )

            if sq.returncode != 0:
                # squeue command failed (timeout, lock contention, etc.)
                # Do NOT fall through to "job vanished" — that causes false
                # COMPLETED when the job is actually still PENDING/RUNNING.
                consecutive_query_failures += 1
                if consecutive_query_failures >= 10:
                    # Persistent failure — try sacct as last resort
                    sacct = _run_slurm_query(
                        ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
                    )
                    if sacct.returncode == 0 and sacct.stdout.strip():
                        for state in sacct.stdout.strip().split("\n"):
                            norm = state.strip().split()[0].rstrip("+")
                            if norm in terminal_states:
                                print(f"[slurm] Job {job_id} finished (via sacct fallback): {state.strip()}")
                                return norm
                    print(f"[slurm] Job {job_id}: {consecutive_query_failures} consecutive query "
                          f"failures, returning FAILED")
                    return "FAILED"
                print(f"[slurm] squeue query failed for job {job_id} "
                      f"(attempt {consecutive_query_failures}/10), retrying in {poll_interval}s...")
                time.sleep(poll_interval)
                continue

            # squeue succeeded — reset failure counter
            consecutive_query_failures = 0

            if sq.stdout.strip():
                # Job is still in the queue — keep waiting
                state = sq.stdout.strip().split("\n")[0].strip()
                # Normalize: "CANCELLED+" -> "CANCELLED", "CANCELLED by ..." -> "CANCELLED"
                norm_state = state.split()[0].rstrip("+")
                if norm_state not in terminal_states:
                    if timeout and (time.time() - start) > timeout:
                        print(f"[slurm] Job {job_id} timed out after {timeout}s")
                        return "TIMEOUT"
                    time.sleep(poll_interval)
                    continue
                # Terminal state detected in squeue; fall through to sacct

            # --- Phase 2: sacct (final state after job leaves queue) ---
            # Only reached when squeue SUCCEEDED (returncode==0) — either with a
            # terminal state or with empty output (job left queue).
            result = _run_slurm_query(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
            )

            if result.returncode == 0 and result.stdout.strip():
                states = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
                for state in states:
                    # Normalize: "CANCELLED by 224356" -> "CANCELLED", "CANCELLED+" -> "CANCELLED"
                    norm_state = state.split()[0].rstrip("+")
                    if norm_state in terminal_states:
                        print(f"[slurm] Job {job_id} finished: {state}")
                        return norm_state

            # sacct may not have data (e.g. accounting disabled).
            # If job is gone from squeue and sacct has no info, treat as COMPLETED
            # after a brief grace period for sacct propagation.
            if not sq.stdout.strip():
                # Job not in squeue — give sacct one more chance, then assume done
                time.sleep(5)
                result2 = _run_slurm_query(
                    ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
                )
                if result2.returncode == 0 and result2.stdout.strip():
                    for state in result2.stdout.strip().split("\n"):
                        norm_state = state.strip().split()[0].rstrip("+")
                        if norm_state in terminal_states:
                            print(f"[slurm] Job {job_id} finished: {state.strip()}")
                            return norm_state
                # sacct still empty — job vanished, assume completed
                print(f"[slurm] Job {job_id} left squeue (sacct unavailable), assuming COMPLETED")
                return "COMPLETED"

            if timeout and (time.time() - start) > timeout:
                print(f"[slurm] Job {job_id} timed out after {timeout}s")
                return "TIMEOUT"

            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Output reading
    # ------------------------------------------------------------------

    def read_outputs(self, out_dir: Path, labels: list[str]) -> dict[str, str]:
        """Read per-label output files from the output directory."""
        outputs = {}
        for label in labels:
            out_file = out_dir / f"{label}.out"
            if out_file.exists():
                outputs[label] = out_file.read_text()
            else:
                outputs[label] = f"[output file not found: {out_file}]"
        return outputs

    def read_exit_codes(self, out_dir: Path) -> dict[str, int]:
        """Read exit codes from the exit_codes.txt file."""
        exit_codes = {}
        exit_file = out_dir / "exit_codes.txt"
        if exit_file.exists():
            for line in exit_file.read_text().strip().split("\n"):
                if ":" in line:
                    label, code = line.rsplit(":", 1)
                    try:
                        exit_codes[label.strip()] = int(code.strip())
                    except ValueError:
                        pass
        return exit_codes

    def read_elapsed(self, out_dir: Path) -> dict[str, int]:
        """Read per-command elapsed seconds from elapsed.txt."""
        elapsed = {}
        elapsed_file = out_dir / "elapsed.txt"
        if elapsed_file.exists():
            for line in elapsed_file.read_text().strip().split("\n"):
                if ":" in line:
                    label, secs = line.rsplit(":", 1)
                    try:
                        elapsed[label.strip()] = int(secs.strip())
                    except ValueError:
                        pass
        return elapsed
