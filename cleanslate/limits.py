"""Resource limits for a workspace, on top of bubblewrap's isolation.

Modes (bubblewrap is mandatory in every mode; these only add resource bounds):

  scope         systemd-run --user --scope with MemoryMax (+ MemorySwapMax=0), TasksMax and CPUQuota.
                The cgroup bounds total memory including the private /tmp (tmpfs pages are charged to
                it), the number of processes and threads, and the CPU share. After start the harness
                reads the scope's own cgroup files and refuses to continue if a required control is
                not in effect (for example when the cpu controller is not delegated to the user).
  prlimit-user  For hosts without a user systemd: per-uid RLIMIT_NPROC plus RLIMIT_CPU, set on the
                worker. RLIMIT_NPROC counts every process of the uid, so this mode only runs as a
                dedicated unprivileged user (named in `dedicated_user`) and refuses otherwise.
  rlimit-only   Only the per-process rlimits below. Allowed for local development; never in the
                worker profile.

In every mode, per process: RLIMIT_AS, RLIMIT_FSIZE, RLIMIT_NOFILE, RLIMIT_CORE=0 and RLIMIT_CPU
(total CPU seconds of the workspace process). After each cell the harness also measures the task
folder and kills the workspace if it is larger than `workdir_max_mb` (a check, not a hard bound: a
hard bound on disk needs the run folder on a size-limited filesystem, which the worker kit can
require with --max-fs-mb).
"""
from __future__ import annotations

import os
import pwd
import shutil
import subprocess
import functools
from dataclasses import dataclass, replace
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")


class LimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    mode: str = "auto"  # auto | scope | prlimit-user | rlimit-only
    memory_max_mb: int = 1024       # cgroup MemoryMax (scope mode): resident memory incl. the private /tmp
    address_space_mb: int = 2048    # RLIMIT_AS per process (virtual; every mode)
    tasks_max: int = 64
    cpu_quota_pct: int = 100
    cpu_seconds: int = 900
    file_size_mb: int = 64
    open_files: int = 256
    workdir_max_mb: int = 512
    require: tuple = ("memory", "pids")  # cgroup controls that must be verified in effect (scope mode)
    allowed_modes: tuple = ("scope", "prlimit-user", "rlimit-only")
    dedicated_user: str | None = None

    @staticmethod
    def worker(dedicated_user: str | None = None) -> "Limits":
        """Paid cells: scope with memory, pids and cpu all verified, or prlimit as a dedicated user."""
        return Limits(mode="auto", require=("memory", "pids", "cpu"), allowed_modes=("scope", "prlimit-user"),
                      dedicated_user=dedicated_user)

    @staticmethod
    def local() -> "Limits":
        """Laptop development: a scope when the user manager can make one (cpu is not delegated
        here, so it is not required); otherwise per-process rlimits only."""
        return Limits()

    def with_(self, **kw) -> "Limits":
        return replace(self, **kw)


@functools.lru_cache(maxsize=1)
def scope_available() -> bool:
    if not shutil.which("systemd-run"):
        return False
    try:
        done = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--collect", "/bin/true"],
                              env=systemd_env(), capture_output=True, timeout=20)
        return done.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def systemd_env() -> dict:
    """Only what systemd-run needs to reach the user manager (no secrets). bwrap --clearenv runs
    after it, so none of this reaches the workspace either."""
    uid = os.getuid()
    env = {"PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")}
    if os.environ.get("DBUS_SESSION_BUS_ADDRESS", "").startswith("unix:"):
        env["DBUS_SESSION_BUS_ADDRESS"] = os.environ["DBUS_SESSION_BUS_ADDRESS"]
    return env


def resolve_mode(limits: Limits) -> str:
    mode = limits.mode
    if mode == "auto":
        if scope_available():
            mode = "scope"
        elif limits.dedicated_user:
            mode = "prlimit-user"
        else:
            mode = "rlimit-only"
    if mode not in limits.allowed_modes:
        raise LimitError(f"resource-limit mode {mode!r} is not allowed here (allowed: {limits.allowed_modes}); "
                         "refusing to start")
    if mode == "prlimit-user":
        me = pwd.getpwuid(os.getuid()).pw_name
        if not limits.dedicated_user or me != limits.dedicated_user or os.getuid() == 0:
            raise LimitError(f"prlimit-user mode needs the dedicated unprivileged user {limits.dedicated_user!r}; "
                             f"running as {me!r}")
    return mode


def scope_prefix(limits: Limits, unit: str) -> list[str]:
    return ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}",
            "-p", f"MemoryMax={limits.memory_max_mb}M", "-p", "MemorySwapMax=0",
            "-p", f"TasksMax={limits.tasks_max}", "-p", f"CPUQuota={limits.cpu_quota_pct}%", "--"]


def preexec(limits: Limits, mode: str):
    def apply():
        import resource
        mb = 1 << 20
        resource.setrlimit(resource.RLIMIT_AS, (limits.address_space_mb * mb, limits.address_space_mb * mb))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_size_mb * mb, limits.file_size_mb * mb))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        if mode == "prlimit-user":
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.tasks_max, limits.tasks_max))
    return apply


def cgroup_of(pid: int) -> Path | None:
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return CGROUP_ROOT / line[3:].lstrip("/")
    except OSError:
        return None
    return None


def verify_scope(pid: int, limits: Limits, unit: str) -> dict:
    """Read the limits actually in effect from the workspace's own cgroup."""
    cg = cgroup_of(pid)
    if cg is None or not cg.name == f"{unit}.scope":
        raise LimitError(f"workspace process {pid} is not in scope {unit}.scope (cgroup {cg})")

    def read(name):
        try:
            return (cg / name).read_text().strip()
        except OSError:
            return None
    found = {"cgroup": str(cg), "memory.max": read("memory.max"), "memory.swap.max": read("memory.swap.max"),
             "pids.max": read("pids.max"), "cpu.max": read("cpu.max")}
    want = {"memory": ("memory.max", str(limits.memory_max_mb << 20)),
            "pids": ("pids.max", str(limits.tasks_max)),
            "cpu": ("cpu.max", f"{limits.cpu_quota_pct * 1000} 100000")}
    found["verified"] = sorted(k for k, (f, v) in want.items() if found[f] == v)
    missing = [k for k in limits.require if k not in found["verified"]]
    if missing:
        raise LimitError(f"required cgroup controls not in effect: {missing} (found {found}); refusing to start")
    return found


def stop_scope(unit: str, cgroup: str | None, timeout: float = 5.0) -> bool:
    """True once the scope's cgroup is gone (verified termination)."""
    import time
    deadline = time.monotonic() + timeout
    while cgroup and Path(cgroup).exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if cgroup and Path(cgroup).exists():
        subprocess.run(["systemctl", "--user", "stop", f"{unit}.scope"], env=systemd_env(),
                       capture_output=True, timeout=20)
        deadline = time.monotonic() + timeout
        while Path(cgroup).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    return not (cgroup and Path(cgroup).exists())


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, names in os.walk(path):
        for n in names:
            try:
                total += os.lstat(os.path.join(root, n)).st_blocks * 512
            except OSError:
                pass
    return total / (1 << 20)


_DEFAULT = [Limits.local()]


def set_default(limits: Limits) -> None:
    """The limits every workspace (live and replay) gets unless given others; the worker kit sets
    the worker profile here once, before anything starts."""
    _DEFAULT[0] = limits


def default() -> Limits:
    return _DEFAULT[0]
