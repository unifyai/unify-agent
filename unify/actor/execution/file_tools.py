"""``read_file`` and ``grep`` for the actor.

Both read only what a sandboxed shell cell could read (unify/sandbox.py):
``read_file`` checks the resolved path against the sandbox policy before
opening it, and ``grep`` runs ripgrep inside the sandbox itself, or, without
ripgrep, a small fixed Python script inside the sandbox (the harness's own
interpreter, isolated). The model's regular expression is never compiled or
matched in the harness process, and without bubblewrap ``grep`` is refused.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from unify import sandbox
from unify.common.tool_errors import ToolInputError

__all__ = ["grep", "read_file"]

MAX_READ_LINES = 2000
MAX_READ_BYTES = 256 * 1024
MAX_GREP_HITS = 1000
MAX_GREP_FILE_BYTES = 10 * 1024 * 1024
GREP_TIMEOUT_S = 60.0
_MAX_LINE_CHARS = 500
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(8192)
    except OSError:
        return False


def _resolve_file(path: str, policy: sandbox.SandboxPolicy) -> Path:
    resolved = sandbox.check_readable(path, policy)
    if not resolved.exists():
        raise ToolInputError(
            f"No such file: {resolved}",
            suggestion="Check the path; relative paths start at the workspace.",
            received={"path": path},
        )
    if not resolved.is_file():
        raise sandbox.SandboxRefusal(
            "regular-files-only",
            f"{resolved} is not a regular file",
            suggestion="List a directory from a shell cell (ls) instead.",
        )
    return resolved


def read_file(
    path: str,
    start: int = 1,
    end: Optional[int] = None,
    *,
    policy: sandbox.SandboxPolicy,
) -> dict[str, Any]:
    """Lines ``start``..``end`` (1-based, inclusive) of *path*, numbered."""
    resolved = _resolve_file(path, policy)
    if _is_binary(resolved):
        raise ToolInputError(
            f"{resolved} is a binary file",
            suggestion="Inspect it from a shell cell (for example with xxd or file).",
            received={"path": path},
        )
    start = max(1, int(start or 1))
    last = start + MAX_READ_LINES - 1
    end = last if end is None else min(int(end), last)
    if end < start:
        raise ToolInputError(
            f"end ({end}) is before start ({start})",
            suggestion="Pass end >= start, or omit end.",
            received={"path": path, "start": start, "end": end},
        )
    lines: list[str] = []
    size = 0
    total = 0
    truncated = False
    with open(resolved, encoding="utf-8", errors="replace") as fh:
        for number, line in enumerate(fh, start=1):
            total = number
            if number < start or number > end or truncated:
                continue
            text = f"{number:>6}\t{line.rstrip(chr(10))}\n"
            if size + len(text) > MAX_READ_BYTES:
                truncated = True
                continue
            lines.append(text)
            size += len(text)
    shown_end = start + len(lines) - 1 if lines else start - 1
    out: dict[str, Any] = {
        "path": str(resolved),
        "start": start,
        "end": shown_end,
        "total_lines": total,
        "content": "".join(lines),
    }
    if shown_end < min(end, total):
        out["next_start"] = shown_end + 1
    return out


async def grep(
    pattern: str,
    path: str = ".",
    max_hits: int = 100,
    *,
    policy: sandbox.SandboxPolicy,
) -> dict[str, Any]:
    """Lines matching the regular expression *pattern* under *path*."""
    target = sandbox.check_readable(path, policy)
    if not target.exists():
        raise ToolInputError(
            f"No such file or directory: {target}",
            suggestion="Check the path; relative paths start at the workspace.",
            received={"path": path},
        )
    max_hits = max(1, min(int(max_hits or 100), MAX_GREP_HITS))
    # Both engines run only inside the sandbox: without bubblewrap, refuse.
    sandbox.require_bwrap()
    rg = shutil.which("rg")
    if rg is not None and policy.readable_violation(Path(rg)) is None:
        hits, truncated = await _grep_ripgrep(rg, pattern, target, max_hits, policy)
        engine = "ripgrep"
    else:
        hits, truncated = await _grep_sandboxed_python(
            pattern,
            target,
            max_hits,
            policy,
        )
        engine = "python-sandboxed"
    return {
        "pattern": pattern,
        "path": str(target),
        "hits": hits,
        "truncated": truncated,
        "engine": engine,
    }


async def _grep_ripgrep(
    rg: str,
    pattern: str,
    target: Path,
    max_hits: int,
    policy: sandbox.SandboxPolicy,
) -> tuple[list[str], bool]:
    argv = [
        rg,
        "--no-heading",
        "--line-number",
        "--color",
        "never",
        "--no-config",
        # Ignore files inside the searched tree apply; those of an enclosing
        # checkout do not, so a workspace inside a repository is searchable.
        "--no-ignore-parent",
        "--max-columns",
        str(_MAX_LINE_CHARS),
        "--glob",
        "!.env",
        "--glob",
        "!.env.*",
        "-e",
        pattern,
        "--",
        str(target),
    ]
    with sandbox.unconfined():  # already wrapped; never wrap twice
        proc = await asyncio.create_subprocess_exec(
            *sandbox.wrap_argv(argv, policy, cwd=str(policy.workspace)),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sandbox.sandbox_env(policy),
            start_new_session=True,
        )
    hits: list[str] = []
    truncated = False

    async def collect() -> None:
        nonlocal truncated
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            if len(hits) >= max_hits:
                truncated = True
                return
            hits.append(line.decode("utf-8", errors="replace").rstrip("\n"))

    try:
        await asyncio.wait_for(collect(), timeout=GREP_TIMEOUT_S)
    except asyncio.TimeoutError:
        truncated = True
    finally:
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
        await proc.wait()
    if proc.returncode == 2 and not hits and not truncated:
        assert proc.stderr is not None
        err = (await proc.stderr.read()).decode("utf-8", errors="replace")
        raise ToolInputError(
            f"grep failed: {err.strip()[:2000]}",
            received={"pattern": pattern, "path": str(target)},
        )
    return hits, truncated


# The fallback's child: fixed code, run as ``python -I -S -c`` inside the
# sandbox. Its one argument is JSON data (pattern, target, limits); it writes
# one JSON object per line: ``{"hit": [file, line, text]}``, then
# ``{"done": true}``, or ``{"error": "invalid-regex", "detail": ...}``.
_GREP_CHILD = r"""
import json, os, re, sys
cfg = json.loads(sys.argv[1])
def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()
try:
    compiled = re.compile(cfg["pattern"])
except re.error as exc:
    emit({"error": "invalid-regex", "detail": str(exc)})
    sys.exit(3)
target, skip = cfg["target"], set(cfg["skip_dirs"])
def files():
    if os.path.isfile(target):
        yield target
        return
    for root, dirs, names in os.walk(target):
        dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith("."))
        for name in sorted(names):
            yield os.path.join(root, name)
for path in files():
    try:
        if not os.path.isfile(path) or os.stat(path).st_size > cfg["max_file_bytes"]:
            continue
        with open(path, "rb") as fh:
            if b"\0" in fh.read(8192):
                continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for number, line in enumerate(fh, start=1):
                if compiled.search(line):
                    text = line.rstrip("\n")[: cfg["max_line_chars"]]
                    emit({"hit": [path, number, text]})
    except OSError:
        continue
emit({"done": True})
"""


async def _grep_sandboxed_python(
    pattern: str,
    target: Path,
    max_hits: int,
    policy: sandbox.SandboxPolicy,
) -> tuple[list[str], bool]:
    """Search without ripgrep: :data:`_GREP_CHILD` under bubblewrap.

    The harness only parses the child's JSON lines and drops any hit in a
    path the policy refuses (a cell could not read it either); every
    directory walk, file read, compile and match happens in the child, which
    is killed (its process group) at the wall limit.
    """
    config = json.dumps(
        {
            "pattern": pattern,
            "target": str(target),
            "skip_dirs": sorted(_SKIP_DIRS),
            "max_file_bytes": MAX_GREP_FILE_BYTES,
            "max_line_chars": _MAX_LINE_CHARS,
        },
    )
    argv = [sys.executable, "-I", "-S", "-c", _GREP_CHILD, config]
    with sandbox.unconfined():  # already wrapped; never wrap twice
        proc = await asyncio.create_subprocess_exec(
            *sandbox.wrap_argv(argv, policy, cwd=str(policy.workspace)),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sandbox.sandbox_env(policy),
            start_new_session=True,
            limit=1 << 20,
        )
    hits: list[str] = []
    truncated = False
    done = False
    error: Optional[str] = None
    refused: dict[str, bool] = {}

    async def collect() -> None:
        nonlocal truncated, done, error
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if "error" in record:
                error = str(record.get("detail", ""))
                return
            if record.get("done"):
                done = True
                return
            file, number, text = record["hit"]
            if file not in refused:
                refused[file] = policy.readable_violation(Path(file)) is not None
            if refused[file]:
                continue
            if len(hits) >= max_hits:
                truncated = True
                return
            hits.append(f"{file}:{number}:{text}")

    timed_out = False
    try:
        await asyncio.wait_for(collect(), timeout=GREP_TIMEOUT_S)
    except asyncio.TimeoutError:
        timed_out = True
    finally:
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
        await proc.wait()
    if error is not None:
        raise ToolInputError(
            f"Invalid regular expression {pattern!r}: {error}",
            suggestion="Escape special characters, or pass a simpler pattern.",
            received={"pattern": pattern},
        )
    if timed_out:
        if hits:
            return hits, True
        raise ToolInputError(
            f"grep timed out after {GREP_TIMEOUT_S:g}s with no matching line; "
            "the search was stopped",
            suggestion=(
                "Use a simpler pattern (nested quantifiers such as (a+)+ can "
                "take exponential time), or search a smaller path."
            ),
            received={"pattern": pattern, "path": str(target)},
        )
    if not done and not truncated:
        assert proc.stderr is not None
        err = (await proc.stderr.read()).decode("utf-8", errors="replace")
        raise ToolInputError(
            f"grep failed: {err.strip()[:2000] or f'exit status {proc.returncode}'}",
            received={"pattern": pattern, "path": str(target)},
        )
    return hits, truncated
