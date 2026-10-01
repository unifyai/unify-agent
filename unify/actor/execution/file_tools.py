"""``read_file`` and ``grep`` for the actor under ``UNIFY_WORKSPACE=sandboxed``.

Both read only what a sandboxed shell cell could read (unify/sandbox.py):
``read_file`` checks the resolved path against the sandbox policy before
opening it, and ``grep`` runs ripgrep inside the sandbox itself, or, without
ripgrep, walks the tree in Python applying the same check to every entry.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
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
    rg = shutil.which("rg")
    if (
        rg is not None
        and sandbox.bwrap_path() is not None
        and policy.readable_violation(Path(rg)) is None
    ):
        hits, truncated = await _grep_ripgrep(rg, pattern, target, max_hits, policy)
        engine = "ripgrep"
    else:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ToolInputError(
                f"Invalid regular expression {pattern!r}: {exc}",
                suggestion="Escape special characters, or pass a simpler pattern.",
                received={"pattern": pattern},
            ) from exc
        hits, truncated = await asyncio.to_thread(
            _grep_python,
            compiled,
            target,
            max_hits,
            policy,
        )
        engine = "python"
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


def _grep_python(
    compiled: re.Pattern,
    target: Path,
    max_hits: int,
    policy: sandbox.SandboxPolicy,
) -> tuple[list[str], bool]:
    hits: list[str] = []

    def files():
        if target.is_file():
            yield target
            return
        for root, dirs, names in os.walk(target):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in _SKIP_DIRS
                and not d.startswith(".")
                and policy.readable_violation(Path(root) / d) is None
            )
            for name in sorted(names):
                yield Path(root) / name

    for file in files():
        if policy.readable_violation(file) is not None or not file.is_file():
            continue
        try:
            if file.stat().st_size > MAX_GREP_FILE_BYTES or _is_binary(file):
                continue
            with open(file, encoding="utf-8", errors="replace") as fh:
                for number, line in enumerate(fh, start=1):
                    if compiled.search(line):
                        if len(hits) >= max_hits:
                            return hits, True
                        text = line.rstrip("\n")[:_MAX_LINE_CHARS]
                        hits.append(f"{file}:{number}:{text}")
        except OSError:
            continue
    return hits, False
