"""Bounded, explicit, pageable views for the writer (spec v2.1 P5, §7.3). Data is never cut silently."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

VIEW_BYTES = 8000
_LINE_CHARS = 300
_PATTERN_MAX = 500


def _is_binary(data: bytes) -> bool:
    head = data[:8192]
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
    except (
        UnicodeDecodeError
    ) as e:  # a multi-byte character cut at the 8 KiB edge is still text
        return e.start < len(head) - 4
    return False


def view(data: bytes, offset: int = 0, length: int = VIEW_BYTES) -> str:
    n = len(data)
    if _is_binary(data):
        return f"[binary: {n} bytes, sha256 {hashlib.sha256(data).hexdigest()}]"
    a = max(0, min(int(offset), n))
    b = min(n, a + max(1, int(length)))
    text = data[a:b].decode("utf-8", errors="replace")
    if b < n:
        text += f"\n[… shown bytes {a}–{b} of {n}; next: offset={b}]"
    return text


def resolve(vpath: str, roots: dict[str, Path]) -> Path:
    for prefix, root in roots.items():
        if vpath == prefix or vpath.startswith(prefix + "/"):
            real_root = Path(os.path.realpath(root))
            real = Path(os.path.realpath(Path(root) / vpath[len(prefix) :].lstrip("/")))
            if real == real_root or real_root in real.parents:
                return real
            break
    raise PermissionError("refused: outside the readable roots")


def read(
    vpath: str,
    roots: dict[str, Path],
    offset: int = 0,
    length: int = VIEW_BYTES,
) -> str:
    p = resolve(vpath, roots)
    if p.is_dir():
        names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
        return view("\n".join(names).encode(), offset, length)
    return view(p.read_bytes(), offset, length)


def _virtual(p: Path, roots: dict[str, Path]) -> str:
    for prefix, root in roots.items():
        real_root = Path(os.path.realpath(root))
        if p == real_root or real_root in p.parents:
            return prefix + "/" + str(p.relative_to(real_root))
    return str(p)


def grep(
    pattern: str,
    vpath: str,
    roots: dict[str, Path],
    offset: int = 0,
    max_hits: int = 50,
) -> str:
    if len(pattern) > _PATTERN_MAX:
        return f"refused: pattern longer than {_PATTERN_MAX} characters"
    rx = re.compile(pattern)
    base = resolve(vpath, roots)
    files = (
        [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file())
    )
    hits: list[str] = []
    for f in files:
        try:
            real = resolve(_virtual(Path(os.path.realpath(f)), roots), roots)
        except PermissionError:
            continue
        data = real.read_bytes()
        if _is_binary(data):
            continue
        for i, line in enumerate(
            data.decode("utf-8", errors="replace").splitlines(),
            1,
        ):
            if rx.search(line):
                hits.append(f"{_virtual(real, roots)}:{i}: {line[:_LINE_CHARS]}")
    a = max(0, int(offset))
    b = min(len(hits), a + max_hits)
    out = "\n".join(hits[a:b]) or "(no hits)"
    if b < len(hits):
        out += f"\n[… hits {a}–{b} of {len(hits)}; next: offset={b}]"
    return out
