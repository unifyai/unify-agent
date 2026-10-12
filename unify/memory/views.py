"""Bounded, explicit, pageable views for the writer's read tools. Data is never cut silently."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

VIEW_BYTES = 8000
_LINE_CHARS = 300
_PATTERN_MAX = 500
#: grep runs in a child process bounded by this wall time: a pathological pattern never stalls the pass.
GREP_TIMEOUT_S = 10.0


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


def _cont(byte: int) -> bool:
    return byte & 0xC0 == 0x80  # a UTF-8 continuation byte


def _bounds(get, n: int, offset: int, length: int) -> tuple[int, int]:
    """The page [a, b) of an *n*-byte text: at most VIEW_BYTES long, its edges moved onto character boundaries
    (a forward, b back), so a page never shows half a character and the read log counts exactly what was shown.
    """
    a = max(0, min(int(offset), n))
    while a < n and _cont(get(a)):
        a += 1
    end = min(n, a + max(1, min(int(length), VIEW_BYTES)))
    b = end
    while a < b < n and _cont(get(b)):
        b -= 1
    if b == a and end > a:  # a single character longer than the page: show it whole
        b = end
        while b < n and _cont(get(b)):
            b += 1
    return a, b


def _page(text: str, a: int, b: int, n: int) -> str:
    if b < n:
        text += f"\n[… shown bytes {a}–{b} of {n}; next: offset={b}]"
    return text


def _binary_line(n: int, sha: str) -> str:
    return f"[binary: {n} bytes, sha256 {sha}]"


def view_range(
    data: bytes,
    offset: int = 0,
    length: int = VIEW_BYTES,
) -> tuple[str, int, int]:
    """A bounded view of *data* and the byte range [a, b) it shows (binary content: its size and sha256, and the
    whole range)."""
    n = len(data)
    if _is_binary(data):
        return _binary_line(n, hashlib.sha256(data).hexdigest()), 0, n
    a, b = _bounds(data.__getitem__, n, offset, length)
    return _page(data[a:b].decode("utf-8", errors="replace"), a, b, n), a, b


def view(data: bytes, offset: int = 0, length: int = VIEW_BYTES) -> str:
    return view_range(data, offset, length)[0]


class PatternProblem(ValueError):
    """A grep that cannot run as asked (its text says why); the tool layer answers it as a refusal."""


#: grep_bounded's child exit codes for a pattern problem and a path outside the roots (the reason on stdout)
_EXIT_PATTERN, _EXIT_OUTSIDE = 3, 4


def resolve(vpath: str, roots: dict[str, Path]) -> Path:
    for prefix, root in roots.items():
        if vpath == prefix or vpath.startswith(prefix + "/"):
            real_root = Path(os.path.realpath(root))
            real = Path(os.path.realpath(Path(root) / vpath[len(prefix) :].lstrip("/")))
            if real == real_root or real_root in real.parents:
                return real
            break
    raise PermissionError("outside the readable roots")


def read(
    vpath: str,
    roots: dict[str, Path],
    offset: int = 0,
    length: int = VIEW_BYTES,
) -> str:
    """The same page as ``view`` of the file's bytes, reading only the page (and a few bytes around it) from disk;
    a directory is listed."""
    p = resolve(vpath, roots)
    if p.is_dir():
        names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
        return view("\n".join(names).encode(), offset, length)
    n = p.stat().st_size
    with p.open("rb") as f:
        if _is_binary(f.read(8192)):
            f.seek(0)
            sha = hashlib.sha256()
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
            return _binary_line(n, sha.hexdigest())
        lo = max(0, min(int(offset), n))
        f.seek(lo)
        buf = f.read(VIEW_BYTES + 8)
        a, b = _bounds(lambda i: buf[i - lo], min(n, lo + len(buf)), offset, length)
        if a == len(buf) + lo:  # the window held only continuation bytes past the end
            a = b = min(n, a)
        return _page(buf[a - lo : b - lo].decode("utf-8", errors="replace"), a, b, n)


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
        raise PatternProblem(f"pattern longer than {_PATTERN_MAX} characters")
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise PatternProblem(f"invalid pattern: {exc}"[:300]) from None
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
        vp, at = _virtual(real, roots), 0
        for i, raw in enumerate(data.split(b"\n"), 1):
            line = raw.rstrip(b"\r").decode("utf-8", errors="replace")
            if rx.search(line):
                hit = f"{vp}:{i}: {line[:_LINE_CHARS]}"
                if len(line) > _LINE_CHARS:
                    hit += f"… [line {i}: {len(line)} chars, shown {_LINE_CHARS}; read({vp}, offset={at})]"
                hits.append(hit)
            at += len(raw) + 1
    a = max(0, int(offset))
    b, used = a, 0
    while b < len(hits) and b - a < max_hits:
        size = len(hits[b].encode()) + (1 if b > a else 0)
        if b > a and used + size > VIEW_BYTES:
            break
        used += size
        b += 1
    out = "\n".join(hits[a:b]) or "(no hits)"
    if b < len(hits):
        out += f"\n[… hits {a}–{b} of {len(hits)}; next: offset={b}]"
    return out


def grep_bounded(
    pattern: str,
    vpath: str,
    roots: dict[str, Path],
    offset: int = 0,
    timeout_s: float | None = None,
) -> str:
    """:func:`grep` in a child Python process (``-I``; this file alone, standard library only), killed after
    *timeout_s* (default :data:`GREP_TIMEOUT_S`). The writer's pattern never runs in the pass's own process.
    """
    limit = GREP_TIMEOUT_S if timeout_s is None else timeout_s
    job = {
        "pattern": pattern,
        "vpath": vpath,
        "roots": {k: str(r) for k, r in roots.items()},
        "offset": offset,
    }
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-S", os.path.abspath(__file__)],
            input=json.dumps(job).encode(),
            capture_output=True,
            timeout=limit,
            env={},
        )
    except subprocess.TimeoutExpired:
        return f"[grep timed out after {limit:g} s: narrow the pattern or the path]"
    if r.returncode != 0:
        last = " ".join(r.stderr.decode(errors="replace").strip().splitlines()[-1:])
        if r.returncode == _EXIT_OUTSIDE:
            raise PermissionError(r.stdout.decode("utf-8", errors="replace")[:300])
        if r.returncode == _EXIT_PATTERN:
            raise PatternProblem(r.stdout.decode("utf-8", errors="replace")[:300])
        raise PatternProblem(f"grep failed ({last})"[:300])
    return r.stdout.decode("utf-8", errors="replace")


if __name__ == "__main__":  # grep_bounded's child: one job on stdin, the page on stdout
    _job = json.loads(sys.stdin.read())
    try:
        _out = grep(
            _job["pattern"],
            _job["vpath"],
            {k: Path(r) for k, r in _job["roots"].items()},
            int(_job["offset"]),
        )
    except PermissionError as _exc:
        sys.stdout.write(str(_exc))
        sys.exit(_EXIT_OUTSIDE)
    except PatternProblem as _exc:
        sys.stdout.write(str(_exc))
        sys.exit(_EXIT_PATTERN)
    sys.stdout.write(_out)
