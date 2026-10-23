"""Bounded, explicit, pageable views for the writer's read tools. Data is never cut silently."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

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


#: grep_bounded's child exit code for a pattern problem (the reason on stdout)
_EXIT_PATTERN = 3


class SearchFailed(RuntimeError):
    """The bounded grep's child ended in a way other than a pattern problem; the tool layer answers it with a fixed
    text (the child's own output can name host paths, so none of it is shown)."""


def _roots(roots):
    from . import safe_paths

    return safe_paths.using(roots)


def read_entry(
    vpath: str,
    roots,
    offset: int = 0,
    length: int = VIEW_BYTES,
) -> tuple[str, str]:
    """``(page, mode)``: the same page as ``view`` of the file's bytes, reading only the page (and a few bytes
    around it) from disk, mode ``read``; or a directory's listing, mode ``list``. *roots*: a
    :class:`.safe_paths.Roots` or a mapping of box prefixes to host directories. The path is opened from the root
    descriptor one component at a time, never through a link (:mod:`.safe_paths`)."""
    from . import safe_paths

    with _roots(roots) as r:
        fd, _, _ = safe_paths.open_path(r, vpath)
        try:
            if safe_paths.is_dir_fd(fd):
                names = safe_paths.listing(fd)
                return view("\n".join(names).encode(), offset, length), "list"
            return _read_fd(fd, offset, length), "read"
        finally:
            os.close(fd)


def read(
    vpath: str,
    roots,
    offset: int = 0,
    length: int = VIEW_BYTES,
) -> str:
    """:func:`read_entry`'s page."""
    return read_entry(vpath, roots, offset, length)[0]


def _read_fd(fd: int, offset: int, length: int) -> str:
    n = os.fstat(fd).st_size
    if _is_binary(os.pread(fd, 8192, 0)):
        sha = hashlib.sha256()
        at = 0
        while chunk := os.pread(fd, 1 << 20, at):
            sha.update(chunk)
            at += len(chunk)
        return _binary_line(n, sha.hexdigest())
    lo = max(0, min(int(offset), n))
    buf = os.pread(fd, VIEW_BYTES + 8, lo)
    a, b = _bounds(lambda i: buf[i - lo], min(n, lo + len(buf)), offset, length)
    if a == len(buf) + lo:  # the window held only continuation bytes past the end
        a = b = min(n, a)
    return _page(buf[a - lo : b - lo].decode("utf-8", errors="replace"), a, b, n)


def _compile(pattern: str) -> "re.Pattern[str]":
    if len(pattern) > _PATTERN_MAX:
        raise PatternProblem(f"pattern longer than {_PATTERN_MAX} characters")
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise PatternProblem(f"invalid pattern: {exc}"[:300]) from None


def _search(rx: "re.Pattern[str]", files, offset: int = 0, max_hits: int = 50) -> str:
    """The page of hits of *rx* in *files* (``(box path, bytes)`` pairs, in order) from hit *offset*."""
    hits: list[str] = []
    for vp, data in files:
        if _is_binary(data):
            continue
        at = 0
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


def _files(roots, vpath: str, deadline: float | None = None, want=None) -> list[tuple[str, bytes]]:
    """The regular files at or under *vpath*, opened from the root descriptors and never through a link; *want*: a
    box path filter applied before a file is read (``safe_paths.iter_files``)."""
    from . import safe_paths

    with _roots(roots) as r:
        return list(safe_paths.iter_files(r, vpath, want=want, deadline=deadline))


def grep(
    pattern: str,
    vpath: str,
    roots,
    offset: int = 0,
    max_hits: int = 50,
) -> str:
    """The page of hits of *pattern* in the files at or under *vpath*, in this process (see :func:`grep_bounded`)."""
    rx = _compile(pattern)
    return _search(rx, _files(roots, vpath), offset, max_hits)


def _frame(pattern: str, offset: int, files: list[tuple[str, bytes]]) -> bytes:
    """The child's input: one JSON header line (pattern, offset, each file's box path and size), then the bytes."""
    head = {
        "pattern": pattern,
        "offset": offset,
        "files": [[vp, len(data)] for vp, data in files],
    }
    return json.dumps(head).encode() + b"\n" + b"".join(data for _, data in files)


def _unframe(raw: bytes) -> tuple[str, int, list[tuple[str, bytes]]]:
    head, _, body = raw.partition(b"\n")
    job = json.loads(head)
    files, at = [], 0
    for vp, n in job["files"]:
        files.append((str(vp), body[at : at + int(n)]))
        at += int(n)
    return str(job["pattern"]), int(job["offset"]), files


def grep_bounded(
    pattern: str,
    vpath: str,
    roots,
    offset: int = 0,
    timeout_s: float | None = None,
    want=None,
) -> str:
    """:func:`grep` with the pattern run in a child Python process (``-I``; this file alone, standard library
    only), killed after *timeout_s* (default :data:`GREP_TIMEOUT_S`, which also bounds the walk). The writer's
    pattern never runs in the pass's own process. The host opens and reads the files from the root descriptors
    (:mod:`.safe_paths`); the child is handed their bytes and box paths, never a host path. *want*: a box path filter
    applied before a file is read (a file it refuses is never read or searched).
    """
    import time

    if len(pattern) > _PATTERN_MAX:  # a length check runs no pattern
        raise PatternProblem(f"pattern longer than {_PATTERN_MAX} characters")
    limit = GREP_TIMEOUT_S if timeout_s is None else timeout_s
    timed_out = f"[grep timed out after {limit:g} s: narrow the pattern or the path]"
    deadline = time.monotonic() + limit
    try:
        files = _files(roots, vpath, deadline, want)
    except TimeoutError:
        return timed_out
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-S", os.path.abspath(__file__)],
            input=_frame(pattern, offset, files),
            capture_output=True,
            timeout=max(0.1, deadline - time.monotonic()),
            env={},
        )
    except subprocess.TimeoutExpired:
        return timed_out
    if r.returncode == _EXIT_PATTERN:
        raise PatternProblem(r.stdout.decode("utf-8", errors="replace")[:300])
    if r.returncode != 0:
        raise SearchFailed("the search process failed")
    return r.stdout.decode("utf-8", errors="replace")


if __name__ == "__main__":  # grep_bounded's child: one job on stdin, the page on stdout
    _pattern, _offset, _files_in = _unframe(sys.stdin.buffer.read())
    try:
        _out = _search(_compile(_pattern), _files_in, _offset)
    except PatternProblem as _exc:
        sys.stdout.write(str(_exc))
        sys.exit(_EXIT_PATTERN)
    sys.stdout.write(_out)
