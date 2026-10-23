"""A pass's agent transcripts, stored whole: every message, never cut, in pageable chunk files.

Each message is one redacted JSON line (:func:`.writer_pass.transcript_lines`). The lines go in order to chunk files
under the pass's record folder, ``<pass_record_root>/<pass_id>/``: the writer's as ``transcript-<n>.jsonl`` with
the list ``transcript.json``, another agent's (a helper the writer launched) as ``transcript-<agent>-<n>.jsonl``
with ``transcript-<agent>.json``. A chunk holds about :data:`CHUNK_BYTES`; a longer line is a chunk of its own,
whole. The list names each chunk with its sha256, byte count and line count, so a reader can page through the
transcript and check each page.

The git note of the writer's transcript (:func:`note_lines`) carries the lines themselves while they fit
:data:`NOTE_BYTES`, else one row naming the chunks and their digests. A transcript that cannot be written to files
goes into the note whole. No message is ever dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

#: about this many bytes per chunk file (a page, never a bound: a longer line is a chunk of its own)
CHUNK_BYTES = 1024 * 1024
#: the git note carries the transcript's lines while they fit this many bytes, else a pointer row
NOTE_BYTES = 256 * 1024
WRITER = "writer"
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")


def _prefix(agent: str) -> str:
    return "transcript" if agent == WRITER else f"transcript-{agent}"


def _check_name(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _NAME.match(value) or ".." in value:
        raise ValueError(
            f"a transcript's {kind} must be letters, digits, '_', '.' or '-', not {value!r}",
        )
    return value


def _folder(root: str | os.PathLike, pass_id: str) -> Path:
    """``<root>/<pass_id>``, made when missing; refused when either is a link or not a directory."""
    folder = Path(root) / _check_name("pass id", pass_id)
    for d in (Path(root), folder):
        d.mkdir(parents=True, exist_ok=True)
        if d.is_symlink() or not d.is_dir():
            raise OSError(f"{d} is not a plain directory")
    return folder


def _write_new(path: Path, data: bytes) -> None:
    """*data* to a regular file created new (never through a link, never over an existing file)."""
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o644,
    )
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _pages(lines: list[str], chunk_bytes: int) -> Iterable[list[str]]:
    page: list[str] = []
    size = 0
    for line in lines:
        n = len(line.encode("utf-8")) + 1
        if page and size + n > chunk_bytes:
            yield page
            page, size = [], 0
        page.append(line)
        size += n
    if page:
        yield page


def write_chunks(
    root: str | os.PathLike,
    pass_id: str,
    agent: str,
    lines: list[str],
    *,
    chunk_bytes: int = CHUNK_BYTES,
) -> dict:
    """Write *lines* (JSON lines, already redacted) whole, in order, as chunk files under ``<root>/<pass_id>/``,
    then the list of chunks, and return that list (the manifest)::

        {"pass_id", "agent", "lines", "bytes", "manifest", "chunks": [{"name", "sha256", "bytes", "lines"}]}

    Every file is created new: a transcript already recorded for this pass and agent is never written over (an
    :class:`OSError`, as for a folder that is a link). The list is written last, so a list names only complete
    chunks."""
    agent = _check_name("agent", agent)
    folder = _folder(root, pass_id)
    prefix = _prefix(agent)
    chunks, total, count = [], 0, 0
    for n, page in enumerate(_pages(list(lines), chunk_bytes)):
        data = ("\n".join(page) + "\n").encode("utf-8")
        name = f"{prefix}-{n}.jsonl"
        _write_new(folder / name, data)
        chunks.append(
            {
                "name": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "lines": len(page),
            },
        )
        total += len(data)
        count += len(page)
    manifest = {
        "pass_id": pass_id,
        "agent": agent,
        "lines": count,
        "bytes": total,
        "manifest": f"{prefix}.json",
        "chunks": chunks,
    }
    _write_new(
        folder / manifest["manifest"],
        (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8"),
    )
    return manifest


def record_transcript(
    root: str | os.PathLike,
    pass_id: str,
    agent: str,
    messages: list,
    *,
    redactor: Any = None,
    tools: list | None = None,
    originals: Mapping[int, str] | None = None,
    store: Callable[[bytes], str] | None = None,
    label: Mapping[str, str] | None = None,
) -> dict:
    """Store an agent's whole conversation *messages* for pass *pass_id*: each message redacted as one JSON line
    (:func:`.writer_pass.transcript_lines`, with its *redactor*, *tools*, *originals*, *store* and *label*), none
    cut or dropped, in chunk files under ``<root>/<pass_id>/`` (:func:`write_chunks`). The one way every agent of
    a pass (the writer, the curator, each helper) stores its transcript. Returns the manifest; when the files
    cannot be written, ``{"pass_id", "agent", "lines", "error"}`` naming the error's class.
    """
    from .writer_pass import transcript_lines

    lines = transcript_lines(
        messages,
        pass_id,
        redactor,
        tools=tools,
        originals=originals,
        store=store,
        label=label,
    )
    try:
        return write_chunks(root, pass_id, agent, lines)
    except (OSError, ValueError) as exc:
        return {
            "pass_id": pass_id,
            "agent": agent,
            "lines": len(lines),
            "error": type(exc).__name__,
        }


def note_lines(
    pass_id: str,
    lines: list[str],
    manifest: dict | None,
    *,
    note_bytes: int = NOTE_BYTES,
) -> list[str]:
    """The git note of a transcript: its *lines* while they fit *note_bytes* (or when no chunk files hold them,
    *manifest* None: then whole, whatever their size), else one pointer row ``{"pass_id", "transcript": {...}}``
    naming the folder under the pass record root, the list file and each chunk with its sha256.
    """
    size = sum(len(ln.encode("utf-8")) + 1 for ln in lines)
    if manifest is None or size <= note_bytes:
        return list(lines)
    pointer = {
        "folder": pass_id,
        **{k: manifest[k] for k in ("agent", "lines", "bytes", "manifest", "chunks")},
    }
    return [json.dumps({"pass_id": pass_id, "transcript": pointer}, sort_keys=True)]
