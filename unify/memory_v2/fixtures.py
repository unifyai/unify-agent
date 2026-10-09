"""Fixture provenance for memory v2.1 (spec §7.4, §9.1, §13.4).

A fixture is a file under ``memory/<package>/tests/data/``. It reaches the library only through the writer's
``fixture`` tool, which copies recorded bytes and records where they came from. The gate re-derives every new or
changed fixture from the recording and refuses one that differs (:func:`verify`).

* **Sources** (:func:`sources_of`, :func:`source_bytes`):
  - an action's recorded response, as stored: a string as its UTF-8 text, anything else as canonical JSON;
  - a work-tree action's file blob;
  - the action's own arguments (``args:<i>``, a value the episode used);
  - each user message (``request:<j>``: the request, then the observations).
  The episode loader resolves capped responses, so a source is never the recorder's excerpt.
* **Copies** (``kind: copy``): the file equals a source, or a contiguous byte slice ``[a, b)`` of one.
* **Assembled** (``kind: assembled``, a ``.jsonl`` file): each line is ``{"input": <value>, "source": {...}}``.
  The value is the JSON-decoded slice, or its text when it is not JSON. A function's recorded-inputs file
  (:func:`inputs_file`) is assembled this way, and the gate appends its drawn inputs to a copy of it (:mod:`.qa`).
* **Trust** (§13.1): a response, blob or message is trust 1 (the environment's own). Arguments are trust 2 (a
  value an episode used).

Pure: no git, no network, no clock. Reasons carry paths, episode ids, source ids and byte offsets, never a
recorded value.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

from .episodes import Episode
from .held_out import _blob_sha

DATA_PATH = re.compile(
    r"^memory/(?:[a-z_][a-z0-9_]*/)+tests/data/"
    r"(?:[A-Za-z0-9_][A-Za-z0-9._-]*/)*[A-Za-z0-9_][A-Za-z0-9._-]*\Z",
)
INPUTS_SUFFIX = ".inputs.jsonl"
DRAWN_DIR = "_drawn/"  # reserved under tests/data/ for the gate's appended drawn inputs
_INDEX = re.compile(r"\d{1,9}\Z")


class FixtureError(ValueError):
    """A fixture that cannot be made or verified (the reason names ids and offsets only)."""


def module_path(item: str) -> str:
    """``memory.office.payroll:export`` -> ``memory/office/payroll.py`` (spec §4.2)."""
    return item.split(":", 1)[0].replace(".", "/") + ".py"


def package_dir(item: str) -> str:
    return module_path(item).rsplit("/", 1)[0]


def inputs_file(item: str) -> str:
    """The function's recorded-inputs file, e.g. ``memory/office/tests/data/payroll.export.inputs.jsonl``."""
    module, name = item.split(":", 1)
    return f"{package_dir(item)}/tests/data/{module.rsplit('.', 1)[1]}.{name}{INPUTS_SUFFIX}"


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode(
        "utf-8",
    )


def sources_of(ep: Episode) -> list[str]:
    """Every source id of *ep*: its messages, then per action its response or file, then its arguments."""
    out = [f"request:{j}" for j in range(len(ep.request))]
    for i, a in enumerate(ep.actions):
        has = (
            _blob_sha(a) is not None if a.kind == "worktree" else a.response is not None
        )
        if has:
            out.append(str(i))
        out.append(f"args:{i}")
    return out


def trust_of(source: str) -> int:
    return 2 if str(source).startswith("args:") else 1


def _index(text: str, n: int, what: str) -> int:
    if not _INDEX.match(text):
        raise FixtureError(f"{what} {text[:20]!r} is not an index")
    i = int(text)
    if i >= n:
        raise FixtureError(f"{what} {i} is out of range ({n} recorded)")
    return i


def source_bytes(ep: Episode, source: object, blob: Callable[[str], bytes]) -> bytes:
    s = str(source)
    if s.startswith("request:"):
        return ep.request[_index(s[8:], len(ep.request), "request")].encode("utf-8")
    if s.startswith("args:"):
        a = ep.actions[_index(s[5:], len(ep.actions), "action")]
        return _json_bytes({"args": list(a.args), "kwargs": dict(a.kwargs or {})})
    a = ep.actions[_index(s, len(ep.actions), "action")]
    if a.kind == "worktree":
        sha = _blob_sha(a)
        if sha is None:
            raise FixtureError(f"action {s} of {ep.episode_id} recorded no file")
        try:
            return blob(sha)
        except (OSError, KeyError) as exc:
            raise FixtureError(
                f"the file of action {s} of {ep.episode_id} is not in the blob store",
            ) from exc
    if a.response is None:
        raise FixtureError(f"action {s} of {ep.episode_id} recorded no response")
    return (
        a.response.encode("utf-8")
        if isinstance(a.response, str)
        else _json_bytes(a.response)
    )


def _cut(data: bytes, cut: object) -> bytes:
    if cut is None:
        return data
    if not (
        isinstance(cut, (list, tuple))
        and len(cut) == 2
        and all(isinstance(x, int) and not isinstance(x, bool) for x in cut)
    ):
        raise FixtureError("slice must be [start, end] byte offsets")
    a, b = cut
    if not 0 <= a < b <= len(data):
        raise FixtureError(
            f"slice [{a}, {b}] is outside the {len(data)} recorded bytes",
        )
    return data[a:b]


def _decoded(piece: bytes) -> Any:
    try:
        text = piece.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FixtureError(
            "an assembled line needs text; these bytes are not UTF-8",
        ) from exc
    try:
        return json.loads(text)
    except ValueError:
        return text


def _where(entry: dict) -> str:
    cut = entry.get("slice")
    return f"episode {entry.get('episode')} source {entry.get('source')}" + (
        f" slice {cut}" if cut else ""
    )


def make(
    ep: Episode,
    source: object,
    dest: str,
    *,
    blob: Callable[[str], bytes],
    cut: object = None,
    append: bool = False,
    existing: dict | None = None,
) -> tuple[bytes, dict]:
    """The bytes to write at *dest* (or, with *append*, the one line to append) and its new provenance entry."""
    if not DATA_PATH.match(dest) or dest.split("/tests/data/", 1)[1].startswith(
        DRAWN_DIR,
    ):
        raise FixtureError(
            f"{dest[:200]} is not a file name under memory/<package>/tests/data/",
        )
    piece = _cut(source_bytes(ep, source, blob), cut)
    where = {
        "episode": ep.episode_id,
        "source": str(source),
        "slice": list(cut) if cut is not None else None,
        "trust": trust_of(str(source)),
    }
    if not append:
        return piece, {
            "kind": "copy",
            **where,
            "sha256": hashlib.sha256(piece).hexdigest(),
        }
    if not dest.endswith(".jsonl"):
        raise FixtureError("append builds a JSON-lines file: dest must end in .jsonl")
    if existing is not None and existing.get("kind") != "assembled":
        raise FixtureError(f"{dest} holds a copied fixture; it cannot be appended to")
    value = _decoded(piece)
    named = {k: where[k] for k in ("episode", "source", "slice")}
    line = json.dumps(
        {"input": value, "source": named},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    lines = list((existing or {}).get("lines", []))
    lines.append(
        {**where, "sha256": hashlib.sha256(canonical(value).encode()).hexdigest()},
    )
    return (line + "\n").encode("utf-8"), {"kind": "assembled", "lines": lines}


def _load(entry: dict, load: Callable[[str], Episode | None]) -> Episode:
    eid = entry.get("episode")
    ep = load(eid) if isinstance(eid, str) else None
    if ep is None:
        raise FixtureError(f"episode {str(eid)[:80]!r} cannot be read")
    return ep


def verify(
    path: str,
    data: bytes,
    entry: object,
    load: Callable[[str], Episode | None],
    blob: Callable[[str], bytes],
) -> str | None:
    """None when *data* at *path* re-derives from the recording as *entry* (the harness's record) says."""
    if not isinstance(entry, dict):
        return (
            f"{path} did not come through fixture(): no provenance is recorded for it"
        )
    try:
        if entry.get("kind") == "copy":
            want = _cut(
                source_bytes(_load(entry, load), entry.get("source"), blob),
                entry.get("slice"),
            )
            return (
                None
                if want == data
                else f"{path} differs from the recorded bytes of {_where(entry)}"
            )
        if entry.get("kind") == "assembled":
            lines = data.decode("utf-8").splitlines()
            rows = entry.get("lines") if isinstance(entry.get("lines"), list) else []
            if len(rows) != len(lines):
                return f"{path} has {len(lines)} lines; fixture() recorded {len(rows)}"
            for n, (text, row) in enumerate(zip(lines, rows), 1):
                got = json.loads(text)
                if not isinstance(got, dict) or set(got) != {"input", "source"}:
                    return f"{path} line {n} is not an input line"
                if got["source"] != {
                    k: row.get(k) for k in ("episode", "source", "slice")
                }:
                    return (
                        f"{path} line {n} names another source than fixture() recorded"
                    )
                want = _decoded(
                    _cut(
                        source_bytes(_load(row, load), row.get("source"), blob),
                        row.get("slice"),
                    ),
                )
                if canonical(want) != canonical(got["input"]):
                    return f"{path} line {n} differs from the recorded value of {_where(row)}"
            return None
    except FixtureError as exc:
        return f"{path}: {exc}"
    except (UnicodeDecodeError, ValueError) as exc:
        return f"{path} cannot be read as recorded lines ({type(exc).__name__})"
    return f"{path} has an unknown provenance kind"


class HashIndex:
    """SHA-256 of every recorded source of some episodes -> ``(episode, source)`` (spec §9.1)."""

    def __init__(self) -> None:
        self.by_sha: dict[str, list[tuple[str, str]]] = {}

    def add(self, ep: Episode, blob: Callable[[str], bytes]) -> None:
        for s in sources_of(ep):
            try:
                data = source_bytes(ep, s, blob)
            except FixtureError:
                continue
            self.by_sha.setdefault(hashlib.sha256(data).hexdigest(), []).append(
                (ep.episode_id, s),
            )

    def find(self, data: bytes) -> list[tuple[str, str]]:
        return list(self.by_sha.get(hashlib.sha256(data).hexdigest(), []))
