"""Fixture provenance for the writer's library.

A fixture is a file under ``memory/<package>/tests/data/``. It reaches the library only through the writer's
``fixture`` tool, which copies recorded bytes and records where they came from. The gate re-derives every new or
changed fixture from the recording and refuses one that differs (:func:`verify`).

* **Sources** (:func:`sources_of`, :func:`source_bytes`):
  - an action's recorded response, as stored: a string as its UTF-8 text, anything else as canonical JSON;
  - a work-tree action's file blob;
  - the action's own arguments (``args:<i>``, a value the episode used);
  - each user message (``request:<j>``: the request, then the observations; ``observation:<i>`` names the same
    message as ``request:<i+1>``, the numbering ``read_episode`` uses);
  - a code cell's recorded output text (``cell:<i>``, ``Episode.cells[i].output``); a cell that raised, or that
    recorded no output, is not a source.
  The episode loader resolves capped responses, so a source is never the recorder's excerpt.
* **Copies** (``kind: copy``): the file equals a source, or a contiguous byte slice ``[a, b)`` of one.
* **Assembled** (``kind: assembled``, a ``.jsonl`` file): each line is ``{"input": <value>, "source": {...}}``.
  The value is the JSON-decoded slice, or its text when it is not JSON. A function's recorded-inputs file
  (:func:`inputs_file`) is assembled this way, and the gate appends its drawn inputs to a copy of it (:mod:`.qa`).
* **Cases** (``kind: case``, a ``.case.json`` file, :func:`make_case`): an input with the output the episode knew for
  it, ``{"input": <text>, "expected": <value>, "kind": "source" | "shown", "source": {...}}``. The input is a recorded
  text (any source, sliced as a copy is). The expected output is either what the request or the environment showed
  the actor (a message, ``request:<j>`` or ``observation:<i>``, including one received after answering, or an
  action's response or file), its JSON value or its text; or ``answer``, the value the episode handed back, which
  pairs only with the whole request it answered (``request:0``, no slice: a *source* case). Never an action's
  arguments or a cell's output (values the episode computed, not outcomes it was shown), and never a signal: a
  grader's verdict is not a source. The file names its sources, so it is its own ledger entry, and the gate
  re-derives it from the recording.
* **Trust**: a response, blob or message is trust 1 (the environment's own). Arguments and cell outputs are trust 2
  (a value an episode used or computed).
* **The ledger is the files** (:func:`derive_ledger`): an assembled file's lines name their own sources, and a copied
  file is listed in ``.pass/fixtures.json`` with its sha256. A file that is deleted, or whose bytes no longer match
  its listing, has no entry. Nothing else is kept, so the ledger cannot drift from what the writer sees; the gate
  re-derives every entry's bytes from the recording, so an edited listing proves nothing by itself.
* **Across passes** (:func:`ledger_for`): the gate keeps each verified entry by ``(path, sha256)``, so a fixture
  made in an earlier pass keeps its provenance; an older file is re-derived from its bytes where it can be, and a
  case file from the sources it names.
* **Views** (:func:`value_views`): the values a test can load from a fixture (its bytes, its text, its parsed value
  and every part of it, its lines and rows and columns), each by the digest an example call's argument has when it
  holds that value, with the episode and source it came from.

Pure: no git, no network, no clock. Reasons carry paths, episode ids, source ids and byte offsets, never a
recorded value.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat as _stat
from dataclasses import dataclass
from pathlib import Path

from . import safe_paths as _safe
from typing import Any, Callable

from .analysis import shapes as _shapes
from .episodes import Action, Episode
from .held_out import _blob_sha

DATA_PATH = re.compile(
    r"^memory/(?:[a-z_][a-z0-9_]*/)+tests/data/"
    r"(?:[A-Za-z0-9_][A-Za-z0-9._-]*/)*[A-Za-z0-9_][A-Za-z0-9._-]*\Z",
)
INPUTS_SUFFIX = ".inputs.jsonl"
#: A function's replay case file (``tests/data/<function>.replay.jsonl``): the inputs of the recorded instances its
#: replay verified, re-verified by replay rather than by bytes (:mod:`.replay_verify`).
REPLAY_SUFFIX = ".replay.jsonl"
_REPLAY_SOURCE = re.compile(r"replay:(\d{1,9}):(\d{1,9})-(\d{1,9})\Z")
DRAWN_DIR = "_drawn/"  # reserved under tests/data/ for the gate's appended drawn inputs
#: A case file (``tests/data/<name>.case.json``): an input with its known output (:func:`make_case`).
CASE_SUFFIX = ".case.json"
#: The expected-output source naming the value the episode handed back (:func:`.captures.handed_back`).
ANSWER = "answer"
#: A case's kind: the episode's own request with the value it handed back, or an input with the output the request
#: or the environment showed.
CASE_KINDS = ("source", "shown")
_INDEX = re.compile(r"\d{1,9}\Z")


class FixtureError(ValueError):
    """A fixture that cannot be made or verified (the reason names ids and offsets only)."""


def module_path(item: str) -> str:
    """``memory.office.payroll:export`` -> ``memory/office/payroll.py``."""
    return item.split(":", 1)[0].replace(".", "/") + ".py"


def package_dir(item: str) -> str:
    return module_path(item).rsplit("/", 1)[0]


def inputs_file(item: str) -> str:
    """The function's recorded-inputs file, e.g. ``memory/office/tests/data/payroll.export.inputs.jsonl``."""
    module, name = item.split(":", 1)
    return f"{package_dir(item)}/tests/data/{module.rsplit('.', 1)[1]}.{name}{INPUTS_SUFFIX}"


def is_replay_file(path: str) -> bool:
    return str(path).endswith(REPLAY_SUFFIX)


def replay_source(source: object) -> tuple[int, int, int] | None:
    """``replay:<cell>:<a>-<b>`` as ``(cell, a, b)``; None for any other source."""
    m = _REPLAY_SOURCE.match(str(source))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


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
    for i, c in enumerate(getattr(ep, "cells", None) or []):
        if not c.error and c.output:
            out.append(f"cell:{i}")
    return out


def trust_of(source: str) -> int:
    return 2 if str(source).startswith(("args:", "cell:", "replay:")) else 1


def _index(text: str, n: int, what: str) -> int:
    if not _INDEX.match(text):
        raise FixtureError(f"{what} {text[:20]!r} is not an index")
    i = int(text)
    if i >= n:
        raise FixtureError(f"{what} {i} is out of range ({n} recorded)")
    return i


#: The copied-fixture listing, relative to the writer's box (``/memory/.pass/fixtures.json``; never committed).
LEDGER_FILE = ".pass/fixtures.json"
_OBSERVATION = re.compile(r"observation:(\d{1,9})\Z")


def canonical_source(source: object) -> str:
    """``observation:<i>`` as the ``request:<i+1>`` it names; any other source id as it is."""
    s = str(source)
    m = _OBSERVATION.match(s)
    return f"request:{int(m.group(1)) + 1}" if m else s


def source_bytes(ep: Episode, source: object, blob: Callable[[str], bytes]) -> bytes:
    s = canonical_source(source)
    if s.startswith("request:"):
        return ep.request[_index(s[8:], len(ep.request), "request")].encode("utf-8")
    if s.startswith("args:"):
        a = ep.actions[_index(s[5:], len(ep.actions), "action")]
        return _json_bytes({"args": list(a.args), "kwargs": dict(a.kwargs or {})})
    if s.startswith("cell:"):
        cells = getattr(ep, "cells", None) or []
        i = _index(s[5:], len(cells), "cell")
        c = cells[i]
        if c.error:
            raise FixtureError(
                f"cell {i} of {ep.episode_id} raised; its output is not a recorded value",
            )
        if not isinstance(c.output, str) or not c.output:
            raise FixtureError(f"cell {i} of {ep.episode_id} has no recorded output")
        return c.output.encode("utf-8")
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
            f"slice [{a}, {b}] is outside the {len(data)} recorded bytes: offsets are bytes of the recorded "
            f"value, 0 <= start < end <= {len(data)}; [0, {len(data)}] or no slice copies all of it",
        )
    return data[a:b]


def _clamped(data: bytes, cut: object, notes: list[str] | None) -> object:
    """A slice whose end is null or left out (``[start, null]`` or ``[start]``) runs to the record's end. A well-formed
    slice that starts inside the record but ends past it is cut at the record's end, and *notes* says so. The clamped
    slice is the one recorded, so the gate re-derives the same bytes. Anything else is left to :func:`_cut`.
    """
    if (
        isinstance(cut, (list, tuple))
        and 1 <= len(cut) <= 2
        and isinstance(cut[0], int)
        and not isinstance(cut[0], bool)
        and (len(cut) == 1 or cut[1] is None)
    ):
        return [cut[0], len(data)]
    if (
        isinstance(cut, (list, tuple))
        and len(cut) == 2
        and all(isinstance(x, int) and not isinstance(x, bool) for x in cut)
        and 0 <= cut[0] < len(data) < cut[1]
    ):
        if notes is not None:
            notes.append(
                f"truncated: slice end {cut[1]} is past the {len(data)} recorded bytes; copied "
                f"[{cut[0]}, {len(data)}]",
            )
        return [cut[0], len(data)]
    return cut


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
    notes: list[str] | None = None,
    parser: str | None = None,
    parse: Callable[[str, str], tuple[str, str]] | None = None,
) -> tuple[bytes, dict]:
    """The bytes to write at *dest* (or, with *append*, the one line to append) and its new provenance entry.
    *notes* (when given) receives what was adjusted, e.g. a slice cut at the record's end.

    With *parser* (a stored reader's id), the recorded bytes, cut as usual and decoded as UTF-8, are given to the
    reader through *parse* (``parse(parser, text) -> (canonical JSON text, reader digest)``, which runs it
    confined): the file holds the JSON of its result (``kind: parsed``, a ``.json`` file), or with *append* one input
    line whose source names the parser.
    """
    if not DATA_PATH.match(dest) or dest.split("/tests/data/", 1)[1].startswith(
        DRAWN_DIR,
    ):
        raise FixtureError(
            f"{dest[:200]} is not a file name under memory/<package>/tests/data/",
        )
    source = canonical_source(source)
    data = source_bytes(ep, source, blob)
    cut = _clamped(data, cut, notes)
    piece = _cut(data, cut)
    where = {
        "episode": ep.episode_id,
        "source": source,
        "slice": list(cut) if cut is not None else None,
        "trust": trust_of(str(source)),
    }
    parsed: dict = {}
    if parser is not None:
        if parse is None:
            raise FixtureError("no reader can be run here")
        try:
            text = piece.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FixtureError("a reader takes text; these bytes are not UTF-8") from exc
        value_text, digest = parse(parser, text)
        parsed = {"parser": parser, "parser_sha256": digest}
        if not append:
            if not dest.endswith(".json"):
                raise FixtureError("a parsed fixture holds JSON: dest must end in .json")
            data = value_text.encode("utf-8")
            return data, {
                "kind": "parsed",
                **where,
                **parsed,
                "sha256": hashlib.sha256(data).hexdigest(),
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
    value = json.loads(value_text) if parser is not None else _decoded(piece)
    named = {k: where[k] for k in ("episode", "source", "slice")}
    if parser is not None:
        named["parser"] = parser
    line = json.dumps(
        {"input": value, "source": named},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    lines = list((existing or {}).get("lines", []))
    lines.append(
        {
            **where,
            **parsed,
            "sha256": hashlib.sha256(canonical(value).encode()).hexdigest(),
        },
    )
    return (line + "\n").encode("utf-8"), {"kind": "assembled", "lines": lines}


def is_case_file(path: object) -> bool:
    return str(path).endswith(CASE_SUFFIX)


def _side(raw: object, what: str) -> tuple[str, object]:
    """A case side as ``(canonical source, slice)``: a source id, or ``{"source": id, "slice": [a, b]}``."""
    if isinstance(raw, dict):
        unknown = sorted(set(raw) - {"source", "slice"})
        if unknown or raw.get("source") is None:
            raise FixtureError(f'the {what} is a source id or {{"source": <id>, "slice": [start, end]}}')
        return canonical_source(raw["source"]), raw.get("slice")
    if isinstance(raw, (str, int)) and not isinstance(raw, bool):
        return canonical_source(raw), None
    raise FixtureError(f'the {what} is a source id or {{"source": <id>, "slice": [start, end]}}')


def _shown_source(source: str) -> bool:
    """Whether *source* holds what the request or the environment showed the actor: a message, or an action's
    response or file (never its arguments, a cell's output or a replayed value)."""
    return source.startswith("request:") or bool(_INDEX.match(source))


def case_doc(
    ep: Episode,
    given: object,
    expected: object,
    *,
    blob: Callable[[str], bytes],
    notes: list[str] | None = None,
) -> dict:
    """The case file's value (module docstring, *Cases*): *given* names the input's recorded bytes and *expected* the
    output's, each a source id or ``{"source", "slice"}``; *expected* may be :data:`ANSWER`. Raises
    :class:`FixtureError` naming ids and offsets only."""
    in_src, in_cut = _side(given, "input")
    if in_src == ANSWER:
        raise FixtureError("a case's input is recorded bytes of a message or an action; answer names an output")
    data = source_bytes(ep, in_src, blob)
    in_cut = _clamped(data, in_cut, notes)
    try:
        text = _cut(data, in_cut).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FixtureError("a case's input is text; these bytes are not UTF-8") from exc
    out_src, out_cut = _side(expected, "expected output")
    if out_src == ANSWER:
        whole = in_cut is None or list(in_cut) == [0, len(data)]
        if out_cut is not None or in_src != "request:0" or not whole:
            raise FixtureError(
                "answer (the value the episode handed back) pairs only with the whole request it answered: input "
                "request:0 with no slice",
            )
        from .captures import handed_back

        found, value = handed_back(ep)
        if not found:
            raise FixtureError(f"episode {ep.episode_id} handed back no value")
        value = json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))
        kind, out_cut = "source", None
    else:
        if not _shown_source(out_src):
            raise FixtureError(
                f"source {out_src[:40]} is not something the request or the environment showed: an expected output "
                "comes from a message (request:<j>, observation:<i>), an action's response or file (<i>), or is "
                "answer, the value the episode handed back",
            )
        piece_src = source_bytes(ep, out_src, blob)
        out_cut = _clamped(piece_src, out_cut, notes)
        value = _decoded(_cut(piece_src, out_cut))
        kind = "shown"
    return {
        "input": text,
        "expected": value,
        "kind": kind,
        "source": {
            "episode": ep.episode_id,
            "input": {"source": in_src, "slice": list(in_cut) if in_cut is not None else None},
            "expected": {"source": out_src, "slice": list(out_cut) if out_cut is not None else None},
        },
    }


def case_bytes(doc: dict) -> bytes:
    """A case file's bytes: its value as sorted, indented JSON with a final newline."""
    return (json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def make_case(
    ep: Episode,
    given: object,
    expected: object,
    dest: str,
    *,
    blob: Callable[[str], bytes],
    notes: list[str] | None = None,
) -> tuple[bytes, dict]:
    """The bytes of the case file at *dest* (:func:`case_doc`) and its provenance entry (:func:`case_entry`)."""
    if not DATA_PATH.match(dest) or not is_case_file(dest) or dest.split("/tests/data/", 1)[1].startswith(DRAWN_DIR):
        raise FixtureError(f"{dest[:200]} is not a case file name under memory/<package>/tests/data/ ending {CASE_SUFFIX}")
    data = case_bytes(case_doc(ep, given, expected, blob=blob, notes=notes))
    entry = case_entry(data)
    assert entry is not None
    return data, entry


def case_entry(data: bytes) -> dict | None:
    """The provenance a case file states (``kind: case``: its episode, the input's source and slice, the expected
    output's, its case kind), or None when *data* is not a case file's value."""
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    src = doc.get("source") if isinstance(doc, dict) else None
    if (
        not isinstance(doc, dict)
        or set(doc) != {"input", "expected", "kind", "source"}
        or doc["kind"] not in CASE_KINDS
        or not isinstance(src, dict)
        or set(src) != {"episode", "input", "expected"}
        or not isinstance(src.get("episode"), str)
        or not all(isinstance(src[k], dict) and set(src[k]) == {"source", "slice"} for k in ("input", "expected"))
    ):
        return None
    return {
        "kind": "case",
        "case": doc["kind"],
        "episode": src["episode"],
        "source": str(src["input"]["source"]),
        "slice": src["input"]["slice"],
        "expected": {"source": str(src["expected"]["source"]), "slice": src["expected"]["slice"]},
        "trust": trust_of(str(src["input"]["source"])),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def read_case(
    path: str,
    data: bytes,
    load: Callable[[str], Episode | None],
    blob: Callable[[str], bytes],
) -> tuple[dict | None, str | None]:
    """``(the case, None)`` when the case file at *path* re-derives from the recording, its value
    ``{"input", "expected", "kind", "episode"}``; else ``(None, why)``."""
    if not is_case_file(path):
        return None, f"{path} is not a case file (a name ending {CASE_SUFFIX}, made with fixture(..., expected=...))"
    entry = case_entry(data)
    if entry is None:
        return None, f"{path} is not a case file made with fixture(..., expected=...)"
    problem = verify(path, data, entry, load, blob)
    if problem is not None:
        return None, problem
    doc = json.loads(data.decode("utf-8"))
    return {"input": doc["input"], "expected": doc["expected"], "kind": doc["kind"], "episode": entry["episode"]}, None


def _load(entry: dict, load: Callable[[str], Episode | None]) -> Episode:
    eid = entry.get("episode")
    ep = load(eid) if isinstance(eid, str) else None
    if ep is None:
        raise FixtureError(f"episode {str(eid)[:80]!r} cannot be read")
    return ep


def _parsed_text(
    entry: dict,
    load: Callable[[str], Episode | None],
    blob: Callable[[str], bytes],
    parse: Callable[[str, str], tuple[str, str]] | None,
) -> str:
    """What the entry's reader gives on its recorded bytes, as canonical JSON text."""
    if parse is None:
        raise FixtureError("it names a reader, which only the gate runs to re-derive it")
    piece = _cut(source_bytes(_load(entry, load), entry.get("source"), blob), entry.get("slice"))
    try:
        text = piece.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FixtureError("a reader takes text; the recorded bytes are not UTF-8") from exc
    return parse(entry.get("parser"), text)[0]


def readers_of(entry: object) -> list[str]:
    """The stored readers a provenance entry names: a parsed file's ``parser``, or its input lines'."""
    if not isinstance(entry, dict):
        return []
    rows = entry.get("lines") if isinstance(entry.get("lines"), list) else [entry]
    return sorted({r["parser"] for r in rows if isinstance(r, dict) and isinstance(r.get("parser"), str)})


REMAKE = "re-make it with fixture(..., parser=...)"


def verify(
    path: str,
    data: bytes,
    entry: object,
    load: Callable[[str], Episode | None],
    blob: Callable[[str], bytes],
    parse: Callable[[str, str], tuple[str, str]] | None = None,
) -> str | None:
    """None when *data* at *path* re-derives from the recording as *entry* (the harness's record) says.

    A parsed fixture (``kind: parsed``, or an input line whose source names a ``parser``) re-derives by running its
    reader again on the recorded bytes through *parse* (:func:`.procedures.reader_parse`); without *parse* it does
    not verify.
    """
    if not isinstance(entry, dict):
        return (
            f"{path} did not come through fixture(): no provenance is recorded for it"
        )
    try:
        if entry.get("kind") == "case":
            exp = entry.get("expected") if isinstance(entry.get("expected"), dict) else {}
            want = case_bytes(
                case_doc(
                    _load(entry, load),
                    {"source": entry.get("source"), "slice": entry.get("slice")},
                    {"source": exp.get("source"), "slice": exp.get("slice")},
                    blob=blob,
                ),
            )
            if want == data:
                return None
            cut = exp.get("slice")
            return (
                f"{path} differs from the case that {_where(entry)} with its expected output from source "
                f"{exp.get('source')}" + (f" slice {cut}" if cut else "") + " gives; make it again with "
                "fixture(..., expected=...)"
            )
        if entry.get("kind") == "parsed":
            want = _parsed_text(entry, load, blob, parse).encode("utf-8")
            return (
                None
                if want == data
                else f"{path} differs from what reader {entry.get('parser')} gives on the recorded bytes of "
                f"{_where(entry)}; {REMAKE}"
            )
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
                named = {k: row.get(k) for k in ("episode", "source", "slice")}
                if row.get("parser") is not None:
                    named["parser"] = row.get("parser")
                if got["source"] != named:
                    return (
                        f"{path} line {n} names another source than fixture() recorded"
                    )
                if row.get("parser") is not None:
                    want = json.loads(_parsed_text(row, load, blob, parse))
                    if canonical(want) != canonical(got["input"]):
                        return (
                            f"{path} line {n} differs from what reader {row.get('parser')} gives on the recorded "
                            f"bytes of {_where(row)}; {REMAKE}"
                        )
                    continue
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
    if isinstance(entry, dict) and entry.get("kind") == "replay":
        return f"{path} is a replay case file; it is verified by replaying its cases, not by its bytes"
    return f"{path} has an unknown provenance kind"


def assembled_entry(data: bytes) -> dict | None:
    """The provenance an assembled file's own lines state, or None when *data* is not a non-empty file of input
    lines (``{"input": value, "source": {"episode", "source", "slice"}}``)."""
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    rows = []
    for text in lines:
        try:
            got = json.loads(text)
        except ValueError:
            return None
        src = got.get("source") if isinstance(got, dict) else None
        if (
            not isinstance(got, dict)
            or set(got) != {"input", "source"}
            or not isinstance(src, dict)
            or set(src) - {"parser"} != {"episode", "source", "slice"}
        ):
            return None
        rows.append(
            {
                **src,  # a parsed line's source names its parser
                "trust": trust_of(str(src.get("source"))),
                "sha256": hashlib.sha256(canonical(got["input"]).encode()).hexdigest(),
            },
        )
    if not rows:
        return None
    replay = [replay_source(r.get("source")) is not None for r in rows]
    if all(replay):
        return {"kind": "replay", "lines": rows}
    return {"kind": "assembled", "lines": rows} if not any(replay) else None


def _read_plain(path, limit: int | None = None) -> bytes | None:
    """*path*'s bytes when it is a regular file reached through no link (None otherwise)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        if not _stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            if limit is not None and sum(map(len, chunks)) > limit:
                return None
        return b"".join(chunks)
    finally:
        os.close(fd)


def _no_link_on_the_way(box, rel: str) -> bool:
    """No directory on the way to *rel* in *box* is a link (checked from *box*'s descriptor, never followed)."""
    try:
        parts = _safe.rel_parts(rel)
        root = os.open(box, _safe.DIR_FLAGS)
    except OSError:
        return False
    try:
        fd = _safe.walk_dirs(root, parts[:-1])
    except _safe.Missing:
        return True
    except OSError:
        return False
    else:
        os.close(fd)
        return True
    finally:
        os.close(root)


def _read_in_box(box, rel: str, limit: int | None = None) -> bytes | None:
    """*rel*'s bytes in *box* when it is a regular file reached through no link (None otherwise), opened from *box*'s
    descriptor one component at a time (:mod:`.safe_paths`)."""
    try:
        return _safe.read_in(box, rel, limit)
    except (OSError, ValueError):
        return None


def read_listing(box) -> dict:
    """``.pass/fixtures.json`` in *box*: dest -> copied-fixture entry ({} when absent or unreadable)."""
    raw = _read_in_box(box, LEDGER_FILE, 16 * 1024 * 1024)
    try:
        val = json.loads(raw.decode("utf-8")) if raw is not None else {}
    except (UnicodeDecodeError, ValueError, RecursionError):
        return {}
    return val if isinstance(val, dict) else {}


def derive_ledger(box) -> dict[str, dict]:
    """The fixture provenance of the files now in *box*, derived from the files alone.

    * copied fixtures: each ``.pass/fixtures.json`` entry whose file still exists and still hashes to the listed
      sha256 (a deleted or rewritten file has no entry);
    * assembled fixtures: every ``.jsonl`` file under ``memory/**/tests/data/`` whose lines are all input lines, with
      the sources those lines name;
    * case files: every ``.case.json`` file there that is a case file's value, with the sources it names.

    Every file is opened from *box*'s descriptor without following a link (:mod:`.safe_paths`).
    """
    box = Path(box)
    out: dict[str, dict] = {}
    for dest, entry in sorted(read_listing(box).items()):
        if (
            not isinstance(dest, str)
            or not isinstance(entry, dict)
            or entry.get("kind") not in ("copy", "parsed")
            or not DATA_PATH.match(dest)
        ):
            continue
        data = _read_in_box(box, dest)
        if data is not None and hashlib.sha256(data).hexdigest() == entry.get("sha256"):
            keys = ("kind", "episode", "source", "slice", "trust", "sha256")
            if entry.get("kind") == "parsed":
                keys += ("parser", "parser_sha256")
            out[dest] = {k: entry.get(k) for k in keys}

    def wanted(vpath: str) -> bool:
        rel = vpath[len("/box/") :]
        return (
            (rel.endswith(".jsonl") or is_case_file(rel))
            and rel not in out
            and bool(DATA_PATH.match(rel))
            and not rel.split("/tests/data/", 1)[1].startswith(DRAWN_DIR)
        )

    try:
        with _safe.Roots({"/box": box}) as roots:
            for vpath, data in _safe.iter_files(roots, "/box/memory", want=wanted):
                entry = case_entry(data) if is_case_file(vpath) else assembled_entry(data)
                if entry is not None:
                    out[vpath[len("/box/") :]] = entry
    except OSError:  # /memory/memory is a link, or the box cannot be opened
        pass
    return out


def write_listing(box, dest: str, entry: dict | None) -> None:
    """Set (or, with None, remove) *dest*'s copied-fixture entry in ``.pass/fixtures.json``; never through a link:
    ``.pass`` is opened from *box*'s descriptor and the listing replaces the file by a rename (:mod:`.safe_paths`).
    """
    try:
        root = os.open(box, _safe.DIR_FLAGS)
    except OSError as exc:
        raise FixtureError(f"the box cannot be opened ({_safe.errno_name(exc)})") from None
    try:
        try:
            pfd = _safe.walk_dirs(root, [".pass"], create=True)
        except _safe.LinkFound:
            raise FixtureError("/memory/.pass is a link") from None
        try:
            try:
                raw = _safe.read_at(pfd, "fixtures.json", 16 * 1024 * 1024)
                listing = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeDecodeError, ValueError, RecursionError):
                listing = {}
            if not isinstance(listing, dict):
                listing = {}
            if entry is None:
                listing.pop(dest, None)
            else:
                listing[dest] = entry
            _safe.check_leaf(pfd, "fixtures.json", LEDGER_FILE)
            data = (json.dumps(listing, indent=1, sort_keys=True) + "\n").encode()
            _safe.replace_at(pfd, "fixtures.json", data)
        finally:
            os.close(pfd)
    finally:
        os.close(root)


class HashIndex:
    """SHA-256 of every recorded source of some episodes -> ``(episode, source)``."""

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


# --- views: the values a test can load from a fixture -----------------------------------------------------------

#: At most this many parts of one fixture's parsed value (subtrees, rows, columns) are views; past it a note says so.
MAX_VIEW_NODES = 5000
#: At most this many assembled files without a kept entry are re-verified from the recording per gate run.
MAX_REVERIFY = 32
_LINES_EXT = (".jsonl", ".ndjson")
_YAML_EXT = (".yaml", ".yml")


@dataclass(frozen=True)
class View:
    """One value a test can load from a fixture: ``name`` says which (``whole``, ``text``, ``value``,
    ``value:a.b[2]``, ``line:4``, ``line:4:input``, ``rows``, ``row:3``, ``column:amount``), and ``episode``,
    ``source`` and ``trust`` where it came from (per line for an assembled file, whose ``line`` is then set)."""

    name: str
    episode: str
    source: str
    trust: int
    line: int | None


def value_digest(value: Any) -> str:
    """SHA-256 of the canonical JSON of *value*: the digest an example call's argument has when it holds *value*."""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _origins(entry: dict) -> list[tuple[str, str, int, int | None]]:
    """(episode, source, trust, line) of a copy (line None) or of each assembled line (line from 1)."""

    def one(row: Any, line: int | None) -> tuple[str, str, int, int | None] | None:
        if not isinstance(row, dict) or not isinstance(row.get("episode"), str):
            return None
        src = canonical_source(row.get("source"))
        trust = row.get("trust")
        if not isinstance(trust, int) or isinstance(trust, bool):
            trust = trust_of(src)
        return row["episode"], src, trust, line

    if entry.get("kind") in ("copy", "parsed"):
        got = one(entry, None)
        return [got] if got else []
    if entry.get("kind") in ("assembled", "replay") and isinstance(entry.get("lines"), list):
        rows = [one(r, n) for n, r in enumerate(entry["lines"], 1)]
        return [r for r in rows if r is not None]
    return []


class _Views:
    def __init__(self, path: str, notes: list[str] | None) -> None:
        self.path = path
        self.notes = notes
        self.out: dict[str, list[View]] = {}
        self.parts = 0
        self.full = False

    def add(
        self,
        value: Any,
        name: str,
        who: list[tuple[str, str, int, int | None]],
        *,
        part: bool = True,
    ) -> bool:
        """Add *value* as view *name* for each origin in *who*; False once the parts bound is reached."""
        if part:
            if self.parts >= MAX_VIEW_NODES:
                if not self.full and self.notes is not None:
                    self.notes.append(
                        f"G2 examples: fixture {self.path} has more than {MAX_VIEW_NODES} parts; only the first "
                        f"{MAX_VIEW_NODES} are matched against arguments",
                    )
                self.full = True
                return False
            self.parts += 1
        try:
            digest = value_digest(value)
        except (TypeError, ValueError, RecursionError):
            return True
        views = self.out.setdefault(digest, [])
        for episode, source, trust, line in who:
            v = View(name, episode, source, trust, line)
            if v not in views:
                views.append(v)
        return True


def _walk(
    acc: _Views,
    value: Any,
    who: list[tuple[str, str, int, int | None]],
    per_item: list[tuple[str, str, int, int | None]] | None = None,
) -> None:
    """Every subtree of *value* as ``value`` / ``value:<path>``, breadth first. *per_item*: for a list of JSON
    lines whose origins differ per line, the origin of each top-level item."""
    queue: list[tuple[Any, str, list]] = [(value, "", who)]
    i = 0
    while i < len(queue):
        node, path, origin = queue[i]
        i += 1
        if not acc.add(node, f"value:{path}" if path else "value", origin):
            return
        if isinstance(node, dict):
            for k, v in node.items():
                queue.append((v, f"{path}.{k}" if path else str(k), origin))
        elif isinstance(node, list):
            for j, v in enumerate(node):
                o = (
                    [per_item[j]]
                    if per_item is not None and not path and j < len(per_item)
                    else origin
                )
                queue.append((v, f"{path}[{j}]", o))


def _table_views(
    acc: _Views,
    text: str,
    delimiter: str,
    header: bool,
    who: list,
) -> None:
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if not acc.add(rows, "rows", who):
        return
    first = rows[0] if rows else []
    body = rows[1:] if header else rows
    if header and not acc.add(body, "rows:body", who):
        return
    dicts: list[dict] = []
    if header:
        for r in body:
            if not r:
                continue
            dicts.append({k: (r[j] if j < len(r) else None) for j, k in enumerate(first)})
        if not acc.add(dicts, "rows:dicts", who):
            return
    for n, r in enumerate(rows, 1):
        if r and not acc.add(r, f"row:{n}", who):
            return
    if header:
        n0 = 2
        for d in dicts:
            while n0 <= len(rows) and not rows[n0 - 1]:
                n0 += 1
            if not acc.add(d, f"row:{n0}", who):
                return
            n0 += 1
    width = max((len(r) for r in rows), default=0)
    for j in range(width):
        name = first[j] if header and j < len(first) else f"#{j + 1}"
        col = [r[j] for r in body if j < len(r)]
        if not acc.add(col, f"column:{name}", who):
            return


def value_views(
    path: str,
    data: bytes,
    entry: dict | None,
    notes: list[str] | None = None,
) -> dict[str, list[View]]:
    """The values a test can load from the fixture at *path* (bytes *data*, provenance *entry*), as digest -> views.

    * the whole bytes as ``{"__bytes__": sha256}``, and the text as one string (UTF-8 only);
    * JSON: the parsed value and every subtree of it; JSON lines (``.jsonl``/``.ndjson``, or any assembled file):
      the list of parsed lines and every subtree, each line, and an assembled line's ``input``;
    * a table (``analysis.shapes`` format ``csv``): the rows as lists (all, and without the header), the rows as
      dicts keyed by the header, each row, and each column (by header name, or ``#<n>`` from 1) as a list;
    * YAML and any other text: the whole bytes and the text only; bytes that are not UTF-8: the whole bytes only.

    A file without an entry (no provenance) has no views: it grounds nothing. Parts past :data:`MAX_VIEW_NODES`
    are not views, and *notes* (when given) says so.
    """
    if not isinstance(entry, dict):
        return {}
    origins = _origins(entry)
    if not origins:
        return {}
    acc = _Views(path, notes)
    acc.add({"__bytes__": hashlib.sha256(data).hexdigest()}, "whole", origins, part=False)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return acc.out
    acc.add(text, "text", origins, part=False)
    ext = _shapes.extension(path)
    assembled = entry.get("kind") in ("assembled", "replay")
    replay = entry.get("kind") == "replay"
    if ext in _YAML_EXT:
        return acc.out
    if assembled or ext in _LINES_EXT:
        values: list[Any] = []
        numbers: list[int] = []
        for n, ln in enumerate(text.splitlines(), 1):
            if not ln.strip():
                continue
            try:
                values.append(json.loads(ln))
            except (ValueError, RecursionError):
                if notes is not None:
                    notes.append(
                        f"G2 examples: fixture {path} line {n} is not JSON; its lines are matched as text only",
                    )
                return acc.out
            numbers.append(n)
        per_line = [
            next((o for o in origins if o[3] == n), None) if assembled else None
            for n in numbers
        ]
        per_item = (
            [p if p is not None else origins[0] for p in per_line] if assembled else None
        )
        for k, (n, v) in enumerate(zip(numbers, values)):
            who = [per_item[k]] if per_item is not None else [o[:3] + (n,) for o in origins]
            acc.add(v, f"line:{n}", who, part=False)
            if assembled and isinstance(v, dict) and "input" in v:
                acc.add(v["input"], f"line:{n}:input", who, part=False)
                if replay and isinstance(v["input"], dict):
                    for param, value in v["input"].items():
                        got = _replay_value(value)
                        if got is not _UNNORMAL:
                            acc.add(got, f"line:{n}:input.{param}", who, part=False)
        _walk(acc, values, origins, per_item)
        return acc.out
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        shape = _shapes.shape(path, data)
        if shape.get("format") == "csv" and isinstance(shape.get("delimiter"), str):
            _table_views(acc, text, shape["delimiter"], bool(shape.get("header")), origins)
        return acc.out
    _walk(acc, parsed, origins)
    return acc.out


_UNNORMAL = object()


def normalise(value: Any, depth: int = 0) -> Any:
    """*value* as the JSON form an example call's argument digest is taken of: a tuple as a list; a set as the list
    of its normalised items in canonical order; bytes as ``{"__bytes__": sha256}``; a Decimal as its text; a date as
    its ISO text; an object with a plain ``__dict__`` as ``{"__type__": name, "vars": ...}`` (depth at most 8). Anything
    else gives a sentinel the caller treats as no digest."""
    import datetime as _dt
    import decimal as _decimal

    if depth > 8:
        return _UNNORMAL
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        items = [normalise(v, depth + 1) for v in value]
        return _UNNORMAL if any(i is _UNNORMAL for i in items) else items
    if isinstance(value, (set, frozenset)):
        items = [normalise(v, depth + 1) for v in value]
        if any(i is _UNNORMAL for i in items):
            return _UNNORMAL
        return sorted(items, key=canonical)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            got = normalise(v, depth + 1)
            if not isinstance(k, str) or got is _UNNORMAL:
                return _UNNORMAL
            out[k] = got
        return out
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, _decimal.Decimal):
        return str(value)
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        got = normalise(dict(attrs), depth + 1)
        return _UNNORMAL if got is _UNNORMAL else {"__type__": type(value).__name__, "vars": got}
    return _UNNORMAL


def _replay_value(encoded: Any) -> Any:
    """A replay case input (tagged JSON, :func:`.mining_behaviour.enc`) as the normalised value the call receives."""
    from .mining_behaviour import dec

    try:
        return normalise(dec(encoded))
    except (TypeError, ValueError, KeyError, RecursionError):
        return _UNNORMAL


# --- the recorded action behind a fixture -----------------------------------------------------------------------

_ACTION_SOURCE = re.compile(r"(?:args:)?(\d{1,9})\Z")


def source_action(
    row: dict,
    load: Callable[[str], Episode | None],
) -> tuple[str, int, Action] | None:
    """The recorded action behind a copy entry or an assembled line: ``(episode, index, action)`` for a source
    ``<i>`` (its response or file) or ``args:<i>`` (its arguments); None for a message or cell source, an unknown
    episode, or an index out of range."""
    if not isinstance(row, dict) or not isinstance(row.get("episode"), str):
        return None
    m = _ACTION_SOURCE.match(canonical_source(row.get("source")))
    if m is None:
        return None
    ep = load(row["episode"])
    if ep is None:
        return None
    i = int(m.group(1))
    if i >= len(ep.actions):
        return None
    return row["episode"], i, ep.actions[i]


# --- provenance across passes ------------------------------------------------------------------------------------


def ledger_for(
    files: dict[str, bytes],
    in_pass: dict[str, dict],
    stored: Callable[[str, str], dict | None],
    load: Callable[[str], Episode | None],
    blob: Callable[[str], bytes],
    episodes: list[str],
) -> tuple[dict[str, dict], list[str]]:
    """The provenance of each fixture in *files* (path -> bytes of the candidate), and notes.

    For each file, the first that applies:
    1. its entry in *in_pass* (G1 verified it in this pass);
    2. a kept entry with the same ``(path, sha256)`` (*stored*, the evidence store's ``fixture_provenance``);
    3. a case file that re-derives from the sources it names, or an assembled file whose lines re-derive from the
       recording now (:func:`verify`; at most :data:`MAX_REVERIFY` assembled files per call, the rest are noted);
    4. bytes equal to a whole recorded source of one of *episodes* (the episodes the package's items cite): a copy
       entry with ``"backfilled": true``, noted;
    5. none: the file grounds no argument, noted.
    """
    out: dict[str, dict] = {}
    notes: list[str] = []
    index: HashIndex | None = None
    reverified = 0
    for path in sorted(files):
        data = files[path]
        entry = in_pass.get(path)
        if isinstance(entry, dict):
            out[path] = entry
            continue
        sha = hashlib.sha256(data).hexdigest()
        kept = stored(path, sha)
        if isinstance(kept, dict):
            out[path] = kept
            continue
        case = case_entry(data) if is_case_file(path) else None
        if case is not None:
            if verify(path, data, case, load, blob) is None:
                out[path] = case
            else:
                notes.append(f"G2 examples: case file {path} does not re-derive from the recording; it grounds nothing")
            continue
        lines = assembled_entry(data) if path.endswith(".jsonl") else None
        if lines is not None and lines.get("kind") == "replay":
            notes.append(
                f"G2 examples: replay case file {path} was not re-verified by replay in this pass; it grounds no "
                "argument",
            )
            continue
        if lines is not None:
            if reverified < MAX_REVERIFY:
                reverified += 1
                if verify(path, data, lines, load, blob) is None:
                    out[path] = lines
                    continue
            else:
                notes.append(
                    f"G2 examples: fixture {path} was not re-verified (more than {MAX_REVERIFY} assembled files "
                    "without kept provenance in one gate run)",
                )
        if index is None:
            index = HashIndex()
            for eid in sorted(set(episodes)):
                ep = load(eid)
                if ep is not None:
                    index.add(ep, blob)
        hits = sorted(index.find(data))
        if hits:
            eid, src = hits[0]
            out[path] = {
                "kind": "copy",
                "episode": eid,
                "source": src,
                "slice": None,
                "trust": trust_of(src),
                "sha256": sha,
                "backfilled": True,
            }
            notes.append(
                f"G2 examples: fixture {path}'s provenance was re-derived from its bytes ({eid} {src})",
            )
            continue
        notes.append(
            f"G2 examples: fixture {path} has no recorded provenance (made before provenance was kept, and its "
            "bytes match no recorded source); it grounds no argument",
        )
    return out, notes
