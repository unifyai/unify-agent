"""The writer's ``land_capture`` tool: turn one captured attempt into library items the writer can change.

A capture (:mod:`.captures`) is a completed attempt's working procedure, stored by the harness outside every box. The
writer reads it under ``/inputs/captures/<id>/``; this tool reads it again from the harness's own store (never from
a path the box names), checks it, and writes into the writer's ``/memory``:

- the reader, ``memory/<package>/readers.py``: the capture's ``read_input`` as ``read_<name>``, unless a reader the
  library already holds gives the same value on this request (each candidate is run confined, as the gate runs a
  reader: :func:`.procedures.run_reader`), which is then reused;
- the procedure, ``memory/<package>/procedures.py``: the capture's ``run`` as ``<name>``, with its ``applies`` and
  ``holds`` (when present) and every other top-level name of the capture kept private under ``_<name>_``;
- two fixtures of the request text: the text itself and the reader's value on it (a parsed fixture,
  ``fixture(..., parser=...)``), and a fixture of the handed-back value, a slice of the recorded arguments of the
  dialogue action that handed it back (``args:<i>``), all made by the fixture tool, so the gate re-derives them from
  the recording;
- a test, ``memory/<package>/tests/test_<name>.py``: the procedure on the reader's value of the recorded text against
  the handed-back value's fixture, the reader on the recorded text, and one refusal of text the reader was not built
  for. When the value was handed back in the final reply, which no fixture source holds, the expected value is
  written in the test instead, and the gate keeps the procedure ``unchecked``;
- the manifest entries, the procedure with ``role: "procedure"`` and ``reader``, a new reader with ``role:
  "reader"``.

When the capture's ``same_as`` names a capture already landed (its marker line is in a ``procedures.py`` of the
library), this request becomes a new case of that procedure instead: two fixtures and one test function more.

Checks, each a fixed refusal: the id is 16 hex digits (optionally ``capture:``-prefixed) and in the store; the store's
files are read without following a link, only as regular files, size-bounded; ``procedure.py``'s sha256 starts with
the id; the capture is not failed and not retired; the gate's safety checks hold (:func:`.captures.safety`); nothing
landed holds a key-shaped string or a URL with a password; *package* and *name* are plain module identifiers. Every
file is written through the writer's ``write`` tool (:func:`.edit_tools.call`: the same path checks, never through a
link) and the fixture tool; the caller then measures the quota as after any write. Model-written code runs only in
:func:`.sandbox_run.run_confined`; on the host it is only parsed as data.

Not in this build: the ``episode`` cover of the ``request`` runner (design I.3.5) on the procedure's entry.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import keyword
import os
import re
import shutil
import stat
import tempfile
import tokenize
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import captures as _captures
from . import edit_tools as _et
from . import fixtures as _fx
from . import safe_paths as _safe
from . import writer_tools as _wt
from .redact import KEY_SHAPED
from .sandbox_run import PYTHON, SandboxResult, run_confined

TOOL = "land_capture"
#: A capture id as the tool takes it: the store's 16 hex digits, optionally prefixed ``capture:``.
_ARG_ID = re.compile(r"(?:capture:)?([0-9a-f]{16})\Z")
#: A package or procedure name: a plain module identifier.
_IDENT = re.compile(r"[a-z][a-z0-9_]{0,47}\Z")
#: The marker line a landed procedure carries in its module, naming the capture and the reader it was landed with.
_MARK = re.compile(
    r"^# Landed from capture ([0-9a-f]{16}): procedure (memory\.[a-z][a-z0-9_]*\.procedures:[a-z][a-z0-9_]*), "
    r"reader (memory\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*:[A-Za-z][A-Za-z0-9_]*)$",
    re.M,
)
_MAX_RECORD_BYTES = 4 * 1024 * 1024
_MAX_MODULE_BYTES = 1024 * 1024


class _NotLanded(Exception):
    def __init__(self, form: str, **fields: object) -> None:
        super().__init__(form)
        self.form, self.fields = form, fields


# --- reading the store ------------------------------------------------------------------------------------------


def _store_file(store: Path, cid: str, name: str, limit: int) -> bytes | None:
    """A capture file's bytes: a regular file reached without a link (the capture dir and the file), at most *limit*
    bytes; None otherwise."""
    try:
        dfd = os.open(store / cid, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError:
            return None
        with os.fdopen(fd, "rb") as fh:
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                return None
            data = fh.read(limit + 1)
    finally:
        os.close(dfd)
    return data if len(data) <= limit else None


def _looks_like_key(text: str) -> bool:
    from unify.sandbox import _looks_like_key as sandbox_key

    return bool(KEY_SHAPED.search(text)) or sandbox_key(text)


def read_capture(store: Path, raw_id: object) -> tuple[str, str, dict]:
    """``(capture id, procedure source, record)`` from the harness's store, checked; raises :class:`_NotLanded`."""
    m = _ARG_ID.match(raw_id) if isinstance(raw_id, str) else None
    if m is None:
        raise _NotLanded("unknown_capture", capture_id=str(raw_id)[:40])
    cid = m.group(1)
    src = _store_file(store, cid, _captures.PROCEDURE_FILE, _captures.MAX_SOURCE_BYTES)
    rec_raw = _store_file(store, cid, _captures.RECORD_FILE, _MAX_RECORD_BYTES)
    if src is None or rec_raw is None:
        raise _NotLanded("unknown_capture", capture_id=cid)
    if hashlib.sha256(src).hexdigest()[:16] != cid:
        raise _NotLanded("not_landed", reason=f"capture {cid}'s procedure does not match its id")
    try:
        source = src.decode("utf-8")
        record = json.loads(rec_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _NotLanded("not_landed", reason=f"capture {cid} cannot be read") from None
    if not isinstance(record, dict):
        raise _NotLanded("not_landed", reason=f"capture {cid} cannot be read")
    if record.get("outcome") == _captures.FAILED or record.get("status") == "retired":
        raise _NotLanded("failed_capture", capture_id=cid)
    why = _captures.safety(source)
    if why is not None or _looks_like_key(source):
        raise _NotLanded("not_landed", reason=f"capture {cid} fails the safety checks ({why or 'key-shaped string'})")
    return cid, source, record


# --- the landed source ------------------------------------------------------------------------------------------


def _lines(source: str) -> list[str]:
    """*source*'s lines with their ends, split at newlines only, as the parser and the tokenizer number them."""
    return io.StringIO(source, newline="").readlines() if "\r" in source else re.findall(r"[^\n]*\n|[^\n]+\Z", source)


def top_names(source: str) -> list[str]:
    """The names *source* binds at module level by ``def``, ``class`` or assignment, in order (parsed, never run)."""
    out: list[str] = []
    for node in ast.parse(source).body:
        found: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found = [node.name]
        elif isinstance(node, ast.Assign):
            found = [n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            found = [node.target.id]
        out += [n for n in found if n not in out]
    return out


def renamed(source: str, names: dict[str, str]) -> str:
    """*source* with each name token of *names* replaced, everything else byte for byte: an attribute (``x.run``)
    and a keyword argument (``f(run=1)``) are left alone; ``from __future__`` imports are dropped (a landed block
    may follow other code)."""
    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    lines = _lines(source)
    edits: list[tuple[int, int, int, str]] = []
    depth = 0
    prev = None
    for i, t in enumerate(toks):
        if t.type == tokenize.OP and t.string in "([{":
            depth += 1
        elif t.type == tokenize.OP and t.string in ")]}":
            depth -= 1
        if t.type == tokenize.NAME and t.string in names:
            after = next(
                (n for n in toks[i + 1 :] if n.type not in (tokenize.NL, tokenize.COMMENT)),
                None,
            )
            attribute = prev is not None and prev.type == tokenize.OP and prev.string == "."
            kwarg = depth > 0 and after is not None and after.type == tokenize.OP and after.string == "="
            if not attribute and not kwarg:
                edits.append((t.start[0], t.start[1], t.end[1], names[t.string]))
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            prev = t
    for row, a, b, new in sorted(edits, reverse=True):
        line = lines[row - 1]
        lines[row - 1] = line[:a] + new + line[b:]
    out = "".join(lines)
    future = [
        n for n in ast.parse(out).body if isinstance(n, ast.ImportFrom) and n.module == "__future__"
    ]
    if future:
        kept = _lines(out)
        for n in sorted(future, key=lambda n: n.lineno, reverse=True):
            del kept[n.lineno - 1 : (n.end_lineno or n.lineno)]
        out = "".join(kept)
    return out


def with_example(source: str, fn: str, test: str) -> str:
    """*source* with an ``Example:`` line naming *test* in top-level function *fn*'s docstring (a new docstring when
    it has none; left as it is when its docstring is not a triple-quoted literal). Parsed as data, never run."""
    node = next(
        (n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn),
        None,
    )
    if node is None or not node.body:
        return source
    lines = _lines(source)
    first = node.body[0]
    indent = " " * first.col_offset
    example = f"Example: {test}"
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
        and first.end_lineno is not None
        and first.end_col_offset is not None
    ):
        # the parser's columns are UTF-8 byte offsets
        row, col = first.end_lineno - 1, first.end_col_offset
        line = lines[row].encode("utf-8")
        quote = line[col - 3 : col].decode("utf-8", "replace")
        start = line[first.col_offset :] if first.end_lineno == first.lineno else b""
        if quote not in ('"""', "'''") or (start and start.decode("utf-8", "replace").count(quote) < 2):
            return source
        if example in first.value.value:
            return source
        head, tail = line[: col - 3].decode("utf-8"), line[col - 3 :].decode("utf-8")
        lines[row] = head + f"\n\n{indent}{example}\n{indent}" + tail
        return "".join(lines)
    if first.lineno - 1 >= len(lines) or first.lineno == node.lineno:
        return source  # a one-line body on the def line: left as it is
    lines.insert(first.lineno - 1, f'{indent}"""{fn.replace("_", " ").capitalize()}.\n\n{indent}{example}\n{indent}"""\n')
    return "".join(lines)


def _private(name: str, orig: str) -> str:
    return f"_{name}_{orig.lstrip('_')}"


def procedure_block(source: str, cid: str, name: str, package: str, reader: str) -> tuple[str, list[str]]:
    """(the procedures.py block, the names it binds): ``run`` as *name*, everything else private."""
    mapping = {n: (name if n == "run" else _private(name, n)) for n in top_names(source)}
    head = f"# Landed from capture {cid}: procedure memory.{package}.procedures:{name}, reader {reader}\n"
    test = f"memory/{package}/tests/test_{name}.py::test_{name}_case1_gives_the_recorded_result"
    return head + with_example(renamed(source, mapping), name, test), list(mapping.values())


def applies_item(source: str, proc_id: str) -> str | None:
    """The item id of the landed procedure *proc_id*'s ``applies`` when the captured *source* defines one
    (:func:`procedure_block` keeps ``run`` as the procedure's name and makes every other name private); None when it
    states none."""
    if "applies" not in top_names(source):
        return None
    module, _, name = proc_id.rpartition(":")
    return f"{module}:{_private(name, 'applies')}"


def reader_block(source: str, name: str, package: str) -> tuple[str, list[str]]:
    """(the readers.py block, the names it binds): ``read_input`` as ``read_<name>``, everything else private."""
    mapping = {n: (f"read_{name}" if n == "read_input" else _private(f"read_{name}", n)) for n in top_names(source)}
    head = f"# The reader of the procedure {name}, landed from its capture.\n"
    test = f"memory/{package}/tests/test_{name}.py::test_{name}_case1_reader_reads_the_recorded_request"
    return head + with_example(renamed(source, mapping), f"read_{name}", test), list(mapping.values())


def _join(old: str | None, block: str) -> str:
    return block if not old else old.rstrip("\n") + "\n\n\n" + block


def _test_head(package: str) -> str:
    return (
        f'"""Landed from captured attempts: each procedure of memory.{package}.procedures on its recorded '
        'request."""\n\nimport json\nfrom pathlib import Path\n\nimport pytest\n\n'
        'DATA = Path(__file__).parent / "data"\n'
    )


def result_fixture(name: str, k: int) -> str:
    """The file name, under ``tests/data/``, of case *k*'s handed-back value."""
    return f"{name}.case{k}.result.json"


def result_slice(ep: Any, i: int, value: Any, blob: Callable[[str], bytes]) -> list[int] | None:
    """The byte slice ``[start, end]`` of action *i*'s recorded arguments (the fixture source ``args:<i>``, the
    canonical JSON ``{"args": [...], "kwargs": {...}}``) that holds *value*, the value :func:`.captures.handed_back`
    reads from that action: its keyword arguments when it has any, else its one argument, else its argument list.
    None when no such slice decodes to *value*."""
    a = ep.actions[i]
    args, kwargs = list(a.args), dict(a.kwargs or {})

    def enc(v: Any) -> bytes:
        return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")

    if kwargs:
        head, part = b'{"args": ' + enc(args) + b', "kwargs": ', kwargs
    elif len(args) == 1:
        head, part = b'{"args": [', args[0]
    else:
        head, part = b'{"args": ', args
    try:
        data = _fx.source_bytes(ep, f"args:{i}", blob)
        body = enc(part)
        start, end = len(head), len(head) + len(body)
        if data[:start] != head or data[start:end] != body:
            return None
        if json.loads(data[start:end]) != json.loads(json.dumps(value, sort_keys=True)):
            return None
    except (_fx.FixtureError, OSError, KeyError, IndexError, TypeError, ValueError, RecursionError):
        return None
    return [start, end]


def case_tests(
    name: str,
    reader_mod: str,
    reader_fn: str,
    package: str,
    k: int,
    result: Any,
    from_fixture: bool = True,
) -> str:
    """The test functions of case *k*: the reader on the recorded text, and the procedure on the reader's value of
    it against the handed-back value. *reader_mod* is the reader's module, dotted (``memory.<package>.<module>``).
    With *from_fixture*, the expected value is loaded from :func:`result_fixture`, a fixture of the recording; else
    *result* is written in the test (a value the fixture tool has no source for)."""
    reader = reader_fn if reader_fn != name else f"read_{name}"
    imported = reader_fn if reader == reader_fn else f"{reader_fn} as {reader}"
    if from_fixture:
        expected = f'json.loads((DATA / "{result_fixture(name, k)}").read_text(encoding="utf-8"))'
    else:
        expected = f"json.loads({json.dumps(result, sort_keys=True, ensure_ascii=False)!r})"
    return (
        f"\n\ndef test_{name}_case{k}_reader_reads_the_recorded_request():\n"
        f"    from {reader_mod} import {imported}\n\n"
        f'    text = (DATA / "{name}.case{k}.txt").read_text(encoding="utf-8")\n'
        f'    value = json.loads((DATA / "{name}.case{k}.json").read_text(encoding="utf-8"))\n'
        f"    assert json.loads(json.dumps({reader}(text))) == value\n"
        f"\n\ndef test_{name}_case{k}_gives_the_recorded_result():\n"
        f"    from {reader_mod} import {imported}\n"
        f"    from memory.{package}.procedures import {name}\n\n"
        f'    text = (DATA / "{name}.case{k}.txt").read_text(encoding="utf-8")\n'
        f"    expected = {expected}\n"
        f"    assert json.loads(json.dumps({name}({reader}(text)))) == expected\n"
    )


def refusal_test(name: str, reader_mod: str, reader_fn: str, package: str, refuses: bool) -> str:
    """The refusal test: the reader raises ``MemoryInputError`` on empty text. When it does not yet (*refuses* is
    False, from a confined run), the test records that as a known failure (a strict expected failure), which turns
    red once the reader refuses such text."""
    mark = (
        ""
        if refuses
        else '@pytest.mark.xfail(strict=True, reason="the reader does not refuse text it was not built for yet")\n'
    )
    return (
        f"\n\n{mark}def test_{name}_reader_refuses_text_it_was_not_built_for():\n"
        f"    from {reader_mod} import {reader_fn}\n\n"
        "    with pytest.raises(Exception) as caught:\n"
        f'        {reader_fn}("")\n'
        '    assert any(c.__name__ == "MemoryInputError" for c in type(caught.value).__mro__)\n'
    )


# --- the box ------------------------------------------------------------------------------------------------------


def _box_text(box: Path, rel: str) -> str | None:
    try:
        return _safe.read_in(box, rel, _MAX_MODULE_BYTES).decode("utf-8")
    except _safe.Missing:
        return None
    except (OSError, UnicodeDecodeError):
        raise _NotLanded("not_landed", reason=f"/memory/{rel} cannot be read as text") from None


def _write(box: Path, rel: str, text: str, written: list[str]) -> None:
    """Through the writer's ``write`` tool: its path checks, never through a link. The final text is scanned for a
    key shape too (the store's source was scanned before the rename; a renamed name or the generated docstring line
    must never carry one into the library the actor sees)."""
    if _looks_like_key(text):
        raise _NotLanded("not_landed", reason=f"/memory/{rel} would hold a key-shaped string")
    reply = _et.call("write", {"path": rel, "content": text}, box)
    if reply.status != "ok":
        raise _NotLanded("not_landed", reason=reply.text[:300])
    written.append(rel)


def _packages(box: Path) -> list[str]:
    # listed from the box's descriptor: a link at /memory/memory is never followed
    try:
        root = os.open(Path(box), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return []
    try:
        mem = os.open("memory", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
        try:
            names = sorted(os.listdir(mem))
        finally:
            os.close(mem)
    except OSError:
        return []
    finally:
        os.close(root)
    return [n for n in names if _IDENT.match(n)]


def _landed(box: Path, cids: list[str]) -> tuple[str, str] | None:
    """(procedure id, reader id) of the first of *cids* already landed in the library at *box*, by its marker."""
    want = set(cids)
    for pkg in _packages(box):
        text = _box_text(box, f"memory/{pkg}/procedures.py")
        for m in _MARK.finditer(text or ""):
            if m.group(1) in want:
                return m.group(2), m.group(3)
    return None


def _readers(box: Path) -> list[str]:
    """The public functions of every ``readers.py`` in the library at *box* (parsed as data, never run here)."""
    out = []
    for pkg in _packages(box):
        text = _box_text(box, f"memory/{pkg}/readers.py")
        if not text:
            continue
        try:
            body = ast.parse(text).body
        except (SyntaxError, ValueError):
            continue
        out += [
            f"memory.{pkg}.readers:{n.name}"
            for n in body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")
        ]
    return out


def _free_case(box: Path, package: str, name: str) -> int:
    k = 1
    while any(
        os.path.lexists(Path(box) / f"memory/{package}/tests/data/{name}.case{k}.{ext}")
        for ext in ("json", "txt", "result.json")
    ):
        k += 1
    return k


def _manifest_add(box: Path, entries: list[dict]) -> None:
    manifest, problem = _wt._read_manifest(box)
    if problem is not None or not isinstance(manifest, dict):
        raise _NotLanded("not_landed", reason=problem or "the draft manifest cannot be read")
    items = manifest.get("items")
    items = items if isinstance(items, list) else []
    for entry in entries:
        old = next((i for i in items if isinstance(i, dict) and i.get("item") == entry["item"]), None)
        if old is None:
            items.append(entry)
            continue
        for key in ("source_episodes", "tests"):
            have = old.get(key) if isinstance(old.get(key), list) else []
            old[key] = have + [v for v in entry.get(key, []) if v not in have]
        for key in ("role", "reader", "kind"):
            old.setdefault(key, entry.get(key))
    manifest["items"] = items
    try:
        _wt.write_manifest(box, manifest)
    except OSError as exc:
        raise _NotLanded("not_landed", reason=f"cannot write the manifest ({_safe.errno_name(exc)})") from None


# --- the tool -----------------------------------------------------------------------------------------------------


def land(
    args: dict,
    *,
    store: Path | None,
    box: Path,
    episode: Callable[[str], Any],
    blob: Callable[[str], bytes],
    batch: list[str] | tuple[str, ...] = (),
    python: Path | None = None,
    runner: Callable[..., SandboxResult] = run_confined,
    parse: Callable[[str, str], tuple[str, str]] | None = None,
) -> _wt.Reply:
    """The ``land_capture`` tool (module docstring). Never raises: a refusal is the reply."""
    where = (_wt.path_row("/memory/memory", "write"),)
    work = Path(tempfile.mkdtemp(prefix="land-capture-"))
    try:
        return _land(args, store, Path(box), episode, blob, batch, python or PYTHON, runner, parse, work)
    except _NotLanded as n:
        return _wt.refused(TOOL, n.form, where, **n.fields)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _land(args, store, box, episode, blob, batch, python, runner, parse, work) -> _wt.Reply:
    package, name = args.get("package"), args.get("name")
    for what, v in (("package", package), ("name", name)):
        if not isinstance(v, str) or not _IDENT.match(v) or keyword.iskeyword(v):
            raise _NotLanded("bad_name", what=what, value=str(v)[:60])
    from .layout import RESERVED_PACKAGES

    if package in RESERVED_PACKAGES or package == "tests":
        raise _NotLanded("bad_name", what="package", value=package)
    if store is None:
        raise _NotLanded("unknown_capture", capture_id=str(args.get("capture_id"))[:40])
    cid, source, record = read_capture(Path(store), args.get("capture_id"))
    # the request the capture was recorded on, and the value its episode handed back
    eid, ep = next(
        ((e, episode(e)) for e in record.get("episodes") or [] if isinstance(e, str) and episode(e) is not None),
        (None, None),
    )
    if ep is None or not getattr(ep, "request", None):
        raise _NotLanded("not_landed", reason=f"no recorded episode of capture {cid} is readable in this pass")
    request_text = str(ep.request[0])
    found, handed = _captures.handed_back(ep)
    if not found:
        raise _NotLanded("not_landed", reason=f"capture {cid}'s episode handed back no value to test against")
    run = _captures.run_box(source, request_text, work=work / "capture", runner=runner, python=python)
    if run.get("outcome") != "returned" or _captures.compare(run, True, handed) != _captures.REPRODUCES:
        raise _NotLanded("not_landed", reason=f"capture {cid} does not give its recorded result on its request now")
    # where the recording holds the handed-back value: a slice of the answering action's arguments (a value the
    # episode used); a value handed back in the final reply has no fixture source, so its test states it instead
    answer = _captures.answered_action(ep)
    result_cut = None if answer is None else result_slice(ep, answer, handed, blob)
    written: list[str] = []
    same = _landed(box, [c for c in record.get("same_as") or [] if isinstance(c, str)])
    if same is not None:
        proc_id, reader_id = same
        package = proc_id.split(".")[1]
        name = proc_id.rsplit(":", 1)[1]
        reused, new_reader = True, False
    else:
        proc_id = f"memory.{package}.procedures:{name}"
        reader_id = ""
        for n, cand in enumerate(_readers(box)):
            values, problem = _run_reader(cand, request_text, box, python, runner, work / f"reader-{n}")
            if problem is None and values and values[0] is not None and json.loads(values[0]) == run.get("value"):
                reader_id = cand
                break
        reused, new_reader = bool(reader_id), not reader_id
        reader_id = reader_id or f"memory.{package}.readers:read_{name}"
        _write_procedure(box, source, cid, package, name, reader_id, new_reader, written)
    reader_mod, reader_fn = reader_id.split(":")
    k = _free_case(box, package, name)
    data_dir = f"memory/{package}/tests/data"
    plan: list[dict] = [
        {"source": "request:0", "dest": f"{data_dir}/{name}.case{k}.txt"},
        {"source": "request:0", "dest": f"{data_dir}/{name}.case{k}.json", "parser": reader_id},
    ]
    if result_cut is not None:
        plan.append(
            {"source": f"args:{answer}", "slice": result_cut, "dest": f"{data_dir}/{result_fixture(name, k)}"},
        )
    for fx_args in plan:
        dest = fx_args["dest"]
        reply = _wt.fixture({"episode": eid, **fx_args}, episode, box, blob, batch, python=python, parse=parse)
        if reply.status != "ok":
            raise _NotLanded("not_landed", reason=f"{dest}: {reply.text[:300]}")
        written.append(dest)
    test_rel = f"memory/{package}/tests/test_{name}.py"
    old = _box_text(box, test_rel)
    text = (old.rstrip("\n") + "\n" if old else _test_head(package)) + case_tests(
        name, reader_mod, reader_fn, package, k, handed, from_fixture=result_cut is not None,
    )
    refuses = None
    if not old:
        # how the reader answers empty text, confined: it refuses (the refusal test), it returns (the refusal test
        # records a known failure), or it raises another exception (no call is written: the gate refuses an example
        # call that raises anything but MemoryInputError)
        _, why = _run_reader(reader_id, "", box, python, runner, work / "refusal")
        refuses = "raises" if why and " raises " in why and " raises MemoryInputError " not in why else (
            bool(why) and " raises MemoryInputError " in why
        )
        if refuses != "raises":
            text += refusal_test(name, reader_mod, reader_fn, package, bool(refuses))
    if _looks_like_key(text):
        raise _NotLanded("not_landed", reason="the recorded result holds a key-shaped string")
    _write(box, test_rel, text, written)
    entries = [
        {
            "item": proc_id,
            "kind": "function",
            "source_episodes": [eid],
            "tests": [test_rel],
            "role": "procedure",
            "reader": reader_id,
        },
    ]
    if new_reader:
        entries.append(
            {"item": reader_id, "kind": "function", "source_episodes": [eid], "tests": [test_rel], "role": "reader"},
        )
    _manifest_add(box, entries)
    # the landing, in the harness's own store: the item records take the capture's signature and applies from it, and
    # the pass closes the capture's pool entry once the procedure lands
    _captures.record_landing(
        Path(store),
        cid,
        {"item": proc_id, "reader": reader_id, "applies": applies_item(source, proc_id), "episode": eid},
    )
    lines = [f"Landed capture {cid} from episode {eid}:"]
    if same is not None:
        lines.append(f"- a new case of {proc_id}, which an earlier capture of the same work landed")
    else:
        lines.append(f"- procedure {proc_id}")
        lines.append(f"- reader {reader_id}" + (" (reused: it gives the same value on this request)" if reused else ""))
    lines += [f"- wrote /memory/{p}" for p in written]
    lines.append("- the manifest entries of " + ", ".join(e["item"] for e in entries) + " in /memory/.pass/manifest.json")
    if refuses is False:
        lines.append(
            f"- {reader_id} does not raise MemoryInputError on empty text: its refusal test is marked as a known "
            "failure until it does",
        )
    elif refuses == "raises":
        lines.append(
            f"- {reader_id} raises another exception than MemoryInputError on empty text, so no refusal test was "
            "written: make it raise MemoryInputError on text it was not built for, and add the test",
        )
    if result_cut is None:
        lines.append(
            "- no fixture source of the recording holds the handed-back value (a value handed back in the final "
            "reply has none), so the recorded-result test states it, and the gate keeps the procedure unchecked "
            "until a test asserts its result against a fixture of the recording",
        )
    lines.append("Change any of them before you finish; check judges them as any other change.")
    return _wt.Reply("\n".join(lines), paths=tuple(_wt.path_row(f"/memory/{p}", "write") for p in written))


def _run_reader(item, text, box, python, runner, work):
    from .procedures import run_reader

    work.mkdir(parents=True, exist_ok=True)
    return run_reader(item, [text], tree=box, python=python, work=work, runner=runner)


def _write_procedure(box, source, cid, package, name, reader_id, new_reader, written) -> None:
    init = f"memory/{package}/__init__.py"
    if _box_text(box, init) is None:
        _write(box, init, '"""Procedures landed from captured attempts, with the readers of their requests."""\n', written)
    plan = [("procedures.py", procedure_block(source, cid, name, package, reader_id))]
    if new_reader:
        plan.append(("readers.py", reader_block(source, name, package)))
    for mod, (block, binds) in plan:
        rel = f"memory/{package}/{mod}"
        old = _box_text(box, rel)
        if old:
            try:
                taken = set(top_names(old)) & set(binds)
            except (SyntaxError, ValueError):
                raise _NotLanded("not_landed", reason=f"/memory/{rel} does not parse") from None
            if taken:
                raise _NotLanded(
                    "not_landed",
                    reason=f"/memory/{rel} already defines {', '.join(sorted(taken))}; choose another name",
                )
        text = _join(old, block)
        if _looks_like_key(text):
            raise _NotLanded("not_landed", reason=f"/memory/{rel} would hold a key-shaped string")
        _write(box, rel, text, written)
