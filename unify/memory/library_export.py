"""The generated files of every library-mode copy of the library: the actor's export and the writer's box.

* ``memory/__init__.py``: :mod:`.library_helper`, byte for byte (``index``, ``show``, ``find``);
* ``INDEX.md`` and ``links.json``: :mod:`.library_index`;
* ``.memory/items.json``: what ``show`` reads per item, namely its kind, path, signature or title, status, a
  function's evidence level with the findings that set it, the verification and use records (None without item
  records), a function's example call (its record's, shown only while its test file still quotes it:
  :mod:`.example_view`) and the subjects of the commits that changed it;
* ``.memory/find.json``: the listed functions with the recorded shapes of their example calls' arguments, each
  naming its parameter, keyed by body digest (:mod:`.shape_rows`); and, once any listed function has an applicability
  signature, ``procedures``: what each such function was saved against, for ``memory.match``
  (:func:`find_data`);
* ``.memory/matcher.py``: v2's matcher (:mod:`.memory_helper` up to its catalogue section, so none of v2's
  catalogue or its footer ships);
* ``.memory/shapes.py``: :mod:`.analysis.shapes`, byte for byte.

None of these is ever committed (the gate's G1 check refuses them). Each is a pure function of the tree, the item
statuses, the recorded shapes and the history, so the same commit gives the same bytes (cache rule).
"""

from __future__ import annotations

import ast
import json
from typing import Any
from collections.abc import Callable, Mapping
from pathlib import Path

from .catalogue import ShapeLookup, body_digest
from .gitio import GitError, Repo
from .layout import (
    INDEX_FILE,
    LINKS_FILE,
    RESERVED_INIT,
    Library,
    classify,
    discover,
    function_bodies,
    module_bodies,
    module_of,
)
from .library_helper import FIND_FILE, ITEMS_FILE, MATCHER_FILE, SHAPES_FILE
from .example_view import ReadFile, example_text, examples_of, tree_reader
from .item_records import (
    effect_of,
    episodes_of,
    level_of,
    level_reasons_of,
    use_totals,
    verification_summary,
)
from .item_states import ON_TRIAL
from .library_index import HIDDEN, build_links, render_index, render_links
from .redact import redact_error

HISTORY_SCAN = 200
HISTORY_SHOW = 5
_MATCHER_START = "from __future__ import annotations"
_MATCHER_END = "\n# --- the catalogue"
_HERE = Path(__file__).resolve().parent
History = tuple[dict[str, list[str]], bool]
#: What a function was saved against (:func:`saved_against_of`): ``signature`` (its applicability signature, the
#: capture store's ``signature.json`` shape), ``asked`` and ``approach`` (its source episode's ``summary.md``
#: sections) and ``level`` (its evidence level); None when it has no signature.
SavedAgainst = Callable[[str], "dict | None"]
#: An episode's ``summary.md`` text by episode id, None when it has none (the integration reads
#: ``<state>/memory-staging/<episode>/summary.md``).
SummaryOf = Callable[[str], "str | None"]
#: Text with its printed identifiers removed (the one rule, ``signatures.strip_identifiers``).
Strip = Callable[[str], str]
#: What a captured attempt was saved against besides its signature, beside it in the actor's copy.
CAPTURE_SAVED_FILE = "saved.json"


def default_strip() -> Strip:
    """The one identifier rule: ``signatures.strip_identifiers``, owned by the capture store. Every text ``match``
    hands its matcher (a docstring, a summary section, what broke) is stripped with it here, on the harness side;
    the generated helper strips nothing."""
    from .signatures import strip_identifiers

    return strip_identifiers


def strip_signature(value: Any, strip: Strip | None = None, _under: str = "") -> Any:
    """A signature (or any JSON-like value) with the one identifier rule applied to every mapping key and to the
    column and header names a file shape lists; other values are kept. A signature the capture store built is
    already stripped, so this changes nothing there; it keeps a signature from any other source to the rule."""
    strip = strip or default_strip()
    if isinstance(value, dict):
        return {
            (strip(k) if isinstance(k, str) else k): strip_signature(v, strip, k if isinstance(k, str) else "")
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            strip(v) if isinstance(v, str) and _under in ("columns", "header") else strip_signature(v, strip)
            for v in value
        ]
    return value


def _section(text: str, heading: str) -> str:
    """The body of the ``## <heading>`` section of a ``summary.md``, stripped of surrounding blank lines."""
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip().lower() == heading.lower()
            continue
        if inside:
            out.append(line)
    return "\n".join(out).strip()


def summary_sections(text: str, strip: Strip | None = None) -> dict[str, str]:
    """The ``Asked`` and ``Approach`` sections of a ``summary.md`` text (design H.4), identifiers removed."""
    strip = strip or default_strip()
    return {
        "asked": strip(_section(text, "Asked")),
        "approach": strip(_section(text, "Approach")),
    }


def _docstring_of(source: str | None, name: str = "run") -> str:
    """The docstring of function *name* in *source*, else its module docstring; read with ``ast``, never run."""
    if not source:
        return ""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            doc = ast.get_docstring(node)
            if doc:
                return doc
    return ast.get_docstring(tree) or ""


def capture_saved(capture_dir: Path, summary: str | None, strip: Strip | None = None) -> dict:
    """What a captured attempt was saved against besides its signature, as ``memory.match`` reads it from
    ``<copy>/.memory/captures/<id>/saved.json``: ``docstring`` (its ``procedure.py``'s ``run``, else module,
    docstring), ``asked`` and ``approach`` (its episode's ``summary.md``, *summary*) and ``broke`` (the lines of its
    ``broke.md``, ``none`` left out), every text with identifiers removed."""
    strip = strip or default_strip()
    d = Path(capture_dir)

    def read(name: str) -> str | None:
        try:
            return (d / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    sections = summary_sections(summary, strip) if summary else {"asked": "", "approach": ""}
    broke = [
        strip(line.strip().lstrip("-* ").strip())
        for line in (read("broke.md") or "").splitlines()
        if line.strip() and line.strip().lower() != "none"
    ]
    return {
        "docstring": strip(_docstring_of(read("procedure.py"))),
        "asked": sections["asked"],
        "approach": sections["approach"],
        "broke": broke,
    }


def capture_saved_bytes(capture_dir: Path, summary: str | None, strip: Strip | None = None) -> bytes:
    """:func:`capture_saved` as the bytes of ``saved.json`` (sorted keys, one trailing newline)."""
    return _json_bytes(capture_saved(capture_dir, summary, strip))


def matcher_source() -> bytes:
    """v2's shape matcher without its module docstring or catalogue (``memory_helper.py`` from its imports up to
    the ``# --- the catalogue`` section), behind a one-line docstring."""
    src = (_HERE / "memory_helper.py").read_text(encoding="utf-8")
    body = src[src.index(_MATCHER_START) : src.index(_MATCHER_END) + 1]
    return (
        '"""Shape matching for memory.find; generated by the harness, never edit it."""\n\n'
        + body
    ).encode("utf-8")


def _bodies(repo: Repo, rev: str | None, path: str) -> dict[str, str]:
    if rev is None:
        return {}
    try:
        return module_bodies(repo.show(rev, path))
    except GitError:
        return {}


def item_history(repo: Repo, sha: str, *, scan: int = HISTORY_SCAN) -> History:
    """item id -> ``"<sha12> <subject>"`` of each of the newest *scan* commits up to *sha* that changed it
    (newest first), and whether that scan reached every such commit.

    A function changes when its unparsed definition differs from its parent's (what ``git log -L`` follows,
    computed by AST instead of git's function-name heuristics). A note changes when its file changes.
    Each subject is redacted (:func:`..redact.redact_error`): ``show`` prints it to the actor, and a commit
    may have come from any merge path.
    """
    raw = repo.run(
        "log",
        f"--max-count={scan + 1}",
        "--format=%H%x1f%P%x1f%s",
        sha,
        "--",
        "memory",
        "notes",
    )
    rows = [line.split("\x1f", 2) for line in raw.splitlines() if line]
    out: dict[str, list[str]] = {}
    for commit, parents, subject in rows[:scan]:
        parent = parents.split()[0] if parents.split() else None
        changed = repo.run(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "--root",
            "-z",
            commit,
            "--",
            "memory",
            "notes",
        )
        for path in (p for p in changed.split("\0") if p):
            kind = classify(path)
            if kind == "note":
                ids = {path}
            elif kind == "module":
                before, after = _bodies(repo, parent, path), _bodies(repo, commit, path)
                module = module_of(path)
                ids = {
                    f"{module}:{n}"
                    for n in set(before) | set(after)
                    if before.get(n) != after.get(n)
                }
            else:
                continue
            for item in sorted(ids):
                out.setdefault(item, []).append(
                    f"{commit[:12]} {redact_error(subject)}",
                )
    return out, len(rows) <= scan


def _status(status_of: Callable[[str], str] | None, item: str) -> str:
    return status_of(item) if status_of is not None else ON_TRIAL


def items_data(
    lib: Library,
    status_of: Callable[[str], str] | None,
    history: dict[str, list[str]],
    complete: bool,
    records: Mapping[str, dict] | None = None,
    examples: Callable[[str], dict | None] | None = None,
) -> dict:
    """``.memory/items.json``: everything :func:`.library_helper.show` needs that the files do not hold.

    *examples* gives a function's shown example call (:func:`.example_view.examples_of`); None shows none.
    """
    rows: dict[str, dict] = {}

    def common(item: str) -> dict:
        h = history.get(item, [])
        rec = records.get(item) if records else None
        return {
            "status": _status(status_of, item),
            "status_reason": (
                rec.get("status_reason") if rec else None
            ),  # why it is out of the index (item record)
            "verification": verification_summary(rec) if rec else None,
            "use": use_totals(rec) if rec else None,
            "history": h[:HISTORY_SHOW],
            "history_more": max(0, len(h) - HISTORY_SHOW),
        }

    for f in lib.functions:
        rows[f.item_id] = {
            "kind": "function",
            "path": f.path,
            "signature": f"{f.module}:{f.signature}",
            "summary": f.summary,
            # the effect the gate derived from its covers (None: not derived)
            "effect": effect_of(records)(f.item_id) if records else None,
            # its evidence level (None: none recorded) and the findings that set it
            "level": level_of(records)(f.item_id) if records else None,
            "level_reasons": level_reasons_of(records.get(f.item_id)) if records else [],
            # {"test", "line", "call"}: a call quoted from its tests, re-checked against them (None: none shown)
            "example": examples(f.item_id) if examples is not None else None,
            **common(f.item_id),
        }
    for n in lib.notes:
        rows[n.item_id] = {
            "kind": "note",
            "path": n.item_id,
            "title": n.title,
            "description": n.description,
            "error": n.error,
            **common(n.item_id),
        }
    return {
        "version": 1,
        "items": rows,
        "history_complete": complete,
        "history_scanned": HISTORY_SCAN,
    }


def saved_against_of(
    records: Mapping[str, dict] | None,
    summary_of: SummaryOf | None = None,
    strip: Strip | None = None,
) -> SavedAgainst:
    """What each function was saved against, from its item record: ``signature`` (the record's ``signature``, an
    applicability signature as the capture store writes it, identifiers already removed; a function without one
    gives None), ``level`` (the record's ``level``), and the ``Asked`` and ``Approach`` sections of the
    ``summary.md`` of the first of its provenance episodes that *summary_of* has one for, identifiers removed with
    *strip* (:func:`default_strip` when None)."""

    def get(item: str) -> dict | None:
        rec = records.get(item) if records else None
        sig = rec.get("signature") if isinstance(rec, dict) else None
        if not isinstance(sig, dict):
            return None
        out = {
            "signature": strip_signature({k: v for k, v in sig.items() if k != "goal"}, strip),
            "asked": "",
            "approach": "",
            "level": rec.get("level") if isinstance(rec.get("level"), str) else None,
        }
        prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}
        episodes = prov.get("episodes") if isinstance(prov.get("episodes"), list) else []
        for episode in episodes if summary_of is not None else []:
            text = summary_of(str(episode)) if isinstance(episode, str) else None
            if text:
                out.update(summary_sections(text, strip))
                break
        if not out["asked"] and isinstance(sig.get("goal"), str):
            out["asked"] = (strip or default_strip())(sig["goal"])
        return out

    return get


def staging_summaries(staging_root: Path) -> SummaryOf:
    """An episode's ``summary.md`` under *staging_root* (``<state>/memory-staging``), as H.4 stages it:
    ``<staging_root>/<episode>/summary.md``; None when it is missing or the episode id is not a single path part."""

    def get(episode: str) -> str | None:
        if not episode or "/" in episode or episode in (".", ".."):
            return None
        try:
            return (Path(staging_root) / episode / "summary.md").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    return get


def _docstrings(tree: Path, lib: Library, wanted: set[str]) -> dict[str, str]:
    """The full docstring of each function in *wanted*, read from its module with ``ast`` (never imported)."""
    out: dict[str, str] = {}
    by_path: dict[str, list] = {}
    for f in lib.functions:
        if f.item_id in wanted:
            by_path.setdefault(f.path, []).append(f)
    for path, fns in by_path.items():
        try:
            mod = ast.parse((Path(tree) / path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
            continue
        defs = {
            n.name: n
            for n in mod.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for f in fns:
            node = defs.get(f.name)
            if node is not None:
                out[f.item_id] = ast.get_docstring(node) or ""
    return out


def find_data(
    tree: Path,
    lib: Library,
    status_of: Callable[[str], str] | None,
    shapes: ShapeLookup | None,
    records: Mapping[str, dict] | None = None,
    saved_against: SavedAgainst | None = None,
    strip: Strip | None = None,
) -> dict:
    """``.memory/find.json``: each listed function with the recorded input shapes of its current body.

    ``input_shapes`` are the shapes of the arguments its example calls returned on, each naming its parameter
    (``"param"``; a row recorded before arguments were shaped has none). *records* are accepted for the export's
    signature and not read: a function's shapes come only from *shapes*.

    With *saved_against*, each listed function it gives a signature for is also a row of ``procedures`` (in id
    order): ``{id, signature, docstring, asked, approach, level, status, summary}``, the docstring and summary with
    identifiers removed by *strip* (:func:`default_strip` when None), which
    ``memory.match`` hands its matcher. The key is left out while no function has one, so the file is then byte for
    byte what it was.
    """
    bodies = function_bodies(tree) if shapes is not None else {}
    out: list[dict] = []
    for f in lib.functions:
        if _status(status_of, f.item_id) in HIDDEN:
            continue
        entry = {
            "id": f.item_id,
            "module": f.module,
            "name": f.name,
            "signature": f.signature,
            "summary": f.summary,
        }
        if shapes is not None and f.item_id in bodies:
            found = shapes(f.item_id, body_digest(bodies[f.item_id]))
            if found and found[0]:
                entry["input_shapes"] = found[0]
        out.append(entry)
    data: dict = {"version": 1, "functions": out}
    saved = {
        e["id"]: s
        for e in out
        if saved_against is not None and (s := saved_against(e["id"])) is not None
    }
    if saved:
        strip = strip or default_strip()
        docs = _docstrings(tree, lib, set(saved))
        data["procedures"] = [
            {
                "id": e["id"],
                "signature": saved[e["id"]].get("signature") or {},
                "docstring": strip(docs.get(e["id"], "")),
                "asked": saved[e["id"]].get("asked") or "",
                "approach": saved[e["id"]].get("approach") or "",
                "level": saved[e["id"]].get("level") or "unknown",
                "status": _status(status_of, e["id"]),
                "summary": strip(e.get("summary", "")),
            }
            for e in sorted(out, key=lambda e: e["id"])
            if e["id"] in saved
        ]
    return data


def _json_bytes(data: dict) -> bytes:
    return (
        json.dumps(data, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def generated_library_mode(
    tree: Path,
    *,
    status_of: Callable[[str], str] | None = None,
    shapes: ShapeLookup | None = None,
    history: History | None = None,
    records: Mapping[str, dict] | None = None,
    tests: ReadFile | None = None,
    saved_against: SavedAgainst | None = None,
    strip: Strip | None = None,
) -> dict[str, bytes]:
    """Every generated file of the copy of the library at *tree*, by relative path (module docstring).

    *records* are the item records at the copy's commit (:func:`.item_records.records_at`); None gives the files
    without item records, byte for byte. *tests* reads a test file of the commit by its tree-relative path, to
    re-check each recorded example call before it is shown; None reads them from *tree* (a copy without its
    tests, as the actor's export is, then shows no example). *saved_against* gives what each function was saved
    against (:func:`saved_against_of`); None reads only the item records' ``signature`` (no summaries). A function
    with a signature gets an ``input:`` line in the index and a ``procedures`` row in ``find.json``."""
    tree = Path(tree)
    lib = discover(tree)
    links = build_links(lib)
    found, complete = history if history is not None else ({}, True)
    shown = examples_of(records, tests if tests is not None else tree_reader(tree))

    def example_of(item: str) -> str | None:
        ex = shown(item)
        return example_text(ex) if ex is not None else None

    if saved_against is None and records is not None:
        saved_against = saved_against_of(records, None, strip)

    def signature_of(item: str) -> dict | None:
        got = saved_against(item) if saved_against is not None else None
        return got.get("signature") if isinstance(got, dict) else None

    return {
        RESERVED_INIT: (_HERE / "library_helper.py").read_bytes(),
        INDEX_FILE: render_index(
            lib,
            links,
            status_of,
            effect_of(records) if records is not None else None,
            example_of if records is not None else None,
            episodes_of(records) if records is not None else None,
            signature_of if saved_against is not None else None,
            level_of(records) if records is not None else None,
        ).encode("utf-8"),
        LINKS_FILE: render_links(links).encode("utf-8"),
        ITEMS_FILE: _json_bytes(
            items_data(lib, status_of, found, complete, records, shown),
        ),
        FIND_FILE: _json_bytes(
            find_data(tree, lib, status_of, shapes, records, saved_against, strip),
        ),
        MATCHER_FILE: matcher_source(),
        SHAPES_FILE: (_HERE / "analysis" / "shapes.py").read_bytes(),
    }
