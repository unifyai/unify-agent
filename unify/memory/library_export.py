"""The generated files of every library-mode copy of the library: the actor's export and the writer's box.

* ``memory/__init__.py``: :mod:`.library_helper`, byte for byte (``memory.get``, ``memory.items``, the ``Item``
  type and, for one build, the earlier calls as aliases), with ``memory/_contract.py``, the contract runtime it
  checks its calls with, and ``memory/__init__.pyi``, its typed stub (:func:`stub_source`);
* ``INDEX.md`` and ``links.json``: :mod:`.library_index`;
* ``.memory/items.json``: what ``memory.get`` reads per item, namely its kind, path, signature or title, status, a
  function's evidence level with the findings that set it, the verification and use records (None without item
  records), a function's example call (its record's, shown only while its test file still quotes it:
  :mod:`.example_view`) and the subjects of the commits that changed it;
* ``.memory/find.json``: the listed functions with the recorded shapes of their example calls' arguments, each
  naming its parameter, keyed by body digest (:mod:`.shape_rows`); and, once any listed function has an applicability
  signature, ``procedures``: what each such function was saved against, for ``memory.match``
  (:func:`find_data`);
* ``.memory/matcher.py``: v2's matcher (:mod:`.memory_helper` up to its catalogue section, so none of v2's
  catalogue or its footer ships);
* ``.memory/shapes.py``: :mod:`.analysis.shapes`, byte for byte;
* with ``UNIFY_MEMORY_LIBRARIAN=on`` only: ``.memory/catalogue.md`` and ``.memory/dispatch.json``, the librarian's
  catalogue and the procedures' recognisers (:func:`librarian_files`), and ``memory.ask`` in the package and its
  stub.

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
    RESERVED_CONTRACT,
    RESERVED_INIT,
    RESERVED_STUB,
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
    recipe_ids,
    recipe_text,
    use_totals,
    verification_summary,
)
from .item_states import ON_TRIAL
from .library_index import HIDDEN, build_links, render_index, render_links
from .recognition import listed, listing
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
    """``.memory/items.json``: everything ``memory.get`` and ``memory.items`` need that the files do not hold.

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
        rec = records.get(f.item_id) if records else None
        recognition = _recognition_line(rec)
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
            # whether memory.match and the librarian list it, and why (only for a procedure tested when it landed)
            **({"recognition": recognition} if recognition else {}),
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
    ``memory.match`` hands its matcher. A procedure its item record does not list (:func:`.recognition.listed`) is no
    row. The key is left out while no function has one, so the file is then byte for byte what it was.
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
        if saved_against is not None
        and listed((records or {}).get(e["id"]))
        and (s := saved_against(e["id"])) is not None
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


# --- the librarian's catalogue and the dispatch file (UNIFY_MEMORY_LIBRARIAN=on) --------------------------------
#
# Both are harness-generated at each export, with no model call: ``.memory/catalogue.md``, one card per stored
# function in module-path order (never a score), then the notes, then the captured procedures; and
# ``.memory/dispatch.json``, each procedure that has a reader or an ``applies``. Neither carries an answer, a raw
# request text or a printed identifier: every free text (docstrings, summary sections, what broke) goes through the
# one identifier rule (``signatures.strip_identifiers``).

CATALOGUE_FILE = ".memory/catalogue.md"
DISPATCH_FILE = ".memory/dispatch.json"
#: The captures' prefix in the catalogue and in ``dispatch.json`` (``memory.match`` and ``memory.run`` use it too).
CAPTURE_PREFIX = "capture:"
_CAPTURE_ID = "[0-9a-f]{16}"
_UNCHECKED = "unchecked"


def _first_paragraph(doc: str) -> str:
    """The first paragraph of a docstring, on one line."""
    return " ".join((doc or "").strip().split("\n\n", 1)[0].split())


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split())


def _provenance_episodes(rec: Mapping | None) -> list[str]:
    prov = rec.get("provenance") if isinstance(rec, Mapping) else None
    eps = prov.get("episodes") if isinstance(prov, dict) else None
    return [e for e in eps if isinstance(e, str)] if isinstance(eps, list) else []


def _saved_lines(rec: Mapping | None, summary_of: SummaryOf | None, strip: Strip) -> list[tuple[str, str]]:
    """``(asked, approach)`` of each provenance episode of *rec* that has a ``summary.md``, identifiers removed,
    without repeats, in the record's episode order; else the signature's own goal (already stripped) as Asked."""
    out: list[tuple[str, str]] = []
    for episode in _provenance_episodes(rec) if summary_of is not None else []:
        text = summary_of(episode)
        if not text:
            continue
        s = summary_sections(text, strip)
        pair = (_one_line(s["asked"]), _one_line(s["approach"]))
        if any(pair) and pair not in out:
            out.append(pair)
    sig = rec.get("signature") if isinstance(rec, Mapping) else None
    if not out and isinstance(sig, dict) and isinstance(sig.get("goal"), str) and sig["goal"].strip():
        out.append((_one_line(sig["goal"]), ""))
    return out


def _readers(rec: Mapping | None) -> list[str]:
    """The readers a function's item record names: the one its landing named (``landed_from``), else the gate's
    (``reader``: a list of ids, sorted; a record holding one id names it)."""
    if not isinstance(rec, Mapping):
        return []
    landed = rec.get("landed_from")
    if isinstance(landed, Mapping) and isinstance(landed.get("reader"), str) and landed["reader"]:
        return [landed["reader"]]
    reader = rec.get("reader")
    named = reader if isinstance(reader, list) else [reader]
    return [r for r in named if isinstance(r, str) and r]


def _no_reader(rec: Mapping | None) -> str:
    """The catalogue's word for a function with no single reader."""
    readers = _readers(rec)
    if not readers:
        return "none"
    return f"none of its {len(readers)} ({', '.join(readers)}): more than one reads its tests' fixtures"


def _reader_of(rec: Mapping | None) -> str | None:
    """The one reader a procedure is read by (:func:`_readers`); None when it names none or several (never one
    chosen of several)."""
    readers = _readers(rec)
    return readers[0] if len(readers) == 1 else None


def _applies_stated(sig: object) -> bool:
    """Whether an applicability signature records an ``applies`` (the capture store writes ``stated``)."""
    cons = sig.get("constraints") if isinstance(sig, dict) else None
    a = cons.get("applies") if isinstance(cons, dict) else None
    return isinstance(a, dict) and bool(a.get("stated") or a.get("exists"))


def _recognition_line(rec: Mapping | None) -> str:
    """Whether ``memory.match`` and the librarian list a procedure tested when it landed, and why
    (:func:`.recognition.listing`); ``""`` for an item with no recognition record."""
    got = rec.get("recognition") if isinstance(rec, Mapping) else None
    return listing(got)[1] if isinstance(got, Mapping) else ""


def _capture_unlisted(rec: Mapping | None) -> str:
    """Why a captured procedure is not listed (its exported record's ``listing``, :func:`.captures.listed_record`);
    ``""`` when it is listed."""
    got = rec.get("listing") if isinstance(rec, Mapping) else None
    if isinstance(got, Mapping) and got.get("listed") is False:
        return str(got.get("why") or "not listed")
    return ""


def _track_line(rec: Mapping | None) -> str:
    """The one-line track record of an item record: its use totals, or that it has none yet."""
    if not isinstance(rec, Mapping) or not isinstance(rec.get("use"), dict) or not rec.get("use"):
        return "no uses recorded yet"
    totals = use_totals(rec)
    return ", ".join(f"{k} {totals[k]}" for k in sorted(totals))


def _capture_docs(captures: Mapping[str, bytes] | None) -> dict[str, dict]:
    """The exported captures (``.memory/captures/<id>/<file>`` -> bytes, :func:`.captures.export_files`) by id:
    their ``signature``, ``record``, ``procedure`` source and ``broke`` text, each None when absent or unreadable."""
    import re

    out: dict[str, dict] = {}
    rx = re.compile(r"\.memory/captures/(" + _CAPTURE_ID + r")/([A-Za-z.]+)")
    for path, data in sorted((captures or {}).items()):
        m = rx.fullmatch(path)
        if m is None:
            continue
        doc = out.setdefault(m.group(1), {"signature": None, "record": None, "procedure": None, "broke": None})
        try:
            text = bytes(data).decode("utf-8")
        except (UnicodeDecodeError, TypeError, ValueError):
            continue
        name = m.group(2)
        if name in ("signature.json", "record.json"):
            try:
                value = json.loads(text)
            except ValueError:
                continue
            doc[name.split(".")[0]] = value if isinstance(value, dict) else None
        elif name == "procedure.py":
            doc["procedure"] = text
        elif name == "broke.md":
            doc["broke"] = text
    return out


def _function_docs(tree: Path, lib: Library) -> dict[str, str]:
    return _docstrings(tree, lib, {f.item_id for f in lib.functions})


def catalogue_text(
    tree: Path,
    lib: Library | None = None,
    *,
    status_of: Callable[[str], str] | None = None,
    records: Mapping[str, dict] | None = None,
    summary_of: SummaryOf | None = None,
    captures: Mapping[str, bytes] | None = None,
    strip: Strip | None = None,
) -> str:
    """``.memory/catalogue.md``: the library's whole catalogue as the librarian reads it (design J(b).3).

    One card per listed function in module-path order (its id, signature and the first paragraph of its docstring;
    what it was saved against, the identifier-stripped ``## Asked`` and ``## Approach`` lines of each provenance
    episode's ``summary.md`` from *summary_of*; its evidence level and status, its reader, whether it has
    ``applies``, its input line and a one-line track record; the notes it links to). Then the notes, then the
    captured procedures from *captures* (the actor's snapshot, ``.memory/captures/...``), each at level
    ``unchecked``; a captured attempt that failed is listed with what broke and is never offered to run. Card fields
    are fixed; the only free text is the docstrings and the summaries, identifiers removed with *strip*. Pure: the
    same inputs give the same bytes."""
    from .library_index import signature_text

    tree = Path(tree)
    lib = lib if lib is not None else discover(tree)
    strip = strip or default_strip()
    links = build_links(lib)
    docs = _function_docs(tree, lib)
    status = {
        **{f.item_id: _status(status_of, f.item_id) for f in lib.functions},
        **{n.item_id: _status(status_of, n.item_id) for n in lib.notes},
    }
    shown = {i for i, s in status.items() if s not in HIDDEN}
    out: list[str] = [
        "# The library's catalogue\n",
        "\nOne card per stored function in module-path order, then the notes, then the captured procedures. "
        "Saved against: what the work each item was built from asked and how it went about it. Evidence: what has "
        "been checked about the item and how it has been used.\n",
        "\n## Functions\n",
    ]
    functions = [f for f in sorted(lib.functions, key=lambda f: (f.module, f.name)) if f.item_id in shown]
    if not functions:
        out.append("\n(none)\n")
    for f in functions:
        rec = records.get(f.item_id) if records else None
        rec = rec if isinstance(rec, dict) else None
        sig = rec.get("signature") if rec else None
        reader = _reader_of(rec)
        level = rec.get("level") if rec and isinstance(rec.get("level"), str) else "not recorded"
        lines = [f"\n### {f.item_id}\n", f"`{f.module}:{f.signature}`\n"]
        doc = _one_line(strip(_first_paragraph(docs.get(f.item_id, ""))))
        if doc:
            lines.append(f"{doc}\n")
        saved = _saved_lines(rec, summary_of, strip)
        lines.append("Saved against:" + ("" if saved else " no summary recorded") + "\n")
        for asked, approach in saved:
            lines.append(f"- Asked: {asked or 'not recorded'}\n")
            if approach:
                lines.append(f"  Approach: {approach}\n")
        input_line = signature_text(sig) if isinstance(sig, dict) else ""
        lines.append(
            f"Evidence: level {level}; status {status[f.item_id]}; reader {reader or _no_reader(rec)}; "
            f"applies: {'yes' if _applies_stated(sig) else 'no'}\n",
        )
        if input_line:
            lines.append(f"Input: {input_line}\n")
        if recipe_text(rec):
            # a procedure's one-call entry, and how it did on the cases its episodes observed when the gate ran it
            lines.append(f"Recipe: {recipe_text(rec)}\n")
        if _recognition_line(rec):
            lines.append(f"Recognition: {_recognition_line(rec)}\n")
        lines.append(f"Record: {_track_line(rec)}\n")
        notes = sorted(
            i
            for i in set(links["items"][f.item_id]["links_to"]) | set(links["items"][f.item_id]["linked_from"])
            if i in shown and links["items"].get(i, {}).get("kind") == "note"
        )
        if notes:
            lines.append("Notes: " + ", ".join(notes) + "\n")
        out.append("".join(lines))
    out.append("\n## Notes\n")
    notes = [n for n in sorted(lib.notes, key=lambda n: n.item_id) if n.item_id in shown]
    if not notes:
        out.append("\n(none)\n")
    for n in notes:
        about = n.description if not n.error else "(front matter unreadable)"
        fns = sorted(
            i
            for i in set(links["items"][n.item_id]["links_to"]) | set(links["items"][n.item_id]["linked_from"])
            if i in shown and links["items"].get(i, {}).get("kind") == "function"
        )
        out.append(
            f"\n### {n.item_id}\n"
            + (f"{_one_line(strip(n.title))}\n" if n.title else "")
            + (f"{_one_line(strip(about))}\n" if about else "")
            + f"Status: {status[n.item_id]}\n"
            + ("Functions: " + ", ".join(fns) + "\n" if fns else ""),
        )
    out.append("\n## Captured procedures\n")
    caps = _capture_docs(captures)
    if not caps:
        out.append("\n(none)\n")
    for cid, doc in caps.items():
        sig = doc["signature"] if isinstance(doc["signature"], dict) else {}
        rec = doc["record"] if isinstance(doc["record"], dict) else {}
        failed = rec.get("outcome") == "failed"
        lines = [f"\n### {CAPTURE_PREFIX}{cid}\n"]
        run_doc = _one_line(strip(_first_paragraph(_docstring_of(doc["procedure"]))))
        if run_doc:
            lines.append(f"{run_doc}\n")
        goal = _one_line(sig.get("goal")) if isinstance(sig.get("goal"), str) else ""
        lines.append(f"Saved against:\n- Asked: {goal or 'not recorded'}\n")
        input_line = signature_text(sig)
        level = _UNCHECKED  # captured procedures are listed at level unchecked (design J(b).3)
        state = rec.get("status") if isinstance(rec.get("status"), str) else "on-trial"
        lines.append(
            f"Evidence: level {level}; status {state}; reader read_input; "
            f"applies: {'yes' if _applies_stated(sig) else 'no'}"
            + ("; tried and failed: never offered to run" if failed else "")
            + "\n",
        )
        if input_line:
            lines.append(f"Input: {input_line}\n")
        why = _capture_unlisted(rec)
        if why:
            lines.append(f"Recognition: {why}\n")
        if failed:
            for line in (doc["broke"] or "").splitlines():
                line = _one_line(strip(line.strip().lstrip("-* ")))
                if line and line.lower() != "none":
                    lines.append(f"Broke: {line}\n")
        out.append("".join(lines))
    return "".join(out)


def dispatch_data(
    lib: Library,
    *,
    status_of: Callable[[str], str] | None = None,
    records: Mapping[str, dict] | None = None,
    captures: Mapping[str, bytes] | None = None,
) -> dict:
    """``.memory/dispatch.json``: each listed procedure with a reader or an ``applies`` (design J(b).3), as
    ``{id, reader, applies, input}``: ``reader`` is the reader's id (an item id from the item record's ``reader``;
    ``capture:<id>`` for a captured procedure, whose ``read_input`` is its reader), ``applies`` whether it has one
    (its applicability signature states one, or its item record names the function), and ``input`` its input line.
    Module-path order, then the captures in id order; a captured attempt that failed is left out (never offered to
    run), and so is a procedure its item record does not list (:func:`.recognition.listed`), a captured procedure
    whose landing is not listed (its record's ``listing``: the same code) and a procedure's recipe (it takes the
    request itself, so no reader reads for it; its procedure's card names it). Beside the rows, ``applies_of`` names, for
    a landed procedure, the library function that is its ``applies`` (the item record's ``applies``); a captured
    procedure's ``applies`` is the one in its own ``procedure.py``."""
    from .library_index import signature_text

    rows: list[dict] = []
    applies_of: dict[str, str] = {}
    recipes = recipe_ids(records)
    for f in sorted(lib.functions, key=lambda f: (f.module, f.name)):
        if _status(status_of, f.item_id) in HIDDEN or f.item_id in recipes:
            continue
        rec = records.get(f.item_id) if records else None
        if not isinstance(rec, dict) or not listed(rec):
            continue
        reader = _reader_of(rec)
        named = rec.get("applies") if isinstance(rec.get("applies"), str) and rec.get("applies") else None
        sig = rec.get("signature")
        applies = named is not None or _applies_stated(sig)
        if reader is None and not applies:
            continue
        if named is not None:
            applies_of[f.item_id] = named
        rows.append(
            {
                "id": f.item_id,
                "reader": reader,
                "applies": applies,
                "input": signature_text(sig) if isinstance(sig, dict) else "",
            },
        )
    for cid, doc in _capture_docs(captures).items():
        rec = doc["record"] if isinstance(doc["record"], dict) else {}
        if rec.get("outcome") == "failed" or not isinstance(doc["procedure"], str) or _capture_unlisted(rec):
            continue
        sig = doc["signature"] if isinstance(doc["signature"], dict) else {}
        target = CAPTURE_PREFIX + cid
        rows.append(
            {
                "id": target,
                "reader": target,
                "applies": _applies_stated(sig),
                "input": signature_text(sig),
            },
        )
    return {"version": 1, "procedures": rows, "applies_of": applies_of}


def librarian_files(
    tree: Path,
    *,
    status_of: Callable[[str], str] | None = None,
    records: Mapping[str, dict] | None = None,
    summary_of: SummaryOf | None = None,
    captures: Mapping[str, bytes] | None = None,
    strip: Strip | None = None,
) -> dict[str, bytes]:
    """The librarian's generated files of the copy at *tree* (``UNIFY_MEMORY_LIBRARIAN=on``): ``catalogue.md``,
    ``dispatch.json``, and the package with ``memory.ask`` (:data:`ASK_SOURCE` after :mod:`.library_helper`) and
    its stub."""
    tree = Path(tree)
    lib = discover(tree)
    return {
        RESERVED_INIT: package_source(librarian=True),
        RESERVED_STUB: stub_source(librarian=True),
        CATALOGUE_FILE: catalogue_text(
            tree,
            lib,
            status_of=status_of,
            records=records,
            summary_of=summary_of,
            captures=captures,
            strip=strip,
        ).encode("utf-8"),
        DISPATCH_FILE: _json_bytes(
            dispatch_data(lib, status_of=status_of, records=records, captures=captures),
        ),
    }


#: The section the generated package gains with the librarian on (``memory.ask``), appended to
#: :mod:`.library_helper`'s bytes; off, the package is that file byte for byte.
ASK_SOURCE = _HERE / "library_ask.py"

#: The contract runtime the package imports as ``memory._contract`` (standard library only), byte for byte.
CONTRACT_SOURCE = _HERE.parent / "common" / "contract_runtime.py"


def package_source(librarian: bool = False) -> bytes:
    """The generated ``memory/__init__.py``: :mod:`.library_helper` byte for byte, followed, with *librarian*, by
    the ``memory.ask`` section (:data:`ASK_SOURCE`)."""
    base = (_HERE / "library_helper.py").read_bytes()
    if not librarian:
        return base
    return base + b"\n\n" + ASK_SOURCE.read_bytes()


#: The stub's opening comment.
STUB_HEADER = (
    "memory: the library's typed contract, generated with every copy of the library from the package's own\n"
    "definitions; edits are discarded. help(memory) shows the same calls and types."
)


def stub_source(librarian: bool = False) -> bytes:
    """The generated ``memory/__init__.pyi``: the package's public contract as typed Python
    (:func:`unify.common.contract_runtime.render_pyi` over the package's own definitions): the types :class:`Fit`,
    :class:`Item` and :class:`Items`, then ``memory.ask`` (with *librarian*), ``memory.items`` and ``memory.get``,
    each with its docstring. The same code gives the same bytes."""
    from unify.common.contract_runtime import render_pyi

    from . import library_helper as lh

    namespace: dict[str, Any] = {"Fit": lh.Fit, "Item": lh.Item, "Items": lh.Items}
    if librarian:
        from . import library_ask

        namespace["ask"] = library_ask.ask
    namespace.update({"items": lh.items, "get": lh.get})
    return render_pyi(namespace, header=STUB_HEADER).encode("utf-8")


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
    librarian: bool = False,
    summary_of: SummaryOf | None = None,
    captures: Mapping[str, bytes] | None = None,
) -> dict[str, bytes]:
    """Every generated file of the copy of the library at *tree*, by relative path (module docstring).

    *librarian* (``UNIFY_MEMORY_LIBRARIAN=on``) adds the librarian's files (:func:`librarian_files`: the catalogue
    with *summary_of*'s summaries and the *captures* snapshot, the dispatch file, and the package with
    ``memory.ask``); off, every byte is as without it.

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

    out = {
        RESERVED_INIT: (_HERE / "library_helper.py").read_bytes(),
        RESERVED_CONTRACT: CONTRACT_SOURCE.read_bytes(),
        RESERVED_STUB: stub_source(),
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
    if librarian:
        out.update(
            librarian_files(
                tree,
                status_of=status_of,
                records=records,
                summary_of=summary_of,
                captures=captures,
                strip=strip,
            ),
        )
    return out
