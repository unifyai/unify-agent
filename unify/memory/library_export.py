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
  catalogue and the procedures' recognisers, and ``.memory/search.json``, the index its ``search`` ranks
  (:mod:`.library_search`) (:func:`librarian_files`), and ``memory.ask`` in the package and its stub.

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
    recipe_of,
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
    # a procedure's recipe, by the procedure and by the recipe itself: its card shows the call that applies it
    recipes: dict[str, str] = {}
    for item, rec in (records or {}).items():
        rid = recipe_of(rec)
        if rid:
            recipes[item] = recipes[rid] = rid

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
            # its evidence label (UNIFY_MEMORY_IMPURE_ITEMS: what its effects, tests and uses show; only when recorded)
            **({"label": rec["label"]} if rec and isinstance(rec.get("label"), dict) else {}),
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
            # the recipe that does a procedure's whole job on the current request (only where there is one)
            **({"recipe": recipes[f.item_id]} if f.item_id in recipes else {}),
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
# ``.memory/dispatch.json``, the structural catalogue: one row per stored procedure from its syntax and records (rev 2). Neither carries an answer, a raw
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


def _withdrawn_rows(captures: Mapping[str, bytes] | None) -> dict[str, dict]:
    """The snapshot's withdrawn captures (``.memory/captures/withdrawn.json``, :func:`.captures.export_files` under
    ``UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES=on``) by capture id; {} when there is none or it is unreadable."""
    from .captures import EXPORT_DIR, WITHDRAWN_FILE

    data = (captures or {}).get(f"{EXPORT_DIR}/{WITHDRAWN_FILE}")
    try:
        doc = json.loads(bytes(data).decode("utf-8")) if data is not None else None
    except (UnicodeDecodeError, TypeError, ValueError):
        return {}
    rows = doc.get("captures") if isinstance(doc, dict) else None
    return {str(k): v for k, v in rows.items() if isinstance(v, dict)} if isinstance(rows, dict) else {}


def in_place_of(captures: Mapping[str, bytes] | None) -> dict[str, list[tuple[str, dict]]]:
    """For each item offered in the place of withdrawn captures (their ``by``), those captures in id order, each
    with its withdrawn row (``asked``, ``docstring``, ``signature``...): the replacement is shown and found where
    they were."""
    out: dict[str, list[tuple[str, dict]]] = {}
    for cid, row in sorted(_withdrawn_rows(captures).items()):
        by = row.get("by")
        if isinstance(by, str) and by:
            out.setdefault(by, []).append((cid, row))
    return out


def worked_example(rec: Mapping | None) -> dict | None:
    """The exported record's worked-example mark (:data:`.captures.EXAMPLE_KEY`, under
    ``UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES=on``), or None: such a capture is read, never called."""
    from .captures import EXAMPLE_KEY

    mark = rec.get(EXAMPLE_KEY) if isinstance(rec, Mapping) else None
    return mark if isinstance(mark, dict) else None


def _function_docs(tree: Path, lib: Library) -> dict[str, str]:
    return _docstrings(tree, lib, {f.item_id for f in lib.functions})


def _evidence_row(serving: Mapping[str, Any] | None, iid: str) -> dict | None:
    """Item *iid*'s row of the serving evidence (``UNIFY_MEMORY_SERVING_FAILURE_RECORD``), or None."""
    if not serving:
        return None
    rows = (serving.get("evidence") or {}).get("items") or {}
    row = rows.get(iid)
    return row if isinstance(row, dict) else None


def failure_line(row: Mapping[str, Any] | None) -> str:
    """An item's failure record on one line: ``none``, or the count and each failure's why, episode and the library
    version that was served (ids shortened)."""
    rows = [r for r in (row or {}).get("failure_record") or [] if isinstance(r, dict)]
    if not rows:
        return "none"
    parts = [
        f"{r.get('why')} in episode {str(r.get('episode') or '?')[:8]}"
        + (f" (version {str(r['pin'])[:7]})" if r.get("pin") else "")
        for r in rows
    ]
    return f"{len(rows)}: " + "; ".join(parts)


def _serving_lines(serving: Mapping[str, Any] | None, iid: str) -> list[str]:
    """A card's evidence lines under ``UNIFY_MEMORY_SERVING_FAILURE_RECORD``: what its status rests on, its recorded
    failures, its contradictions and its stated conditions; [] with the switch off."""
    if not serving:
        return []
    row = _evidence_row(serving, iid) or {}
    from .trust import basis_text

    status = row.get("status_text") or basis_text("on-trial", "no_outcome_signals", {})
    out = [f"Status by evidence: {status}\n", f"Recorded failures: {failure_line(row)}\n"]
    for c in row.get("contradictions") or []:
        out.append(f"Contradicted: {c}\n")
    if row.get("conditions"):
        out.append(f"States that it applies when: {row['conditions']}\n")
    return out


def items_in_place_of(serving: Mapping[str, Any] | None) -> dict[str, list[tuple[str, dict]]]:
    """For each library item offered in the place of withdrawn library items (their ``by``), those items in id order
    with their withdrawn rows (``UNIFY_MEMORY_SERVING_FAILURE_RECORD`` and the withdrawal switch on)."""
    out: dict[str, list[tuple[str, dict]]] = {}
    for iid, row in sorted(((serving or {}).get("withdrawn_items") or {}).items()):
        by = row.get("by") if isinstance(row, dict) else None
        if isinstance(by, str) and by:
            out.setdefault(by, []).append((iid, row))
    return out


def _item_in_place_lines(rows: list[tuple[str, dict]] | None, docs: Mapping[str, str], strip: Strip) -> list[str]:
    """A card's lines for the withdrawn library items it is offered in place of: each one's id and what it said it
    was for."""
    out = []
    for iid, row in rows or []:
        doc = _one_line(strip(_first_paragraph(docs.get(iid, ""))))
        out.append(f"- In place of {iid} ({row.get('status') or 'withdrawn'}): {doc or 'no docstring recorded'}\n")
    return out


def _in_place_lines(rows: list[tuple[str, dict]] | None, strip: Strip) -> list[str]:
    """A card's lines for the withdrawn captures it is offered in place of: each one's id and what it was asked."""
    out = []
    for cid, row in rows or []:
        asked = _one_line(strip(str(row.get("asked") or "")))
        status = row.get("status") or "closed"
        out.append(f"- In place of {CAPTURE_PREFIX}{cid} ({status}): Asked: {asked or 'not recorded'}\n")
    return out


def catalogue_text(
    tree: Path,
    lib: Library | None = None,
    *,
    status_of: Callable[[str], str] | None = None,
    records: Mapping[str, dict] | None = None,
    summary_of: SummaryOf | None = None,
    captures: Mapping[str, bytes] | None = None,
    strip: Strip | None = None,
    serving: Mapping[str, Any] | None = None,
) -> str:
    """``.memory/catalogue.md``: the library's whole catalogue as the librarian reads it (design J(b).3).

    One card per listed function in module-path order (its id, signature and the first paragraph of its docstring;
    what it was saved against, the identifier-stripped ``## Asked`` and ``## Approach`` lines of each provenance
    episode's ``summary.md`` from *summary_of*; its evidence level and status, its reader, whether it has
    ``applies``, its input line and a one-line track record; the notes it links to). Then the notes, then the
    captured procedures from *captures* (the actor's snapshot, ``.memory/captures/...``), each at level
    ``unchecked``; a captured attempt that failed is listed with what broke and is never offered to run. Card fields
    are fixed; the only free text is the docstrings and the summaries, identifiers removed with *strip*. Pure: the
    same inputs give the same bytes. With *serving* (``UNIFY_MEMORY_SERVING_FAILURE_RECORD``: the evidence and the
    withdrawn library items), each card also says what its status rests on, its recorded failures, its
    contradictions and its stated conditions, and a library item offered in the place of withdrawn ones names them;
    a withdrawn item has no card."""
    from .library_index import signature_text

    tree = Path(tree)
    lib = lib if lib is not None else discover(tree)
    strip = strip or default_strip()
    links = build_links(lib)
    docs = _function_docs(tree, lib)
    gone_items = set(((serving or {}).get("withdrawn_items") or {}))
    replaced_items = items_in_place_of(serving)
    status = {
        **{f.item_id: _status(status_of, f.item_id) for f in lib.functions},
        **{n.item_id: _status(status_of, n.item_id) for n in lib.notes},
    }
    shown = {i for i, s in status.items() if s not in HIDDEN and i not in gone_items}
    replaced = in_place_of(captures)
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
        lines += _in_place_lines(replaced.get(f.item_id), strip)
        lines += _item_in_place_lines(replaced_items.get(f.item_id), docs, strip)
        input_line = signature_text(sig) if isinstance(sig, dict) else ""
        lines.append(
            f"Evidence: level {level}; status {status[f.item_id]}; reader {reader or _no_reader(rec)}; "
            f"applies: {'yes' if _applies_stated(sig) else 'no'}\n",
        )
        lines += _serving_lines(serving, f.item_id)
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
            + "".join(_serving_lines(serving, n.item_id))
            + ("Functions: " + ", ".join(fns) + "\n" if fns else ""),
        )
    out.append("\n## Captured procedures\n")
    caps = _capture_docs(captures)
    if not caps:
        out.append("\n(none)\n")
    for cid, doc in caps.items():
        sig = doc["signature"] if isinstance(doc["signature"], dict) else {}
        rec = doc["record"] if isinstance(doc["record"], dict) else {}
        failed = rec.get("outcome") in ("failed", "contrary")
        lines = [f"\n### {CAPTURE_PREFIX}{cid}\n"]
        run_doc = _one_line(strip(_first_paragraph(_docstring_of(doc["procedure"]))))
        if run_doc:
            lines.append(f"{run_doc}\n")
        goal = _one_line(sig.get("goal")) if isinstance(sig.get("goal"), str) else ""
        lines.append(f"Saved against:\n- Asked: {goal or 'not recorded'}\n")
        lines += _in_place_lines(replaced.get(CAPTURE_PREFIX + cid), strip)
        input_line = signature_text(sig)
        level = _UNCHECKED  # captured procedures are listed at level unchecked (design J(b).3)
        state = rec.get("status") if isinstance(rec.get("status"), str) else "on-trial"
        example = worked_example(rec)
        lines.append(
            f"Evidence: level {level}; status {state}; "
            + ("reader none (a worked example)" if example else "reader read_input")
            + f"; applies: {'yes' if _applies_stated(sig) and not example else 'no'}"
            + ("; tried and failed: never offered to run" if failed else "")
            + (f"; a worked example, never called: {example.get('why')}" if example else "")
            + "\n",
        )
        lines += _serving_lines(serving, CAPTURE_PREFIX + cid)
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


def _code_facts(source: str | None, applies_source: str | None, strip: Strip) -> dict:
    """What a procedure's code says of itself, from its syntax alone (nothing run): the ``env.<channel>.<method>``
    chains its entry function's tool namespace names (``effects``), whether its entry reads its input
    (``reads_input``) and whether it gives a fixed answer (``constant``, :func:`.fixed_answer.fixed_answer`), and its
    ``applies`` source as text through the identifier rule (``applies_text``)."""
    from .capture_shape import entry_env_chains
    from .fixed_answer import NEVER_READS, fixed_answer

    fixed = fixed_answer(source) if isinstance(source, str) else None
    return {
        "effects": entry_env_chains(source) if isinstance(source, str) else [],
        "reads_input": not (fixed is not None and fixed.get("why") == NEVER_READS),
        "constant": fixed is not None,
        "applies_text": strip(applies_source) if isinstance(applies_source, str) and applies_source else "",
    }


def _function_source(source: str | None, name: str) -> str | None:
    """The source of the module-level function *name* of *source* (None when absent or unparseable)."""
    if not isinstance(source, str):
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ast.unparse(node)
    return None


def procedure_bodies(tree: Path, records: Mapping[str, dict] | None = None) -> dict[str, str]:
    """Item id -> source of every public function of the copy at *tree* (:func:`.layout.function_bodies`), and of each
    function an item record names as its ``applies`` (a private one included), read from its module's file."""
    out = dict(function_bodies(tree))
    for rec in (records or {}).values():
        named = rec.get("applies") if isinstance(rec, Mapping) else None
        if not isinstance(named, str) or ":" not in named or named in out:
            continue
        module, _, name = named.partition(":")
        try:
            text = (Path(tree) / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found = _function_source(text, name)
        if found is not None:
            out[named] = found
    return out


def _outcome_counts(rec: Mapping | None) -> dict:
    """A capture's occurrences by outcome (``positive``, ``contrary``, ``none``), from its record's cases."""
    out = {"positive": 0, "contrary": 0, "none": 0}
    for c in (rec.get("cases") if isinstance(rec, Mapping) else None) or []:
        if isinstance(c, Mapping):
            o = c.get("outcome")
            out["contrary" if o in ("contrary", "failed") else "positive" if o == "positive" else "none"] += 1
    return out


def dispatch_data(
    lib: Library,
    *,
    status_of: Callable[[str], str] | None = None,
    records: Mapping[str, dict] | None = None,
    captures: Mapping[str, bytes] | None = None,
    bodies: Mapping[str, str] | None = None,
    strip: Strip | None = None,
) -> dict:
    """``.memory/dispatch.json`` (rev 2): the structural catalogue, one row per stored procedure, from the code's
    syntax and the harness's records only (nothing is run to make it, and nothing reads it to run anything):

    ``{id, kind, form, signature, applies_text, effects, reads_input, constant, occurrences, outcomes, provenance}``:
    ``kind`` is ``function`` or ``capture-shape``; ``form`` a capture's entry form (``value``, ``state``,
    ``state_env``) or ``function``; ``signature`` the input line (:func:`.library_index.signature_text`) and
    ``applies_text`` the ``applies`` source, both through the identifier rule; ``effects``, ``reads_input`` and
    ``constant`` from the syntax (:func:`_code_facts`); for a capture, ``occurrences`` (its folded occurrences),
    ``outcomes`` (their counts by signal outcome) and ``provenance`` (how many episodes and segments it came from;
    never an id).

    Module-path order, then the captures in id order. A function its record does not list, a hidden one and a
    procedure's recipe are left out, as are a captured attempt that failed or met a contrary signal and a captured
    procedure whose landing is not listed. *bodies* are the functions' sources by item id
    (:func:`.layout.function_bodies`); without them a function's code facts are empty."""
    from .library_index import signature_text

    strip = strip or default_strip()
    rows: list[dict] = []
    recipes = recipe_ids(records)
    for f in sorted(lib.functions, key=lambda f: (f.module, f.name)):
        if _status(status_of, f.item_id) in HIDDEN or f.item_id in recipes:
            continue
        rec = records.get(f.item_id) if records else None
        if not isinstance(rec, dict) or not listed(rec):
            continue
        named = rec.get("applies") if isinstance(rec.get("applies"), str) and rec.get("applies") else None
        sig = rec.get("signature")
        source = (bodies or {}).get(f.item_id)
        applies_source = (bodies or {}).get(named) if named else None
        rows.append(
            {
                "id": f.item_id,
                "kind": "function",
                "form": "function",
                "signature": strip(signature_text(sig)) if isinstance(sig, dict) else "",
                **_code_facts(source, applies_source, strip),
                "occurrences": None,
                "outcomes": None,
                "provenance": None,
            },
        )
    for cid, doc in _capture_docs(captures).items():
        rec = doc["record"] if isinstance(doc["record"], dict) else {}
        if rec.get("outcome") in ("failed", "contrary") or not isinstance(doc["procedure"], str) or _capture_unlisted(rec):
            continue
        sig = doc["signature"] if isinstance(doc["signature"], dict) else {}
        entry = rec.get("entry_point") if isinstance(rec.get("entry_point"), dict) else {}
        cases = [c for c in rec.get("cases") or [] if isinstance(c, dict)]
        episodes = rec.get("episodes") if isinstance(rec.get("episodes"), list) else []
        rows.append(
            {
                "id": CAPTURE_PREFIX + cid,
                "kind": "capture-shape",
                "form": entry.get("form") if isinstance(entry.get("form"), str) else "value",
                "signature": strip(signature_text(sig)),
                **_code_facts(doc["procedure"], _function_source(doc["procedure"], "applies"), strip),
                "occurrences": rec.get("occurrences") if isinstance(rec.get("occurrences"), int) else len(episodes),
                "outcomes": _outcome_counts(rec),
                "provenance": {
                    "episodes": len({c.get("episode") for c in cases if c.get("episode")}) or len(episodes),
                    "segments": sum(1 for c in cases if c.get("segment") is not None),
                },
            },
        )
    return {"version": 2, "procedures": rows}


def librarian_files(
    tree: Path,
    *,
    status_of: Callable[[str], str] | None = None,
    records: Mapping[str, dict] | None = None,
    summary_of: SummaryOf | None = None,
    captures: Mapping[str, bytes] | None = None,
    strip: Strip | None = None,
    search: Any = None,
    serving: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """The librarian's generated files of the copy at *tree* (``UNIFY_MEMORY_LIBRARIAN=on``): ``catalogue.md``,
    ``dispatch.json``, ``search.json`` (:func:`.library_search.index_files`, its vectors made with *search*, a
    :class:`.library_search.Backend`; None: none, the ranking lexical), and the package with ``memory.ask``
    (:data:`ASK_SOURCE` after :mod:`.library_helper`) and its stub."""
    from .library_search import index_files

    tree = Path(tree)
    lib = discover(tree)
    return {
        **index_files(
            tree,
            lib,
            status_of=status_of,
            records=records,
            captures=captures,
            use=search,
            **({"serving": serving} if serving else {}),
        ),
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
            serving=serving,
        ).encode("utf-8"),
        DISPATCH_FILE: _json_bytes(
            dispatch_data(lib, status_of=status_of, records=records, captures=captures,
                          bodies=procedure_bodies(tree, records), strip=strip),  # fmt: skip
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
    :class:`Item` and :class:`Items`, then ``memory.ask`` (with *librarian*), ``memory.items``, ``memory.get`` and
    ``memory.request``, each with its docstring. The same code gives the same bytes."""
    from unify.common.contract_runtime import render_pyi

    from . import library_helper as lh

    namespace: dict[str, Any] = {"Fit": lh.Fit, "Item": lh.Item, "Items": lh.Items}
    if librarian:
        from . import library_ask

        namespace["ask"] = library_ask.ask
    namespace.update({"items": lh.items, "get": lh.get, "request": lh.request})
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
    search: Any = None,
    serving: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Every generated file of the copy of the library at *tree*, by relative path (module docstring).

    *librarian* (``UNIFY_MEMORY_LIBRARIAN=on``) adds the librarian's files (:func:`librarian_files`: the catalogue
    with *summary_of*'s summaries and the *captures* snapshot, the dispatch file, the search index with *search*'s
    vectors, and the package with ``memory.ask``); off, every byte is as without it.

    *records* are the item records at the copy's commit (:func:`.item_records.records_at`); None gives the files
    without item records, byte for byte. *tests* reads a test file of the commit by its tree-relative path, to
    re-check each recorded example call before it is shown; None reads them from *tree* (a copy without its
    tests, as the actor's export is, then shows no example). *saved_against* gives what each function was saved
    against (:func:`saved_against_of`); None reads only the item records' ``signature`` (no summaries). A function
    with a signature gets an ``input:`` line in the index and a ``procedures`` row in ``find.json``.

    *serving* (``UNIFY_MEMORY_SERVING_FAILURE_RECORD``: ``{"evidence", "withdrawn_items"}``, :mod:`.serving`) adds
    the evidence file and the withdrawn library items' file, and every file then treats a withdrawn item as hidden
    (it leaves the index, ``find.json``, the catalogue, the search index and the dispatch file); None gives every
    byte as without it."""
    tree = Path(tree)
    lib = discover(tree)
    if serving:
        gone = set(serving.get("withdrawn_items") or {})
        if gone:
            base_status = status_of
            status_of = lambda item: "retired" if item in gone else _status(base_status, item)  # noqa: E731
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
    if serving:
        from .serving import EVIDENCE_FILE, WITHDRAWN_ITEMS_FILE, file_bytes

        out[EVIDENCE_FILE] = file_bytes(serving.get("evidence") or {"v": 1, "items": {}})
        out[WITHDRAWN_ITEMS_FILE] = file_bytes({"v": 1, "items": dict(serving.get("withdrawn_items") or {})})
    if librarian:
        out.update(
            librarian_files(
                tree,
                status_of=status_of,
                records=records,
                summary_of=summary_of,
                captures=captures,
                strip=strip,
                search=search,
                **({"serving": serving} if serving else {}),
            ),
        )
    return out
