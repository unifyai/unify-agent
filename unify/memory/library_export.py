"""The generated files of every library-mode copy of the library: the actor's export and the writer's box.

* ``memory/__init__.py``: :mod:`.library_helper`, byte for byte (``memory.get``, ``memory.items``, the ``Item``
  type and, for one build, the earlier calls as aliases), with ``memory/_contract.py``, the contract runtime it
  checks its calls with, and ``memory/__init__.pyi``, its typed stub (:func:`stub_source`);
* ``INDEX.md`` and ``links.json``: :mod:`.library_index`;
* ``.memory/items.json``: what ``memory.get`` reads per item, namely its kind, path, signature or title, status, a
  function's evidence level with the findings that set it, the verification and use records (None without item
  records), a function's example call (its record's, shown only while its test file still quotes it:
  :mod:`.example_view`) and the subjects of the commits that changed it;
* ``.memory/find.json``: the listed functions with the known shapes of their recorded inputs, each naming its
  parameter when that is known, keyed by body digest (:mod:`.shape_rows`); and, once any listed function has an applicability
  signature, ``procedures``: what each such function was saved against, for ``Item.fits`` (:func:`find_data`);
* ``.memory/matcher.py``: the shape comparison ``Item.fits`` uses (:mod:`.memory_helper` from its imports on);
* ``.memory/shapes.py``: :mod:`.analysis.shapes`, byte for byte;
* only: ``.memory/catalogue.md`` and ``.memory/dispatch.json``, the librarian's
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
    level_of,
    recipe_ids,
    recipe_of,
    recipe_text,
    support,
    support_text,
    use_totals,
)
from .item_states import ON_TRIAL
from .library_index import HIDDEN, build_links, render_index, render_links
from .recognition import listed, listing
from .redact import redact_error

HISTORY_SCAN = 200
HISTORY_SHOW = 5
_MATCHER_START = "from __future__ import annotations"
_HERE = Path(__file__).resolve().parent
History = tuple[dict[str, list[str]], bool]
#: What a function was saved against (:func:`saved_against_of`): ``signature`` (its applicability signature, the
#: capture store's ``signature.json`` shape), ``asked`` and ``approach`` (its source episode's ``summary.md``
#: sections) and ``level`` (its evidence level); None when it has no signature.
SavedAgainst = Callable[[str], "dict | None"]
#: An episode's ``summary.md`` text by episode id, None when it has none (the integration reads
#: ``<state>/memory-staging/<episode>/summary.md``).
SummaryOf = Callable[[str], "str | None"]


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


def summary_sections(text: str) -> dict[str, str]:
    """The ``Asked`` and ``Approach`` sections of a ``summary.md`` text (design H.4), as written."""
    return {
        "asked": _section(text, "Asked"),
        "approach": _section(text, "Approach"),
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


def matcher_source() -> bytes:
    """The shape matcher without its module docstring (``memory_helper.py`` from its imports on), behind a one-line
    docstring."""
    src = (_HERE / "memory_helper.py").read_text(encoding="utf-8")
    body = src[src.index(_MATCHER_START) :]
    return (
        '"""Shape comparison for Item.fits; generated by the harness, never edit it."""\n\n'
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


#: The parts of an item's evidence label that are never served: what its tests gave and the gate's findings.
UNSERVED_LABEL_KEYS = frozenset({"tests", "findings"})


def served_label(label: Mapping[str, Any]) -> dict:
    """An item's evidence label as the actor and the librarian may read it: without :data:`UNSERVED_LABEL_KEYS`."""
    return {k: v for k, v in label.items() if k not in UNSERVED_LABEL_KEYS}


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
            "use": use_totals(rec) if rec else None,
            # what it rests on: the episodes it comes from and its confirmed uses (every card shows it)
            "support": support(rec),
            # the episodes it comes from, by id: the recorded procedures of those episodes are the ones it generalises
            "episodes": _provenance_episodes(rec),
            "history": h[:HISTORY_SHOW],
            "history_more": max(0, len(h) - HISTORY_SHOW),
            # its evidence label as served (only when recorded): what the writer stated and what its uses show; the
            # results of its tests and the gate's findings are the writer's, never served
            **({"label": served_label(rec["label"])} if rec and isinstance(rec.get("label"), dict) else {}),
        }

    for f in lib.functions:
        rec = records.get(f.item_id) if records else None
        recognition = _recognition_line(rec)
        rows[f.item_id] = {
            "kind": "function",
            "path": f.path,
            "signature": f"{f.module}:{f.signature}",
            "summary": f.summary,
            # its evidence level (None: none recorded): unchecked, or confirmed by its uses' outcomes
            "level": level_of(records)(f.item_id) if records else None,
            # {"test", "line", "call"}: a call quoted from its tests, re-checked against them (None: none shown)
            "example": examples(f.item_id) if examples is not None else None,
            # whether Item.fits and the librarian list it, and why (only for a procedure tested when it landed)
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
            # its scope (front matter ``applies_to``/``stated_by``), shown wherever the note is served
            "applies_to": n.applies_to,
            "stated_by": n.stated_by,
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
) -> SavedAgainst:
    """What each function was saved against, from its item record: ``signature`` (the record's ``signature``, an
    applicability signature as the capture store writes it; a function without one gives None), ``level`` (the
    record's ``level``), and the ``Asked`` and ``Approach`` sections of the ``summary.md`` of the first of its
    provenance episodes that *summary_of* has one for, as written."""

    def get(item: str) -> dict | None:
        rec = records.get(item) if records else None
        sig = rec.get("signature") if isinstance(rec, dict) else None
        if not isinstance(sig, dict):
            return None
        out = {
            "signature": {k: v for k, v in sig.items() if k != "goal"},
            "asked": "",
            "approach": "",
            "level": rec.get("level") if isinstance(rec.get("level"), str) else None,
        }
        prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}
        episodes = prov.get("episodes") if isinstance(prov.get("episodes"), list) else []
        for episode in episodes if summary_of is not None else []:
            text = summary_of(str(episode)) if isinstance(episode, str) else None
            if text:
                out.update(summary_sections(text))
                break
        if not out["asked"] and isinstance(sig.get("goal"), str):
            out["asked"] = sig["goal"]
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
) -> dict:
    """``.memory/find.json``: each listed function with the recorded input shapes of its current body.

    ``input_shapes`` are the known shapes of its recorded inputs (:mod:`.shape_rows`), each naming its parameter
    (``"param"``) when that is known. *records* are accepted for the export's
    signature and not read: a function's shapes come only from *shapes*.

    With *saved_against*, each listed function it gives a signature for is also a row of ``procedures`` (in id
    order): ``{id, signature, docstring, asked, approach, level, status, summary}``, as written, which
    ``Item.fits`` compares with. A procedure its item record does not list (:func:`.recognition.listed`) is no
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
        docs = _docstrings(tree, lib, set(saved))
        data["procedures"] = [
            {
                "id": e["id"],
                "signature": saved[e["id"]].get("signature") or {},
                "docstring": docs.get(e["id"], ""),
                "asked": saved[e["id"]].get("asked") or "",
                "approach": saved[e["id"]].get("approach") or "",
                "level": saved[e["id"]].get("level") or "unknown",
                "status": _status(status_of, e["id"]),
                "summary": e.get("summary", ""),
            }
            for e in sorted(out, key=lambda e: e["id"])
            if e["id"] in saved
        ]
    return data


def _json_bytes(data: dict) -> bytes:
    return (
        json.dumps(data, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    ).encode("utf-8")


# --- the librarian's catalogue and the dispatch file --------------------------------
#
# Both are harness-generated at each export, with no model call: ``.memory/catalogue.md``, one card per stored
# function in module-path order (never a score), then the notes, then the captured procedures; and
# ``.memory/dispatch.json``, the structural catalogue: one row per stored procedure from its syntax and records
#. Neither carries an answer or a raw request text; the free texts (docstrings, summary sections, what broke)
# are shown as written, identifiers included (nothing strips or masks an identifier).

CATALOGUE_FILE = ".memory/catalogue.md"
DISPATCH_FILE = ".memory/dispatch.json"
#: The captures' prefix in the catalogue and in ``dispatch.json`` (the package's item ids use it too).
CAPTURE_PREFIX = "capture:"
_CAPTURE_ID = "[0-9a-f]{16}"


def _first_paragraph(doc: str) -> str:
    """The first paragraph of a docstring, on one line."""
    return " ".join((doc or "").strip().split("\n\n", 1)[0].split())


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split())


def _provenance_episodes(rec: Mapping | None) -> list[str]:
    prov = rec.get("provenance") if isinstance(rec, Mapping) else None
    eps = prov.get("episodes") if isinstance(prov, dict) else None
    return [e for e in eps if isinstance(e, str)] if isinstance(eps, list) else []


def _saved_lines(rec: Mapping | None, summary_of: SummaryOf | None) -> list[tuple[str, str]]:
    """``(asked, approach)`` of each provenance episode of *rec* that has a ``summary.md``, as written, without
    repeats, in the record's episode order; else the signature's own goal as Asked."""
    out: list[tuple[str, str]] = []
    for episode in _provenance_episodes(rec) if summary_of is not None else []:
        text = summary_of(episode)
        if not text:
            continue
        s = summary_sections(text)
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
    """Whether ``Item.fits`` and the librarian list a procedure tested when it landed, and why
    (:func:`.recognition.listing`); ``""`` for an item with no recognition record."""
    got = rec.get("recognition") if isinstance(rec, Mapping) else None
    return listing(got)[1] if isinstance(got, Mapping) else ""


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
    by capture id; {} when there is none or it is unreadable."""
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


def _function_docs(tree: Path, lib: Library) -> dict[str, str]:
    return _docstrings(tree, lib, {f.item_id for f in lib.functions})


def _evidence_row(serving: Mapping[str, Any] | None, iid: str) -> dict | None:
    """Item *iid*'s row of the serving evidence, or None."""
    if not serving:
        return None
    rows = (serving.get("evidence") or {}).get("items") or {}
    row = rows.get(iid)
    return row if isinstance(row, dict) else None


def capture_support(serving: Mapping[str, Any] | None, cid: str, rec: Mapping | None) -> dict | None:
    """What recorded procedure *cid* rests on: its serving evidence row's ``support`` (its episodes, the accepted ones
    and its confirmed uses, :func:`.serving.capture_support`), else its exported record's (its episodes and the
    accepted ones; its uses are not known without the evidence); None when neither has it."""
    row = _evidence_row(serving, CAPTURE_PREFIX + cid) or {}
    got = row.get("support") if isinstance(row.get("support"), Mapping) else None
    if got is None and isinstance(rec, Mapping) and isinstance(rec.get("support"), Mapping):
        got = rec["support"]
    return dict(got) if got is not None else None


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
    """A card's evidence lines: what its status rests on, its recorded failures, its later uses and its stated
    conditions; [] with the switch off."""
    if not serving:
        return []
    row = _evidence_row(serving, iid) or {}
    from .trust import basis_text

    status = row.get("status_text") or basis_text("on-trial", "no_outcome_signals", {})
    out = [
        f"Status by evidence: {status}\n",
        f"Recorded failures: {failure_line(row)}\n",
        f"Later uses: {row.get('later_uses_text') or 'none yet'}\n",
    ]
    if row.get("conditions"):
        out.append(f"States that it applies when: {row['conditions']}\n")
    return out


def items_in_place_of(serving: Mapping[str, Any] | None) -> dict[str, list[tuple[str, dict]]]:
    """For each library item offered in the place of withdrawn library items (their ``by``), those items in id order
    with their withdrawn rows ."""
    out: dict[str, list[tuple[str, dict]]] = {}
    for iid, row in sorted(((serving or {}).get("withdrawn_items") or {}).items()):
        by = row.get("by") if isinstance(row, dict) else None
        if isinstance(by, str) and by:
            out.setdefault(by, []).append((iid, row))
    return out


def _item_in_place_lines(rows: list[tuple[str, dict]] | None, docs: Mapping[str, str]) -> list[str]:
    """A card's lines for the withdrawn library items it is offered in place of: each one's id and what it said it
    was for."""
    out = []
    for iid, row in rows or []:
        doc = _one_line(_first_paragraph(docs.get(iid, "")))
        out.append(f"- In place of {iid} ({row.get('status') or 'withdrawn'}): {doc or 'no docstring recorded'}\n")
    return out


def _in_place_lines(rows: list[tuple[str, dict]] | None) -> list[str]:
    """A card's lines for the withdrawn captures it is offered in place of: each one's id and what it was asked."""
    out = []
    for cid, row in rows or []:
        asked = _one_line(str(row.get("asked") or ""))
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
    serving: Mapping[str, Any] | None = None,
) -> str:
    """``.memory/catalogue.md``: the library's whole catalogue as the librarian reads it (design J(b).3).

    One card per listed function in module-path order (its id, signature and the first paragraph of its docstring;
    what it was saved against, the ``## Asked`` and ``## Approach`` lines of each provenance
    episode's ``summary.md`` from *summary_of*; its evidence level and status, its reader, whether it has
    ``applies``, its input line and a one-line track record; the notes it links to). Then the notes, then the
    captured procedures from *captures* (the actor's snapshot, ``.memory/captures/...``), each at level
    ``unchecked``; a captured attempt that failed is listed with what broke and is never offered to run. Card fields
    are fixed; the only free text is the docstrings and the summaries, as written. Pure: the
    same inputs give the same bytes. With *serving* (the evidence and the
    withdrawn library items), each card also says what its status rests on, its recorded failures, its
    contradictions and its stated conditions, and a library item offered in the place of withdrawn ones names them;
    a withdrawn item has no card."""
    from .library_index import signature_text

    tree = Path(tree)
    lib = lib if lib is not None else discover(tree)
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
        "Saved against: what the work each item was built from asked and how it went about it. Evidence: its kind "
        "(curated item or recorded procedure), the episodes it comes from and the uses whose outcome "
        "confirmed it, its status, and how it has been used.\n",
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
        lines = [f"\n### {f.item_id}\n", f"`{f.module}:{f.signature}`\n"]
        doc = _one_line(_first_paragraph(docs.get(f.item_id, "")))
        if doc:
            lines.append(f"{doc}\n")
        saved = _saved_lines(rec, summary_of)
        lines.append("Saved against:" + ("" if saved else " no summary recorded") + "\n")
        for asked, approach in saved:
            lines.append(f"- Asked: {asked or 'not recorded'}\n")
            if approach:
                lines.append(f"  Approach: {approach}\n")
        lines += _in_place_lines(replaced.get(f.item_id))
        lines += _item_in_place_lines(replaced_items.get(f.item_id), docs)
        input_line = signature_text(sig) if isinstance(sig, dict) else ""
        lines.append(
            f"Evidence: {support_text('curated item', support(rec))}; status {status[f.item_id]}; "
            f"reader {reader or _no_reader(rec)}; "
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
            + (f"{_one_line(n.title)}\n" if n.title else "")
            + (f"{_one_line(about)}\n" if about else "")
            + (f"Scope: {n.scope}\n" if n.scope and not n.error else "")
            + f"Status: {status[n.item_id]}; "
            + support_text("curated item", support(records.get(n.item_id) if records else None))
            + "\n"
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
        run_doc = _one_line(_first_paragraph(_docstring_of(doc["procedure"])))
        if run_doc:
            lines.append(f"{run_doc}\n")
        goal = _one_line(sig.get("goal")) if isinstance(sig.get("goal"), str) else ""
        lines.append(f"Saved against:\n- Asked: {goal or 'not recorded'}\n")
        lines += _in_place_lines(replaced.get(CAPTURE_PREFIX + cid))
        input_line = signature_text(sig)
        lines.append(
            f"Evidence: {support_text('recorded procedure', capture_support(serving, cid, rec))}; "
            + _capture_reader(rec, doc.get("procedure"))
            + f"; applies: {'yes' if _applies_stated(sig) else 'no'}"
            + ("; tried and failed: never offered to run" if failed else "")
            + "\n",
        )
        lines += _serving_lines(serving, CAPTURE_PREFIX + cid)
        if input_line:
            lines.append(f"Input: {input_line}\n")
        if failed:
            for line in (doc["broke"] or "").splitlines():
                line = _one_line(line.strip().lstrip("-* "))
                if line and line.lower() != "none":
                    lines.append(f"Broke: {line}\n")
        out.append("".join(lines))
    return "".join(out)


def _provenance_episodes(rec: Mapping | None) -> list[str]:
    """An item's provenance episodes (its record's ``provenance.episodes``), in order; [] when none are recorded."""
    prov = rec.get("provenance") if isinstance(rec, Mapping) else None
    eps = prov.get("episodes") if isinstance(prov, Mapping) else None
    return [e for e in eps if isinstance(e, str)] if isinstance(eps, list) else []


def _capture_reader(rec: Mapping, source: Any) -> str:
    """How a recorded procedure takes a request, as its catalogue line says it (from its record's entry form and its
    code's own definitions; nothing is run)."""
    entry = rec.get("entry_point") if isinstance(rec.get("entry_point"), Mapping) else {}
    form = entry.get("form")
    reads = False
    if isinstance(source, str):
        import ast

        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            tree = None
        reads = tree is not None and any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "read_input" for n in tree.body
        )
    if form == "state_env":
        return (
            "acts through the episode's tools (run(state, env)): no call runs it and nothing supplies env; "
            "env.<name>.<method>(...) in its code stands for a call its request made of <name> through whatever "
            "reached <name> there (not necessarily a harness tool); print(memory.get(id).text) shows the code whose "
            "calls, made in a cell through whatever reaches <name> here, do its work"
        )
    if form == "state":
        return "reader: the request's state (.request, .messages, .last, .data); a no-argument call runs it on the request"
    if reads:
        return "reader read_input; a no-argument call runs it on the request"
    return "no reader: it is called on a value"


def _code_facts(source: str | None, applies_source: str | None) -> dict:
    """What a procedure's code names, from its syntax alone (nothing run, nothing judged): the
    ``env.<channel>.<method>`` chains its tool namespace names in its entry function and in every helper it passes the
    namespace to (``calls``, :func:`.capture_shape.entry_env_chains`, as its shape address reads them), and its
    ``applies`` source as written (``applies_text``). No judgement of the code (pure or not, a fixed answer or not,
    whether it reads its input) is made: none can be told reliably, and the librarian ranks from these rows."""
    from .capture_shape import entry_env_chains

    return {
        "calls": entry_env_chains(source) if isinstance(source, str) else [],
        "applies_text": applies_source if isinstance(applies_source, str) and applies_source else "",
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
) -> dict:
    """``.memory/dispatch.json``: the structural catalogue, one row per stored procedure, from the code's
    syntax and the harness's records only (nothing is run to make it, and nothing reads it to run anything):

    ``{id, kind, form, signature, applies_text, calls, occurrences, outcomes, provenance}``:
    ``kind`` is ``function`` or ``capture-shape``; ``form`` a capture's entry form (``value``, ``state``,
    ``state_env``) or ``function``; ``signature`` the input line (:func:`.library_index.signature_text`) and
    ``applies_text`` the ``applies`` source, both as written; ``calls`` from the syntax (:func:`_code_facts`); for a
    capture, ``occurrences`` (its folded occurrences), ``outcomes`` (their counts by signal outcome) and ``provenance`` (how many episodes it came from;
    never an id).

    Module-path order, then the captures in id order. A function its record does not list, a hidden one and a
    procedure's recipe are left out, as is a captured attempt that failed or met a contrary signal. *bodies* are the functions' sources by item id
    (:func:`.layout.function_bodies`); without them a function's code facts are empty."""
    from .library_index import signature_text

    rows: list[dict] = []
    recipes = recipe_ids(records)
    docs = _capture_docs(captures)
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
        # a procedure landed from a capture: that capture's occurrences and their outcomes, from signals
        landed = rec.get("landed_from") if isinstance(rec.get("landed_from"), Mapping) else {}
        cap = docs.get(landed.get("capture")) if isinstance(landed.get("capture"), str) else None
        cap_rec = cap["record"] if cap is not None and isinstance(cap["record"], dict) else None
        rows.append(
            {
                "id": f.item_id,
                "kind": "function",
                "form": "function",
                "signature": signature_text(sig) if isinstance(sig, dict) else "",
                **_code_facts(source, applies_source),
                "occurrences": (cap_rec.get("occurrences") if isinstance(cap_rec.get("occurrences"), int)
                                else len(cap_rec.get("episodes") or [])) if cap_rec is not None else None,
                "outcomes": _outcome_counts(cap_rec) if cap_rec is not None else None,
                "provenance": None,
            },
        )
    for cid, doc in docs.items():
        rec = doc["record"] if isinstance(doc["record"], dict) else {}
        if rec.get("outcome") in ("failed", "contrary") or not isinstance(doc["procedure"], str):
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
                "signature": signature_text(sig),
                **_code_facts(doc["procedure"], _function_source(doc["procedure"], "applies")),
                "occurrences": rec.get("occurrences") if isinstance(rec.get("occurrences"), int) else len(episodes),
                "outcomes": _outcome_counts(rec),
                "provenance": {
                    "episodes": len({c.get("episode") for c in cases if c.get("episode")}) or len(episodes),
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
    search: Any = None,
    serving: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """The librarian's generated files of the copy at *tree*: ``catalogue.md``,
    ``dispatch.json``, ``search.json`` (:func:`.library_search.index_files`, its vectors made with *search*, a
    :class:`.library_search.Backend`; None: none, and a search then ranks nothing), and the package with ``memory.ask``
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
            serving=serving,
        ).encode("utf-8"),
        DISPATCH_FILE: _json_bytes(
            dispatch_data(lib, status_of=status_of, records=records, captures=captures,
                          bodies=procedure_bodies(tree, records)),  # fmt: skip
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
    librarian: bool = False,
    summary_of: SummaryOf | None = None,
    captures: Mapping[str, bytes] | None = None,
    search: Any = None,
    serving: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Every generated file of the copy of the library at *tree*, by relative path (module docstring).

    *librarian* adds the librarian's files (:func:`librarian_files`: the catalogue
    with *summary_of*'s summaries and the *captures* snapshot, the dispatch file, the search index with *search*'s
    vectors, and the package with ``memory.ask``); off, every byte is as without it.

    *records* are the item records at the copy's commit (:func:`.item_records.records_at`); None gives the files
    without item records, byte for byte. *tests* reads a test file of the commit by its tree-relative path, to
    re-check each recorded example call before it is shown; None reads them from *tree* (a copy without its
    tests, as the actor's export is, then shows no example). *saved_against* gives what each function was saved
    against (:func:`saved_against_of`); None reads only the item records' ``signature`` (no summaries). A function
    with a signature gets an ``input:`` line in the index and a ``procedures`` row in ``find.json``.

    *serving* (``{"evidence", "withdrawn_items"}``, :mod:`.serving`) adds
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
        saved_against = saved_against_of(records, None)

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
            example_of if records is not None else None,
            signature_of if saved_against is not None else None,
            (lambda item: support(records.get(item))) if records is not None else None,
        ).encode("utf-8"),
        LINKS_FILE: render_links(links).encode("utf-8"),
        ITEMS_FILE: _json_bytes(
            items_data(lib, status_of, found, complete, records, shown),
        ),
        FIND_FILE: _json_bytes(
            find_data(tree, lib, status_of, shapes, records, saved_against),
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
                search=search,
                **({"serving": serving} if serving else {}),
            ),
        )
    return out
