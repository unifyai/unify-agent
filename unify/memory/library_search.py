"""Search over the library's "when to use" text: the index each export writes, and the ranking ``search`` returns.

Every listed item of a library copy has one row in ``.memory/search.json`` (:func:`index_files`): its id, its kind,
its path in the copy, a one-line card (its first docstring line or description and, for a procedure with a recipe,
the call that applies it to the current request: :func:`.item_records.recipe_call`) and the text it is searched by:
its "When to use" section (:mod:`.when_to_use`), else its first docstring paragraph or description. A function that
is a procedure's recipe has no row of its own (its procedure's card names it), as in the dispatch file.

With an embedding backend (:func:`backend`), the export also stores one vector per row, computed with the harness's
embedder (:func:`unify.common.embeddings.embed`, whose cache makes a commit's vectors a one-time cost) and labelled
with its model; the vectors live only in the copy, never in a prompt, and a search ranks by meaning. Without one (no
backend configured, the paper profile, or an embedding call that failed) the index holds the rows alone and says why,
and :func:`search_text` ranks nothing: it lists every row in the library's order, says why, and leaves the judgement to
the reader of the cards (no word matching ranks anything). Each request an embedding sends to a provider is
told to *on_call* (:func:`backend`), which books it (``integration.cost``).

Nothing here caps what is returned: every row is listed, and a long result is paged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The index file of a library copy.
INDEX_FILE = ".memory/search.json"
#: The library copy as the librarian's box sees it.
LIBRARY_ROOT = "/memory"
#: The bytes of one page of a result (as a cell's output is shown).
PAGE_BYTES = 8000
#: Why the rows are not ranked.
NOT_CONFIGURED = "no embedding backend is configured"
OTHER_MODEL = "the index was built with another embedding model"
_ROUND = 6


@dataclass(frozen=True)
class Backend:
    """How texts are embedded: *model* labels the vectors (None: none are made, and *why* says why) and *embed*
    turns texts into unit vectors."""

    model: str | None
    why: str = ""
    embed: Callable[[list[str]], Any] | None = None


NO_BACKEND = Backend(None, NOT_CONFIGURED)


def backend(on_call: Callable[[dict], None] | None = None) -> Backend:
    """The run's embedding backend: the harness's embedder (:func:`unify.common.embeddings.embedder`), each request it
    sends told to *on_call*."""
    from unify.common import embeddings

    def embed(texts: list[str]) -> Any:
        return embeddings.embed(list(texts), on_call=on_call)

    return Backend(embeddings.embedder().model, "", embed)


# --- the rows ----------------------------------------------------------------------------------------------------


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def _first_paragraph(doc: str) -> str:
    return _one_line((doc or "").strip().split("\n\n", 1)[0])


def _section_line(section: str | None) -> str:
    return _one_line(section) if section else ""


def rows(
    tree: Path,
    lib: Any = None,
    *,
    status_of: Callable[[str], str] | None = None,
    records: Mapping[str, dict] | None = None,
    captures: Mapping[str, bytes] | None = None,
    serving: Mapping[str, Any] | None = None,
) -> list[dict]:
    """One row per listed item of the copy at *tree*, in module-path order, then the notes, then the captured
    procedures (*captures*: the snapshot's files): ``{id, kind, path, card, text}``. Retired items have none, nor
    has a procedure's recipe. An item offered in the place of withdrawn captures (the snapshot's ``withdrawn.json``)
    also carries what each was asked in its text, so it is found where they were; a captured procedure is a row of
    kind ``captured procedure``. With *serving*, each card also ends with what its status rests on and its recorded
    failures, and an item offered in the place of withdrawn library items carries their docstrings in its text. Cards
    and texts are as written, identifiers included."""
    from . import when_to_use as wtu
    from .item_records import recipe_call, recipe_of
    from .layout import discover
    from .library_export import (
        CAPTURE_PREFIX,
        _capture_docs,
        capture_support,
        _docstring_of,
        _docstrings,
        _status,
        in_place_of,
        record_why,
    )
    from .item_records import support, support_text
    from .library_index import HIDDEN

    tree = Path(tree)
    lib = lib if lib is not None else discover(tree)
    recipes = {r for r in (recipe_of(rec) for rec in (records or {}).values()) if r}
    docs = _docstrings(tree, lib, {f.item_id for f in lib.functions})
    out: list[dict] = []
    for f in sorted(lib.functions, key=lambda f: (f.module, f.name)):
        if _status(status_of, f.item_id) in HIDDEN or f.item_id in recipes:
            continue
        doc = docs.get(f.item_id, "")
        summary = _one_line(f.summary) or _first_paragraph(doc)
        recipe = recipe_of(records.get(f.item_id) if records else None)
        card = summary + (f" | applies to the current request: {recipe_call(recipe)}" if recipe else "")
        card += " | " + support_text("curated item", support(records.get(f.item_id) if records else None))
        text = _section_line(wtu.docstring_section(doc)) or _first_paragraph(doc) or summary
        out.append({"id": f.item_id, "kind": "function", "path": f.path, "card": card, "text": text})
    for n in sorted(lib.notes, key=lambda n: n.item_id):
        if _status(status_of, n.item_id) in HIDDEN:
            continue
        try:
            body = (tree / n.item_id).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            body = ""
        about = _one_line(n.description) or _one_line(n.title)
        if n.scope:
            about += f" ({n.scope})"
        text = _section_line(wtu.note_section(body)) or about
        card = about + " | " + support_text("curated item", support(records.get(n.item_id) if records else None))
        out.append({"id": n.item_id, "kind": "note", "path": n.item_id, "card": card, "text": text})
    for cid, doc in _capture_docs(captures).items():
        sig = doc["signature"] if isinstance(doc["signature"], dict) else {}
        rec = doc["record"] if isinstance(doc["record"], dict) else {}
        goal = _one_line(sig.get("goal")) if isinstance(sig.get("goal"), str) else ""
        run_doc = _first_paragraph(_docstring_of(doc["procedure"])) if isinstance(doc["procedure"], str) else ""
        negative = rec.get("outcome") in ("failed", "contrary")
        why = record_why(rec)
        card = (run_doc or goal) + (" | tried and failed: never offered to run" if negative else "")
        if why and not negative:
            # a record (capture_shape.record_kind): what was done last time, never a fit for a different request
            card += f" | a record of what was done last time, not a procedure: {why}"
        card += " | " + support_text("record" if why else "recorded procedure", capture_support(serving, cid, rec))
        out.append(
            {
                "id": CAPTURE_PREFIX + cid,
                "kind": "captured procedure",
                "path": f".memory/captures/{cid}/procedure.py",
                "card": card,
                "text": " ".join(t for t in (goal, run_doc) if t),
            },
        )
    replaced = in_place_of(captures)
    for row in out:
        for _cid, gone in replaced.get(row["id"], []):
            extra = " ".join(_one_line(gone.get(k)) for k in ("asked", "docstring") if _one_line(gone.get(k)))
            if extra:
                row["text"] = f"{row['text']} {extra}".strip()
    if serving:
        from .library_export import _evidence_row, failure_line, items_in_place_of
        from .library_helper import use_counts_text

        replaced_items = items_in_place_of(serving)
        for row in out:
            ev = _evidence_row(serving, row["id"]) or {}
            if ev.get("status_text"):
                row["card"] = f"{row['card']} | status by evidence: {ev['status_text']}"
            row["card"] = f"{row['card']} | {use_counts_text(ev.get('counts'))}"
            row["card"] = f"{row['card']} | recorded failures: {failure_line(ev)}"
            row["card"] = f"{row['card']} | later uses: {ev.get('later_uses_text') or 'none yet'}"
            for old, _gone in replaced_items.get(row["id"], []):
                extra = _first_paragraph(docs.get(old, ""))
                if extra:
                    row["text"] = f"{row['text']} {extra}".strip()
    return out


def index_data(items: list[dict], use: Backend = NO_BACKEND) -> dict:
    """``.memory/search.json``: the rows, with one vector each when *use* embeds (rounded, so the same vectors give
    the same bytes); else the reason it holds none (then nothing is ranked). An embedding that fails leaves the index
    without vectors, with the failure's kind as the reason."""
    out = {"version": 1, "backend": "none", "model": None, "why": use.why or NOT_CONFIGURED, "items": items}
    if use.embed is None or use.model is None or not items:
        if use.embed is not None and use.model is not None:
            out.update(backend="embedding", model=use.model, why="")
        return out
    try:
        vectors = use.embed([r["text"] or r["card"] or r["id"] for r in items])
        rowed = [[round(float(x), _ROUND) for x in v] for v in vectors]
    except Exception as exc:  # noqa: BLE001 - nothing is ranked then, and the index says why
        out["why"] = f"the embedding call failed ({type(exc).__name__})"
        return out
    if len(rowed) != len(items):
        out["why"] = "the embedding call returned another number of vectors"
        return out
    return {
        **out,
        "backend": "embedding",
        "model": use.model,
        "why": "",
        "items": [{**r, "vector": v} for r, v in zip(items, rowed)],
    }


def index_bytes(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def index_files(
    tree: Path,
    lib: Any = None,
    *,
    status_of: Callable[[str], str] | None = None,
    records: Mapping[str, dict] | None = None,
    captures: Mapping[str, bytes] | None = None,
    use: Backend | None = None,
    serving: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """The index file of the copy at *tree* (:func:`rows`, :func:`index_data`), by its path in the copy."""
    found = rows(
        tree,
        lib,
        status_of=status_of,
        records=records,
        captures=captures,
        **({"serving": serving} if serving else {}),
    )
    return {INDEX_FILE: index_bytes(index_data(found, use or NO_BACKEND))}


def read_index(copy: Path) -> dict | None:
    """The copy's index, or None when it cannot be read."""
    try:
        data = json.loads((Path(copy) / INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("items"), list) else None


# --- the ranking -------------------------------------------------------------------------------------------------


def _dot(a: list, b: list) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def ranking(data: dict, query: str, use: Backend = NO_BACKEND) -> tuple[list[tuple[float | None, dict]], str]:
    """Every row of the index *data* with its score, best first (ties in index order), and how they were ranked:
    by meaning when the index holds vectors of *use*'s model and the query embeds; else not ranked at all: every row
    in the index's own order with no score, and the reason (no word matching ranks anything)."""
    items = [r for r in data.get("items") or [] if isinstance(r, dict)]
    why = ""
    if data.get("backend") == "embedding":
        if use.embed is None or use.model is None:
            why = use.why or NOT_CONFIGURED
        elif use.model != data.get("model"):
            why = OTHER_MODEL
        else:
            try:
                (q,) = [list(map(float, v)) for v in use.embed([query])]
                scored = [(_dot(q, r.get("vector") or []), r) for r in items]
                how = f"ranked by meaning (the embedding model {use.model}), best first"
                return _sorted(scored), how
            except Exception as exc:  # noqa: BLE001 - nothing is ranked then, and the result says why
                why = f"the embedding call failed ({type(exc).__name__})"
    else:
        why = str(data.get("why") or NOT_CONFIGURED)
    how = f"not ranked, because {why}: every item in the library's order; judge by the cards"
    return [(None, r) for r in items], how


def _sorted(scored: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))
    return [scored[i] for i in order]


def search_text(data: dict | None, query: str, use: Backend = NO_BACKEND, offset: int = 0) -> str:
    """The result of ``search(query)``: how the items were ranked (or why they were not), then one line per item from
    *offset*, best first (its rank, its path in the box, its id, kind and score when there is one, and its card), one
    page of :data:`PAGE_BYTES`; a last line says where the next page starts."""
    if data is None:
        return "search: the library's search index cannot be read; use tools.grep and tools.read instead."
    ranked, how = ranking(data, query, use)
    total = len(ranked)
    if not total:
        return "search: the library holds no items."
    offset = max(0, min(int(offset), total))
    lines = [f"search: {total} item(s), {how}."]
    size = len(lines[0]) + 1
    end = offset
    for i in range(offset, total):
        score, r = ranked[i]
        scored = f", score {score:.3f}" if score is not None else ""
        line = (
            f"{i + 1}. {LIBRARY_ROOT}/{r.get('path')}  {r.get('id')} ({r.get('kind')}{scored}): "
            f"{r.get('card') or '(no summary)'}"
        )
        if end > offset and size + len(line.encode("utf-8")) + 1 > PAGE_BYTES:
            break
        lines.append(line)
        size += len(line.encode("utf-8")) + 1
        end = i + 1
    if end < total:
        lines.append(f"[items {end + 1} to {total} follow: search again with offset={end}]")
    return "\n".join(lines)


__all__ = [
    "Backend",
    "INDEX_FILE",
    "NO_BACKEND",
    "NOT_CONFIGURED",
    "OTHER_MODEL",
    "backend",
    "index_bytes",
    "index_data",
    "index_files",
    "ranking",
    "read_index",
    "rows",
    "search_text",
]
