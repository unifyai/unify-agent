"""Search over the library's "when to use" text: the index each export writes, and the ranking ``search`` returns.

Every listed item of a library copy has one row in ``.memory/search.json`` (:func:`index_files`): its id, its kind,
its path in the copy, a one-line card (its first docstring line or description and, for a procedure with a recipe,
the call that applies it to the current request: :func:`.item_records.recipe_call`) and the text it is searched by:
its "When to use" section (:mod:`.when_to_use`), else its first docstring paragraph or description. A function that
is a procedure's recipe has no row of its own (its procedure's card names it), as in the dispatch file.

With an embedding backend (:func:`backend`), the export also stores one vector per row, computed with the harness's
embedder (:func:`unify.common.embeddings.embed`, whose cache makes a commit's vectors a one-time cost) and labelled
with its model; the vectors live only in the copy, never in a prompt. Without one (no backend configured, the paper
profile, or an embedding call that failed) the index holds the rows alone and says why, and :func:`search_text`
ranks by a lexical score over the same text (:func:`lexical_scores`) and says so in its result. Each request an
embedding sends to a provider is told to *on_call* (:func:`backend`), which books it (``integration.cost``).

Nothing here caps what is returned: every row is ranked, and a long result is paged.
"""

from __future__ import annotations

import json
import math
import re
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
#: Why a ranking is lexical.
NOT_CONFIGURED = "no embedding backend is configured"
PAPER = "the paper profile makes no embedding calls"
OTHER_MODEL = "the index was built with another embedding model"
_WORD = re.compile(r"[^\W_]+")
_ROUND = 6


def paper_profile() -> bool:
    """Whether the run's profile is the paper's (``UNIFY_PROFILE=paper``): one model only, so no embedding call."""
    try:
        from unify.settings import SETTINGS
    except Exception:  # noqa: BLE001 - no settings: not the paper profile
        return False
    return str(getattr(SETTINGS, "UNIFY_PROFILE", "") or "").strip().lower() == "paper"


@dataclass(frozen=True)
class Backend:
    """How texts are embedded: *model* labels the vectors (None: none are made, and *why* says why) and *embed*
    turns texts into unit vectors."""

    model: str | None
    why: str = ""
    embed: Callable[[list[str]], Any] | None = None


LEXICAL = Backend(None, NOT_CONFIGURED)


def backend(on_call: Callable[[dict], None] | None = None) -> Backend:
    """The run's embedding backend: none under the paper profile, nor unless one is configured
    (``UNIFY_LOCAL_EMBEDDINGS``, the local model, or ``UNIFY_EMBED_URL``, an endpoint of the provider's form);
    otherwise the harness's embedder (:func:`unify.common.embeddings.embedder`), each request it sends told to
    *on_call*."""
    if paper_profile():
        return Backend(None, PAPER)
    try:
        from unify.settings import SETTINGS
    except Exception:  # noqa: BLE001 - no settings: no backend
        return LEXICAL
    local = bool(getattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", False))
    url = str(getattr(SETTINGS, "UNIFY_EMBED_URL", "") or "").strip()
    if not local and not url:
        return LEXICAL
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
) -> list[dict]:
    """One row per listed item of the copy at *tree*, in module-path order, then the notes, then the captured
    procedures (*captures*: the snapshot's files): ``{id, kind, path, card, text}``. Retired items have none, nor
    has a procedure's recipe. A captured procedure the snapshot offers as a worked example is a row of kind
    ``worked example``."""
    from . import when_to_use as wtu
    from .item_records import recipe_call, recipe_of
    from .layout import discover
    from .library_export import (
        CAPTURE_PREFIX,
        _capture_docs,
        _docstring_of,
        _docstrings,
        _status,
        worked_example,
    )
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
        text = _section_line(wtu.note_section(body)) or about
        out.append({"id": n.item_id, "kind": "note", "path": n.item_id, "card": about, "text": text})
    for cid, doc in _capture_docs(captures).items():
        sig = doc["signature"] if isinstance(doc["signature"], dict) else {}
        rec = doc["record"] if isinstance(doc["record"], dict) else {}
        goal = _one_line(sig.get("goal")) if isinstance(sig.get("goal"), str) else ""
        run_doc = _first_paragraph(_docstring_of(doc["procedure"])) if isinstance(doc["procedure"], str) else ""
        card = (run_doc or goal) + (" | tried and failed: never offered to run" if rec.get("outcome") == "failed" else "")
        example = worked_example(rec)
        if example:
            card += f" | a worked example to read, never called: {example.get('why')}"
        out.append(
            {
                "id": CAPTURE_PREFIX + cid,
                "kind": "worked example" if example else "captured procedure",
                "path": f".memory/captures/{cid}/procedure.py",
                "card": card,
                "text": " ".join(t for t in (goal, run_doc) if t),
            },
        )
    return out


def index_data(items: list[dict], use: Backend = LEXICAL) -> dict:
    """``.memory/search.json``: the rows, with one vector each when *use* embeds (rounded, so the same vectors give
    the same bytes); else the reason the ranking is lexical. An embedding that fails leaves the index lexical, with
    the failure's kind as the reason."""
    out = {"version": 1, "backend": "lexical", "model": None, "why": use.why or NOT_CONFIGURED, "items": items}
    if use.embed is None or use.model is None or not items:
        if use.embed is not None and use.model is not None:
            out.update(backend="embedding", model=use.model, why="")
        return out
    try:
        vectors = use.embed([r["text"] or r["card"] or r["id"] for r in items])
        rowed = [[round(float(x), _ROUND) for x in v] for v in vectors]
    except Exception as exc:  # noqa: BLE001 - the ranking falls back to lexical, and says why
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
) -> dict[str, bytes]:
    """The index file of the copy at *tree* (:func:`rows`, :func:`index_data`), by its path in the copy."""
    found = rows(tree, lib, status_of=status_of, records=records, captures=captures)
    return {INDEX_FILE: index_bytes(index_data(found, use or LEXICAL))}


def read_index(copy: Path) -> dict | None:
    """The copy's index, or None when it cannot be read."""
    try:
        data = json.loads((Path(copy) / INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("items"), list) else None


# --- the ranking -------------------------------------------------------------------------------------------------


def words(text: str) -> list[str]:
    """The words of *text*, lower case (letters and digits, any script)."""
    return [w.lower() for w in _WORD.findall(text or "")]


def lexical_scores(query: str, texts: list[str]) -> list[float]:
    """A score per text: the words it shares with *query*, each weighted by how rare it is among *texts* (its
    inverse document frequency) and by how often it occurs, over the text's length (BM25, k1 1.2, b 0.75)."""
    docs = [words(t) for t in texts]
    asked = set(words(query))
    n = len(docs)
    if not n or not asked:
        return [0.0] * n
    avg = sum(len(d) for d in docs) / n or 1.0
    df = {w: sum(1 for d in docs if w in d) for w in asked}
    scores = []
    for d in docs:
        counts: dict[str, int] = {}
        for w in d:
            if w in asked:
                counts[w] = counts.get(w, 0) + 1
        s = 0.0
        for w, tf in counts.items():
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append(s)
    return scores


def _dot(a: list, b: list) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def ranking(data: dict, query: str, use: Backend = LEXICAL) -> tuple[list[tuple[float, dict]], str]:
    """Every row of the index *data* with its score, best first (ties in index order), and how they were ranked:
    by meaning when the index holds vectors of *use*'s model and the query embeds; else by a lexical score, with
    the reason."""
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
            except Exception as exc:  # noqa: BLE001 - falls back to lexical, and says why
                why = f"the embedding call failed ({type(exc).__name__})"
    else:
        why = str(data.get("why") or NOT_CONFIGURED)
    scores = lexical_scores(query, [str(r.get("text") or r.get("card") or "") for r in items])
    how = f"ranked by a lexical score (shared words, rarer words weighing more), best first, because {why}"
    return _sorted(list(zip(scores, items))), how


def _sorted(scored: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))
    return [scored[i] for i in order]


def search_text(data: dict | None, query: str, use: Backend = LEXICAL, offset: int = 0) -> str:
    """The result of ``search(query)``: how the items were ranked, then one line per item from *offset*, best
    first (its rank, its path in the box, its id, kind and score, and its card), one page of :data:`PAGE_BYTES`;
    a last line says where the next page starts."""
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
        line = (
            f"{i + 1}. {LIBRARY_ROOT}/{r.get('path')}  {r.get('id')} ({r.get('kind')}, score {score:.3f}): "
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
    "LEXICAL",
    "NOT_CONFIGURED",
    "OTHER_MODEL",
    "PAPER",
    "backend",
    "index_bytes",
    "index_data",
    "index_files",
    "lexical_scores",
    "paper_profile",
    "ranking",
    "read_index",
    "rows",
    "search_text",
    "words",
]
