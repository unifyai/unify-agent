"""The librarian navigates the library (``UNIFY_MEMORY_LIBRARIAN=on``): a constant pointer in its prompt in place of
the catalogue, a minimal set of tools (``search`` over the items' "When to use" text, ``read`` and ``grep`` inside the
library, ``finish``), and a procedure's recipe as its one-call entry on the current request.

The libraries are real copies as the export writes them (``library_export.generated_library_mode`` over the files a
writer writes: each function and note with its When to use section, each folder with its README.md); the ones with
item records are the production-built worlds of :mod:`tests.memory.test_librarian` (:mod:`tests.memory.gate_records`).
Two worlds that are not puzzles of any kind: a library for calling a web service (an API-style request) and one for
working with text documents. Keyless: the local embedder (``UNIFY_LOCAL_EMBEDDINGS``) or a stubbed provider endpoint,
and the librarian's model is a scripted turn; nothing leaves the machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.memory import test_librarian as T
from tests.memory.test_librarian_answer_once import Turn, _ask, _ctx
from tests.memory.test_library_helper import _py
from unify.common import embeddings
from unify.common import tool_registry as R
from unify.memory import librarian as L
from unify.memory import library_export as lx
from unify.memory import library_search as ls
from unify.memory.catalogue import write_files
from unify.memory.integration import cost
from unify.memory.item_records import recipe_call
from unify.settings import SETTINGS

# --- two libraries that are not puzzles: a web service and text documents ------------------------------------------

PAGES = "memory.web.pages:all_pages"
RETRY = "memory.web.retry:with_retries"
OUTLINE = "memory.docs.sections:outline"
TONE = "notes/writing/replies.md"


def _function(name: str, args: str, summary: str, saved: str, recognise: str, not_for: str, asked: str) -> str:
    return (
        f"def {name}({args}):\n"
        f'    """{summary}\n\n'
        "    When to use:\n"
        f"        Saved for: {saved}\n"
        f"        Recognise: {recognise}\n"
        f"        Not for: {not_for}\n"
        f"        Example request: {asked}\n"
        '    """\n'
        "    return None\n"
    )


WEB = {
    "memory/web/__init__.py": '"""Calls to web services."""\n',
    "memory/web/README.md": "Calling web services: paging through list endpoints, and retrying failed calls.\n",
    "memory/web/pages.py": '"""Paging through list endpoints."""\n\n\n'
    + _function(
        "all_pages",
        "fetch, first_url",
        "Follow each page's next link and gather every record.",
        "collecting every record that a paginated list endpoint of a web service returns.",
        "the response carries a next link or a cursor, and the request wants all the records, not one page.",
        "a single lookup by key, or an endpoint that returns everything in one response.",
        "List every invoice of the account from the billing service.",
    ),
    "memory/web/retry.py": '"""Retrying calls that fail for a while."""\n\n\n'
    + _function(
        "with_retries",
        "call, attempts",
        "Call again after a short wait when the service answers that it is busy.",
        "calling a web service that sometimes answers busy or times out.",
        "the service answers with a temporary failure status, and the same call later succeeds.",
        "a call refused for bad credentials or bad arguments, which a retry cannot fix.",
        "Fetch the user's profile; the service is often overloaded.",
    ),
}
DOCS = {
    "memory/docs/__init__.py": '"""Working with long text documents."""\n',
    "memory/docs/README.md": "Long text documents: finding their sections and summarising them.\n",
    "memory/docs/sections.py": '"""The sections of a long document."""\n\n\n'
    + _function(
        "outline",
        "text",
        "The headings of a document with the line each starts at.",
        "summarising a long report or agreement section by section.",
        "the request hands over a long document with headings and asks what each part says.",
        "a short message, or a document without headings.",
        "Summarise each section of this rental agreement for me.",
    ),
    "notes/writing/README.md": "How to write replies.\n",
    TONE: (
        "---\ntitle: Replies to complaints\ndescription: How a polite reply to an unhappy customer was written.\n---\n"
        "Thank them, say what went wrong, say what happens next.\n\n"
        "## When to use\n"
        "- Saved for: answering an unhappy customer's message politely.\n"
        "- Recognise: a request to draft a reply to a complaint.\n"
        "- Not for: internal notes or messages to colleagues.\n"
        "- Example request: Write back to this customer whose parcel arrived late.\n"
    ),
}
API_REQUEST = "Gather all the records the orders endpoint returns; each response has a next link to the following page."
DOCUMENT_REQUEST = "Here is a long agreement with headings: summarise what each section of the document says."


def _copy(root: Path, files: dict[str, str], search: ls.Backend | None = None) -> Path:
    """A library copy as the export writes it with the librarian on: the files and every generated file."""
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    write_files(root, lx.generated_library_mode(root, librarian=True, search=search))
    return root


def _bulk(n: int) -> dict[str, str]:
    """*n* more modules, each one function with its When to use section."""
    return {
        f"memory/bulk/part_{i:03d}.py": _function(
            f"step_{i}", "x", f"Step {i}.", f"step {i} of a long job.", f"a request for step {i}.", "anything else.",
            f"Do step {i}.",
        )
        for i in range(n)
    } | {"memory/bulk/__init__.py": '"""Many steps."""\n', "memory/bulk/README.md": "Many steps.\n"}


def _state(copy: Path) -> SimpleNamespace:
    ids, failed, recipes = L.library_items(copy)
    return SimpleNamespace(state=L.LibrarianCalls(ids=ids, failed=failed, copy=copy, recipes=recipes))


def _first(result: str) -> str:
    """The id on the first ranked line of a search result."""
    line = next(ln for ln in result.splitlines() if ln.startswith("1. "))
    return line.split()[2]


@pytest.fixture
def embed_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFY_EMBED_CACHE", str(tmp_path / "embeddings.sqlite"))


@pytest.fixture
def no_backend(monkeypatch, embed_cache):
    """No embedding backend configured, and an embedder that fails the test if anything calls it."""
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_EMBED_URL", "")
    calls: list = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("an embedding was made")

    monkeypatch.setattr(embeddings, "embed", refuse)
    monkeypatch.setattr(embeddings, "_local", refuse)
    monkeypatch.setattr(embeddings, "_openrouter", refuse)
    return calls


def _paper(monkeypatch) -> None:
    """``UNIFY_PROFILE=paper``. The setting comes with its own change; until it is merged it is set on the instance."""
    if "UNIFY_PROFILE" in type(SETTINGS).model_fields:
        monkeypatch.setattr(SETTINGS, "UNIFY_PROFILE", "paper", raising=False)
    else:
        monkeypatch.setitem(SETTINGS.__dict__, "UNIFY_PROFILE", "paper")


# --- no catalogue in the prompt ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_system_message_is_the_same_bytes_for_a_small_and_a_large_library(tmp_path, no_backend):
    small = _copy(tmp_path / "small", WEB)
    large = _copy(tmp_path / "large", {**WEB, **DOCS, **_bulk(80)})
    assert len(L.library_items(large)[0]) > len(L.library_items(small)[0]) + 80
    sessions = [L.LibrarianSession(copy=c, turn=Turn([]), compact=False) for c in (small, large)]
    try:
        first, second = (s.messages[0]["content"] for s in sessions)
    finally:
        for s in sessions:
            await s.close()
    assert first == second == L.system_prompt() == L.opening() + "\n\n" + L.BRIEF
    for item in (PAGES, RETRY, OUTLINE, TONE, "memory.bulk.part_000:step_0"):
        assert item not in first
    # the librarian's tools: search, read, grep and finish, in its own box and nowhere else
    assert R.allow_list("librarian") == ("read", "grep", "finish", "search")
    assert R.registry()["search"].roles == ("librarian",)
    assert R.handler_for("librarian", "search") == ("unify.memory.librarian:search", True)
    assert R.handler_for("librarian", "read") == ("unify.memory.librarian:read", True)
    assert R.handler_for("librarian", "grep") == ("unify.memory.librarian:grep", True)
    assert "search" not in R.allow_list("actor", frozenset({"librarian", "store", "helper"}))
    assert "search" not in R.allow_list("writer", frozenset({"pool", "similar", "helper", "edit"}))


# --- search ------------------------------------------------------------------------------------------------------


def test_search_finds_items_by_their_when_to_use_text_with_the_local_embedder(tmp_path, monkeypatch, embed_cache):
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", True)
    use = ls.backend()
    assert use.model == embeddings.LOCAL.model and use.embed is not None
    copy = _copy(tmp_path / "lib", {**WEB, **DOCS}, search=use)
    index = ls.read_index(copy)
    assert index["backend"] == "embedding" and index["model"] == embeddings.LOCAL.model
    assert [r["id"] for r in index["items"]] == [OUTLINE, PAGES, RETRY, TONE]
    assert all(len(r["vector"]) == 384 for r in index["items"])
    pages = next(r for r in index["items"] if r["id"] == PAGES)
    assert pages["text"].startswith("Saved for: collecting every record") and pages["path"] == "memory/web/pages.py"
    # an API-style request and a document request, in words their sections do not share
    for request, want in (
        ("Fetch everything from a service whose replies come back one chunk at a time with a link onwards.", PAGES),
        ("Tell me what every part of this long contract covers.", OUTLINE),
        ("Draft an answer to this angry shopper.", TONE),
    ):
        got = L.search(_state(copy), {"query": request})
        assert got.status == "ok", got.text
        assert "ranked by meaning" in got.text.splitlines()[0], got.text
        assert _first(got.text) == want, got.text
    # every item is listed, best first, each with its path in the box and its one-line card
    lines = [ln for ln in got.text.splitlines()[1:] if ln[:1].isdigit()]
    assert len(lines) == 4 and f"/memory/{TONE}  {TONE} (note, score " in lines[0]


def test_search_ranks_lexically_without_an_embedding_backend_and_says_so(tmp_path, no_backend):
    copy = _copy(tmp_path / "lib", {**WEB, **DOCS})
    index = ls.read_index(copy)
    assert index["backend"] == "lexical" and index["why"] == ls.NOT_CONFIGURED
    assert all("vector" not in r for r in index["items"])
    got = L.search(_state(copy), {"query": API_REQUEST})
    head = got.text.splitlines()[0]
    assert "ranked by a lexical score" in head and "because no embedding backend is configured" in head
    assert _first(got.text) == PAGES
    assert _first(L.search(_state(copy), {"query": DOCUMENT_REQUEST}).text) == OUTLINE
    assert no_backend == []  # no embedding was made, at the export or at the search
    refused = L.search(_state(copy), {"query": "  "})
    assert refused.status == "refused" and refused.form == "no_query"


def test_the_paper_profile_makes_no_embedding_call_at_export_or_at_search(tmp_path, monkeypatch, embed_cache):
    calls: list = []

    def counted(*args, **kwargs):
        calls.append(args)
        raise AssertionError("an embedding was made under the paper profile")

    monkeypatch.setattr(embeddings, "embed", counted)
    monkeypatch.setattr(embeddings, "_local", counted)
    monkeypatch.setattr(embeddings, "_openrouter", counted)
    # a backend is configured, and the paper profile still makes no embedding call
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_EMBED_URL", "http://127.0.0.1:9/embeddings")
    _paper(monkeypatch)
    assert ls.paper_profile() and ls.backend() == ls.Backend(None, ls.PAPER)
    copy = _copy(tmp_path / "lib", {**WEB, **DOCS}, search=ls.backend())
    index = ls.read_index(copy)
    assert index["backend"] == "lexical" and index["why"] == ls.PAPER
    got = L.search(_state(copy), {"query": API_REQUEST})
    assert "ranked by a lexical score" in got.text and ls.PAPER in got.text and _first(got.text) == PAGES
    # an index built with vectors elsewhere is searched lexically too, with no call
    fake = ls.Backend("fake-model", "", lambda texts: [[1.0, 0.0] for _ in texts])
    write_files(copy, ls.index_files(copy, use=fake))
    assert ls.read_index(copy)["backend"] == "embedding"
    got = L.search(_state(copy), {"query": DOCUMENT_REQUEST})
    assert ls.PAPER in got.text and _first(got.text) == OUTLINE
    assert calls == []


def test_an_embedding_that_fails_leaves_a_lexical_index_that_says_why(tmp_path, no_backend):
    def broken(texts):
        raise ConnectionError("unreachable")

    copy = _copy(tmp_path / "lib", WEB, search=ls.Backend("some-model", "", broken))
    index = ls.read_index(copy)
    assert index["backend"] == "lexical" and index["why"] == "the embedding call failed (ConnectionError)"
    assert "the embedding call failed (ConnectionError)" in L.search(_state(copy), {"query": API_REQUEST}).text


def test_a_long_result_is_paged_and_nothing_is_left_out(tmp_path, no_backend):
    copy = _copy(tmp_path / "lib", {**WEB, **_bulk(150)})
    ranked = 0
    offset, seen = 0, []
    while True:
        text = L.search(_state(copy), {"query": "step of a long job", "offset": offset}).text
        assert len(text.encode()) <= ls.PAGE_BYTES + 200
        seen += [ln.split()[2] for ln in text.splitlines() if ln[:1].isdigit()]
        tail = text.splitlines()[-1]
        if not tail.startswith("[items "):
            break
        offset = int(tail.rsplit("offset=", 1)[1].rstrip("]"))
        ranked += 1
    assert ranked >= 1 and len(seen) == len(set(seen)) == len(L.library_items(copy)[0])


def test_each_embedding_request_is_booked_as_an_embedding_cost_row(tmp_path, monkeypatch, embed_cache):
    """The provider path, its endpoint stubbed: each request is told to ``on_call`` with the tokens and the cost its
    reply reports and booked as an ``embedding`` row; a request that fails is booked with its cost unknown."""
    import httpx

    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_EMBED_URL", "http://127.0.0.1:9/embeddings")
    fail = {"now": False}

    def post(url, *, headers, json, timeout):
        if fail["now"]:
            raise httpx.ConnectError("refused")
        words = [set(t.lower().split()) for t in json["input"]]
        data = [{"index": i, "embedding": [float(len(w)), 1.0, float(i)]} for i, w in enumerate(words)]
        return httpx.Response(
            200,
            json={"data": data, "usage": {"prompt_tokens": 11, "cost": 0.00002}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(embeddings.httpx, "post", post)
    told: list[dict] = []
    copy = _copy(tmp_path / "lib", WEB, search=ls.backend(on_call=told.append))
    assert ls.read_index(copy)["backend"] == "embedding"
    assert told == [{"model": "openai/text-embedding-3-small", "texts": 2, "prompt_tokens": 11, "cost": 0.00002}]
    listener = cost.CostListener()
    listener.activate()
    for call in told:
        listener.add_embedding(call)
    assert listener.rows == [cost.CostRow("embedding", "openai/text-embedding-3-small", 11, None, "0.00002")]
    # the search's own request goes to the request's cost record through book_embedding
    monkeypatch.setattr(cost, "_LISTENER", listener)
    L.search(_state(copy), {"query": "page through every record"})
    assert listener.rows[-1] == cost.CostRow("embedding", "openai/text-embedding-3-small", 11, None, "0.00002")
    fail["now"] = True
    got = L.search(_state(copy), {"query": "a query never embedded before"})
    assert "the embedding call failed (ConnectError)" in got.text and _first(got.text) in (PAGES, RETRY)
    assert listener.rows[-1] == cost.CostRow("embedding", "openai/text-embedding-3-small", None, None, "unknown")
    listener.deactivate()


# --- read, grep and finish stay inside the library ------------------------------------------------------------


def test_read_and_grep_refuse_a_path_outside_the_library_and_read_inside_it(tmp_path, no_backend):
    copy = _copy(tmp_path / "lib", {**WEB, **DOCS})
    outside = tmp_path / "outside.txt"
    outside.write_text("NOT-THE-LIBRARY")
    (copy / "memory" / "web" / "escape.md").symlink_to(outside)
    ctx = _state(copy)
    for path in ("/etc/passwd", "/memory/../etc/passwd", str(outside), "/tmp", "memory/web/README.md"):
        got = L.read(ctx, {"path": path})
        assert got.status == "refused" and got.form == "outside_library", (path, got)
        assert "is outside the library, /memory" in got.text and "NOT-THE-LIBRARY" not in got.text
        got = L.grep(ctx, {"pattern": "root", "path": path})
        assert got.status == "refused" and got.form == "outside_library", (path, got)
    link = L.read(ctx, {"path": "/memory/memory/web/escape.md"})
    assert link.status == "refused" and "NOT-THE-LIBRARY" not in link.text
    readme = L.read(ctx, {"path": "/memory/memory/web/README.md"})
    assert readme.status == "ok" and "paging through list endpoints" in readme.text
    listing = L.read(ctx, {"path": "/memory/notes/writing"})
    assert "replies.md" in listing.text and "README.md" in listing.text
    hits = L.grep(ctx, {"pattern": r"def all_pages"})  # the whole library by default
    assert hits.status == "ok" and "/memory/memory/web/pages.py:" in hits.text, hits.text
    assert "NOT-THE-LIBRARY" not in L.grep(ctx, {"pattern": "NOT-THE"}).text


def test_finish_refuses_an_id_that_is_not_an_item_of_the_library(tmp_path, no_backend):
    copy = _copy(tmp_path / "lib", {**WEB, **DOCS})
    ids, failed, recipes = L.library_items(copy)
    assert set(ids) == {PAGES, RETRY, OUTLINE, TONE} and not failed and not recipes
    for answer, why in (
        ({"fits": [{"id": "memory.web.pages:ghost", "call": "c", "why": "w"}]}, "is not an item of the library"),
        ({"fits": [], "nearest": [{"id": "notes/writing/ghost.md", "differs": "d"}]}, "is not an item of the library"),
        ({"fits": [{"id": "memory/web/pages.py", "call": "c", "why": "w"}]}, "is not an item of the library"),
    ):
        got, reason = L.check_answer(answer, ids, failed, recipes)
        assert got is None and why in reason, (answer, reason)
    fine = {"fits": [{"id": PAGES, "call": "memory.get('...')(fetch, url)", "why": "a paginated endpoint"},
                     {"id": RETRY, "call": "c", "why": "the service may be busy"}]}
    assert L.check_answer(fine, ids, failed, recipes) == (fine, "")
    # a retired item is no longer an item the librarian may name
    items = json.loads((copy / ".memory/items.json").read_text())
    items["items"][RETRY]["status"] = "retired"
    (copy / ".memory/items.json").write_text(json.dumps(items))
    ids, failed, recipes = L.library_items(copy)
    assert RETRY not in ids and L.check_answer(fine, ids, failed, recipes)[0] is None
    assert L.library_items(tmp_path / "no-copy") is None


# --- the recipe as the one-call entry -------------------------------------------------------------------------


@T.needs_box
def test_an_answer_naming_a_procedure_carries_its_recipe_call_and_the_card_shows_it(tmp_path, no_backend):
    copy = T._copy(tmp_path)
    entry = recipe_call(T.RECIPE)
    assert entry == f"memory.get({T.RECIPE!r})(memory.request())"
    ids, failed, recipes = L.library_items(copy)
    assert recipes == {T.TOTAL: entry, T.RECIPE: entry}
    for named in (T.TOTAL, T.RECIPE):
        got, why = L.check_answer({"fits": [{"id": named, "call": f"memory.run({named!r}, text)", "why": "w"}]},
                                  ids, failed, recipes)
        assert got is None and entry in why, why
        answer = {"fits": [{"id": named, "call": entry, "why": "its reader parsed the text"}]}
        assert L.check_answer(answer, ids, failed, recipes) == (answer, "")
    # the search card names the same call; the recipe has no row of its own
    text = L.search(_state(copy), {"query": "add up every number of a table"}).text
    line = next(ln for ln in text.splitlines() if f" {T.TOTAL} (function" in ln)
    assert line.endswith(f"| applies to the current request: {entry}")
    assert f" {T.RECIPE} (" not in text
    # library_helper, the actor's own lookups, shows it on the item's card (the procedure's and the recipe's)
    out = _py(copy, f"import memory\nprint(memory.get({T.TOTAL!r}))\nprint(memory.get({T.RECIPE!r}))\n")
    assert out.count(f"Applies to the current request: {entry}") == 2, out
    assert "Applies to the current request" not in _py(copy, f"import memory\nprint(memory.get({T.COUNT!r}))\n")


def test_memory_request_is_the_current_request_as_a_recipe_takes_it(tmp_path, no_backend):
    copy = _copy(tmp_path / "lib", WEB)
    request = 'Total these: {"rows": [1, 2]} and [3]'
    tools = (
        "import json, sys, types\n"
        "tools = types.ModuleType('tools')\n"
        f"tools._call = lambda tool, args: json.dumps({{'text': {request!r}, 'data': [{{'rows': [1, 2]}}, [3]]}})\n"
        "sys.modules['tools'] = tools\n"
    )
    out = _py(
        copy,
        tools + "import memory\nr = memory.request()\n"
        "print(json.dumps([isinstance(r, str), str(r), r.text, r.data]))\n",
    )
    assert json.loads(out.splitlines()[-1]) == [True, request, request, [{"rows": [1, 2]}, [3]]]
    out = _py(
        copy,
        "import memory\ntry:\n    memory.request()\nexcept Exception as exc:\n    print(type(exc).__name__, exc)\n",
    )
    assert out.startswith("NotFound") and "the current request is not available" in out
    stub = (copy / "memory" / "__init__.pyi").read_text()
    assert "def request()" in stub and "memory.request().data" in stub


@T.needs_box
def test_memory_ask_shows_each_fits_call_beside_its_reason(tmp_path):
    entry = recipe_call(T.RECIPE)
    answer = json.dumps({"fits": [{"id": T.TOTAL, "call": entry, "why": "its reader parsed it"}]})
    fake = T.FAKE_TOOLS.replace(repr(T.FAKE_ANSWER), repr(answer))
    assert fake != T.FAKE_TOOLS
    out = _py(T._copy(tmp_path), fake + "import memory\na = memory.ask()\nprint(a)\nprint(json.dumps(a.calls))\n")
    assert f"- {T.TOTAL} (function): its reader parsed it\n  call: {entry}" in out, out
    assert "Nothing has run." in out and json.loads(out.splitlines()[-1]) == {T.TOTAL: entry}


# --- the answer cache keys on the commit and the ask ---------------------------------------------------------


def _session(copy: Path, turn: Turn, library: str) -> L.LibrarianSession:
    session = L.LibrarianSession(copy=copy, turn=turn, compact=False, library=library)

    async def tool_call(tc: dict) -> str:
        answer = json.loads(json.loads(tc["function"]["arguments"])["code"])
        return (await L.finish(SimpleNamespace(state=session.state), {"answer": answer})).text

    session._tool_call = tool_call
    return session


@pytest.mark.asyncio
async def test_the_answer_cache_keys_on_the_commit_and_the_ask_and_a_new_commit_misses(tmp_path, no_backend):
    copy = _copy(tmp_path / "lib", {**WEB, **DOCS})
    fits = {"fits": [{"id": PAGES, "call": "memory.get(...)(fetch, url)", "why": "a paginated endpoint"}]}
    other = {"fits": [{"id": OUTLINE, "call": "memory.get(...)(text)", "why": "a long document"}]}
    turn = Turn([fits, fits, other])
    session = _session(copy, turn, "c0ffee")
    ctx = _ctx(session, API_REQUEST)
    try:
        _, one = await _ask(ctx, [], API_REQUEST)
        _, again = await _ask(ctx, [], API_REQUEST)  # the same ask at the same commit: the answer already given
        session.library = "beef"  # the library at another commit
        _, moved = await _ask(ctx, [], API_REQUEST)
        _, doc = await _ask(ctx, [], DOCUMENT_REQUEST)  # another ask
    finally:
        await session.close()
    assert one == fits and again == {**fits, L.ALREADY_GIVEN: True}
    assert moved == fits and L.ALREADY_GIVEN not in moved and doc == other
    assert len(turn.calls) == 3 and session.from_cache == 1
    sent = [L.sent_to_librarian(API_REQUEST, [], 0, {"commit": c, "dispatch": L._digest_of({})}) for c in ("a", "b")]
    assert L.answer_key(sent[0]) != L.answer_key(sent[1])
    # what the librarian is sent for a question carries no catalogue: only the question and its facts
    asked = [m["content"] for m in session.messages if m.get("role") == "user"]
    assert all(PAGES not in m and OUTLINE not in m for m in asked)
