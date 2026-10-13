"""``memory.ask`` end to end (design J(b).4): an actor request whose cell asks the librarian, with a scripted model
answering both the actor and the librarian.

The actor's cells run in their persistent confined interpreter and reach the host through the tool channel; the
host handler opens the request's librarian session on the first question, keeps it for the request's follow-ups and
ends it with the request. The librarian runs on the shared agent runtime in its own box, which mounts the library
copy read-only, with no network and no write root; it answers with ``tools.finish(answer)``; a malformed answer, or
one naming an id the catalogue does not hold, goes back to it as a refusal.

Requests are read at unillm's transport (``tests/scripted_model.py``), headers included, so nothing leaves the
process. The boxes need bubblewrap, as the other box tests do.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    ENV_SECRET,
    SSH_SECRET,
    STATE_SECRET,
    TOKEN_VALUE,
    needs_bwrap,
    world,
)
from tests.actor.code_act.test_in_code_tools_wiring import (
    OPENING_HEADING,
    _assert_ended,
    _channel_rows,
    _Collect,
    _rows,
    _text,
    code,
)
from tests.common.test_request_metadata_headers import transport
from tests.memory import librarian_world as w
from tests.memory.test_episodes import _ep
from tests.memory.test_library_helper import _commit
from tests.scripted_model import KINDS, Always, ScriptedModel, reply
from unify.actor import in_code_tools
from unify.memory import item_records
from unify.memory.captures import EXPORT_DIR, store_dir
from unify.memory.gitio import Repo
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import State
from unify.sandbox import LIBRARY_MOUNT
from unify.settings import SETTINGS

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
SWITCH = "UNIFY_MEMORY_LIBRARIAN"
ASK_TOOL = "ask_library"
CATALOGUE = w.CATALOGUE_FILE
#: The call kind the librarian's model calls carry (design J(b).4, accounting).
CALL_KIND = "librarian"
#: The use record's fields (J(b).4): the count, the rows (each: the ids listed as fitting, whether it said nothing
#: fits, and the facts' outcomes) and, on a later run of an item, whether an ask listed it.
ASK_CALLS, ASK_ROWS = "ask_calls", "ask_rows"
ROW_LISTED, ROW_NOTHING_FITS, ROW_FACTS = "listed", "nothing_fits", "facts"


def listed_by_ask(record: dict, item: str) -> bool | None:
    """Whether the use record marks a later run of *item* as listed by an ask (``run_rows[].listed_by_ask``)."""
    rows = [r for r in record.get("run_rows") or [] if r.get("target") == item]
    return rows[0].get("listed_by_ask") if rows else None


def librarian_answer(answer: dict) -> str:
    """The librarian's cell that answers: ``tools.finish(answer)`` with the answer as a plain object."""
    return f"tools.finish({answer!r})\n"


def librarian_brief() -> str:
    from unify.memory import librarian

    return librarian.BRIEF


def scripted(answer) -> ScriptedModel:
    """A model answering every request through *answer*, whatever kind the classifier gives it (the librarian's
    requests may be a kind of their own)."""
    kinds = {k: Always(answer) for k in ("actor", "helper", "query_llm", "last_word") if k in KINDS}
    if "librarian" in KINDS:
        kinds["librarian"] = Always(answer)
    return ScriptedModel(**kinds)


def plain(expr: str) -> str:
    """Code turning what ``memory.ask`` returned (a list of ``Item``, each with its reason; when it is empty,
    ``.nearest``, each with what differs) into plain JSON data in a cell, in the answer's order."""
    return (
        f"(lambda r: {{'fits': [{{'id': i.id, 'why': i.reason}} for i in r]}} if r or not r.nearest else "
        f"{{'fits': [], 'nearest': [{{'id': i.id, 'differs': i.reason}} for i in r.nearest]}})({expr})"
    )


def seen(answer: dict) -> dict:
    """What a cell reads of the librarian's *answer* through :func:`plain`: each fit's id and reason (the item
    itself is the call), or the nearest with what differs."""
    if answer.get("fits"):
        return {"fits": [{"id": f["id"], "why": f["why"]} for f in answer["fits"]]}
    return answer


# ==============================================================================================================

ACTOR_MARK = "Total the order table in the attached text, please."
Q1 = (
    "Which stored items fit this text?\nTABLE\n" + json.dumps(w.ROWS) + f"\nMARKED\n{w.PLANTED_B64}\n"
)
Q2 = "And is there one that matches ledger rows against payments?"
ANSWER_ONE = {
    "fits": [
        {"id": w.READ_ORDERS, "call": "rows = read_orders(text)", "why": "its reader parsed the text"},
        {"id": w.TOTAL, "call": "total_amount(read_orders(text))", "why": "saved against totalling an order table"},
    ],
}
ANSWER_TWO = {
    "fits": [],
    "nearest": [{"id": w.RECONCILE, "differs": "it needs ledger and payment lists; the text has an order table"}],
}
GHOST = "memory.orders.nope:ghost"


def _set(monkeypatch, name: str, value: str) -> None:
    if hasattr(SETTINGS, name):  # while the setting does not exist yet, the test fails on what it checks
        monkeypatch.setattr(SETTINGS, name, value)


def _open_request(paths: Paths, sha: str, monkeypatch) -> request_mod.RequestRun:
    """A library-mode request opened on *sha*, as ``RequestRun.begin`` opens it, made the current one."""
    run = request_mod.RequestRun("r", paths)
    run.state = State.load(paths.state)
    run.pin = sha
    run.library_mode = True
    run._shape_rows = lambda consolidate, library_mode=False: {}  # no evidence store here
    run._open_library_mode(None)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    return run


@pytest.fixture
def librarian(core_world, monkeypatch):  # noqa: F811
    """Tools in code, a persistent interpreter, library mode, the librarian, request headers; a library with item
    records, its episodes' summaries staged and two captured procedures in the store; every in-code session kept."""
    for name, value in (
        ("UNIFY_TOOLS_IN_CODE", "on"),
        ("UNIFY_CELL_STATE", "persistent"),
        ("UNIFY_MEMORY", "on"),
        ("UNIFY_MEMORY_LIBRARY", "on"),
        ("UNIFY_REQUEST_METADATA_HEADERS", "on"),
    ):
        monkeypatch.setattr(SETTINGS, name, value)
    _set(monkeypatch, SWITCH, "on")
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    paths = Paths.under(core_world["state"])
    mem = Repo.init_bare(paths.memory)
    sha = _commit(mem, w.FILES, "seed")
    item_records.write_records(mem, sha, w.records())
    for episode, text in w.SUMMARIES.items():
        d = Path(paths.state_dir) / "memory-staging" / episode
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.md").write_text(text)
    store = store_dir(paths.state_dir)
    for rel, data in w.capture_files().items():
        dest = store / rel[len(EXPORT_DIR) + 1 :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    yield {"world": core_world, "opened": collect.opened, "paths": paths, "sha": sha, "monkeypatch": monkeypatch}


class Script:
    """Answers the actor by step (its cells, in order) and the librarian from a queue of cells, one per request."""

    def __init__(self, actor_cells: list[str], librarian_cells: list[str]) -> None:
        self.actor_cells = list(actor_cells)
        self.librarian_cells = list(librarian_cells)
        self.librarian_calls: list = []
        self.actor_calls: list = []

    @staticmethod
    def is_actor(call) -> bool:
        users = [_text(m.get("content")) for m in call.messages if m.get("role") == "user"]
        return bool(users) and ACTOR_MARK in users[0]

    def __call__(self, call):
        if self.is_actor(call):
            self.actor_calls.append(call)
            step = sum(1 for m in call.messages if m.get("role") == "assistant")
            if step < len(self.actor_cells):
                return reply(calls=[code(self.actor_cells[step])])
            return reply("not finished")
        self.librarian_calls.append(call)
        if self.librarian_cells:
            return reply(calls=[code(self.librarian_cells.pop(0))])
        return reply(calls=[code(librarian_answer({"fits": []}))])


def _json(result: str, model: ScriptedModel):
    """The actor's answer as JSON; when it is not, the last tool results show why."""
    try:
        return json.loads(result)
    except ValueError:
        tools = "\n".join(
            _text(m.get("content")) for m in model.calls[-1].messages if m.get("role") == "tool"
        ) if model.calls else ""
        try:  # a cell's result is JSON with its traceback under "error": show its last lines
            tail = "\n".join(json.loads(tools).get("error", "").splitlines()[-4:]) or tools[-3000:]
        except (ValueError, AttributeError):
            tail = tools[-3000:]
        raise AssertionError(f"the actor answered {result[:300]!r}; its last tool results:\n{tail}") from None


def _system(call) -> str:
    return "\n".join(_text(m.get("content")) for m in call.messages if m.get("role") == "system")


def _not_assistant(call) -> str:
    return "\n".join(_text(m.get("content")) for m in call.messages if m.get("role") != "assistant")


def _headers(sent: list[dict], model: ScriptedModel, call) -> dict:
    return sent[model.calls.index(call)].get("extra_headers") or {}


def _actor_session(opened: list):
    actors = [s for s in opened if s.agent == "actor"]
    assert len(actors) == 1, [s.agent for s in opened]
    return actors[0]


async def _ask_act(env: dict, script: Script) -> tuple[str, ScriptedModel, list[dict]]:
    """One actor request answered by *script*, every request's transport arguments (headers included) kept."""
    from unify.actor.code_act_actor import CodeActActor

    model = scripted(script)
    with transport(model) as sent:
        actor = CodeActActor()
        try:
            handle = await actor.act(ACTOR_MARK, persist=False)
            result = await asyncio.wait_for(handle.result(), 480)
        finally:
            await actor.close()
    return result, model, sent


def _boxes() -> set[int]:
    """The live processes below this one that are sandboxes (their command line names bubblewrap)."""
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] != "Z":
            kids.setdefault(int(fields[1]), []).append(int(d))
    out, stack = set(), [os.getpid()]
    while stack:
        for c in kids.get(stack.pop(), []):
            stack.append(c)
            try:
                with open(f"/proc/{c}/cmdline", "rb") as fh:
                    if b"bwrap" in fh.read():
                        out.add(c)
            except OSError:
                pass
    return out


# --- the whole path --------------------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_ask_returns_the_librarians_answers_reuses_the_session_and_is_accounted(librarian):
    env = librarian
    run = _open_request(env["paths"], env["sha"], env["monkeypatch"])
    actor_cell = (
        "import json\n"
        f"_a = memory.ask({Q1!r})\n"
        f"_b = memory.ask({Q2!r})\n"
        f"_r = memory.run({w.TOTAL!r}, {w.ROWS!r})\n"
        f"tools.answer(json.dumps({{'a': {plain('_a')}, 'b': {plain('_b')}}}))\n"
    )
    script = Script([actor_cell], [librarian_answer(ANSWER_ONE), librarian_answer(ANSWER_TWO)])
    result, model, sent = await _ask_act(env, script)

    got = _json(result, model)
    assert got == {"a": seen(ANSWER_ONE), "b": seen(ANSWER_TWO)}, result[:2000]
    lib = script.librarian_calls
    assert len(lib) == 2, [c.kind for c in model.calls]

    # the librarian's system prompt: the shared opening, the brief, then the whole catalogue, in that order
    catalogue = (env["paths"].checkout / CATALOGUE).read_text(encoding="utf-8")
    system = _system(lib[0])
    assert system.startswith(OPENING_HEADING)
    brief = librarian_brief()
    assert brief in system and catalogue in system
    assert system.index(brief) < system.index(catalogue)
    assert all(_system(c) == system for c in lib)
    # the question and the facts reach it; the value a reader parsed never does
    assert "Which stored items fit this text?" in _not_assistant(lib[0])
    marked = f"capture:{w.capture_id(w.CAPTURE_MARKED)}"
    assert marked in _not_assistant(lib[0]) or marked.split(":")[1] in _not_assistant(lib[0])
    for call in model.calls:
        assert w.PLANTED_VALUE not in json.dumps(call.request)

    # the follow-up reuses the session: its request continues the first question's conversation
    assert "Which stored items fit this text?" in _not_assistant(lib[1]) and Q2 in _not_assistant(lib[1])
    one, two = (_headers(sent, model, c) for c in lib)
    assert one.get("X-Unify-Session") and one.get("X-Unify-Session") == two.get("X-Unify-Session")

    # the librarian runs on the actor's own model and effort
    first_actor = sent[model.calls.index(script.actor_calls[0])]
    for call in lib:
        mine = sent[model.calls.index(call)]
        assert mine.get("model") == first_actor.get("model")
        assert mine.get("reasoning_effort") == first_actor.get("reasoning_effort")

    # every librarian call carries its own call kind; the actor's never do
    for call in lib:
        assert _headers(sent, model, call).get("X-Unify-Call-Kind") == CALL_KIND
    for call in script.actor_calls:
        assert _headers(sent, model, call).get("X-Unify-Call-Kind") != CALL_KIND

    # the actor called the in-code tool once per question; its evidence holds no parsed value
    session = _actor_session(env["opened"])
    assert ASK_TOOL in session.allow
    tools_called = [r["tool"] for r in _channel_rows(_rows(session))]
    assert tools_called.count(ASK_TOOL) == 2, tools_called
    assert w.PLANTED_VALUE not in json.dumps(_rows(session))
    _assert_ended(env["opened"])

    # the use record: two asks, their rows, the helper counted apart, and the later run marked as listed by an ask
    record = request_mod.memory_use(
        _ep(transcript=[], actions=[]),
        run.item_ids,
        runtime_calls=run.runtime_calls,
        library_mode=True,
    )
    assert "error" not in record, record
    assert record[ASK_CALLS] == 2
    rows = record[ASK_ROWS]
    assert [sorted(r[ROW_LISTED]) for r in rows] == [sorted([w.READ_ORDERS, w.TOTAL]), []]
    assert [bool(r[ROW_NOTHING_FITS]) for r in rows] == [False, True]
    assert all(ROW_FACTS in r for r in rows)
    blob = json.dumps(rows)
    for text in (w.PLANTED_VALUE, w.PLANTED_B64, "Which stored items", Q2):
        assert text not in blob  # outcomes and ids only, never text
    assert record["library_helper_calls"]["ask"]["ok"] == 2
    assert "memory:ask" not in run.runtime_calls["counts"]
    assert listed_by_ask(record, w.TOTAL) is True


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_a_malformed_answer_or_an_unknown_id_goes_back_to_the_librarian_as_a_refusal(librarian):
    env = librarian
    _open_request(env["paths"], env["sha"], env["monkeypatch"])
    actor_cell = f"import json\n_a = memory.ask({Q1!r})\ntools.answer(json.dumps({plain('_a')}))\n"
    script = Script(
        [actor_cell],
        [
            librarian_answer({"fit": [{"id": w.TOTAL}]}),  # not the answer's form
            librarian_answer({"fits": [{"id": GHOST, "call": "ghost()", "why": "made up"}]}),  # no such item
            librarian_answer(ANSWER_ONE),
        ],
    )
    result, model, _ = await _ask_act(env, script)

    assert _json(result, model) == seen(ANSWER_ONE), result[:2000]
    lib = script.librarian_calls
    assert len(lib) == 3, [c.kind for c in model.calls]
    # each refusal reached the librarian in the next request of its session, outside its own turns
    assert len(lib[1].messages) > len(lib[0].messages)
    assert GHOST in _not_assistant(lib[2])
    # the actor never saw a refused answer
    for call in script.actor_calls:
        assert GHOST not in json.dumps(call.request)
    _assert_ended(env["opened"])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_the_session_ends_with_the_request_and_the_next_request_opens_a_new_one(librarian):
    env = librarian
    before = _boxes()
    _open_request(env["paths"], env["sha"], env["monkeypatch"])
    first_cell = f"import json\n_a = memory.ask({Q1!r})\ntools.answer(json.dumps({plain('_a')}))\n"
    first = Script([first_cell], [librarian_answer(ANSWER_ONE)])
    result, model_one, sent_one = await _ask_act(env, first)
    assert _json(result, model_one) == seen(ANSWER_ONE)
    _assert_ended(env["opened"])
    assert _boxes() <= before  # no librarian box is left

    _open_request(env["paths"], env["sha"], env["monkeypatch"])
    second_cell = f"import json\n_a = memory.ask({Q2!r})\ntools.answer(json.dumps({plain('_a')}))\n"
    second = Script([second_cell], [librarian_answer(ANSWER_TWO)])
    result, model_two, sent_two = await _ask_act(env, second)
    assert _json(result, model_two) == seen(ANSWER_TWO)
    (lib_one,), (lib_two,) = first.librarian_calls, second.librarian_calls
    # the second request's librarian starts afresh: nothing of the first request's question
    assert "Which stored items fit this text?" not in _not_assistant(lib_two)
    s1 = _headers(sent_one, model_one, lib_one).get("X-Unify-Session")
    s2 = _headers(sent_two, model_two, lib_two).get("X-Unify-Session")
    assert s1 and s2 and s1 != s2
    _assert_ended(env["opened"])


# --- confinement and what agents see ------------------------------------------------------------------------------

PROBE = f"""
import json, os, socket
out = {{}}
def attempt(name, fn):
    try:
        fn()
        out[name] = "ok"
    except BaseException as exc:
        out[name] = type(exc).__name__
attempt("write_library", lambda: open({LIBRARY_MOUNT + '/memory/orders/planted.py'!r}, "w").write("x = 1\\n"))
attempt("write_library_root", lambda: open({LIBRARY_MOUNT + '/planted.txt'!r}, "w").write("x"))
attempt("write_workspace", lambda: open("/workspace/librarian-wrote.txt", "w").write("x"))
attempt("connect", lambda: socket.create_connection(("1.1.1.1", 53), timeout=3).close())
attempt("resolve", lambda: socket.getaddrinfo("example.com", 80))
attempt("tool_write", lambda: tools.write("/workspace/librarian-wrote.txt", "x"))
out["show"] = memory.show({w.TOTAL!r})[:400]
out["env"] = sorted(os.environ)
out["env_secret"] = any(s in v for v in os.environ.values() for s in ({ENV_SECRET!r}, {TOKEN_VALUE!r}))
print("PROBE " + json.dumps(out))
"""


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_the_librarians_box_is_read_only_on_the_library_offline_and_has_no_write_root(librarian):
    env = librarian
    _open_request(env["paths"], env["sha"], env["monkeypatch"])
    checkout = env["paths"].checkout
    files_before = sorted(p.relative_to(checkout).as_posix() for p in checkout.rglob("*"))
    actor_cell = f"import json\n_a = memory.ask({Q1!r})\ntools.answer(json.dumps({plain('_a')}))\n"
    script = Script([actor_cell], [PROBE, librarian_answer(ANSWER_ONE)])
    result, model, _ = await _ask_act(env, script)

    assert _json(result, model) == seen(ANSWER_ONE), result[:2000]
    lib = script.librarian_calls
    assert len(lib) >= 2
    shown = _not_assistant(lib[1])
    line = next((ln for ln in shown.splitlines() if ln.startswith("PROBE ")), None)
    assert line is not None, shown[-3000:]
    probe = json.loads(line[len("PROBE ") :])
    for name in ("write_library", "write_library_root", "write_workspace", "connect", "resolve", "tool_write"):
        assert probe[name] != "ok", (name, probe)
    assert "total_amount" in probe["show"]  # it reads items with memory.show
    assert probe["env_secret"] is False
    # nothing was written on the host
    assert sorted(p.relative_to(checkout).as_posix() for p in checkout.rglob("*")) == files_before
    assert not (env["world"]["workspace"] / "librarian-wrote.txt").exists()
    _assert_ended(env["opened"])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_no_host_path_planted_secret_or_parsed_value_reaches_the_librarian_or_the_actor(librarian):
    env = librarian
    _open_request(env["paths"], env["sha"], env["monkeypatch"])
    actor_cell = (
        "import json\n"
        f"_a = memory.ask({Q1!r})\n"
        f"print(json.dumps({plain('_a')}))\n"
    )
    answer = "tools.answer('done')\n"
    script = Script([actor_cell, answer], [librarian_answer(ANSWER_ONE)])
    result, model, _ = await _ask_act(env, script)

    assert result == "done"
    assert script.librarian_calls, [c.kind for c in model.calls]
    world = env["world"]
    roots = {str(world[k]) for k in ("home", "state", "workspace")} | {
        os.path.realpath(world[k]) for k in ("home", "state", "workspace")
    }
    for call in model.calls:
        sent = json.dumps(call.request)
        for planted in (ENV_SECRET, SSH_SECRET, STATE_SECRET, TOKEN_VALUE, w.PLANTED_VALUE, *w.IDENTIFIERS, w.ANSWER):
            assert planted not in sent, (planted, call.kind)
        for root in roots:
            assert root not in sent, (root, call.kind)
    # the catalogue the librarian reads names no host path either
    catalogue = (env["paths"].checkout / CATALOGUE).read_text(encoding="utf-8")
    for root in roots:
        assert root not in catalogue


# --- off ----------------------------------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_off_binds_no_ask_offers_no_tool_and_generates_no_catalogue(librarian):
    env = librarian
    _set(env["monkeypatch"], SWITCH, "off")
    run = _open_request(env["paths"], env["sha"], env["monkeypatch"])
    actor_cell = "print('HAS_ASK', hasattr(memory, 'ask'))\ntools.answer('done')\n"
    script = Script([actor_cell], [])
    result, model, sent = await _ask_act(env, script)

    assert result == "done"
    assert script.librarian_calls == []
    first = script.actor_calls[0]
    assert "memory.ask" not in _system(first)
    assert _system(first).endswith(run.index)  # the index section, as without the librarian
    session = _actor_session(env["opened"])
    assert ASK_TOOL not in session.allow
    assert not (env["paths"].checkout / CATALOGUE).exists()
    assert not (env["paths"].checkout / w.DISPATCH_FILE).exists()
    for call in model.calls:
        assert _headers(sent, model, call).get("X-Unify-Call-Kind") != CALL_KIND
    _assert_ended(env["opened"])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_off_the_cell_has_no_ask(librarian):
    env = librarian
    _set(env["monkeypatch"], SWITCH, "off")
    _open_request(env["paths"], env["sha"], env["monkeypatch"])
    actor_cell = "tools.answer('HAS_ASK ' + str(hasattr(memory, 'ask')))\n"
    script = Script([actor_cell], [])
    result, _, _ = await _ask_act(env, script)
    assert result == "HAS_ASK False"
