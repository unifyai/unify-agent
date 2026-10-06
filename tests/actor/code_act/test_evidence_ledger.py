"""Symbolic: ``UNIFY_EVIDENCE_LEDGER``, evidence that arrived during a session is kept and shown on a return.

On Continual-ARC LOW (cost decomposition, 6 Oct) Overhauled Unify bought 4.05
demonstration pairs per return visit where the program library bought 0.74:
Unify kept nothing that arrived during a session, so a return paid for it
again. With the switch on, a top-level session keeps every later message and
clarification under its request (redacted and capped), and a later session
whose request is the same or shares a rare whole identifier gets that
evidence as plain data, ``seen_before`` in the sandbox, with one sentence in
its first message. These tests pin what is kept, under which request, what a
return sees (and what an unrelated request and the session itself do not),
the prerequisite, and the path through ``act()``. The model is the scripted
transport of ``tests/cache_discipline_helpers``: nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import sqlite3

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act import test_evidence_list as tel
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import evidence_ledger as el
from unify.function_manager import task_origin
from unify.settings import ProductionSettings, SETTINGS

PREAMBLE = (
    "You are working through a stream of grid puzzles. Each instance names its "
    "task id and gives a test grid. Demonstrations cost budget."
)


def _visit(task: str, grid: str) -> str:
    return f"{PREAMBLE}\n\nNew instance. Task id: {task}\nTest grid: {grid}\n"


VISIT = _visit("task-3d61a9", "1 2 / 3 4")
RETURN = _visit("task-3d61a9", "5 6 / 7 8")
OTHERS = [_visit("task-7b02c4", "0 0 / 1 1"), _visit("task-e19f50", "2 2 / 0 0")]
UNRELATED = _visit("task-55c0d1", "9 9 / 9 9")
embed_calls = tel.embed_calls  # the concept fake through the real embedding cache

DEMOS = "Demos received (request 1):\n\nDemo pair 1 input: 1 0 / 0 1 output: 0 1 / 1 0"
FAKE_KEY = "sk-test-4f3e2d1c0b9a8f7e6d5c"  # pragma: allowlist secret
FAKE_PASSWORD = "correct-horse-battery-77"  # pragma: allowlist secret
ENV_SECRET = "envheld-credential-5150aa"  # pragma: allowlist secret


@pytest.fixture
def ledger(monkeypatch):
    """The ledger with request records on (or the switch *on*/off)."""

    def set_(*, on=True, origin=True):
        monkeypatch.setattr(SETTINGS, "UNIFY_EVIDENCE_LEDGER", on)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


@contextlib.contextmanager
def _session(request):
    """A keyed top-level request with its ledger session, as ``act()`` starts one."""
    token = task_origin.enter(request)
    ledger_token = el.enter() if token is not None else None
    try:
        yield el.current_session()
    finally:
        el.leave(ledger_token)
        task_origin.leave(token)


def _log(*requests):
    for request in requests:
        with _session(request):
            pass


def _rows() -> list[dict]:
    path = task_origin.request_log_path()
    if not path.exists():
        return []
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'evidence'",
        ).fetchone():
            return []
        return [dict(r) for r in conn.execute("SELECT * FROM evidence ORDER BY seq")]


def _table_exists() -> bool:
    path = task_origin.request_log_path()
    if not path.exists():
        return False
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'evidence'",
            ).fetchone(),
        )


# ── the setting and its prerequisite ─────────────────────────────────────


def test_the_switch_is_off_by_default_and_parses():
    assert ProductionSettings().UNIFY_EVIDENCE_LEDGER is False
    assert ProductionSettings(UNIFY_EVIDENCE_LEDGER="1").UNIFY_EVIDENCE_LEDGER is True


def test_the_ledger_refuses_to_start_without_request_records(ledger):
    ledger(origin=False)
    with pytest.raises(
        ValueError,
        match="UNIFY_EVIDENCE_LEDGER needs UNIFY_TASK_ORIGIN",
    ):
        task_origin.require_origin_link_prerequisites()
    ledger()
    task_origin.require_origin_link_prerequisites()


# ── off ──────────────────────────────────────────────────────────────────


def test_off_keeps_reads_and_says_nothing(ledger):
    ledger(on=False)
    _log(*OTHERS)
    with _session(VISIT) as session:
        assert session is None
        assert el.enter() is None
        assert not el.record(el.MESSAGE, DEMOS)
    fake = el.Session(key="0" * 16, text=VISIT)
    assert not el.record(el.MESSAGE, DEMOS, key=fake)
    assert not _table_exists()
    with _session(RETURN):
        assert el.earlier(task_origin.bounded_text(RETURN)) == []
    assert el.line([]) is None


# ── what is kept ─────────────────────────────────────────────────────────


def test_a_message_is_kept_under_its_request_redacted(ledger, monkeypatch):
    ledger()
    monkeypatch.setenv("LEDGER_TEST_SERVICE_TOKEN", ENV_SECRET)
    text = (
        f"{DEMOS}\nUse api_key={FAKE_KEY} and password: {FAKE_PASSWORD}; "
        f"also {ENV_SECRET} and bare {FAKE_KEY}."
    )
    with _session(VISIT) as session:
        assert el.record(el.MESSAGE, text)
    [row] = _rows()
    assert row["text_key"] == task_origin.text_key(task_origin.bounded_text(VISIT))
    assert row["task_key"] == task_origin.task_key(VISIT)
    assert row["session"] == session.id
    assert row["kind"] == el.MESSAGE and row["label"] is None
    content = row["content"]
    assert DEMOS in content
    for secret in (FAKE_KEY, FAKE_PASSWORD, ENV_SECRET):
        assert secret not in content
    assert "api_key=<redacted:" in content and "password: <redacted:" in content
    assert "[REDACTED:LEDGER_TEST_SERVICE_TOKEN]" in content
    # The same key, bare, gets the same placeholder as where it was named.
    placeholders = set(part.split(">")[0] for part in content.split("<redacted:")[1:])
    assert len(placeholders) == 2
    assert row["chars"] == len(content)
    assert row["created_at"]


def test_items_and_sessions_are_capped(ledger, monkeypatch):
    ledger()
    with _session(VISIT) as session:
        assert el.record(el.MESSAGE, "x" * (el.MAX_ITEM_CHARS + 500))
        big = _rows()[-1]
        assert el.MAX_ITEM_CHARS - 10 < len(big["content"]) <= el.MAX_ITEM_CHARS
        assert big["content"].endswith("more characters not kept]")
        assert big["chars"] == el.MAX_ITEM_CHARS + 500
        while el.record(el.MESSAGE, "y" * el.MAX_ITEM_CHARS):
            pass
        assert session.kept_chars <= el.MAX_SESSION_CHARS
        assert sum(len(r["content"]) for r in _rows()) <= el.MAX_SESSION_CHARS
        assert not el.record(el.MESSAGE, "one more")
    # Another session keeps its own allowance.
    with _session(RETURN):
        assert el.record(el.MESSAGE, "fresh")
    # The table keeps the latest LEDGER_SIZE items.
    monkeypatch.setattr(el, "LEDGER_SIZE", 3)
    with _session(UNRELATED):
        for i in range(5):
            el.record(el.MESSAGE, f"item {i}")
    assert [r["content"] for r in _rows()] == ["item 2", "item 3", "item 4"]


def test_a_clarification_is_kept_as_its_question_and_answer(ledger):
    ledger()
    _log(*OTHERS)
    with _session(VISIT) as session:
        session.question("Which colour fills the border?")
        assert session.answer("Blue, always.")
    [row] = _rows()
    assert row["kind"] == el.CLARIFICATION
    assert row["content"] == "Q: Which colour fills the border?\nA: Blue, always."
    with _session(RETURN):
        items = el.earlier(task_origin.bounded_text(RETURN))
    assert [i["content"] for i in items] == [row["content"]]
    assert el.line(items) == (
        "Evidence received while handling 1 earlier request like this one is in "
        "`seen_before` (1 clarification)."
    )


# ── what a return sees ───────────────────────────────────────────────────


def test_a_return_sees_the_earlier_evidence_and_others_do_not(ledger):
    ledger()
    _log(*OTHERS)
    with _session(VISIT) as first:
        el.record(el.MESSAGE, DEMOS)
        el.record(el.MESSAGE, "Feedback: wrong, the border stays.")
        # The session never sees its own evidence.
        assert el.earlier(first.text) == []
    with _session(RETURN) as second:
        items = el.earlier(second.text)
    assert [i["content"] for i in items] == [
        "Feedback: wrong, the border stays.",
        DEMOS,
    ]
    item = items[0]
    assert set(item) == {"request", "visit", "kind", "label", "content", "age_seconds"}
    assert item["request"] == task_origin.bounded_text(VISIT)[: el.REQUEST_CHARS]
    assert item["visit"] == 1 and item["kind"] == el.MESSAGE
    assert isinstance(item["age_seconds"], int) and item["age_seconds"] >= 0
    assert json.loads(json.dumps(items)) == items  # plain data
    assert el.line(items) == (
        "Evidence received while handling 1 earlier request like this one is in "
        "`seen_before` (2 messages)."
    )
    # A request sharing no rare identifier sees nothing.
    with _session(UNRELATED) as other:
        assert el.earlier(other.text) == []
    # The same request again sees what both earlier sessions kept.
    with _session(VISIT) as again:
        el.record(el.MESSAGE, "mine")
        items = el.earlier(again.text)
    assert "mine" not in [i["content"] for i in items]
    assert len(items) == 2


def test_an_identifier_every_request_names_links_nothing(ledger):
    ledger()
    shared = "Stream batch run-2024q3."
    _log(*(f"{shared} {o}" for o in OTHERS))
    with _session(f"{shared} first job"):
        el.record(el.MESSAGE, DEMOS)
    with _session(f"{shared} second job") as later:
        assert el.earlier(later.text) == []


def test_what_a_return_sees_is_capped(ledger, monkeypatch):
    ledger()
    _log(*OTHERS)
    monkeypatch.setattr(el, "MAX_SHOWN_ITEMS", 3)
    with _session(VISIT):
        for i in range(5):
            el.record(el.MESSAGE, f"pair {i}")
    with _session(RETURN) as later:
        items = el.earlier(later.text)
    assert [i["content"] for i in items] == ["pair 4", "pair 3", "pair 2"]
    monkeypatch.setattr(el, "MAX_SHOWN_CHARS", 13)
    with _session(RETURN) as later:
        assert [i["content"] for i in el.earlier(later.text)] == ["pair 4", "pair 3"]


def test_a_sqlite_error_only_warns(ledger, monkeypatch):
    ledger()
    warnings: list[str] = []

    def broken(path):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(el, "_connect", broken)
    monkeypatch.setattr(el.logger, "warning", warnings.append)
    with _session(VISIT):
        assert not el.record(el.MESSAGE, DEMOS)
    assert warnings == ["evidence not written: OperationalError: disk I/O error"]


# ── through the actor ────────────────────────────────────────────────────


def _first_user(requests: list[dict]) -> str:
    return next(m["content"] for m in requests[0]["messages"] if m["role"] == "user")


def _tool_outputs(requests: list[dict]) -> str:
    return "\n".join(
        json.dumps(m["content"])
        for r in requests
        for m in r["messages"]
        if m.get("role") == "tool"
    )


async def _act(actor, task, replies, *, interject=None, **kwargs):
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act(task, persist=False, **kwargs)
            if interject is not None:
                # As the CLI's stdin reader sends it: from a task that does
                # not carry the request's context.
                await asyncio.create_task(
                    handle.interject(interject),
                    context=contextvars.Context(),
                )
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return h.session_requests(provider.requests)


def _done(n=16):
    return [lambda: h.completion(content="done")] * n


def _print_seen_before():
    return lambda: h.completion(
        calls=[
            (
                "execute_code",
                {
                    "thought": "Look.",
                    "code": "import json\nprint(json.dumps(seen_before))",
                },
            ),
        ],
    )


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_session_keeps_its_interjection_and_a_return_sees_it(
    ledger,
    embed_calls,
    monkeypatch,
):
    ledger()
    _log(*OTHERS)
    monkeypatch.setenv("LEDGER_TEST_SERVICE_TOKEN", ENV_SECRET)
    message = f"{DEMOS}\nauth_token={FAKE_KEY}"
    await _act(caa.CodeActActor(), VISIT, _done(), interject=message)
    [row] = _rows()
    assert row["task_key"] == task_origin.task_key(VISIT)
    assert DEMOS in row["content"] and FAKE_KEY not in row["content"]

    requests = await _act(
        caa.CodeActActor(),
        RETURN,
        [_print_seen_before()] + _done(),
    )
    first = _first_user(requests)
    assert (
        "Evidence received while handling 1 earlier request like this one is in "
        "`seen_before` (1 message)."
    ) in first
    outputs = _tool_outputs(requests)
    assert "Demo pair 1 input" in outputs and FAKE_KEY not in outputs

    # An unrelated request is told nothing and has no ``seen_before``.
    requests = await _act(
        caa.CodeActActor(),
        UNRELATED,
        [_print_seen_before()] + _done(),
    )
    assert "seen_before" not in _first_user(requests)
    assert "NameError" in _tool_outputs(requests)


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_off_the_actor_keeps_and_says_nothing(ledger, embed_calls):
    ledger(on=False)
    _log(*OTHERS)
    await _act(caa.CodeActActor(), VISIT, _done(), interject=DEMOS)
    requests = await _act(caa.CodeActActor(), RETURN, _done())
    assert "seen_before" not in _first_user(requests)
    assert not _table_exists()


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_clarification_through_the_actor_is_kept(ledger, embed_calls):
    ledger()
    _log(*OTHERS)
    up, down = asyncio.Queue(), asyncio.Queue()
    actor = caa.CodeActActor()
    replies = [
        lambda: h.completion(
            calls=[("request_clarification", {"question": "Keep the border?"})],
        ),
    ] + _done()
    try:
        with h.scripted(replies):
            handle = await actor.act(
                VISIT,
                persist=False,
                _clarification_up_q=up,
                _clarification_down_q=down,
            )
            assert await asyncio.wait_for(up.get(), 30) == "Keep the border?"
            await down.put("Yes, keep it.")
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    [row] = _rows()
    assert row["kind"] == el.CLARIFICATION
    assert row["content"] == "Q: Keep the border?\nA: Yes, keep it."


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_the_worker_sandbox_gets_seen_before_as_plain_data(
    core_world,
    ledger,
    monkeypatch,
):
    ledger()
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    _log(*OTHERS)
    with _session(VISIT):
        el.record(el.MESSAGE, DEMOS)
    requests = await _act(new_actor(), RETURN, [_print_seen_before()] + _done())
    assert "is in `seen_before` (1 message)." in _first_user(requests)
    outputs = _tool_outputs(requests)
    assert "Demo pair 1 input" in outputs and "age_seconds" in outputs
