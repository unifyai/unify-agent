"""The run's agent record behind the actor's record tools satisfies the ``Record`` interface without changing what it
does. A refusal comes back as ``RecordRefused`` with the record tool's form code. Then the actor's record handlers,
over the agent record, answer as the memory agents' do."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Optional, Protocol, runtime_checkable

import pytest

from unify.actor import tool_handlers as th
from unify.agents.pool import Pool
from tests.agents.records import run_record
from unify.common.record_adapters import AgentRecordAdapter
from unify.common.tool_registry import RecordRefused


@runtime_checkable
class Record(Protocol):
    """The record the record tools use, whichever agent calls them: a memory pass's in-memory record
    (:class:`unify.common.record_adapters.PassRecordAdapter`) or the actor's agent record
    (:class:`unify.common.record_adapters.AgentRecordAdapter`). A refused operation raises ``RecordRefused``."""

    def post(self, sender: str, text: str, to: Optional[list] = None) -> dict: ...

    def unseen(self, reader: str) -> list: ...

    async def wait(self, reader: str, timeout: float) -> list: ...

    def roster(self) -> list: ...

    async def start(self, spawner: str, task: str, **kw: Any) -> dict: ...

    async def stop(self, by: str, name: str) -> dict: ...


def _agent_pool(replies: dict | None = None, **options):
    record = run_record(**options)

    async def start(name, spawner, request, seq):
        await asyncio.sleep(0)
        return (replies or {}).get(name, f"{name} did: {request}")

    return Pool(record, start_helper=start)


def test_the_agent_record_is_the_record_interface():
    assert isinstance(AgentRecordAdapter(_agent_pool()), Record)


def test_the_agent_record_starts_a_teammate_and_its_reply_arrives():
    async def go():
        pool = _agent_pool()
        rec = AgentRecordAdapter(pool)
        started = await rec.start("root", "summarise the second report in two lines")
        got = await rec.wait("root", 5)
        await pool.close("done")
        return started, got, rec.roster()

    started, got, roster = asyncio.run(go())
    assert started == {"name": "h1"}
    assert got and got[-1]["author"] == "h1" and got[-1]["kind"] == "reply"
    assert {r["name"]: r["role"] for r in roster} == {"root": "actor", "h1": "teammate"}


def test_the_agent_records_refusals_carry_the_tools_form_codes():
    async def go():
        pool = _agent_pool(max_live=1)
        rec = AgentRecordAdapter(pool)
        forms = []
        for spawner, task in (("root", "too short"), ("root", "a request that is long enough to run")):
            try:
                await rec.start(spawner, task)
            except RecordRefused as exc:
                forms.append(exc.form)
        for spawner, task in (("h1", "a request that is long enough to run"), ("root", "another long request here")):
            try:
                await rec.start(spawner, task)
            except RecordRefused as exc:
                forms.append(exc.form)
        try:
            await rec.stop("root", "nobody")
        except RecordRefused as exc:
            forms.append(exc.form)
        await pool.close("done")
        return forms

    # any agent may start agents (h1 too); the record's live limit refuses both later starts
    assert asyncio.run(go()) == ["short_request", "helper_limit", "helper_limit", "unknown_helper"]


def test_the_agent_record_refuses_an_oversized_post_with_its_form():
    rec = AgentRecordAdapter(_agent_pool())
    with pytest.raises(RecordRefused) as refused:
        rec.post("root", "@user " + "x" * 20000)
    assert refused.value.form == "too_large"


def test_unseen_entries_are_given_once():
    pool = _agent_pool()
    rec = AgentRecordAdapter(pool)
    pool.record.add_agent("h1", spawner="root")
    rec.post("h1", "@root found it")
    assert [e["text"] for e in rec.unseen("root")] == ["@root found it"]
    assert rec.unseen("root") == []


# --- the actor's record handlers over the agent record ---


def _ctx(pool, name="root", **state):
    return SimpleNamespace(
        agent="actor" if name == "root" else "teammate",
        name=name,
        record=AgentRecordAdapter(pool),
        deadline=time.monotonic() + 60,
        state=th.ActorRequestState(**state),
    )


def test_the_actors_record_tools_answer_as_the_memory_agents_do():
    async def go():
        pool = _agent_pool()
        ctx = _ctx(pool)
        out = {}
        out["no_text"] = th.send(ctx, {"text": " "})
        out["delegate"] = await th.delegate(ctx, {"task": "summarise the second report in two lines"})
        out["no_timeout"] = await th.wait(ctx, {})
        out["wait"] = await th.wait(ctx, {"timeout": 5})
        out["who"] = await th.who(ctx, {})
        out["reply"] = th.reply(ctx, {"agent": "nobody", "text": "hi"})
        out["short"] = await th.delegate(ctx, {"task": "too short"})
        out["messages"] = th.messages(ctx, {})
        await pool.close("done")
        return out

    out = asyncio.run(go())
    assert out["no_text"].status == "refused" and out["no_text"].form == "no_text"
    assert out["delegate"].status == "ok" and out["delegate"].text.startswith("started h1")
    assert out["no_timeout"].form == "bad_timeout"
    assert "h1 (done): h1 did: summarise" in out["wait"].text
    assert "- h1 (teammate, replied)" in out["who"].text
    assert out["reply"].form == "unknown_name"
    assert out["short"].form == "short_request" and out["short"].text.startswith("refused: no helper started:")
    assert out["messages"].text.startswith("no new messages")


def test_a_teammates_question_that_names_no_one_goes_to_whoever_started_it():
    async def go():
        pool = _agent_pool()
        pool.record.add_agent("h1", spawner="root")
        ctx = _ctx(pool, "h1", spawner="root")
        ctx.deadline = time.monotonic() + 0.5
        answer = await th.ask(ctx, {"question": "should totals include tax?"})
        return answer, pool.record.entries[-1]

    answer, entry = asyncio.run(go())
    assert entry.mentions == ("root",) and answer.status == "ok"


