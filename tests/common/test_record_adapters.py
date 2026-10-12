"""Both records behind the record tools satisfy the one ``Record`` interface, without changing what they do: a memory
pass's in-memory record and the run's agent record. A refusal comes back as ``RecordRefused`` with the record tool's
form code. Then the actor's record handlers, over the agent record, answer as the memory agents' do."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from unify.actor import tool_handlers as th
from unify.agents.options import Options
from unify.agents.pool import Pool
from unify.agents.record import Record as AgentRecord
from unify.common.record_adapters import AgentRecordAdapter, PassRecordAdapter
from unify.common.tool_registry import Record, RecordRefused, form_of
from unify.memory import messaging


def _pass_record():
    pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 600)
    for name in ("call-0", "call-1"):
        pr.add_helper(name, "a task")
        pr.record.add_agent(name, spawner="writer")
    return pr


def _agent_pool(replies: dict | None = None, **options):
    record = AgentRecord(None, options=Options(min_post_interval_s=0.0, **options), root="root")

    async def start(name, spawner, request, seq):
        await asyncio.sleep(0)
        return (replies or {}).get(name, f"{name} did: {request}")

    return Pool(record, start_helper=start, spawn_allowed=True)


def test_both_records_are_the_one_interface():
    assert isinstance(PassRecordAdapter(_pass_record()), Record)
    assert isinstance(AgentRecordAdapter(_agent_pool()), Record)


def test_the_pass_record_posts_reads_and_refuses_as_before():
    pr = _pass_record()
    rec = PassRecordAdapter(pr)
    out = rec.post("call-0", "@writer done reading e1")
    assert out["to"] == ["writer"] and out["text"].startswith("sent to writer")
    got = rec.unseen("writer")
    assert [e["text"] for e in got] == ["@writer done reading e1"] and rec.unseen("writer") == []
    with pytest.raises(RecordRefused) as refused:
        rec.post("writer", "to whom?")
    assert refused.value.form == "no_recipient" and form_of(refused.value.text) == "no_recipient"
    assert rec.post("writer", "go on", to=["call-1"])["to"] == ["call-1"]
    assert {r["name"] for r in rec.roster()} == {"writer", "call-0", "call-1"}


def test_the_pass_records_wait_returns_at_once_when_nothing_can_arrive():
    pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 600)
    started = time.monotonic()
    assert asyncio.run(PassRecordAdapter(pr).wait("writer", 30)) == []
    assert time.monotonic() - started < 5


def test_the_pass_record_refuses_a_start_without_a_helper_lane():
    with pytest.raises(RecordRefused) as refused:
        asyncio.run(PassRecordAdapter(_pass_record()).start("writer", "read e1 and e2 for totals"))
    assert refused.value.form == "no_lane"


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

    assert asyncio.run(go()) == ["short_request", "not_main_agent", "helper_limit", "unknown_helper"]


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


def test_store_skills_starts_the_review_as_host_work_that_outlives_the_call():
    done = asyncio.Event()

    class Handle:
        async def result(self):
            await done.wait()
            return "stored one function"

    async def start(request):
        return Handle()

    async def go():
        ctx = _ctx(_agent_pool(), store_skills=start)
        out = await th.store_skills(ctx, {"request": "store the parser with its retry loop"})
        running = [t for t in ctx.state.host_tasks if not t.done()]
        done.set()
        await asyncio.gather(*running)
        unavailable = await th.store_skills(_ctx(_agent_pool()), {"request": "store it"})
        no_context = await th.store_skills(
            _ctx(_agent_pool(), store_skills=lambda r: _text("No active execution context to snapshot.")),
            {"request": "store it"},
        )
        return out, len(running), unavailable, no_context

    async def _text(t):
        return t

    out, running, unavailable, no_context = asyncio.run(go())
    assert out.status == "ok" and running == 1
    assert unavailable.form == "unavailable" and no_context.form == "no_context"
