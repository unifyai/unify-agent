"""Rev 2 (PLAN-REV2 item 1): the librarian as a role of the run record.

``@librarian`` names the role once the run seeds it (``run_record.seed_role``); each message to it is answered by one
short-lived instance, ``librarian-<hex>``, started by the request's role dispatcher (``integration.request``), which
posts exactly one reply (``reply_to`` the message, mentioning its author) and ends. ``memory.ask``'s instance is the
role's synchronous client and shares its state (the exact-match answer cache). An instance's address, like every
agent's, is refused once it has ended, with a pointer; ``@librarian`` in a run without the role is refused too.

Keyless. The instances here are stub sessions (what ``answer_message`` needs of a session: ``name``, ``run_record``,
``answer``, ``close``, ``answers``); one test runs a real librarian session in its box on the scripted model.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from types import SimpleNamespace

import pytest

from unify.agents import run_record as rr
from unify.agents.options import Options
from unify.agents.record import HARNESS
from unify.common.tool_registry import RecordRefused
from unify.memory import librarian as L
from unify.memory.integration import request as R

FAST = Options(max_live=16, max_total=64, max_entries=None, max_record_bytes=rr.MAX_RECORD_BYTES, min_post_interval_s=0.0)


def _record(home):
    return rr.join_run_record(home, options=FAST)


class Instance:
    """A stub librarian instance: joins the record as its name, answers by *script*, leaves at close."""

    def __init__(self, rec, name, script, home=None):
        self.run_record, self.name, self.script = rec, name, script
        rec.join(name, "librarian")
        self.answers: dict = {}
        self.messages: list = []
        self.library = "c0ffee"
        self.closed = False
        self.questions: list[str] = []
        if home is not None:
            L._role_state(self, home)

    async def answer(self, question, *, schema=None, last_message=None):
        self.questions.append(question)
        key = f"k:{question}"
        if key in self.answers:
            return {**self.answers[key], L.ALREADY_GIVEN: True}
        got = await self.script(question)
        self.answers[key] = got
        return got

    async def close(self):
        self.closed = True
        self.run_record.leave(self.name)


def _opener(rec, home, script, log):
    async def open_session(*, mode="ask"):
        assert mode == "message"
        inst = Instance(rec, rec.unique_name("librarian", os.urandom(8).hex()), script, home)
        log.append(inst)
        return inst

    return open_session


async def _fits(question):
    return {"fits": [], "nearest": []}


def _seeded(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    assert rr.seed_role(home, "librarian", by="test") is True
    assert rr.seed_role(home, "librarian", by="again") is False  # once per run
    rec.join("actor-aaaa", "actor", episode="ep-a")
    return home, rec


# --- addresses ------------------------------------------------------------------------------------------------------


def test_at_librarian_is_the_role_once_seeded_and_refused_with_a_pointer_without_it(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("actor-aaaa", "actor", episode="ep-a")
    rec.join("writer", "writer", state="idle")
    view = rr.RunRecordView(rec, "actor-aaaa", "actor")
    with pytest.raises(RecordRefused) as no_role:
        view.post("actor-aaaa", "@librarian which stored items fit this?")
    assert no_role.value.form == "no_role" and "there is no librarian in this run" in no_role.value.text
    assert "@writer" in no_role.value.text  # the pointer: what reaches someone now
    assert rr.read_entries(home) == []  # nothing was posted
    rr.seed_role(home, "librarian", by="test")
    out = view.post("actor-aaaa", "@librarian which stored items fit this?")
    assert out["to"] == ["librarian"] and out["ended"] == []
    row = rr.read_entries(home)[-1]
    assert row["mentions"] == ["librarian"] and row["role"] == "actor"
    files = sorted(p.name for p in rr.role_dir(home, "librarian").iterdir())
    assert {"role.json", "answers.jsonl", "notes.md", "claims.jsonl", "events.jsonl"} <= set(files)
    assert oct(os.stat(rr.role_dir(home, "librarian")).st_mode & 0o777) == "0o700"


def test_a_post_to_an_ended_instance_is_refused_with_a_pointer_and_the_writer_is_kept(tmp_path):
    home, rec = _seeded(tmp_path)
    rec.join("librarian-1a2b", "librarian")
    rec.join("fork-3c4d", "fork", episode="ep-a")
    rec.join("h1", "helper")
    rec.join("writer", "writer", state="idle")
    for n in ("librarian-1a2b", "fork-3c4d", "h1"):
        rec.leave(n)
    rec.set_state("writer", "ended")
    view = rr.RunRecordView(rec, "actor-aaaa", "actor")
    before = len(rr.read_entries(home))
    with pytest.raises(RecordRefused) as inst:
        view.post("actor-aaaa", "@librarian-1a2b one more question")
    assert inst.value.form == "ended" and "address the role: @librarian" in inst.value.text
    assert re.search(r"@librarian-1a2b ended at \S+", inst.value.text)
    with pytest.raises(RecordRefused) as fork:
        view.post("actor-aaaa", "@fork-3c4d what did the episode do?")
    assert re.search(r"the fork @fork-3c4d ended at \S+", fork.value.text)
    with pytest.raises(RecordRefused) as helper:
        view.post("actor-aaaa", "@h1 @librarian done?")  # one ended name refuses the whole post
    assert "@h1 ended at" in helper.value.text
    assert len(rr.read_entries(home)) == before  # nothing was posted
    out = view.post("actor-aaaa", "@writer the served item was wrong here")  # the writer: kept for when it next runs
    assert out["to"] == ["writer"] and out["ended"] == ["writer"]


# --- one reply per message ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_message_to_the_role_gets_exactly_one_reply_from_one_instance_that_then_ends(tmp_path):
    home, rec = _seeded(tmp_path)
    view = rr.RunRecordView(rec, "actor-aaaa", "actor")
    asked = view.post("actor-aaaa", "@librarian which stored items fit a list of numbers?")
    opened: list[Instance] = []
    done = await R._role_round(_record(home), home, _opener(_record(home), home, _fits, opened))
    assert done == 1 and len(opened) == 1
    inst = opened[0]
    assert inst.questions == ["which stored items fit a list of numbers?"]  # the role's address is not the question
    replies = [e for e in rr.read_entries(home) if e.get("reply_to") == asked["seq"]]
    assert len(replies) == 1
    (rep,) = replies
    assert rep["kind"] == "reply" and rep["author"] == inst.name and rep["mentions"] == ["actor-aaaa"]
    assert json.loads(rep["text"].split(": ", 1)[1]) == {"fits": [], "nearest": []}
    assert rr.role_claims(home, "librarian")[asked["seq"]]["state"] == "answered"
    # the instance ended: its address is refused from now on, pointing at the role
    assert inst.closed and rec._row(inst.name)["state"] == "ended"
    with pytest.raises(RecordRefused, match="address the role: @librarian"):
        view.post("actor-aaaa", f"@{inst.name} thanks")
    # the delivery log shows the instance was given the message; the author gets the reply at its next boundary
    deliveries = [d for d in rr.read_deliveries(home) if d["agent"] == inst.name]
    assert deliveries and deliveries[0]["seqs"] == [asked["seq"]] and deliveries[0]["via"] == "messages"
    assert f"#{rep['seq']} {inst.name} (reply)" in rec.take_block("actor-aaaa")
    # a second round finds nothing to do: never a second reply
    assert await R._role_round(_record(home), home, _opener(_record(home), home, _fits, opened)) == 0
    assert len([e for e in rr.read_entries(home) if e.get("reply_to") == asked["seq"]]) == 1


@pytest.mark.asyncio
async def test_a_withdrawn_message_gets_no_reply(tmp_path):
    home, rec = _seeded(tmp_path)
    view = rr.RunRecordView(rec, "actor-aaaa", "actor")
    asked = view.post("actor-aaaa", "@librarian anything for this?")
    rec.append("actor-aaaa", "never mind", kind="cancel", mentions=[], reply_to=asked["seq"])
    opened: list = []
    await R._role_round(_record(home), home, _opener(_record(home), home, _fits, opened))
    assert opened == []  # no instance started for it
    assert [e for e in rr.read_entries(home) if e.get("reply_to") == asked["seq"] and e["kind"] == "reply"] == []
    assert rr.role_claims(home, "librarian")[asked["seq"]]["state"] == "withdrawn"


@pytest.mark.asyncio
async def test_a_storm_of_mentions_is_one_instance_each_one_at_a_time_in_order(tmp_path):
    home, rec = _seeded(tmp_path)
    rec.join("writer", "writer")
    view_a = rr.RunRecordView(rec, "actor-aaaa", "actor")
    view_w = rr.RunRecordView(rec, "writer", "writer")
    seqs = []
    for i in range(6):
        v, who = (view_a, "actor-aaaa") if i % 2 == 0 else (view_w, "writer")
        seqs.append(v.post(who, f"@librarian question {i}")["seq"])
    running: list[int] = []
    peak = [0]

    async def slow(question):
        running.append(1)
        peak[0] = max(peak[0], len(running))
        await asyncio.sleep(0.01)
        running.pop()
        return {"fits": [], "nearest": []}

    opened: list[Instance] = []
    assert await R._role_round(_record(home), home, _opener(_record(home), home, slow, opened)) == 6
    assert peak[0] == 1 and len(opened) == 6 and len({i.name for i in opened}) == 6
    assert [i.questions[0] for i in opened] == [f"question {i}" for i in range(6)]  # queue order kept
    replies = [e for e in rr.read_entries(home) if e["kind"] == "reply"]
    assert [r["reply_to"] for r in replies] == seqs
    assert [r["mentions"] for r in replies] == [["actor-aaaa"], ["writer"]] * 3


@pytest.mark.asyncio
async def test_another_process_holding_the_instance_lock_defers_the_message_to_a_later_round(tmp_path):
    home, rec = _seeded(tmp_path)
    asked = rr.RunRecordView(rec, "actor-aaaa", "actor").post("actor-aaaa", "@librarian anything?")
    opened: list = []
    with rr.role_lock(home, "librarian"):
        with pytest.raises(rr.RoleBusy):
            with rr.role_lock(home, "librarian", wait=False):
                pass
        assert await R._role_round(_record(home), home, _opener(_record(home), home, _fits, opened)) == 0
    assert opened == [] and asked["seq"] not in rr.role_claims(home, "librarian")
    assert await R._role_round(_record(home), home, _opener(_record(home), home, _fits, opened)) == 1


@pytest.mark.asyncio
async def test_a_stopped_instance_marks_its_message_abandoned_and_the_next_round_answers_it(tmp_path):
    home, rec = _seeded(tmp_path)
    asked = rr.RunRecordView(rec, "actor-aaaa", "actor").post("actor-aaaa", "@librarian anything?")
    started = asyncio.Event()

    async def hang(question):
        started.set()
        await asyncio.sleep(3600)

    opened: list[Instance] = []
    task = asyncio.ensure_future(R._role_round(_record(home), home, _opener(_record(home), home, hang, opened)))
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rr.role_claims(home, "librarian")[asked["seq"]]["state"] == "abandoned"
    assert opened[0].closed and [e for e in rr.read_entries(home) if e.get("reply_to") == asked["seq"]] == []
    assert await R._role_round(_record(home), home, _opener(_record(home), home, _fits, opened)) == 1
    assert rr.role_claims(home, "librarian")[asked["seq"]]["state"] == "answered"


@pytest.mark.asyncio
async def test_an_instance_that_fails_or_cannot_start_still_gives_one_reply_saying_why(tmp_path):
    home, rec = _seeded(tmp_path)
    view = rr.RunRecordView(rec, "actor-aaaa", "actor")
    a = view.post("actor-aaaa", "@librarian first?")["seq"]

    async def no_answer(question):
        raise L.NoAnswer("the operational step guard was reached")

    await R._role_round(_record(home), home, _opener(_record(home), home, no_answer, []))
    (r1,) = [e for e in rr.read_entries(home) if e.get("reply_to") == a]
    assert r1["text"].startswith(f"no answer to #{a}: ") and rr.role_claims(home, "librarian")[a]["state"] == "failed"
    b = view.post("actor-aaaa", "@librarian second?")["seq"]

    async def broken(question):
        raise ConnectionError("upstream detail that is never shown")

    await R._role_round(_record(home), home, _opener(_record(home), home, broken, []))
    (r2,) = [e for e in rr.read_entries(home) if e.get("reply_to") == b]
    assert r2["author"] == HARNESS and "ConnectionError" in r2["text"] and "upstream detail" not in r2["text"]
    c = view.post("actor-aaaa", "@librarian third?")["seq"]

    async def unavailable(*, mode="ask"):
        raise L.Unavailable("no library is mounted in this request")

    await R._role_round(_record(home), home, unavailable)
    (r3,) = [e for e in rr.read_entries(home) if e.get("reply_to") == c]
    assert r3["author"] == HARNESS and r3["mentions"] == ["actor-aaaa"] and "no library is mounted" in r3["text"]


# --- the role's state -----------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_ask_and_a_mention_share_the_roles_exact_match_answer_cache(tmp_path):
    home, rec = _seeded(tmp_path)
    calls: list[str] = []

    async def counted(question):
        calls.append(question)
        return {"fits": [], "nearest": []}

    ask = Instance(rec, "librarian-a5c0", counted, home)
    ask.role_home = home
    await ask.answer("which items fit?")
    assert L.keep_answers(ask) == 1 and L.keep_answers(ask) == 0  # kept once
    assert [r["key"] for r in rr.role_answers(home, "librarian")] == ["k:which items fit?"]
    # a message instance started later reads the role's state at its start: the same ask is answered from it
    asked = rr.RunRecordView(rec, "actor-aaaa", "actor").post("actor-aaaa", "@librarian which items fit?")
    opened: list[Instance] = []
    await R._role_round(_record(home), home, _opener(_record(home), home, counted, opened))
    assert calls == ["which items fit?"]  # no second model answer
    (rep,) = [e for e in rr.read_entries(home) if e.get("reply_to") == asked["seq"]]
    assert json.loads(rep["text"].split(": ", 1)[1])[L.ALREADY_GIVEN] is True
    # an ask that differs in anything (another question) is never served from it
    rr.RunRecordView(rec, "actor-aaaa", "actor").post("actor-aaaa", "@librarian which items fit a table?")
    await R._role_round(_record(home), home, _opener(_record(home), home, counted, opened))
    assert calls == ["which items fit?", "which items fit a table?"]


def test_an_instance_reads_the_notes_earlier_instances_kept(tmp_path):
    home, rec = _seeded(tmp_path)
    (rr.role_dir(home, "librarian") / rr.ROLE_NOTES).write_text("- the captures under shapes repeat often\n")
    inst = Instance(rec, "librarian-77aa", _fits, home)
    assert inst.messages == [{"role": "user", "content": f"{L.NOTES_HEAD}\n- the captures under shapes repeat often"}]
    empty_home = tmp_path / "other"
    rec2 = _record(empty_home)
    rr.seed_role(empty_home, "librarian", by="test")
    assert Instance(rec2, "librarian-88bb", _fits, empty_home).messages == []


def test_memory_asks_go_on_the_record_as_an_ask_and_its_reply_mentioning_no_one(tmp_path):
    home, rec = _seeded(tmp_path)
    rec.join("librarian-ep0a", "librarian", episode="ep-a")
    me = SimpleNamespace(run_record=rec, name="librarian-ep0a", episode="ep-a")
    L.AskInstance._post_ask(me, "which items fit?", json.dumps({"value": {"steps": ["a"]}, "repairs": 0}))
    ask, rep = rr.read_entries(home)[-2:]
    assert ask["author"] == HARNESS and ask["mentions"] == [] and "actor-aaaa asked the librarian (memory.ask): which items fit?" == ask["text"]
    assert rep["author"] == "librarian-ep0a" and rep["kind"] == "reply" and rep["reply_to"] == ask["seq"] and rep["mentions"] == []
    assert json.loads(rep["text"].split(": ", 1)[1]) == {"value": {"steps": ["a"]}, "repairs": 0}
    assert rec.take_block("actor-aaaa") is None  # the asker already has the answer: nothing is delivered again


def test_a_long_answer_is_cut_to_the_records_entry_limit():
    from unify.agents.entry import MAX_TEXT_BYTES

    text = L.reply_text(9, {"fits": [{"id": "x", "why": "y" * (2 * MAX_TEXT_BYTES)}]})
    assert len(text.encode()) <= MAX_TEXT_BYTES and text.startswith("answer to #9: ") and "cut at" in text


# --- waiting --------------------------------------------------------------------------------------------------------


def test_a_question_to_the_role_keeps_its_author_waiting_until_the_reply(tmp_path):
    home, rec = _seeded(tmp_path)
    view = rr.RunRecordView(rec, "actor-aaaa", "actor")
    assert rec.nobody_can_answer("actor-aaaa") is True
    asked = view.post("actor-aaaa", "@librarian anything?")
    assert rec.nobody_can_answer("actor-aaaa") is False  # an instance can still reply
    rec.join("librarian-ab12", "librarian")
    rec.append("librarian-ab12", "answer", kind="reply", mentions=["actor-aaaa"], reply_to=asked["seq"])
    rec.take_unseen("actor-aaaa", "wait")
    rec.leave("librarian-ab12")
    assert rec.nobody_can_answer("actor-aaaa") is True


# --- the request's dispatcher ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_request_seeds_the_role_and_dispatches(tmp_path, monkeypatch):
    home = tmp_path / "home"
    run = SimpleNamespace(paths=SimpleNamespace(home=home), episode_id="ep-a")
    assert not rr.role_seeded(home, "librarian")
    seen = []

    async def fake_dispatcher(r, **kw):
        seen.append(r)

    monkeypatch.setattr(R, "role_dispatcher", fake_dispatcher)
    task = R.start_role_dispatcher(run)
    assert task is not None and rr.role_seeded(home, "librarian")
    await task
    assert seen == [run]


def test_with_no_running_loop_nothing_is_seeded(tmp_path, monkeypatch):
    home = tmp_path / "home"
    run = SimpleNamespace(paths=SimpleNamespace(home=home), episode_id="ep-a")
    assert R.start_role_dispatcher(run) is None and not rr.role_seeded(home, "librarian")


# --- texts -----------------------------------------------------------------------------------------------------------


def test_role_names_and_refusal_texts_name_no_world(tmp_path):
    from tests.memory.test_no_world_vocabulary import GUARD

    home, rec = _seeded(tmp_path)
    rec.join("librarian-1a2b", "librarian")
    rec.leave("librarian-1a2b")
    texts = [L.NOTES_HEAD, L.reply_text(1, {"fits": []}), L.question_of("@librarian x"), *rr.INSTANCE_ROLES, *rr.CLAIM_STATES]
    view = rr.RunRecordView(rec, "actor-aaaa", "actor")
    for text in ("@librarian-1a2b x",):
        with pytest.raises(RecordRefused) as exc:
            view.post("actor-aaaa", text)
        texts.append(exc.value.text)
    other = tmp_path / "other"
    rec2 = _record(other)
    rec2.join("actor-bbbb", "actor")
    with pytest.raises(RecordRefused) as exc:
        rr.RunRecordView(rec2, "actor-bbbb", "actor").post("actor-bbbb", "@librarian x")
    texts.append(exc.value.text)
    for text in texts:
        assert not GUARD.search(text), text


# --- one real instance, in its box, on the scripted model ------------------------------------------------------------

needs_box = pytest.mark.skipif(shutil.which("bwrap") is None or not os.access("/usr/bin/prlimit", os.X_OK),
                               reason="bubblewrap and prlimit required (the library is built confined)")


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_real_message_instance_answers_from_the_library_on_the_record(tmp_path, monkeypatch):
    from tests.memory.test_librarian import LIBRARIAN_SECOND, TOTAL, _copy, code
    from tests.scripted_model import ScriptedModel, reply, scripted
    from unify.memory.writer_pass import unillm_turn
    # rev 2: the run record is always on (no run_record_on switch to patch)
    home, rec = _seeded(tmp_path)
    copy = _copy(tmp_path)
    run = SimpleNamespace(copy_root=copy, library_mode=True, paths=SimpleNamespace(home=home), episode_id="ep-a",
                          pin="c0ffee", librarian_sessions=[])
    monkeypatch.setattr(R, "current", lambda: run)
    open_session = L.opener(turn_factory=lambda: unillm_turn("openai/gpt-5.6-sol@openrouter", "low", origin=L.ORIGIN))
    asked = rr.RunRecordView(rec, "actor-aaaa", "actor").post("actor-aaaa", "@librarian which items add up a table?")
    model = ScriptedModel(librarian=[reply(calls=[code(LIBRARIAN_SECOND)])])  # the librarian's own kind
    with scripted(model):
        assert await R._role_round(_record(home), home, open_session) == 1
    (rep,) = [e for e in rr.read_entries(home) if e.get("reply_to") == asked["seq"]]
    assert rep["author"].startswith("librarian-") and rep["mentions"] == ["actor-aaaa"]
    assert json.loads(rep["text"].split(": ", 1)[1])["fits"][0]["id"] == TOTAL
    (session,) = run.librarian_sessions
    assert session.mode == "message" and session.closed and not session.box_alive()
    assert rec._row(rep["author"])["state"] == "ended" and session.episode is None
    assert "which items add up a table?" in model.calls[0].messages[-1]["content"]
    assert [r["key"] for r in rr.role_answers(home, "librarian")] == list(session.answers)


def test_a_request_opens_at_most_its_bound_of_ask_sessions(tmp_path, monkeypatch):
    """The actor and its teammates each keep their own ``memory.ask`` session; one request opens at most
    MAX_ASK_SESSIONS of them, so the librarian's spend per request is bounded; a later one is refused, naming where to
    ask instead."""
    from tests.memory.test_librarian import _copy

    run = SimpleNamespace(copy_root=_copy(tmp_path), library_mode=True, paths=SimpleNamespace(home=tmp_path / "home"),
                          episode_id="ep-a", pin="c0ffee",
                          librarian_sessions=[SimpleNamespace(mode="ask") for _ in range(L.MAX_ASK_SESSIONS)])  # fmt: skip
    monkeypatch.setattr(R, "current", lambda: run)
    with pytest.raises(L.Unavailable, match="its bound; send the question to the actor"):
        asyncio.run(L.opener()())
    assert len(run.librarian_sessions) == L.MAX_ASK_SESSIONS
