"""The redesign's conformance suite: three synthetic surfaces that name no world,
each run end to end through the CLI with ``UNIFY_MEMORY_PROFILE=redesign``, the writer service in its own process and
every memory lane scripted (:mod:`.fake_lanes`, :mod:`.scripts`):

1. a conversation with follow-up messages and no reward;
2. a tool API with read and write calls and no returned value, with an outcome at the episode's end;
3. a request whose outcome arrives two requests later.

Every surface ends as a kit ends a cell (the drain, verified) and is checked (:func:`.harness.check_run`): no world
vocabulary in what the memory agents were sent, the coverage guarantee, metering sums (A3), verified termination
(A2). Each asserts the acceptance lines of the elements its surface exercises:

* A1 (surface 2): a fork -> writer message, a writer -> fork question answered, an actor -> writer report, a writer ->
  helper delegation and a helper -> writer reply, all on one record;
* A2 (all): one service, its passes, ``Terminated.verified`` and no process left in its group;
* A3 (all): the entries' shares plus the unread zeros equal the passes' known USD exactly; ids on every memory call
  (``X-Unify-Pass-Id`` and ``-Entries`` on the writer's, ``-Episode`` on the fork's);
* traced captures (surface 2): the capture record's form, trace, value, own check and outcome from signals;
* the attention ledger (all): every entry closed, unreadable or not reached; a late signal makes ``late_evidence`` (surface 3);
* labelled items (surface 2): a landed item is served;
* one fork per episode (surface 1): a follow-up message makes no entry of its own; the episode's end makes one;
* serving on evidence (surfaces 2, 3): served items carry their evidence; an item the writer retires in a pass leaves every
  listing and stays on record, retired by the writer's decision, its file still in the library;
* inferred regimes (surfaces 1, 2, 3): each episode's regime is inferred from its own signals (``none`` with none, ``dense``
  with a verdict that names the answer it judged, changed and logged by a late one), never defaulted; a hidden grader
  decides nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401 (fixtures)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory.conformance import fake_lanes as FL
from tests.memory.conformance import harness as H
from tests.memory.conformance import scripts as SX
# the writer's lane refuses while unillm records OTel spans, which the test runner turns on for every session; the
# fixture takes UNILLM_OTEL out of the environment, so the service and the fork worker start with it off too
from tests.memory.test_writer_route import _otel_off_unless_a_test_turns_it_on  # noqa: F401 - an autouse fixture
from tests.scripted_model import ScriptedModel, cell, reply
from tests.memory.arrivals import read_arrivals, regime_changes
from tests.memory import store_reads

ID16 = re.compile(r"[0-9a-f]{16}")


def _fake_embed(texts, on_call=None):
    """Unit vectors of hashed lower-case words (texts that share words are close): no provider is called."""
    import hashlib

    import numpy as np

    rows = []
    for text in texts:
        vector = np.zeros(512, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", str(text).lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 512] += 1.0
        if not vector.any():
            vector[0] = 1.0
        rows.append(vector / np.linalg.norm(vector))
    return np.stack(rows) if rows else np.zeros((0, 512), dtype=np.float32)


@pytest.fixture
def lanes(monkeypatch, core_world):  # noqa: F811
    import unify.common.embeddings as embeddings

    lanes = FL.Lanes()
    server, port = FL.serve(lanes)
    H.apply_settings(monkeypatch, H.redesign_env(port))
    H.settle_lanes(monkeypatch, port)
    monkeypatch.setattr(embeddings, "embed", _fake_embed)
    try:
        yield lanes
    finally:
        server.shutdown()


def _forever(lanes: FL.Lanes, lane: str, fn) -> None:
    lanes.script(lane, *([fn] * 300))


def _stores(home: Path):
    from unify.memory.integration.consolidate import open_stores
    from unify.memory.integration.paths import Paths

    return open_stores(Paths.under(home))


def _served(home: Path, seen: list, text: str = "Noted."):
    """An actor turn that records what the request's library copy serves (items, withdrawn items, evidence)."""

    def turn(call):
        from unify.memory.integration.paths import Paths

        copy = Paths.under(home).checkout
        copy = copy / "current" if (copy / "current").exists() else copy  # in-episode memory: the pinned copy
        meta = copy / ".memory"
        seen.append({
            "items": sorted(json.loads((meta / "items.json").read_text())["items"]),
            "withdrawn": json.loads((meta / "withdrawn_items.json").read_text())
            if (meta / "withdrawn_items.json").is_file() else None,
            "evidence": json.loads((meta / "item_evidence.json").read_text())
            if (meta / "item_evidence.json").is_file() else None,
        })
        return reply(text)

    return turn


def _check_ids_on_calls(lanes: FL.Lanes, *, writer_ran: bool = True) -> None:
    """A3: every writer call carries its pass and entries; every fork call its episode.
    *writer_ran* False: the surface left no capture to write up, so no writer call was made at all."""
    writer = lanes.of("writer")
    if not writer_ran:
        assert not writer, [c.meta for c in writer][:3]
    assert (writer or not writer_ran) and all(ID16.fullmatch(c.meta.get("x-unify-pass-id", "")) for c in writer), \
        [c.meta for c in writer][:3]
    assert all(all(ID16.fullmatch(e) for e in c.meta.get("x-unify-entries", "").split(".")) for c in writer)
    forks = lanes.of("fork")
    assert forks and all(c.meta.get("x-unify-episode") for c in forks), [c.meta for c in forks][:3]


def _entries(home: Path) -> list:
    return list(H.ledger(home).entries().values())


def check_mesh(home: Path) -> dict:
    """A1's acceptance on the run record: who said what to whom, by role (counts)."""
    from unify.agents import run_record as rr

    rows = rr.read_entries(home)
    roles = {r["author"]: r.get("role") for r in rows}

    def to(role_from: str, role_to: str) -> list:
        return [r for r in rows if r.get("role") == role_from
                and any(m == role_to or roles.get(m) == role_to for m in r.get("mentions") or [])]

    fork_to_writer = to("fork", "writer")
    return {"fork_to_writer": len(fork_to_writer), "writer_to_fork": len(to("writer", "fork")),
            "fork_answer": sum(1 for r in fork_to_writer if SX.FORK_ANSWER[len("@writer "):] in r["text"]),
            "actor_to_writer": len(to("actor", "writer")), "writer_to_helper": len(to("writer", "helper")),
            "helper_to_writer": len(to("helper", "writer"))}


# --- surface 1 -------------------------------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_surface_1_a_conversation_with_follow_ups_and_no_reward(core_world, monkeypatch, lanes):  # noqa: F811
    home = Path(core_world["state"])
    _forever(lanes, "fork", SX.FORK_REPLY)
    _forever(lanes, "writer", SX.writer_policy(land=False))
    first = "Plan a weekly reading schedule for the novels on my list."
    follow = ["Move the longest one to the end.", "Thanks, that works."]
    model = ScriptedModel(actor=[
        reply(calls=[cell("schedule = {'Mon': 'A', 'Tue': 'B'}\nprint(schedule)")]),
        reply("Here is a plan: Monday A, Tuesday B."),
        reply("Moved: the longest is now on Sunday."),
        reply("Glad it works."),
    ])
    s = await H.cli_session(core_world, monkeypatch, model, first,
                            H.replies([{"message": follow[0]}], [{"message": follow[1]}]))
    assert s.code == 0, s.lines[-5:]
    assert s.episode_id
    H.wait_until_settled(home)
    facts = H.check_run(home, lanes, actor_model=model, own_texts=[first, *follow])
    print("CONFORMANCE", json.dumps(H.run_facts(home, lanes, facts), default=str, sort_keys=True)[:3000])
    # nothing arrived for the episode, so its regime is none, by that rule (never a default)
    assert store_reads.regime_record(_stores(home).evidence, s.episode_id) == ("none", "nothing")
    # the episode's end forks once, and its experience is one entry; a follow-up message makes none of its own
    entries = _entries(home)
    ends = [e for e in entries if e.kind == "new_experience"]
    assert len(ends) == 1 and ends[0].refs.episodes == (s.episode_id,), [(e.kind, e.refs.row()) for e in entries]
    # a conversation leaves no procedure: the harness closes each entry, and no writer pass runs on nothing
    assert all(e.state == "closed" and e.decision == "nothing_to_keep" for e in entries)
    assert all(e.reason_ref == "no_capture" for e in entries)
    _check_ids_on_calls(lanes, writer_ran=False)


# --- surface 2 -------------------------------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_surface_2_a_tool_api_with_reads_and_writes_and_an_outcome_at_the_end(core_world, monkeypatch, lanes):  # noqa: F811
    from unify.memory import captures as C
    from unify.memory.integration.paths import Paths

    home = Path(core_world["state"])
    paths = Paths.under(home)
    (Path(core_world["workspace"]) / "inbox.csv").write_text(SX.INBOX)
    _forever(lanes, "fork", SX.fork_mesh)
    _forever(lanes, "writer", SX.writer_mesh)
    _forever(lanes, "helper", SX.helper_mesh)
    first = "Sort the rows of inbox.csv by id into sorted.csv."
    shown = "The check passed: every row is in place."
    code = ("import csv\n"
            "rows = list(csv.DictReader(open('inbox.csv')))\n"
            "rows.sort(key=lambda r: int(r['id']))\n"
            "with open('sorted.csv', 'w', newline='') as fh:\n"
            "    w = csv.DictWriter(fh, fieldnames=['id', 'amount'])\n"
            "    w.writeheader(); w.writerows(rows)\n"
            "print(len(rows), 'rows')\n"
            f"print(tools.send({SX.ACTOR_REPORT!r}))\n")
    model = ScriptedModel(actor=[reply(calls=[cell(code)]), reply("Done: sorted.csv holds 3 rows."), reply("Good.")])
    # the outcome at the episode's end as the actor sees it (a verdict on the episode as a whole, then the message
    # that shows it), and the hidden grader's, which decides nothing
    host = H.replies([{"checker": {"label": "pass"}}, {"message": shown}],
                     [{"outcome": SX.OUTCOME_SOLVED}, {"quit": True}])
    s = await H.cli_session(core_world, monkeypatch, model, first, host)
    assert s.code == 0, s.lines[-5:]
    H.wait_until_settled(home)
    # one capture alone is never written up: the same kind of work again gives it a related capture, and the pass
    again = f"Sort the rows of inbox.csv by id into {SX.SECOND_FILE}."
    again_code = code.split("print(tools.send(")[0].replace("sorted.csv", SX.SECOND_FILE)
    s1b = await H.cli_session(core_world, monkeypatch,
                              ScriptedModel(actor=[reply(calls=[cell(again_code)]), reply("Done."), reply("Good.")]),
                              again, H.replies())
    assert s1b.code == 0, s1b.lines[-5:]
    H.wait_until_settled(home)
    seen: list = []
    s2 = await H.cli_session(core_world, monkeypatch, ScriptedModel(actor=[_served(home, seen)]),
                             "Anything new in the inbox?", H.replies())
    assert s2.code == 0
    H.wait_until_settled(home)
    facts = H.check_run(home, lanes, actor_model=model, own_texts=[first, shown, again, "Anything new in the inbox?"])
    print("CONFORMANCE", json.dumps(H.run_facts(home, lanes, facts), default=str, sort_keys=True)[:3000])
    stores = _stores(home)
    # one verdict the actor saw, right after its answer: rev 2 records the answers as dialogue actions on the env
    # channel (the redesign arms' UNIFY_MEMORY_DIALOGUE=env, now fixed), so the pairing is certain and the verdict
    # names the action it judged, which makes the regime dense, from its signals; the hidden one is posted unseen
    # and decides nothing
    assert store_reads.regime_record(stores.evidence, s.episode_id) == ("dense", "signals")
    sigs = stores.evidence.signals_for(s.episode_id)
    assert {(x.visible_to_actor, x.label, x.timing) for x in sigs} == {(True, "pass", "in_episode"),
                                                                     (False, "pass", "in_episode")}
    [shown_verdict] = [x for x in sigs if x.visible_to_actor]
    assert str(shown_verdict.refers_to).startswith(f"{s.episode_id}/actions/"), shown_verdict
    # the capture records what the episode did, with the value it handed back (rev 2 records each answered reply
    # as a dialogue action, so the answer is the handed-back value), no own check (rev 2: a capture is never run to
    # classify it), and its outcome from the signals only
    caps = [d.get("record") or {} for d in C.load(C.store_dir(paths.state_dir)).values()]
    case = next(c for r in caps for c in r.get("cases") or [] if c.get("episode") == s.episode_id)
    assert case["form"] == "state_env" and case["value"]["present"] is True and ID16.fullmatch(case["value"]["digest"])
    assert case["own_check"] == {"ran": False, "why": "a capture is never run to classify it"}
    # the file's read and write, and the answer, a dialogue action whose effect is not known
    assert case["effects_seen"] == {"read": 1, "write": 1, "unknown": 1}
    # its trace names the two file actions and the answer (action 2)
    assert case["outcome"] == "positive" and "action:0..2" in case["trace"]["parts"]
    # the note the writer landed from the episode is served to the next request, with evidence served
    assert seen and SX.NOTE_ID in seen[0]["items"] and seen[0]["evidence"] is not None
    landed = [p for p in H._jsonl(paths.state_dir / "pass-results.jsonl") if (p.get("items") or {}).get("landed")]
    assert landed and SX.NOTE_ID in landed[0]["items"]["landed"]
    # the pass that landed it read the two related captures together
    groups = [r for r in H._jsonl(paths.state_dir / "memory-attention" / "ledger.jsonl") if r.get("op") == "group"]
    assert groups and all(len(r["members"]) + len(r["context"]) >= 2 for r in groups), groups
    assert groups[0]["reasons"]["sequence"] >= 1, groups
    # A1: the five exchanges, on one record
    mesh = check_mesh(home)
    assert mesh["fork_to_writer"] >= 1 and mesh["writer_to_fork"] >= 1 and mesh["fork_answer"] == 1, mesh
    assert mesh["actor_to_writer"] == 1 and mesh["writer_to_helper"] >= 1 and mesh["helper_to_writer"] >= 1, mesh
    _check_ids_on_calls(lanes)


# --- surface 3 -------------------------------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1200)
async def test_surface_3_an_outcome_that_arrives_two_requests_later(core_world, monkeypatch, lanes):  # noqa: F811
    from unify.memory.integration.paths import Paths

    home = Path(core_world["state"])
    paths = Paths.under(home)
    (Path(core_world["workspace"]) / "inbox.csv").write_text(SX.INBOX)
    _forever(lanes, "fork", SX.fork_sorting())
    _forever(lanes, "writer", SX.writer_policy())
    code = ("import csv\n"
            "rows = sorted(csv.DictReader(open('inbox.csv')), key=lambda r: int(r['id']))\n"
            "w = csv.DictWriter(open('sorted.csv', 'w', newline=''), fieldnames=['id', 'amount'])\n"
            "w.writeheader(); w.writerows(rows)\nprint(len(rows))\n")
    texts = ["Sort the rows of inbox.csv by id into sorted.csv.", "How many rows were there?", "Thanks.",
             "Anything else?", f"Sort the rows of inbox.csv by id into {SX.SECOND_FILE}."]
    # request 0: the same kind of work earlier, so request 1's capture has a related capture to be written up with
    z = await H.cli_session(core_world, monkeypatch,
                            ScriptedModel(actor=[reply(calls=[cell(code.replace("sorted.csv", SX.SECOND_FILE))]),
                                                 reply("Sorted.")]),
                            texts[4], H.replies())
    assert z.code == 0
    H.wait_until_settled(home)
    # request 1: the work, and no verdict
    a = await H.cli_session(core_world, monkeypatch, ScriptedModel(actor=[reply(calls=[cell(code)]), reply("Sorted.")]),
                            texts[0], H.replies())
    assert a.code == 0 and a.episode_id
    H.wait_until_settled(home)
    assert store_reads.regime_record(_stores(home).evidence, a.episode_id) == ("none", "nothing")
    # request 2: other work
    b = await H.cli_session(core_world, monkeypatch, ScriptedModel(actor=[reply("Three.")]), texts[1], H.replies())
    assert b.code == 0
    H.wait_until_settled(home)
    # request 3: the verdict on request 1's episode arrives, named by the id the CLI announced
    late = {"checker": {"label": "fail"}, "episode": a.episode_id}
    c = await H.cli_session(core_world, monkeypatch, ScriptedModel(actor=[reply("You're welcome.")]), texts[2],
                            H.replies([late, {"quit": True}]))
    assert c.code == 0
    assert any(x.get("type") == "checker" and x.get("accepted") and x.get("episode") == a.episode_id for x in c.lines)
    H.wait_until_settled(home)
    # request 4: what the library serves now, and what the actor's own listing shows
    seen: list = []
    listed: list = []

    def after_listing(call):
        listed.append(json.dumps([m.get("content") for m in call.messages if m.get("role") == "tool"][-1:]))
        return _served(home, seen)(call)

    d = await H.cli_session(core_world, monkeypatch,
                            ScriptedModel(actor=[reply(calls=[cell("print(sorted(i.id for i in memory.items()))")]),
                                                 after_listing]), texts[3], H.replies())
    assert d.code == 0
    H.wait_until_settled(home)
    facts = H.check_run(home, lanes, own_texts=texts)
    print("CONFORMANCE", json.dumps(H.run_facts(home, lanes, facts), default=str, sort_keys=True)[:3000])
    stores = _stores(home)
    # the arrival, after the episode, and the regime change it caused, logged with the signal's id
    (arrival,) = [r for r in read_arrivals(paths.state_dir) if r["episode_id"] == a.episode_id]
    assert arrival["timing"] == "after_episode" and arrival["label"] == "fail" and arrival["visible_to_actor"] is True
    assert (arrival["regime_before"], arrival["regime_after"]) == ("none", "sparse")
    changes = regime_changes(stores.evidence, a.episode_id)
    assert [(x["before"], x["after"]) for x in changes] == [(None, "none"), ("none", "sparse")]
    assert changes[-1]["signal_id"] == arrival["signal_id"]
    # the late signal reopened what the episode used or produced, as one late_evidence entry, decided
    (le,) = [e for e in _entries(home) if e.kind == "late_evidence"]
    assert a.episode_id in le.refs.episodes and SX.NOTE_ID in le.refs.items and arrival["signal_id"] in le.refs.signals
    assert le.state == "closed" and le.decision == "retire"
    # retired through the writer's normal pass, the item stays on record and is never deleted: its file is still in the
    # library's main, and main's records mark it retired by the writer's decision
    from unify.memory.item_records import read_records

    head = stores.memory.head()
    assert stores.memory.show(head, SX.NOTE_ID).strip(), head
    record = (read_records(stores.memory, head) or {}).get(SX.NOTE_ID) or {}
    assert (record.get("status"), record.get("status_rule")) == ("retired", "writer"), record
    # the item the writer retired is withdrawn: retired in the copy's records, named in its withdrawn file with
    # the decision, and gone from the actor's listing
    assert seen and SX.NOTE_ID in (seen[0]["withdrawn"] or {}).get("items", {}), seen
    assert seen[0]["withdrawn"]["items"][SX.NOTE_ID]["status"] == "retired", seen
    assert listed and "sort-rows" not in listed[0] and "[" in listed[0], listed
    _check_ids_on_calls(lanes)
