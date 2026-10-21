"""The redesign's conformance suite: four synthetic surfaces that name no world,
each run end to end through the CLI with ``UNIFY_MEMORY_PROFILE=redesign``, the writer service in its own process and
every memory lane scripted (:mod:`.fake_lanes`, :mod:`.scripts`):

1. a conversation with follow-up messages and no reward;
2. a tool API with read and write calls and no returned value, with an outcome at the episode's end;
3. one long episode with checkpoints (a compaction and an actor call);
4. a request whose outcome arrives two requests later.

Every surface ends as a kit ends a cell (the drain, verified) and is checked (:func:`.harness.check_run`): no world
vocabulary in what the memory agents were sent, the coverage guarantee, metering sums (A3), verified termination
(A2). Each asserts the acceptance lines of the elements its surface exercises:

* A1 (surface 2): a fork -> writer message, a writer -> fork question answered, an actor -> writer report, a writer ->
  helper delegation and a helper -> writer reply, all on one record;
* A2 (all): one service, its passes, ``Terminated.verified`` and no process left in its group;
* A3 (all): the entries' shares plus the unread zeros equal the passes' known USD exactly; ids on every memory call
  (``X-Unify-Pass-Id`` and ``-Entries`` on the writer's, ``-Episode`` / ``-Segment`` on the fork's);
* traced captures (surface 2): the capture record's form, trace, value, own check and outcome from signals;
* the attention ledger (all): every entry closed, unreadable or not reached; a late signal makes ``late_evidence`` (surface 4);
* labelled items (surface 2): a landed item is served;
* in-episode memory (surfaces 1, 3, which set the in-episode arm's switch on): every pause (a new message, an actor call,
  a compaction) starts a segment fork, whose experience is an entry, or an occurrence of the entry with the same
  capture; a commit landed while the episode runs is served from the next cell with one notice;
* serving on evidence (surfaces 2, 4): served items carry their evidence; an item the writer retires leaves every listing;
* inferred regimes (surfaces 1, 2, 4): each episode's regime is inferred from its own signals (``none`` with none, ``sparse`` with
  a verdict on the whole episode, changed and logged by a late one), never defaulted; a hidden grader decides nothing.
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
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.memory.arrivals import read_arrivals

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


def _check_ids_on_calls(lanes: FL.Lanes) -> None:
    """A3: every writer call carries its pass and entries; every fork call its episode (and a segment's, its number)."""
    writer = lanes.of("writer")
    assert writer and all(ID16.fullmatch(c.meta.get("x-unify-pass-id", "")) for c in writer), \
        [c.meta for c in writer][:3]
    assert all(all(ID16.fullmatch(e) for e in c.meta.get("x-unify-entries", "").split(".")) for c in writer)
    forks = lanes.of("fork")
    assert forks and all(c.meta.get("x-unify-episode") for c in forks), [c.meta for c in forks][:3]
    for c in forks:
        if SX.is_segment(c.body):
            assert re.fullmatch(r"[1-9][0-9]*", c.meta.get("x-unify-segment", "")), c.meta


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
    H.apply_settings(monkeypatch, H.in_episode_env())
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
    assert _stores(home).evidence.regime_record(s.episode_id) == ("none", "nothing")
    # the follow-up message after a cell is a pause: a segment fork and its entry; the episode's end, another
    entries = _entries(home)
    segs = sorted(e.refs.segment for e in entries if e.kind == "new_experience" and e.refs.segment)
    ends = [e for e in entries if e.kind == "new_experience" and not e.refs.segment]
    assert segs == [1] and len(ends) == 1, [(e.kind, e.refs.row()) for e in entries]
    assert all(e.state == "closed" and e.decision == "nothing_to_keep" for e in entries)
    _check_ids_on_calls(lanes)


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
    seen: list = []
    s2 = await H.cli_session(core_world, monkeypatch, ScriptedModel(actor=[_served(home, seen)]),
                             "Anything new in the inbox?", H.replies())
    assert s2.code == 0
    H.wait_until_settled(home)
    facts = H.check_run(home, lanes, actor_model=model, own_texts=[first, shown, "Anything new in the inbox?"])
    print("CONFORMANCE", json.dumps(H.run_facts(home, lanes, facts), default=str, sort_keys=True)[:3000])
    stores = _stores(home)
    # one verdict the actor saw, on the episode as a whole: sparse, from its signals; the hidden one is posted
    # unseen and decides nothing
    assert stores.evidence.regime_record(s.episode_id) == ("sparse", "signals")
    sigs = stores.evidence.signals_for(s.episode_id)
    assert {(x.visible_to_actor, x.label, x.timing) for x in sigs} == {(True, "pass", "in_episode"),
                                                                     (False, "pass", "in_episode")}
    # the capture records what the episode did, with no value (none was handed back), no own check (rev 2: a
    # capture is never run to classify it), and its outcome from the signals only
    caps = [d.get("record") or {} for d in C.load(C.store_dir(paths.state_dir)).values()]
    case = next(c for r in caps for c in r.get("cases") or [] if c.get("episode") == s.episode_id)
    assert case["form"] == "state_env" and case["value"] == {"present": False}
    assert case["own_check"] == {"ran": False, "why": "a capture is never run to classify it"}
    assert case["effects_seen"] == {"read": 1, "write": 1, "unknown": 0}
    assert case["outcome"] == "positive" and "action:0..1" in case["trace"]["parts"]
    # the note the writer landed from the episode is served to the next request, with evidence served
    assert seen and SX.NOTE_ID in seen[0]["items"] and seen[0]["evidence"] is not None
    landed = [p for p in H._jsonl(paths.state_dir / "pass-results.jsonl") if (p.get("items") or {}).get("landed")]
    assert landed and SX.NOTE_ID in landed[0]["items"]["landed"]
    # A1: the five exchanges, on one record
    mesh = check_mesh(home)
    assert mesh["fork_to_writer"] >= 1 and mesh["writer_to_fork"] >= 1 and mesh["fork_answer"] == 1, mesh
    assert mesh["actor_to_writer"] == 1 and mesh["writer_to_helper"] >= 1 and mesh["helper_to_writer"] >= 1, mesh
    _check_ids_on_calls(lanes)


# --- surface 3 -------------------------------------------------------------------------------------------------------


async def _wait_for_item_served(home: Path, item: str, timeout_s: float = 300.0) -> str | None:
    """Wait (without blocking the request's event loop) until the commit requests now pin (``served_head``: what the
    next cell re-pins to) holds *item* (its path in the memory); that commit, or None when it was not served in time.
    The service lands on main and publishes ``served`` after it, and under load it can group the segment entries into
    fewer passes, so neither main moving nor the item on main is enough: the wait is for the item where cells read it."""
    import asyncio
    import time

    from unify.memory.gitio import Repo
    from unify.memory.integration.paths import Paths
    from unify.memory.memory_writer import served_head

    repo = Repo(Paths.under(home).memory)
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        head = served_head(repo)
        if repo.run("ls-tree", "--name-only", head, "--", item).strip() == item:
            return head
        await asyncio.sleep(0.25)
    return None


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1200)
async def test_surface_3_a_long_episode_with_checkpoints(core_world, monkeypatch, lanes):  # noqa: F811
    from unify.memory.gitio import Repo
    from unify.memory.integration.paths import Paths

    H.apply_settings(monkeypatch, H.in_episode_env())
    home = Path(core_world["state"])
    paths = Paths.under(home)
    (Path(core_world["workspace"]) / "inbox.csv").write_text(SX.INBOX)
    _forever(lanes, "fork", SX.fork_by_request(SX.TOOL_FORK_REPLY, when="inbox.csv", segment_reply=SX.TOOL_FORK_REPLY))
    _forever(lanes, "writer", SX.writer_policy())
    sort_code = ("import csv\n"
                 "rows = sorted(csv.DictReader(open('inbox.csv')), key=lambda r: int(r['id']))\n"
                 "w = csv.DictWriter(open('sorted.csv', 'w', newline=''), fieldnames=['id', 'amount'])\n"
                 "w.writeheader(); w.writerows(rows)\nprint(len(rows))\n")
    start: dict = {}

    def first_cell(call):
        start["head"] = Repo(paths.memory).run("rev-parse", "main").strip()
        return reply(calls=[cell(sort_code)])

    async def after_pause(call):
        # the pause's segment fork runs now; the service lands from it while the episode goes on
        start["landed"] = await _wait_for_item_served(home, SX.NOTE_ID)
        return reply(calls=[cell("total = sum(float(r['amount']) for r in rows)\nprint(total)")])

    model = ScriptedModel(
        actor=[
            first_cell,
            reply(calls=[cell("print(tools.checkpoint(note='rows sorted'))")]),
            after_pause,
            # a context-full turn: the loop compacts before the next call, a pause of its own
            reply(calls=[cell("print('totals done')")], prompt_tokens=900_000),
            reply(calls=[cell("print('checked')")]),
            reply("Sorted, totalled and checked."),
        ],
        compression_fork=Always(reply(SX.SUMMARY)),
        compress_turn=Always(reply(calls=[("compress_context", {})])),
    )
    first = "Sort the rows of inbox.csv by id, then total the amounts."
    s = await H.cli_session(core_world, monkeypatch, model, first, H.replies())
    assert s.code == 0, s.lines[-5:]
    H.wait_until_settled(home)
    facts = H.check_run(home, lanes, actor_model=model, own_texts=[first, SX.SUMMARY])
    print("CONFORMANCE", json.dumps(H.run_facts(home, lanes, facts), default=str, sort_keys=True)[:3000])
    assert "compression_fork" in model.kinds()
    # the actor's call and the compaction each started a segment fork; the compaction's fork captured the same
    # procedure as the actor call's, so its experience joins that entry (one entry per capture: a ref while the
    # entry is open, an occurrence once it is closed)
    from unify.memory.segments import SEGMENTS_DIR

    entries = _entries(home)
    checkpoints = [
        c for c in H._jsonl(paths.state_dir / SEGMENTS_DIR / "checkpoints.jsonl") if c.get("type") == "checkpoint"
    ]
    started = [(c.get("segment"), c.get("sources")) for c in checkpoints if c.get("fork") == "started"]
    assert started[:2] == [(1, ["actor_call"]), (2, ["compaction"])], checkpoints
    first = [e for e in entries if e.kind == "new_experience" and e.refs.segment == 1]
    assert len(first) == 1 and first[0].sources, [(e.kind, e.refs.row(), e.sources) for e in entries]
    # what the service landed from the first segment while the episode ran is served from the next cell, with
    # one notice naming exactly what changed
    assert start.get("landed") and start["landed"] != start["head"], "the note was not served while the episode ran"
    notices = [m for c in model.of("actor") for m in c.messages
               if m.get("role") == "tool" and "memory changed" in json.dumps(m.get("content"))]
    assert notices and SX.NOTE_ID in json.dumps(notices[0].get("content")), notices[:1]
    _check_ids_on_calls(lanes)


# --- surface 4 -------------------------------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1200)
async def test_surface_4_an_outcome_that_arrives_two_requests_later(core_world, monkeypatch, lanes):  # noqa: F811
    from unify.memory.integration.paths import Paths

    home = Path(core_world["state"])
    paths = Paths.under(home)
    (Path(core_world["workspace"]) / "inbox.csv").write_text(SX.INBOX)
    _forever(lanes, "fork", SX.fork_by_request(SX.TOOL_FORK_REPLY, when="inbox.csv"))
    _forever(lanes, "writer", SX.writer_policy())
    code = ("import csv\n"
            "rows = sorted(csv.DictReader(open('inbox.csv')), key=lambda r: int(r['id']))\n"
            "w = csv.DictWriter(open('sorted.csv', 'w', newline=''), fieldnames=['id', 'amount'])\n"
            "w.writeheader(); w.writerows(rows)\nprint(len(rows))\n")
    texts = ["Sort the rows of inbox.csv by id into sorted.csv.", "How many rows were there?", "Thanks.",
             "Anything else?"]
    # request 1: the work, and no verdict
    a = await H.cli_session(core_world, monkeypatch, ScriptedModel(actor=[reply(calls=[cell(code)]), reply("Sorted.")]),
                            texts[0], H.replies())
    assert a.code == 0 and a.episode_id
    H.wait_until_settled(home)
    assert _stores(home).evidence.regime_record(a.episode_id) == ("none", "nothing")
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
    changes = stores.evidence.regime_changes(a.episode_id)
    assert [(x["before"], x["after"]) for x in changes] == [(None, "none"), ("none", "sparse")]
    assert changes[-1]["signal_id"] == arrival["signal_id"]
    # the late signal reopened what the episode used or produced, as one late_evidence entry, decided
    (le,) = [e for e in _entries(home) if e.kind == "late_evidence"]
    assert a.episode_id in le.refs.episodes and SX.NOTE_ID in le.refs.items and arrival["signal_id"] in le.refs.signals
    assert le.state == "closed" and le.decision == "retire"
    # the item the writer retired is withdrawn: retired in the copy's records, named in its withdrawn file with
    # the decision, and gone from the actor's listing
    assert seen and SX.NOTE_ID in (seen[0]["withdrawn"] or {}).get("items", {}), seen
    assert seen[0]["withdrawn"]["items"][SX.NOTE_ID]["status"] == "retired", seen
    assert listed and "sort-rows" not in listed[0] and "[" in listed[0], listed
    _check_ids_on_calls(lanes)


# --- frozen memory (held-out evaluation) -----------------------------------------------------------------------------


def _memory_fingerprint(home: Path) -> dict:
    """What serving reads, as it stands: the memory repo's and the episodes repo's refs, every row of every evidence
    table, the capture store's files, the run's pass results and the attention ledger's entries."""
    import hashlib
    import sqlite3

    from unify.memory.gitio import Repo
    from unify.memory.integration.paths import Paths

    paths = Paths.under(home)
    db = sqlite3.connect(f"file:{paths.evidence}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        rows = {t: sorted(map(repr, db.execute(f"SELECT * FROM {t}").fetchall())) for t in tables}
    finally:
        db.close()
    caps = paths.captures
    files = {str(p.relative_to(caps)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(caps.rglob("*")) if p.is_file()} if caps.is_dir() else {}
    return {
        "memory_refs": Repo(paths.memory).run("for-each-ref").strip(),
        "episodes_refs": Repo(paths.episodes).run("for-each-ref").strip(),
        "evidence": rows,
        "captures": files,
        "pass_results": H._jsonl(paths.state_dir / "pass-results.jsonl"),
        "entries": sorted(H.ledger(home).entries()),
    }


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1200)
async def test_frozen_memory_serves_what_it_holds_and_changes_nothing(core_world, monkeypatch, lanes):  # noqa: F811
    """The held-out mode (``UNIFY_MEMORY_FROZEN=on``): after a request that lands a note, the kit stops the service
    (the drain) and freezes memory. Two frozen requests are served the same library and evidence, start no writer,
    fork or service, record nothing (their uses and verdicts reach no store), and a message to the writer is
    answered "not available: memory is frozen"."""
    from unify.memory import writer_service as ws
    from unify.memory.integration.paths import Paths

    home = Path(core_world["state"])
    paths = Paths.under(home)
    (Path(core_world["workspace"]) / "inbox.csv").write_text(SX.INBOX)
    _forever(lanes, "fork", SX.fork_by_request(SX.TOOL_FORK_REPLY, when="inbox.csv"))
    _forever(lanes, "writer", SX.writer_policy())
    code = ("import csv\n"
            "rows = sorted(csv.DictReader(open('inbox.csv')), key=lambda r: int(r['id']))\n"
            "w = csv.DictWriter(open('sorted.csv', 'w', newline=''), fieldnames=['id', 'amount'])\n"
            "w.writeheader(); w.writerows(rows)\nprint(len(rows))\n")
    texts = ["Sort the rows of inbox.csv by id into sorted.csv.", "Sort the rows of inbox.csv again.",
             "And once more, please."]
    a = await H.cli_session(core_world, monkeypatch, ScriptedModel(actor=[reply(calls=[cell(code)]), reply("Sorted.")]),
                            texts[0], H.replies())
    assert a.code == 0 and a.episode_id
    H.wait_until_settled(home)
    # the kit's freeze: the service stopped (verified), then the setting on for every later request
    H.check_terminated(H.drain(home), home)
    before = _memory_fingerprint(home)
    assert SX.NOTE_ID in json.dumps(before["pass_results"]), "the training request landed nothing"
    calls_before = {lane: len(lanes.of(lane)) for lane in ("fork", "writer", "helper")}
    events_before = len(ws.events(home))
    H.apply_settings(monkeypatch, {"UNIFY_MEMORY_FROZEN": "on"})
    seen: list = []
    said: list = []

    def frozen_turn(call):
        return reply(calls=[cell(code + "for step in (lambda: tools.send('@writer the ids are text here as well'),\n"
                                 "             lambda: tools.checkpoint(note='sorted')):\n"
                                 "    try:\n"
                                 "        print(step())\n"
                                 "    except Exception as exc:\n"
                                 "        print(type(exc).__name__, exc)\n")])

    def after(call):
        said.append(json.dumps([m.get("content") for m in call.messages if m.get("role") == "tool"][-1:]))
        return _served(home, seen, "Sorted again.")(call)

    sessions = []
    for text in texts[1:]:
        s = await H.cli_session(core_world, monkeypatch, ScriptedModel(actor=[frozen_turn, after]), text,
                                H.replies([{"checker": {"label": "fail"}}, {"quit": True}]))
        assert s.code == 0, s.lines[-5:]
        sessions.append(s)
    after_fp = _memory_fingerprint(home)
    # nothing that serving reads changed: no episode, signal, use, capture, pass or entry was added
    assert after_fp == before, {k: (before[k] == after_fp[k]) for k in before}
    # both frozen requests were served the same library and evidence, the landed note among it
    assert len(seen) == 2 and seen[0] == seen[1] and SX.NOTE_ID in seen[0]["items"], seen
    # no writer, fork or helper call, and no service started
    assert {lane: len(lanes.of(lane)) for lane in ("fork", "writer", "helper")} == calls_before
    assert len(ws.events(home)) == events_before, ws.events(home)[events_before:]
    # the writer is not available on the record; a pause is not taken: rev 2's main profile has no in-episode memory,
    # so the role has no checkpoint at all (the frozen refusal of a checkpoint is test_frozen_memory's)
    assert all("not available: memory is frozen" in x for x in said), said
    assert all("tools.checkpoint isn't available in your role" in x for x in said), said
    frozen_rows = [e for e in H._jsonl(paths.state_dir / "events.jsonl") if e.get("phase") == "frozen"]
    assert sorted(e["episode_id"] for e in frozen_rows) == sorted(s.episode_id for s in sessions), frozen_rows
    from unify.agents import run_record as rr

    assert not [r for r in rr.read_entries(home) if "writer" in (r.get("mentions") or []) and r.get("episode") in
                {s.episode_id for s in sessions}]
    H.check_terminated(H.drain(home), home)
