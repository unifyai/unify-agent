"""The outcomes a world shows the actor reach memory.

* Within a request, a visible ``fail`` verdict (a ``{"checker": ...}`` line, with or without the verdict's text) that
  follows the use of an item (its call, or ``memory.get('<id>')`` in a cell) counts as a failed use of that item in this
  request: the librarian is not offered it again here. A call made after the verdict is not counted; a ``pass``
  excludes nothing. Uses are read from the runtime's call counts and from the cells' code, never from text.
* A checker line may carry the verdict's text exactly as the environment showed it; the label stays the verdict.
* A verdict on an earlier episode (named by its id) is recorded as late evidence on that episode and on the items
  it used; the writer's late-evidence material carries the verdict's text.

Real components throughout (the request run, its stores, the actor's ``ask_library`` handler, the attention ledger and
its home sources); only the librarian's model and the world are faked.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.memory.test_checker_visible import _arc_lines
from tests.memory.test_episodes import _ep
from tests.memory.integration.test_request import _Run, _drive
from tests.memory.test_inferred_regime import ENDED, ITEM, _record, _run
from unify import transcripts
from unify.actor import tool_handlers
from unify.memory import attention as at
from unify.memory import lifecycle, trust
from unify.memory.episodes import EpisodeWriter
from unify.memory.integration import outcomes
from unify.memory.integration import request as request_mod
from unify.memory.redact import Redactor
from unify.settings import SETTINGS

CAPTURE = "capture:0123456789abcdef"
LATER = "memory.reports.totals:sum_columns"
TEXT = "Sofia Osei: The receipts forecast counted Dunmore Studios' invoices on their due dates; that cash did not come in."


def _cell(call_id: str, code: str) -> list[dict]:
    """One executed code cell as the transcript holds it: the tool call and its result."""
    return [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "execute_code", "arguments": json.dumps({"code": code})},
                    },
                ],
            },
        },
        {"type": "message", "message": {"role": "tool", "tool_call_id": call_id, "content": "ran"}},
    ]


def _calls(item: str, outcome: str = "ok") -> dict:
    """A cell's drained audit block with one library call of *item* (``adapters.audit``)."""
    row = {"ok": 0, "refused": 0, "error": 0}
    row[outcome] = 1
    return {"calls": {"counts": {item: row}}}


@pytest.fixture
def run(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch)
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    monkeypatch.setattr(transcripts, "transcripts_dir", lambda: tdir)
    run.transcript_path = tdir / f"{run.episode_id}.jsonl"
    monkeypatch.setattr(run, "_observations_now", lambda: 1)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    return run


def _write(run, rows: list[dict]) -> None:
    run.transcript_path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class _Librarian:
    """The request's librarian session with its model faked: records what each question carried."""

    def __init__(self) -> None:
        self.asked: list[dict] = []

    async def ask(self, question, *, request=None, **extra):
        self.asked.append({"question": question, "request": request, **extra})
        return json.dumps({"fits": [], "nearest": []})


def _ask(run) -> dict:
    session = _Librarian()
    ctx = SimpleNamespace(state=SimpleNamespace(librarian=session, request_text=run.request), agent="actor")
    asyncio.run(tool_handlers.ask_library(ctx, {"question": "What fits this request now?"}))
    (asked,) = session.asked
    return asked


# --- 1. within a request ------------------------------------------------------------------------------------------


def test_a_fail_verdict_with_its_text_after_a_use_is_a_failed_use_in_this_request(run):
    # the actor looked a capture up and called a function, then the world's verdict came
    _write(run, _cell("c1", f"card = memory.get({CAPTURE!r})\nprint(card)"))
    run.note_calls(_calls(ITEM))
    answer = run.take_checker({"label": "fail", "text": TEXT})
    assert answer == {"type": "checker", "accepted": True, "label": "fail", "count": 1}
    assert run.checker_seen[0]["text"] == TEXT
    assert run.checker_seen[0]["used_before"] == sorted([ITEM, CAPTURE])
    # a call made after the verdict is not a use the verdict followed
    run.note_calls(_calls(LATER))
    assert tool_handlers._memory_run_failed() == sorted([ITEM, CAPTURE])
    # the re-ask carries them as failed in this request
    assert _ask(run)["failed"] == sorted([ITEM, CAPTURE])


def test_an_arc_style_label_only_fail_excludes_the_item_and_a_pass_excludes_nothing(run):
    run.note_calls(_calls(ITEM))
    assert run.take_checker({"label": "pass"})["accepted"] is True
    assert "used_before" not in run.checker_seen[0]
    assert tool_handlers._memory_run_failed() == []
    assert "failed" not in _ask(run)
    # the same line form ARC's runner sends, with no world-specific code anywhere
    assert run.take_checker({"label": "fail"})["accepted"] is True
    assert tool_handlers._memory_run_failed() == [ITEM]
    assert _ask(run)["failed"] == [ITEM]


def test_a_fail_before_any_use_excludes_nothing(run):
    assert run.take_checker({"label": "fail", "text": TEXT})["accepted"] is True
    run.note_calls(_calls(ITEM))
    assert run.failed_after_verdicts() == [] and tool_handlers._memory_run_failed() == []


def test_lookups_are_read_from_code_never_from_text():
    rows = [
        *_cell("c1", f'print("see memory.get({CAPTURE!r})")  # memory.get({LATER!r})'),
        *_cell("c2", f"x = memory.get({ITEM!r})\ny = other.get('notes/a/b.md')"),
        *_cell("c3", "this is not python ("),
    ]
    assert request_mod.cells_get_ids(rows) == [ITEM]


# --- 2. the typed line -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, why",
    [
        ({"label": "fail", "text": ""}, "non-empty"),
        ({"label": "fail", "text": "   "}, "non-empty"),
        ({"label": "fail", "text": 3}, "non-empty"),
        ({"label": "fail", "text": "x" * (outcomes.MAX_TEXT + 1)}, "longer than"),
        ({"label": "fail", "text": TEXT, "score": 0.4}, "optionally with"),
        ({"text": TEXT}, "optionally with"),
    ],
)
def test_a_lines_text_is_checked_and_never_quoted(run, raw, why):
    answer = run.take_checker(raw)
    assert answer["accepted"] is False and why in answer["reason"]
    assert TEXT not in json.dumps(answer) and run.checker_seen == []


def test_an_in_episode_verdicts_text_is_kept_beside_its_signal(run):
    assert run.take_checker({"label": "fail", "text": TEXT})["accepted"] is True
    ep = _ep(episode_id=run.episode_id, ended_at="2999-01-01T00:00:00Z", regime="none")
    basis, visible = run._infer_regime(ep, _arc_lines())
    sha = EpisodeWriter(run.stores.episodes, run.stores.blobs, Redactor()).write(ep)
    run.stores.evidence.index_episode(ep, sha)
    run._post_inferred(ep, sha, ep.ended_at, basis, visible)
    (sig,) = run.stores.evidence.signals_for(run.episode_id)
    assert (sig.label, sig.visible_to_actor) == ("fail", True)
    assert outcomes.signal_texts(run.paths.state_dir, [sig.signal_id]) == {sig.signal_id: TEXT}


# --- 3. a verdict on an earlier episode: late evidence the writer reads ---------------------------------------------


def test_a_late_verdict_with_text_is_late_evidence_on_the_episode_and_its_items_with_the_text(run, tmp_path):
    _record(run.stores, "e1", regime="none", minute=0, ended=ENDED)
    run.stores.evidence.record_regime("e1", "none", "nothing", ENDED)
    answer = run.take_checker({"label": "fail", "text": TEXT}, episode="e1")
    assert answer["accepted"] is True and run.checker_seen == []
    ep = _ep(episode_id=run.episode_id, ended_at="2999-01-01T00:00:00Z", regime="none")
    basis, visible = run._infer_regime(ep, [])
    sha = EpisodeWriter(run.stores.episodes, run.stores.blobs, Redactor()).write(ep)
    run.stores.evidence.index_episode(ep, sha)
    run._post_inferred(ep, sha, ep.ended_at, basis, visible)
    sid = f"e1.late.{run.episode_id}.1"
    assert [s.signal_id for s in run.stores.evidence.signals_for("e1")] == [sid]
    # the attention ledger, over the home's own records, opens late evidence on e1 naming the item it used
    src = at.HomeSources(run.paths.home, state_dir=run.paths.state_dir)
    led = at.Ledger(run.paths.home)
    led.sync(src)
    late = [e for e in led.hand("p1", at.K) if e.kind == "late_evidence"]
    assert len(late) == 1 and late[0].refs.episodes == ("e1",) and ITEM in late[0].refs.items
    out = tmp_path / "inputs" / "attention"
    at.export_material(late, out, ledger=led, sources=src)
    doc = json.loads((out / late[0].entry / "signal.json").read_text())
    (row,) = doc["signals"]
    assert (row["signal"], row["label"], row["timing"], row["text"]) == (sid, "fail", "after_episode", TEXT)
    # the ledger's own rows still hold ids and kinds only
    assert TEXT not in led.path.read_text()


# --- 4. a persistent session of several requests: a verdict bears on the request it followed ------------------------

OTHER = "memory.reports.totals:count_rows"


def _judged(stores, eid: str) -> dict:
    """Each item's trust verdict in episode *eid*, from the evidence (``lifecycle.trust_uses``)."""
    (f,) = [f for f in lifecycle.facts_from_evidence(stores.evidence) if f.episode_id == eid]
    return {i: [trust.judge(u)[:2] for u in lifecycle.trust_uses(f, u)] for i, u in f.items.items()}


def _uses(*items) -> dict:
    return {
        "items": {i: {"imported": 1, "called": 1, "refused": 0, "errored": 0} for i in items},
        "outcomes_known": True,
    }


def test_a_fail_after_request_two_marks_only_request_ones_items(run):
    # request 1 (the session's first): a lookup and a call; then its feedback arrives with request 2's message
    _write(run, _cell("c1", f"card = memory.get({CAPTURE!r})"))
    run.note_calls(_calls(ITEM))
    assert run.take_checker({"label": "fail", "text": TEXT})["accepted"] is True
    assert run.checker_seen[0]["used_before"] == sorted([CAPTURE, ITEM])
    run.mark_request()  # the CLI submits request 2
    # request 2 calls another item (and ITEM is not called again); its feedback comes before request 3
    _write(run, [*_cell("c1", f"card = memory.get({CAPTURE!r})"), *_cell("c2", "print(1)")])
    run.note_calls(_calls(LATER))
    assert run.take_checker({"label": "fail"})["accepted"] is True
    assert run.checker_seen[1]["used_before"] == [LATER]
    run.mark_request()  # request 3: uses OTHER, and no verdict follows it
    run.note_calls(_calls(OTHER))
    # recorded: each verdict bears on its own request's items; OTHER, used in request 3, is not judged by either
    ep = _ep(episode_id=run.episode_id, ended_at="2999-01-01T00:00:00Z", regime="none")
    ep.memory_use = _uses(ITEM, LATER, OTHER)
    basis, visible = run._infer_regime(ep, _arc_lines())
    sha = EpisodeWriter(run.stores.episodes, run.stores.blobs, Redactor()).write(ep)
    run.stores.evidence.index_episode(ep, sha)
    run._post_inferred(ep, sha, ep.ended_at, basis, visible)
    scoped = run.stores.evidence.signal_items(run.episode_id)
    assert sorted(scoped.values(), key=len) == [frozenset({LATER}), frozenset({CAPTURE, ITEM})]
    judged = _judged(run.stores, run.episode_id)
    assert judged[ITEM] == [(trust.FAILED, "negative")]
    assert judged[LATER] == [(trust.FAILED, "negative")]
    assert judged[OTHER] == [(trust.CLEAN, None)]
    # the session's last request is kept for a verdict that arrives after it ended
    assert run.stores.evidence.last_request_items(run.episode_id) == frozenset({OTHER})


def test_the_in_request_exclusion_resets_at_the_next_request(run):
    run.note_calls(_calls(ITEM))
    assert run.take_checker({"label": "fail"})["accepted"] is True
    assert tool_handlers._memory_run_failed() == [ITEM]  # the request it was taken in
    run.mark_request()
    assert tool_handlers._memory_run_failed() == [ITEM]  # the request that shows it
    assert _ask(run)["failed"] == [ITEM]
    run.mark_request()
    assert tool_handlers._memory_run_failed() == []  # never later
    assert "failed" not in _ask(run)


def test_a_late_verdict_on_a_session_bears_on_its_last_request_only(run, tmp_path):
    _record(run.stores, "e1", regime="none", minute=0, ended=ENDED, also=(LATER,))
    run.stores.evidence.record_regime("e1", "none", "nothing", ENDED)
    run.stores.evidence.set_last_request("e1", 3, [LATER])
    assert run.take_checker({"label": "fail", "text": TEXT}, episode="e1")["accepted"] is True
    ep = _ep(episode_id=run.episode_id, ended_at="2999-01-01T00:00:00Z", regime="none")
    basis, visible = run._infer_regime(ep, [])
    sha = EpisodeWriter(run.stores.episodes, run.stores.blobs, Redactor()).write(ep)
    run.stores.evidence.index_episode(ep, sha)
    run._post_inferred(ep, sha, ep.ended_at, basis, visible)
    judged = _judged(run.stores, "e1")
    assert judged[LATER] == [(trust.FAILED, "negative")] and judged[ITEM] == [(trust.CLEAN, None)]
    src = at.HomeSources(run.paths.home, state_dir=run.paths.state_dir)
    led = at.Ledger(run.paths.home)
    led.sync(src)
    late = [e for e in led.hand("p1", at.K) if e.kind == "late_evidence"]
    assert len(late) == 1 and late[0].refs.items == (LATER,)


def test_an_episode_of_one_request_keeps_the_episode_wide_verdict(run):
    """ARC's label-only line in an episode the CLI never marked: unchanged, the verdict bears on the whole episode."""
    run.note_calls(_calls(ITEM))
    assert run.take_checker({"label": "fail"})["accepted"] is True
    ep = _ep(episode_id=run.episode_id, ended_at="2999-01-01T00:00:00Z", regime="none")
    ep.memory_use = _uses(ITEM, OTHER)
    basis, visible = run._infer_regime(ep, _arc_lines())
    sha = EpisodeWriter(run.stores.episodes, run.stores.blobs, Redactor()).write(ep)
    run.stores.evidence.index_episode(ep, sha)
    run._post_inferred(ep, sha, ep.ended_at, basis, visible)
    assert run.stores.evidence.signal_items(run.episode_id) == {}
    assert run.stores.evidence.last_request_items(run.episode_id) is None
    judged = _judged(run.stores, run.episode_id)
    assert judged[ITEM] == judged[OTHER] == [(trust.FAILED, "negative")]


class _MarkRun(_Run):
    def take_checker(self, raw, **where):
        self.log.append(("take_checker", raw))
        return {"type": "checker", "accepted": True, "label": raw["label"], "count": 1}

    def mark_request(self):
        self.log.append(("mark_request",))


def _drive_marks(monkeypatch, stdin: bytes) -> list:
    from tests.memory.integration import test_request

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    log: list = []

    async def submit(self, text):
        log.append(("submit", text))

    monkeypatch.setattr(test_request._Handle, "submit", submit, raising=False)
    code, out, log, handle = _drive(
        monkeypatch,
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "first request"],
        stdin,
        _MarkRun(log),
    )
    assert code == 0
    return [e[0] for e in log]


def test_the_cli_marks_a_new_request_only_on_the_worlds_typed_marker(monkeypatch):
    """The world says where a request starts: ``{"message": ..., "request": "new"}``. A plain message is more of the
    same request (an observation, feedback on the same task) and marks nothing."""
    plain = _drive_marks(
        monkeypatch,
        b'{"checker": {"label": "fail"}}\n{"message": "your answer was wrong"}\n'
        b'{"checker": {"label": "fail"}}\n{"message": "wrong again"}\n{"quit": true}\n',
    )
    assert plain.count("mark_request") == 0 and plain.count("take_checker") == 2
    marked = _drive_marks(
        monkeypatch,
        b'{"checker": {"label": "fail", "text": "it went wrong"}}\n'
        b'{"message": "second request", "request": "new"}\n{"message": "more on it"}\n{"quit": true}\n',
    )
    assert marked.count("mark_request") == 1
    assert marked.index("take_checker") < marked.index("mark_request")


def test_plain_follow_up_messages_keep_the_visit_wide_exclusion(run):
    """No marker (every world but a driver that marks requests): one request, the episode-wide behaviour as before."""
    run.note_calls(_calls(ITEM))
    assert run.take_checker({"label": "fail"})["accepted"] is True
    run.note_calls(_calls(OTHER))  # a later turn of the same request
    assert run.take_checker({"label": "fail"})["accepted"] is True
    assert run.checker_seen[1]["used_before"] == sorted([ITEM, OTHER])
    run.note_calls(_calls(LATER))
    assert run.requests == 0
    assert tool_handlers._memory_run_failed() == sorted([ITEM, OTHER])
    assert _ask(run)["failed"] == sorted([ITEM, OTHER])
