"""Memory v2.1 P9 (spec §5): the verdicts a bed shows the actor, posted as agent-visible checker signals.

The bed's runner sends ``{"checker": {"label": "pass"|"fail"}}`` just before the message that shows the actor the
same verdict. Under ``UNIFY_MEMORY_V21`` and ``UNIFY_MEMORY_V21_CHECKER_VISIBLE`` the request keeps it, and at
episode write it becomes one checker signal per line with ``visible_to_actor=True``, referring to the dialogue
action that message answered (or to the episode when that is not certain). With the switch off nothing is kept.
The grader channel (``{"outcome": ...}``) is unchanged and never visible. Every other signal's note and every
store without a visible signal keep their bytes and schema.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tests.memory_v2.integration.test_request import (  # noqa: F401
    _Run,
    _abort,
    _begin,
    _drive,
    mv2,
)
from tests.memory_v2.test_episodes import _ep
from unify.memory_v2.episodes import Action
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import consolidate, switch
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.adapters import dialogue
from unify.memory_v2.redact import Redactor
from unify.memory_v2.signals import Signal, note_text
from unify.settings import SETTINGS

SENTINEL = "SENTINEL-c4e1"


def _msg(role: str, content: str, **extra) -> dict:
    return {"type": "message", "message": {"role": role, "content": content, **extra}}


def _arc_lines() -> list[dict]:
    """An ARC-like request: the instance, a wrong submit, its verdict, a right submit, its verdict, a closing
    reply. Observations: 0 the instance, 1 the first verdict, 2 the second verdict."""
    return [
        _msg("user", "Instance 1. Demo pairs ..."),
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1"}],
            },
        },
        _msg("tool", "ran"),
        _msg("assistant", 'Submitting.\n{"action": "submit", "grid": [[1]]}'),
        _msg("user", "Your submission was INCORRECT. Wrong attempts so far: 1."),
        _msg("assistant", 'Trying again.\n{"action": "submit", "grid": [[2]]}'),
        _msg(
            "user",
            "Your submission was CORRECT. Instance solved with 1 wrong attempt(s).",
        ),
        _msg("assistant", '{"action": "finish"}'),
    ]


# ── the switch ───────────────────────────────────────────────────────────────


def test_switch_defaults_off_and_parses():
    assert (
        type(SETTINGS).model_fields["UNIFY_MEMORY_V21_CHECKER_VISIBLE"].default == "off"
    )
    assert switch.PARSERS[switch.CHECKER_VISIBLE] is switch.parse_checker_visible
    assert switch.parse_checker_visible("") == "off"
    assert switch.parse_checker_visible("ON") == "on"
    with pytest.raises(ValueError):
        switch.parse_checker_visible("maybe")


@pytest.mark.parametrize(
    "v21,visible,expected",
    [
        ("off", "off", False),
        ("off", "on", False),
        ("on", "off", False),
        ("on", "on", True),
    ],
)
def test_checker_visible_needs_both_switches(monkeypatch, v21, visible, expected):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", v21)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21_CHECKER_VISIBLE", visible)
    assert request_mod.checker_visible_on() is expected


# ── the signal and the store ─────────────────────────────────────────────────


def test_signal_note_bytes_unchanged_when_not_visible():
    sig = Signal("e.checker", "e", "checker", "pass", "t", refers_to="e")
    row = asdict(sig)
    del row["visible_to_actor"]
    assert note_text(sig) == json.dumps(row, sort_keys=True)  # the note as before P9
    assert "visible_to_actor" not in note_text(sig)
    vis = Signal("e.checker.1", "e", "checker", "fail", "t", visible_to_actor=True)
    assert json.loads(note_text(vis))["visible_to_actor"] is True


def _schema(path) -> list:
    con = sqlite3.connect(path)
    try:
        return sorted(con.execute("SELECT name, sql FROM sqlite_master").fetchall())
    finally:
        con.close()


def test_store_schema_unchanged_until_a_visible_signal(tmp_path):
    path = tmp_path / "evidence.sqlite"
    ev = EvidenceStore(path)
    ev.index_episode(_ep(episode_id="e0"), "0" * 40)
    before = _schema(path)
    ev.add_signal(
        Signal("e0.checker", "e0", "checker", "pass", "t"),
    )  # the grader channel
    assert _schema(path) == before
    assert ev.signals_for("e0")[0].visible_to_actor is False
    ev.add_signal(
        Signal("e0.checker.1", "e0", "checker", "fail", "t2", visible_to_actor=True),
    )
    assert "signal_visible" in {name for name, _ in _schema(path)}
    got = {s.signal_id: s.visible_to_actor for s in ev.signals_for("e0")}
    assert got == {"e0.checker": False, "e0.checker.1": True}


# ── pairing a verdict with the action its message answers ────────────────────


def test_answered_observations_follow_the_dialogue_actions():
    lines = _arc_lines()
    acts = dialogue.dialogue_actions(
        lines,
        "env",
        redactor=Redactor(),
        answered_only=True,
    )
    pairs = dialogue.answered_observations(lines)
    assert len(pairs) == len(acts) == 2  # the closing reply is unanswered: no action
    assert pairs == [1, 2]
    assert dialogue.observation_count(lines) == 3
    assert (
        dialogue.observation_count(lines[:4]) == 1
    )  # before the first verdict is shown


def test_loop_authored_user_messages_are_not_observations():
    lines = [
        *_arc_lines()[:4],
        _msg("user", "harness note", _loop_authored=True),
        *_arc_lines()[4:],
    ]
    assert dialogue.observation_count(lines) == 3
    assert dialogue.answered_observations(lines) == [1, 2]


# ── posting ──────────────────────────────────────────────────────────────────


def _stores(tmp_path, regime: str = "dense"):
    repo = Repo.init_bare(tmp_path / "ep.git")
    ev = EvidenceStore(tmp_path / "evidence.sqlite")
    paths = SimpleNamespace(errors=tmp_path / "errors.jsonl")
    return SimpleNamespace(episodes=repo, evidence=ev, paths=paths), repo


def _episode(eid: str, n_dialogue: int, regime: str = "dense"):
    tool = Action(0, "fs", "write", [], {}, None, "ok", "write")
    dlg = [
        Action(-1, "env", "submit", [], {}, "v", "ok", kind="dialogue")
        for _ in range(n_dialogue)
    ]
    return _ep(episode_id=eid, regime=regime, actions=[tool, *dlg])


def _entries(*obs_labels):
    return [
        {"label": lab, "obs": obs, "at": f"2026-10-09T00:00:0{i}Z"}
        for i, (obs, lab) in enumerate(obs_labels)
    ]


def test_one_visible_signal_per_submit_referring_to_its_action(tmp_path):
    stores, repo = _stores(tmp_path)
    ep = _episode("ep1", 2)
    stores.evidence.index_episode(ep, repo.head())
    n = consolidate.post_visible_checkers(
        stores,
        ep,
        repo.head(),
        _entries((1, "fail"), (2, "pass")),
        _arc_lines(),
        "env",
    )
    assert n == 2
    sigs = stores.evidence.signals_for("ep1")
    assert [(s.signal_id, s.label, s.refers_to, s.visible_to_actor) for s in sigs] == [
        ("ep1.checker.1", "fail", "ep1/actions/1", True),
        ("ep1.checker.2", "pass", "ep1/actions/2", True),
    ]
    notes = [json.loads(x) for x in repo.notes(repo.head())]
    assert all(
        x["visible_to_actor"] is True and set(x) >= {"label", "refers_to"}
        for x in notes
    )


def test_refers_to_episode_when_dialogue_is_off(tmp_path):
    stores, repo = _stores(tmp_path)
    ep = _episode("ep2", 2)
    stores.evidence.index_episode(ep, repo.head())
    consolidate.post_visible_checkers(
        stores,
        ep,
        repo.head(),
        _entries((1, "pass")),
        _arc_lines(),
        "",
    )
    assert stores.evidence.signals_for("ep2")[0].refers_to == "ep2"


@pytest.mark.parametrize(
    "entries,n_dialogue",
    [
        (_entries((1, "fail"), (1, "pass")), 2),  # not strictly increasing: uncertain
        (
            _entries((1, "pass")),
            1,
        ),  # the episode's dialogue actions and the pairing disagree
        (_entries((0, "pass")), 2),  # observation 0 (the instance) answers no reply
    ],
)
def test_never_a_guessed_action(tmp_path, entries, n_dialogue):
    stores, repo = _stores(tmp_path)
    ep = _episode("ep3", n_dialogue)
    stores.evidence.index_episode(ep, repo.head())
    consolidate.post_visible_checkers(
        stores,
        ep,
        repo.head(),
        entries,
        _arc_lines(),
        "env",
    )
    assert {s.refers_to for s in stores.evidence.signals_for("ep3")} == {"ep3"}


def test_masked_regime_drops(tmp_path):
    stores, repo = _stores(tmp_path)
    ep = _episode("ep4", 2, regime="implicit")
    stores.evidence.index_episode(ep, repo.head())
    assert (
        consolidate.post_visible_checkers(
            stores,
            ep,
            repo.head(),
            _entries((1, "fail")),
            _arc_lines(),
            "env",
        )
        == 0
    )
    assert stores.evidence.signals_for("ep4") == []


def test_outcome_channel_unchanged_and_not_visible(tmp_path):
    stores, repo = _stores(tmp_path)
    ep = _episode("ep5", 0)
    stores.evidence.index_episode(ep, repo.head())
    assert consolidate.post_checker(stores, "ep5", repo.head(), True, "t") is True
    (sig,) = stores.evidence.signals_for("ep5")
    assert sig.signal_id == "ep5.checker" and sig.visible_to_actor is False
    assert "visible_to_actor" not in repo.notes(repo.head())[0]


# ── the request ──────────────────────────────────────────────────────────────


def _on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21_CHECKER_VISIBLE", "on")


def test_checker_line_refused_when_switch_off(mv2, monkeypatch):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21_CHECKER_VISIBLE", "off")
    run = _begin(mv2, "hi")
    try:
        answer = run.take_checker({"label": "pass"})
        assert answer["type"] == "checker" and answer["accepted"] is False
        assert run.checker_seen == []
    finally:
        _abort(mv2, run)


def test_checker_line_refused_without_v21(mv2, monkeypatch):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21_CHECKER_VISIBLE", "on")
    run = _begin(mv2, "hi")
    try:
        assert run.take_checker({"label": "pass"})["accepted"] is False
        assert run.checker_seen == []
    finally:
        _abort(mv2, run)


@pytest.mark.parametrize(
    "raw",
    [
        {"label": "maybe"},
        {"label": "pass", "grid": SENTINEL},
        {"label": "pass", "task": SENTINEL},
        "pass",
        None,
        {"label": True},
        {},
    ],
)
def test_label_validation(mv2, monkeypatch, raw):  # noqa: F811
    _on(monkeypatch)
    run = _begin(mv2, "hi")
    try:
        answer = run.take_checker(raw)
        assert answer["accepted"] is False and SENTINEL not in json.dumps(answer)
        assert run.checker_seen == []
    finally:
        _abort(mv2, run)


def test_accepted_entries_carry_the_observation_count(mv2, monkeypatch):  # noqa: F811
    _on(monkeypatch)
    run = _begin(mv2, "hi")
    try:
        lines = _arc_lines()
        monkeypatch.setattr(
            run,
            "_observations_now",
            lambda: dialogue.observation_count(lines[:4]),
        )
        first = run.take_checker({"label": "fail"})
        monkeypatch.setattr(
            run,
            "_observations_now",
            lambda: dialogue.observation_count(lines[:6]),
        )
        second = run.take_checker({"label": "pass"})
        assert first == {
            "type": "checker",
            "accepted": True,
            "label": "fail",
            "count": 1,
        }
        assert second["count"] == 2
        assert [(e["label"], e["obs"]) for e in run.checker_seen] == [
            ("fail", 1),
            ("pass", 2),
        ]
    finally:
        _abort(mv2, run)


def test_observations_now_reads_this_requests_transcript(
    mv2,
    monkeypatch,
):  # noqa: F811
    _on(monkeypatch)
    run = _begin(mv2, "hi")
    try:
        from unify import transcripts

        path = transcripts.transcripts_dir() / f"{run.episode_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(x) + "\n" for x in _arc_lines()[:4]))
        assert run._observations_now() == 1
    finally:
        _abort(mv2, run)


def test_finish_posts_visible_checkers_only_when_any_were_taken(
    mv2,
    monkeypatch,
):  # noqa: F811
    from tests.memory_v2.integration.test_request import _finish, _transcript

    calls = []
    # the request imports consolidate at call time: under the mv2 fixture that is fake_tracks' module, so the
    # recorder goes on whichever module is installed now
    import sys

    cons = sys.modules["unify.memory_v2.integration.consolidate"]
    monkeypatch.setattr(
        cons,
        "post_visible_checkers",
        lambda stores, ep, sha, entries, lines, counterpart: calls.append(entries)
        or len(entries),
        raising=False,
    )
    _on(monkeypatch)
    run = _begin(mv2, "none taken")
    _transcript(run)
    _finish(mv2, run)
    assert calls == []  # no checker line: the episode write is as before P9
    run = _begin(mv2, "one taken")
    monkeypatch.setattr(run, "_observations_now", lambda: 1)
    run.take_checker({"label": "pass"})
    _transcript(run)
    _finish(mv2, run)
    assert (
        len(calls) == 1 and calls[0][0]["label"] == "pass" and calls[0][0]["obs"] == 1
    )


# ── the CLI ──────────────────────────────────────────────────────────────────


class _CheckerRun(_Run):
    def take_checker(self, raw):
        self.log.append(("take_checker", raw))
        return {"type": "checker", "accepted": True, "label": raw["label"], "count": 1}


def test_cli_hands_a_checker_line_to_the_run_and_never_to_the_actor(monkeypatch):
    _on(monkeypatch)
    log: list = []
    code, out, log, handle = _drive(
        monkeypatch,
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"],
        b'{"checker": {"label": "pass"}}\n{"quit": true}\n',
        _CheckerRun(log),
    )
    assert code == 0
    assert [e[0] for e in log] == ["begin", "act", "take_checker", "finish", "abort"]
    assert ("take_checker", {"label": "pass"}) in log
    assert all(e != ("act", '{"checker": {"label": "pass"}}') for e in log)
    assert out[0] == {"type": "checker", "accepted": True, "label": "pass", "count": 1}


def test_cli_without_a_memory_run_refuses_the_checker_line(monkeypatch):
    _on(monkeypatch)
    code, out, log, handle = _drive(
        monkeypatch,
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"],
        b'{"checker": {"label": "pass"}}\n{"quit": true}\n',
        None,
    )
    assert code == 0
    refused = [o for o in out if o.get("type") == "checker"]
    assert refused == [
        {
            "type": "checker",
            "accepted": False,
            "reason": "this session records no memory",
        },
    ]


# ── review S4: off-path equivalence of the stdin channel ────────────────────


def _off(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21_CHECKER_VISIBLE", "off")


def test_cli_with_the_switches_off_ignores_a_checker_line_as_before(monkeypatch):
    _off(monkeypatch)
    log: list = []
    code, out, log, handle = _drive(
        monkeypatch,
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"],
        b'{"checker": {"label": "pass"}}\n{"quit": true}\n',
        _CheckerRun(log),
    )
    assert code == 0
    assert [e[0] for e in log] == [
        "begin",
        "act",
        "finish",
        "abort",
    ]  # not handed to the run
    assert [
        o for o in out if o.get("type") == "checker"
    ] == []  # and no answer: no control line at all


@pytest.mark.parametrize("switches", ["off", "on"])
def test_a_line_that_also_quits_is_never_swallowed(monkeypatch, switches):
    (_on if switches == "on" else _off)(monkeypatch)
    log: list = []
    code, out, log, handle = _drive(
        monkeypatch,
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"],
        b'{"checker": {"label": "pass"}, "quit": true}\n',
        _CheckerRun(log),
    )
    assert (
        code == 0
    )  # the quit was honoured (a swallowed quit leaves the persistent session waiting)
    assert "take_checker" not in [e[0] for e in log]
    assert [e[0] for e in log] == ["begin", "act", "finish", "abort"]


# ── review S2: only verdicts the actor saw are posted ───────────────────────


def _errors(tmp_path):
    path = tmp_path / "errors.jsonl"
    return (
        [json.loads(x)["error"] for x in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def test_a_verdict_the_actor_never_saw_is_dropped_and_logged(tmp_path):
    stores, repo = _stores(tmp_path)
    ep = _episode("ep6", 2)
    stores.evidence.index_episode(ep, repo.head())
    lines = _arc_lines()[
        :4
    ]  # the request ended after the first submit, before its verdict was shown
    n = consolidate.post_visible_checkers(
        stores,
        ep,
        repo.head(),
        _entries((1, "fail")),
        lines,
        "env",
    )
    assert n == 0 and stores.evidence.signals_for("ep6") == []
    (err,) = _errors(tmp_path)
    assert "checker line 1 dropped" in err and "observation 1 of 1 never shown" in err


def test_an_unreadable_count_is_dropped_not_posted(tmp_path):
    stores, repo = _stores(tmp_path)
    ep = _episode("ep7", 2)
    stores.evidence.index_episode(ep, repo.head())
    n = consolidate.post_visible_checkers(
        stores,
        ep,
        repo.head(),
        _entries((None, "pass")),
        _arc_lines(),
        "env",
    )
    assert n == 0 and stores.evidence.signals_for("ep7") == []
    assert "count unreadable" in _errors(tmp_path)[0]


def test_delivered_entries_still_pair_when_a_later_one_is_dropped(tmp_path):
    stores, repo = _stores(tmp_path)
    ep = _episode("ep8", 2)
    stores.evidence.index_episode(ep, repo.head())
    entries = _entries(
        (1, "fail"),
        (2, "pass"),
        (3, "pass"),
    )  # the third came after the last shown verdict
    n = consolidate.post_visible_checkers(
        stores,
        ep,
        repo.head(),
        entries,
        _arc_lines(),
        "env",
    )
    assert n == 2
    assert [(s.signal_id, s.refers_to) for s in stores.evidence.signals_for("ep8")] == [
        ("ep8.checker.1", "ep8/actions/1"),
        ("ep8.checker.2", "ep8/actions/2"),
    ]
    assert len(_errors(tmp_path)) == 1
