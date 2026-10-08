"""The whole trajectory from the session transcript, and the episode built from it (integration Task 23).

The fixture is written in the line format of ``unify/transcripts.py`` at this base: ``seq``, ``ts``,
``session``, ``type`` and, on ``message``/``message_update`` lines, ``loop``, ``in_context`` and ``message``.
"""

import datetime as dt
import json
import time
from types import SimpleNamespace

from unify.memory_v2.analysis.cells import cells_from_transcript
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action, CostRow, EpisodeWriter, load_episode
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import trajectory
from unify.memory_v2.integration.trajectory import (
    TimedObserver,
    assemble,
    cell_at,
    dialogue_text,
    fold,
    learned_secrets,
    read_jsonl,
    timed_cells,
)

SID = "20261008T120000-0a1b2c3d"
T0 = dt.datetime(2026, 10, 8, 12, 0, 0, tzinfo=dt.UTC)
TOKEN = "tok-abcdefgh123"  # pragma: allowlist secret
FAKE_PASS = "longer-pass"  # pragma: allowlist secret
SHORT = "short"  # under 8 characters: never learned
FIRST = "Say hi to ada and list the workspace"
FOLLOW = "Now say bye"
FINAL = "hi sent; the workspace has one file"
PY = "print(hello(apis, 'ada'))\nprint(apis.spotify.login(username='ada'))"


def _ts(s: float) -> str:
    return (T0 + dt.timedelta(seconds=s)).isoformat()


def _line(seq, s, kind, **rest):
    return {"seq": seq, "ts": _ts(s), "session": SID, "type": kind, **rest}


def _msg(seq, s, message, kind="message"):
    extra = {"in_context": True} if kind == "message" else {}
    return _line(seq, s, kind, loop="CodeActActor.act", message=message, **extra)


def _call(cid, args):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": "execute_code", "arguments": json.dumps(args)},
    }


def _result(text, stderr=None):
    blocks = [
        {"type": "text", "text": json.dumps({"duration_ms": 12}, indent=2)},
        {"type": "text", "text": "\n--- stdout ---\n"},
        {"type": "text", "text": text},
    ]
    if stderr:
        blocks += [
            {"type": "text", "text": "\n--- stderr ---\n"},
            {"type": "text", "text": stderr},
        ]
    return blocks


def _lines():
    return [
        _line(0, 0, "session_start", parent=None, origin="CodeActActor.act"),
        _line(1, 0.1, "system_prompt", loop="CodeActActor.act", content="sys"),
        _msg(2, 1, {"role": "user", "content": FIRST}),
        _msg(
            3,
            2,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("c1", {"thought": "t", "code": PY})],
            },
        ),
        _msg(
            4,
            2.5,
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "execute_code",
                "content": json.dumps({"_placeholder": "pending"}),
            },
        ),
        _msg(
            5,
            4,
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "execute_code",
                "content": _result(f"hi ada\n{{'access_token': '{TOKEN}'}}\n"),
            },
            kind="message_update",
        ),
        _msg(
            6,
            5,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("c2", {"thought": "t", "code": "ls", "language": "bash"}),
                ],
            },
        ),
        _msg(
            7,
            7,
            {
                "role": "tool",
                "tool_call_id": "c2",
                "name": "execute_code",
                "content": _result("notes.txt\n", stderr="warn: x"),
            },
        ),
        _msg(8, 8, {"role": "assistant", "content": [{"type": "text", "text": FINAL}]}),
        _msg(9, 9, {"role": "user", "content": FOLLOW}),
    ]


def test_read_jsonl_skips_unreadable_lines(tmp_path):
    p = tmp_path / f"{SID}.jsonl"
    p.write_text(
        "\n".join(json.dumps(ln) for ln in _lines()) + "\nnot json\n[1]\n\n",
    )
    assert read_jsonl(p) == _lines()
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_fold_keeps_the_final_tool_output_at_the_first_position():
    folded = fold(_lines())
    assert [r["seq"] for r in folded] == [2, 3, 4, 6, 7, 8, 9]
    tool = folded[2]
    assert tool["message"]["content"] == _result(
        f"hi ada\n{{'access_token': '{TOKEN}'}}\n",
    )
    assert tool["end_ts"] == _ts(4)
    assert "end_ts" not in folded[4]


def test_two_cells_the_second_in_bash_with_their_spans():
    cells = timed_cells(fold(_lines()))
    assert [c.cell.language for c in cells] == ["python", "bash"]
    assert [c.call_id for c in cells] == ["c1", "c2"]
    assert cells[0].cell.code == PY and cells[1].cell.code == "ls"
    assert cells[0].cell.output.startswith("hi ada\n")
    assert cells[1].cell.output == "notes.txt\n" and cells[1].cell.error == "warn: x"
    assert cells[0].start == T0.timestamp() + 2 and cells[0].end == T0.timestamp() + 4
    assert cells[1].start == T0.timestamp() + 5 and cells[1].end == T0.timestamp() + 7


def test_cell_times_are_epoch_seconds_on_the_harness_clock():
    """The work-tree records are stamped with time.time(); a cell's span must be on that clock."""
    now = round(time.time(), 3)  # the ts keeps microseconds
    stamp = dt.datetime.fromtimestamp(
        now,
        dt.UTC,
    ).isoformat()  # as unify.transcripts writes ts
    lines = [
        {
            **_msg(
                0,
                0,
                {"role": "assistant", "tool_calls": [_call("c9", {"code": "1"})]},
            ),
            "ts": stamp,
        },
        {
            **_msg(1, 0, {"role": "tool", "tool_call_id": "c9", "content": "1"}),
            "ts": stamp,
        },
    ]
    (cell,) = timed_cells(fold(lines))
    assert abs(cell.start - now) < 1e-3 and abs(cell.end - now) < 1e-3
    assert cell_at(now, [cell]) == 0
    naive = {**lines[0], "ts": "2026-10-08T12:00:02"}  # no offset: read as UTC
    (c2,) = timed_cells(fold([naive, lines[1]]))
    assert c2.start == T0.timestamp() + 2


def test_timed_cells_agree_with_the_analysis_reader():
    folded = fold(_lines())
    ours = [(c.cell.code, c.cell.output) for c in timed_cells(folded)]
    theirs = [(c.code, c.output) for c in cells_from_transcript(folded)]
    assert ours == theirs


def test_a_result_left_as_a_placeholder_keeps_no_output():
    lines = [ln for ln in _lines() if ln["type"] != "message_update"]
    cells = timed_cells(fold(lines))
    assert cells[0].cell.output == "" and "no final result" in cells[0].cell.error


def test_requests_and_replies():
    assert dialogue_text(fold(_lines())) == ([FIRST, FOLLOW], [FINAL])


def test_cell_at_uses_the_spans():
    cells = timed_cells(fold(_lines()))
    t = T0.timestamp()
    assert cell_at(t + 3, cells) == 0
    assert cell_at(t + 6, cells) == 1
    assert cell_at(t + 4.5, cells) == -1
    assert cell_at(None, cells) == -1
    assert cell_at(float("nan"), cells) == -1


def test_learned_secrets_are_labelled_by_structure_never_by_value():
    got = learned_secrets(
        [
            ("spotify.login", {"access_token": TOKEN, "user": {"api_key": "k" * 9}}),
            ("spotify.login", {"access_token": "another-token-1"}),
            ("venmo.pay", [{"password": SHORT}, {"Password": FAKE_PASS}]),
            ("x.y", {"keys": ["not-a-string-key-under-a-list"]}),
        ],
    )
    assert got == {
        "spotify.login.access_token": TOKEN,
        "spotify.login.user.api_key": "k" * 9,
        "spotify.login.access_token#2": "another-token-1",
        "venmo.pay.Password": FAKE_PASS,
    }


def _env_call(method, kwargs):
    return SimpleNamespace(
        namespace="apis",
        method=method,
        via="global",
        args=(),
        kwargs=kwargs,
        effect=None,
    )


def _observer(monkeypatch, at):
    obs = TimedObserver()
    monkeypatch.setattr(trajectory.time, "time", lambda: at)
    call = _env_call("spotify.login", {"username": "ada"})
    obs.before(call)
    obs.after(
        call,
        result={"access_token": TOKEN, "user": "ada"},
        error=None,
        intercepted=False,
        started=0.0,
        elapsed_s=0.0,
    )
    return obs


def _run(obs):
    return SimpleNamespace(
        episode_id=SID,
        started_at=_ts(0),
        build="a7465b99d",
        model="openai/gpt-6-luna",
        effort="high",
        pin="0" * 40,
        request=FIRST,
        observer=obs,
        costs=SimpleNamespace(rows=[CostRow("actor", "m", 1, 2, "unknown")]),
    )


def test_a_call_inside_cell_two_is_attributed_to_it(monkeypatch):
    obs = _observer(monkeypatch, T0.timestamp() + 6)
    ep, _ = assemble(_run(obs), _lines(), "", _ts(10))
    assert [(a.channel, a.method, a.cell) for a in ep.actions] == [
        ("spotify", "login", 1),
    ]


def test_episode_holds_the_whole_trajectory_and_extra_actions(monkeypatch):
    obs = _observer(monkeypatch, T0.timestamp() + 3)
    wt = Action(
        0,
        "worktree:workspace",
        "write",
        ["out/report.json"],
        {},
        {"blob_after": "b" * 64, "size": 3, "shape": {"format": "json"}},
        "ok",
        "write",
        kind="worktree",
    )
    ep, _ = assemble(
        _run(obs),
        _lines(),
        "diff --git a/x b/x\n",
        _ts(10),
        extra_actions=[wt],
        worktree_before="1" * 40,
        worktree_after="2" * 40,
        worktree_diff="--- a/out/report.json\n",
    )
    assert ep.request == [FIRST, FOLLOW] and ep.replies == [FINAL]
    assert [c.language for c in ep.cells] == ["python", "bash"]
    assert [a.kind for a in ep.actions] == ["tool", "worktree"]
    assert ep.actions[0].cell == 0
    assert (ep.worktree_before, ep.worktree_after) == ("1" * 40, "2" * 40)
    assert ep.worktree_diff == "--- a/out/report.json\n"
    assert ep.memory_diff == "diff --git a/x b/x\n"
    assert ep.transcript == _lines()  # raw lines, not folded
    assert ep.costs == [CostRow("actor", "m", 1, 2, "unknown")]
    assert ep.regime == "dense" and ep.effort == "high" and ep.memory_main == "0" * 40
    assert "spotify.login" in ep.fingerprints
    assert any(k.startswith("worktree:workspace.") for k in ep.fingerprints)


def test_no_user_message_falls_back_to_the_request_and_no_observer_is_fine():
    run = _run(None)
    lines = [ln for ln in _lines() if ln.get("message", {}).get("role") != "user"]
    ep, _ = assemble(run, lines, "", _ts(10))
    assert ep.request == [FIRST] and ep.actions == []


def test_learned_credential_never_reaches_the_written_episode(tmp_path, monkeypatch):
    obs = _observer(monkeypatch, T0.timestamp() + 3)
    ep, redactor = assemble(_run(obs), _lines(), "", _ts(10))
    repo = Repo.init_bare(tmp_path / "episodes.git")
    blobs = BlobStore(tmp_path / "blobs")
    sha = EpisodeWriter(repo, blobs, redactor).write(ep)
    history = repo.run("log", "-p", "--all")
    assert TOKEN not in history
    assert "<secret:spotify.login.access_token>" in history
    back = load_episode(repo, sha, f"2026/10/{SID}", blobs)
    assert back.actions[0].response["access_token"] == (
        "<secret:spotify.login.access_token>"
    )
    assert TOKEN not in json.dumps([c.output for c in back.cells])
