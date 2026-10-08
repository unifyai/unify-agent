"""``UNIFY_MEMORY_V2_DIALOGUE`` in the request's memory run (RequestRun, with the other tracks faked).

With ``env`` the episode's actions are the work-tree rows, then the transcript's dialogue actions on
``env``, redacted with the run's redactor. Off (empty or ``off``) the episode gets the work-tree rows
only and the dialogue adapter is never called, as in the base build (9deefbfd1).
"""

from __future__ import annotations

import json

import pytest

from tests.memory_v2.arc_transcript import visit
from tests.memory_v2.integration import fake_tracks
from tests.memory_v2.integration.test_request import (  # noqa: F401 (fixture)
    _begin,
    _finish,
    mv2,
)
from unify.memory_v2.integration import switch
from unify.memory_v2.integration.adapters import dialogue
from unify.settings import SETTINGS


def _write_transcript(run, lines) -> None:
    path = run.paths.home / "transcripts" / f"{run.episode_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))


def test_env_appends_the_dialogue_actions_after_the_worktree_rows(mv2, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_DIALOGUE", "env")
    v = visit()
    run = _begin(mv2, "an ARC visit")
    _write_transcript(run, v["lines"])
    _finish(mv2, run)
    _, kw = mv2.fakes.of("assemble")
    extra = kw["extra_actions"]
    assert extra[0] == fake_tracks.WT_ACTION
    recorded = extra[1:]
    assert recorded == dialogue.dialogue_actions(v["lines"], "env")
    assert [(a.kind, a.channel, a.method, a.status) for a in recorded] == [
        ("dialogue", "env", "request_demos", "ok"),
        ("dialogue", "env", "submit", "ok"),
        ("dialogue", "env", "submit", "ok"),
        ("dialogue", "env", "finish", "unrecorded"),
    ]
    assert not mv2.paths.errors.exists()


def test_env_redacts_with_the_runs_redactor(mv2, monkeypatch):
    secret = "arc-proxy-secret-0123456789abcdef"  # pragma: allowlist secret
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_DIALOGUE", "env")
    monkeypatch.setenv("ARC_PROXY_TOKEN", secret)
    v = visit()
    lines = list(v["lines"])
    lines[3] = {**lines[3], "message": {"role": "user", "content": f"proxy {secret}"}}
    run = _begin(mv2, "an ARC visit")
    _write_transcript(run, lines)
    _finish(mv2, run)
    _, kw = mv2.fakes.of("assemble")
    first, *_rest = [a for a in kw["extra_actions"] if a.kind == "dialogue"]
    assert first.response == "proxy <secret:ARC_PROXY_TOKEN>"


@pytest.mark.parametrize("raw", ["", "off"])
def test_off_records_the_worktree_rows_only(mv2, monkeypatch, raw):
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_MEMORY_V2_DIALOGUE",
        switch.parse_dialogue(raw),
    )

    def never(*_a, **_k):
        raise AssertionError("the dialogue adapter ran with the switch off")

    monkeypatch.setattr(dialogue, "dialogue_actions", never)
    run = _begin(mv2, "an ARC visit")
    _write_transcript(run, visit()["lines"])
    _finish(mv2, run)
    _, kw = mv2.fakes.of("assemble")
    assert kw["extra_actions"] == [fake_tracks.WT_ACTION]
    assert not mv2.paths.errors.exists()
