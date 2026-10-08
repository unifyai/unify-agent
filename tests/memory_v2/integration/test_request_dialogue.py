"""``UNIFY_MEMORY_V2_DIALOGUE`` in the request's memory run (RequestRun, with the other tracks faked).

With ``env`` the episode's actions are the work-tree rows, then the transcript's answered dialogue actions
on ``env``, redacted with the run's redactor: a reply the counterpart never answered (an ARC session's
closing ``finish``, a single-turn office request's final reply) adds nothing, so such an episode's bytes
and experience are those recorded with the setting off. Off (empty or ``off``) the episode gets the
work-tree rows only and the dialogue adapter is never called, as in the base build (9deefbfd1).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.memory_v2.arc_transcript import code_call, line, visit
from tests.memory_v2.integration import fake_tracks
from tests.memory_v2.integration.test_request import (  # noqa: F401 (fixture)
    _begin,
    _finish,
    mv2,
)
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import EpisodeWriter
from unify.memory_v2.experience import experience_tokens
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration import switch
from unify.memory_v2.integration.adapters import dialogue
from unify.memory_v2.integration.trajectory import assemble
from unify.memory_v2.redact import Redactor
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
    assert recorded == dialogue.dialogue_actions(v["lines"], "env", answered_only=True)
    # the closing finish, which nothing answered, is not recorded online
    assert [(a.kind, a.channel, a.method, a.status) for a in recorded] == [
        ("dialogue", "env", "request_demos", "ok"),
        ("dialogue", "env", "submit", "ok"),
        ("dialogue", "env", "submit", "ok"),
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


# --- a single-turn office request: its final reply is never answered -------------------------------------

_FINALS = {
    "text": "I wrote report.md with the three regional totals and the overall growth.",
    "json": 'Done; the report is written.\n{"status": "done", "file": "report.md", "rows": 3}',
}


def _office(final: str) -> list[dict]:
    """One user request, two cells and a final reply with nothing after it (as ``unify act`` records it)."""
    return [
        line(0, "user", "Summarise sales.csv by region into report.md."),
        code_call(
            1,
            "c1",
            "import csv; rows = list(csv.DictReader(open('sales.csv')))",
        ),
        line(2, "tool", "--- stdout ---\n", tool_call_id="c1", name="execute_code"),
        code_call(3, "c2", "open('report.md', 'w').write('# Sales by region')"),
        line(4, "tool", "--- stdout ---\n17\n", tool_call_id="c2", name="execute_code"),
        line(5, "assistant", _FINALS[final]),
    ]


@pytest.mark.parametrize("final", sorted(_FINALS))
def test_env_records_no_dialogue_for_a_single_turn_request(mv2, monkeypatch, final):
    lines = _office(final)
    # the adapter's own semantics keep the reply as unrecorded; the online recorder drops it
    (kept,) = dialogue.dialogue_actions(lines, "env")
    assert kept.status == "unrecorded"
    assert request_mod.recorded_dialogue(lines, "env", Redactor()) == []
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_DIALOGUE", "env")
    run = _begin(mv2, "Summarise sales.csv by region into report.md.")
    _write_transcript(run, lines)
    _finish(mv2, run)
    _, kw = mv2.fakes.of("assemble")
    assert kw["extra_actions"] == [fake_tracks.WT_ACTION]
    assert not mv2.paths.errors.exists()


def _office_run() -> SimpleNamespace:
    return SimpleNamespace(
        episode_id="20261008T130000-0c1d2e3f",
        started_at="2026-10-08T13:00:00+00:00",
        build="memory-v2-final",
        model="openai/gpt-6-luna",
        effort="low",
        pin="0" * 40,
        request="(unused: the transcript holds the request)",
        observer=None,
        costs=None,
    )


def _written(tmp_path, name: str, lines: list[dict], extra: list):
    ep, redactor = assemble(
        _office_run(),
        lines,
        "",
        "2026-10-08T13:05:00+00:00",
        extra_actions=extra,
    )
    repo = Repo.init_bare(tmp_path / name / "episodes.git")
    blobs = tmp_path / name / "blobs"
    sha = EpisodeWriter(repo, BlobStore(blobs), redactor).write(ep)
    tree = repo.run("rev-parse", f"{sha}^{{tree}}").strip()
    stored = sorted(p.name for p in blobs.rglob("*") if p.is_file())
    return ep, tree, stored


@pytest.mark.parametrize("final", sorted(_FINALS))
def test_a_single_turn_episode_is_byte_identical_with_dialogue_off(tmp_path, final):
    """The episode the online path assembles under ``env`` (work-tree rows plus :func:`recorded_dialogue`,
    here none) has the same stored bytes (episode tree and blobs) and experience tokens as with it off.
    """
    lines = _office(final)
    on_extra = [
        fake_tracks.WT_ACTION,
        *request_mod.recorded_dialogue(lines, "env", Redactor()),
    ]
    on, on_tree, on_blobs = _written(tmp_path, "on", lines, on_extra)
    off, off_tree, off_blobs = _written(tmp_path, "off", lines, [fake_tracks.WT_ACTION])
    assert not any(a.kind == "dialogue" for a in on.actions)
    assert on_tree == off_tree and on_blobs == off_blobs
    assert experience_tokens(on) == experience_tokens(off)
