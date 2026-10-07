"""Symbolic: a checker's outcome never reaches anything a later task's cell can read.

The storage review of a session reads the environment's checked outcome in a
section of its own (``test_outcome_channel.py``). Model code must stay
isolated from the checker and its output, so the review's session, and every
other harness-internal session (the review fork, the review gate), is
transcribed to ``<UNIFY_HOME>/internal-transcripts/``, which no cell can
reach, never to the cell-readable ``transcripts/``. Every transcript line also
has the outcome section the harness built replaced by ``[REDACTED:outcome]``.

Task 1 is a scripted persistent session whose outcome is posted, so its review
sees it. Task 2 is a cell in the real sandboxed worker (bubblewrap), which
searches everything the sandbox mounts for it -- the workspace, the
transcripts, the store, the venv, the state directory as it sees it, the
home -- its own environment and ``/proc/self/environ``, and then opens the
review's transcript by its absolute path.
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
from pathlib import Path

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    needs_bwrap,
    world,
)
from tests.actor.code_act.test_outcome_channel import FAILED, _persistent_review
from unify import db, outcome as outcome_mod, sandbox, transcripts
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import parts_to_text
from unify.settings import SETTINGS

MARKER = "outcome-marker-c41d07"
OUTCOME = {**FAILED, "summary": MARKER}
# What the checker wrote, as the review read it: the summary, a check's
# reason and the section's header.
NEEDLES = (MARKER, "no email to Kim", outcome_mod.OUTCOME_HEADER)

# Run in the worker: every needle found in what the cell can read.
PROBE = r"""
import json, os
needles = json.loads({needles!r})
roots = json.loads({roots!r})
found = []
for name, value in os.environ.items():
    for n in needles:
        if n in value:
            found.append(("env", name, n))
try:
    with open("/proc/self/environ", "rb") as fh:
        raw = fh.read()
    for n in needles:
        if n.encode() in raw:
            found.append(("/proc/self/environ", "", n))
except OSError:
    pass
scanned = 0
for root in roots:
    for dirpath, dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > 20_000_000:
                    continue
                with open(full, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            scanned += 1
            for n in needles:
                if n.encode() in data:
                    found.append(("file", full, n))
print(json.dumps({{"found": found, "scanned": scanned}}))
"""

OPEN = r"""
import json, os
target = {target!r}
directory = os.path.dirname(target)
out = {{}}
try:
    with open(target, "rb") as fh:
        out["read"] = fh.read().decode("utf-8", "replace")[:200]
except OSError as exc:
    out["error"] = type(exc).__name__
try:
    out["listing"] = sorted(os.listdir(directory))
except OSError as exc:
    out["listing_error"] = type(exc).__name__
print(json.dumps(out))
"""


@pytest.fixture
def outcome_world(world, monkeypatch):  # noqa: F811
    """The sandbox world with worker Python, transcripts on and a real store."""
    (world["state"] / "store.sqlite").unlink()
    db.reset_store()
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)
    yield world
    db.reset_store()


def _switch(monkeypatch, review: str) -> None:
    fork = review == "fork"
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", fork)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", fork)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", "")


def _sessions(directory: Path) -> list[dict]:
    """The ``session_start`` line of every session file in *directory*."""
    out = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.name == "index.jsonl":
            continue
        first = json.loads(path.read_text().splitlines()[0])
        out.append({**first, "path": str(path)})
    return out


def _review_sessions(directory: Path) -> list[dict]:
    return [
        row
        for row in _sessions(directory)
        if str(row.get("origin", "")).startswith("StorageCheck")
    ]


async def _task_one(monkeypatch, review: str) -> list[dict]:
    """A scripted session, its posted outcome and its review, ended."""
    _switch(monkeypatch, review)
    _note, requests, _handle, posted = await _persistent_review(outcome=OUTCOME)
    assert posted is not None and posted["summary"] == MARKER
    # the review read the outcome
    review_request, session_request = requests[3], requests[2]
    assert MARKER in json.dumps(review_request, default=str)
    if review == "fork":
        # a fork: the session's own requests are its prefix
        sent = session_request["messages"]
        assert review_request["messages"][: len(sent)] == sent
    else:
        assert "## Completed Trajectory" in review_request["messages"][0]["content"]
    return requests


async def _cell(code: str) -> dict:
    """Task 2: one cell in a fresh sandboxed worker."""
    ex = SessionExecutor(environments={})
    try:
        res = await ex.execute(code=code, state_mode="stateful", session_id=0)
        assert res["error"] is None, res["error"]
        assert ex.python_session(session_id=0)._worker is not None
        return json.loads(parts_to_text(res["stdout"]).strip().splitlines()[-1])
    finally:
        await ex.close()


def _mounts(world) -> list[str]:  # noqa: F811
    """Every path the policy mounts into a cell, plus the state dir and home."""
    policy = sandbox.build_policy(fresh=True)
    return [
        str(p)
        for p in (
            policy.workspace,
            *policy.readonly_state,
            policy.state_dir,
            world["home"],
        )
    ]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("review", ["standalone", "fork"])
async def test_a_later_cell_finds_no_outcome_anywhere_it_can_read(
    outcome_world,
    monkeypatch,
    review,
):
    await _task_one(monkeypatch, review)
    roots = _mounts(outcome_world)
    report = await _cell(
        PROBE.format(needles=json.dumps(NEEDLES), roots=json.dumps(roots)),
    )
    assert report["scanned"] > 0
    assert report["found"] == [], report["found"]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("workspace", ["default", "the state directory"])
async def test_a_later_cell_cannot_open_the_internal_transcripts(
    outcome_world,
    monkeypatch,
    workspace,
):
    if workspace == "the state directory":
        # The widest workspace a deployment can configure: UNIFY_HOME itself,
        # a parent of internal-transcripts, bound writable.
        monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", str(outcome_world["state"]))
        monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    await _task_one(monkeypatch, "standalone")
    internal = outcome_world["state"] / "internal-transcripts"
    sessions = _review_sessions(internal)
    assert sessions, f"no review session indexed in {internal}"
    target = sessions[0]["path"]
    assert Path(target).is_file()
    report = await _cell(OPEN.format(target=target))
    assert "read" not in report, report
    assert report.get("error") in ("FileNotFoundError", "PermissionError"), report
    assert Path(target).name not in report.get("listing", []), report


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("review", ["standalone", "fork"])
async def test_the_internal_transcripts_keep_the_review_session(
    unify_home,
    monkeypatch,
    review,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)
    await _task_one(monkeypatch, review)
    internal, readable = unify_home / "internal-transcripts", unify_home / "transcripts"
    assert transcripts.internal_transcripts_dir() == internal
    sessions = _review_sessions(internal)
    assert len(sessions) == 1, sessions
    text = Path(sessions[0]["path"]).read_text()
    assert "## Final Result" in text
    # the outcome section is kept out even here
    assert "[REDACTED:outcome]" in text
    assert all(n not in text for n in NEEDLES)
    # the task's own session stays where its cells can read it; no review does
    assert _review_sessions(readable) == []
    task = [r for r in _sessions(readable) if r["origin"] == "CodeActActor.act"]
    assert len(task) == 1
    assert Path(task[0]["path"]).parent == readable


def test_the_policy_mounts_nothing_that_holds_the_internal_transcripts(
    unify_home,
    monkeypatch,
    tmp_path,
):
    """The full mount list, for the default workspace and for UNIFY_HOME itself."""
    from unify import environment

    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", "sandboxed")
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)
    internal = Path(os.path.realpath(unify_home)) / "internal-transcripts"
    for local_root in ("", str(unify_home)):
        monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", local_root)
        policy = sandbox.build_policy(fresh=True)
        assert policy.readable_violation(internal) is not None
        assert policy.readable_violation(internal / "x.jsonl") is not None
        if not sys.platform.startswith("linux") or sandbox.bwrap_path() is None:
            continue
        argv = sandbox.wrap_argv(
            ["true"],
            policy,
            writable=[environment.environment_dir(), environment.installer_cache()],
        )
        # Replay the mounts in order: what is visible at internal-transcripts
        # once every mount is made.
        visible = False
        i = 1
        while argv[i] != "--chdir":
            flag = argv[i]
            if flag in ("--ro-bind", "--bind"):
                dest = Path(argv[i + 2])
                if internal == dest or internal.is_relative_to(dest):
                    visible = True
                i += 3
            elif flag == "--tmpfs":
                dest = Path(argv[i + 1])
                if internal == dest or internal.is_relative_to(dest):
                    visible = False
                i += 2
            elif flag in ("--dev", "--proc"):
                i += 2
            else:
                i += 1
        assert not visible, (local_root, argv)


def test_every_transcript_line_drops_the_outcome_section_the_harness_built(
    unify_home,
    monkeypatch,
):
    """Keyed on the note the harness rendered, never on words in model text."""
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)
    note = outcome_mod.render(outcome_mod.normalize(OUTCOME))
    client = h.new_client("system")
    cfg = type("Cfg", (), {"label": "probe", "loop_id": "probe"})()
    # In a context of its own, so no later test's session counts it a parent.
    session = contextvars.copy_context().run(transcripts.attach, client, cfg)
    assert session is not None and session.path.parent == unify_home / "transcripts"
    session.observe(
        [
            {"role": "user", "content": f"before\n{note}after"},
            # the model's own words about the outcome are not the note
            {"role": "assistant", "content": f"I saw {MARKER}; no email to Kim."},
        ],
    )
    text = session.path.read_text()
    assert "[REDACTED:outcome]" in text
    assert outcome_mod.OUTCOME_HEADER not in text
    assert "before\\n[REDACTED:outcome]after" in text
    assert f"I saw {MARKER}; no email to Kim." in text
