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
transcripts, the venv, the state directory as it sees it, the home -- and the
store file's path (not mounted: cells use the functions/guidance API), its own
environment and ``/proc/self/environ``, and then opens the review's transcript
by its absolute path.

Every session's transcript is readable from a cell, other sessions' and other
agents' too, by the lead's design (8 Oct); what keeps the outcome out is the
redaction and ``internal-transcripts/``, never a narrower mount.
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import uuid
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

# Fresh per process: logs of earlier runs cannot match it.
MARKER = f"outcome-marker-{uuid.uuid4().hex[:12]}"
# The review reads the verdict only: the marker is in the source it names,
# and in the summary, which it does not read.
OUTCOME = {**FAILED, "source": MARKER, "summary": MARKER}
# What the checker wrote: the marker, a check's reason and the section's
# header.
NEEDLES = (MARKER, "no email to Kim", outcome_mod.OUTCOME_HEADER)

# Run in the worker: for each place, what was read and every needle found.
PROBE = r"""
import json, os
needles = json.loads({needles!r})
roots = json.loads({roots!r})
pointer = {pointer!r}

def hits(data):
    return [n for n in needles if n.encode() in data]

report = {{"env": [], "roots": {{}}}}
for name, value in os.environ.items():
    report["env"] += [[name, n] for n in needles if n in value]
try:
    with open("/proc/self/environ", "rb") as fh:
        report["proc_environ"] = hits(fh.read())
except OSError as exc:
    report["proc_environ"] = type(exc).__name__
for label, root in roots.items():
    seen = {{"files": 0, "found": []}}
    if os.path.isfile(root):
        walk = [(os.path.dirname(root), [], [os.path.basename(root)])]
    else:
        seen["missing"] = not os.path.isdir(root)
        walk = os.walk(root)
    for dirpath, dirs, files in walk:
        for name in files:
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > 20_000_000:
                    continue
                with open(full, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            seen["files"] += 1
            seen["found"] += [[full, n] for n in hits(data)]
    report["roots"][label] = seen
try:
    with open(pointer, "rb") as fh:
        data = fh.read()
    report["pointer"] = {{"bytes": len(data), "found": hits(data)}}
except OSError as exc:
    report["pointer"] = {{"error": type(exc).__name__}}
print(json.dumps(report))
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
    yield world
    db.reset_store()


def _switch(monkeypatch, review: str) -> None:
    """The review forks the session when it can; ``standalone`` refuses the
    fork, as a compressed session or unanswered calls do."""
    from unify.actor import code_act_actor as caa

    if review == "standalone":
        monkeypatch.setattr(
            caa,
            "_review_fork_source",
            lambda inner, actor: (None, "the test refuses the fork"),
        )
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


def _places(world) -> dict[str, str]:  # noqa: F811
    """Every path the policy mounts into a cell, the internal transcripts, every
    configured log directory, the state directory as a whole and the home."""
    policy = sandbox.build_policy(fresh=True)
    state = policy.state_dir
    places = {
        "workspace": policy.workspace,
        **{
            f"mounted {p.relative_to(state) if p.is_relative_to(state) else p}": p
            for p in policy.readonly_state
        },
        "internal-transcripts": state / "internal-transcripts",
        # Not mounted; searched by its path all the same.
        "store.sqlite": Path(db.store_path()),
        "state directory (UNIFY_HOME)": state,
        "home": world["home"],
    }
    for name, value in zip(sandbox.LOG_DIR_SETTINGS, sandbox._log_dir_settings()):
        if value:
            places[name] = value
    return {k: str(v) for k, v in places.items()}


def _has_needle(directory: str) -> bool:
    for dirpath, _dirs, files in os.walk(directory):
        for name in files:
            try:
                data = Path(dirpath, name).read_bytes()
            except OSError:
                continue
            if MARKER.encode() in data:
                return True
    return False


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
    places = _places(outcome_world)
    readable = outcome_world["state"] / "transcripts"
    task = [r for r in _sessions(readable) if r.get("origin") == "CodeActActor.act"]
    assert len(task) == 1
    # The file a compaction pointer names for the task's session.
    pointer = task[0]["path"]
    # The outcome is on the host where only the harness reads it: the LLM
    # request log, which the cell must not see.
    llm_log = places.get("UNILLM_LOG_DIR")
    assert llm_log and _has_needle(llm_log), "the review's request was not logged"
    report = await _cell(
        PROBE.format(
            needles=json.dumps(NEEDLES),
            roots=json.dumps(places),
            pointer=pointer,
        ),
    )
    print(json.dumps(report, indent=1))
    assert report["env"] == [] and report["proc_environ"] in ([], "PermissionError")
    found = {k: v["found"] for k, v in report["roots"].items() if v["found"]}
    assert found == {}, found
    assert report["roots"]["mounted transcripts"]["files"] > 0
    # The store is on the host (task 1 wrote it) and absent in the cell.
    assert Path(places["store.sqlite"]).is_file()
    assert not any(
        "store.sqlite" in label for label in places if label != "store.sqlite"
    )
    assert report["roots"]["store.sqlite"]["files"] == 0
    assert report["roots"]["workspace"]["files"] >= 0
    # The task's own transcript stays readable, and holds none of it.
    assert report["pointer"].get("bytes", 0) > 0, report["pointer"]
    assert report["pointer"]["found"] == []
    # What the harness keeps hidden reads as nothing at all.
    assert report["roots"]["internal-transcripts"]["files"] == 0
    assert report["roots"]["UNILLM_LOG_DIR"]["files"] <= 1  # the mask's notice


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


# Run in the worker: another session's transcript, everything under the
# mounted transcripts/ searched for the rendered outcome section, and the
# internal transcripts.
ACROSS = r"""
import json, os
other = {other!r}
transcripts = {transcripts!r}
internal = {internal!r}
review = {review!r}
forms = [f.encode() for f in json.loads({forms!r})]
out = {{"files": 0, "found": []}}
with open(other, "rb") as fh:
    out["other"] = fh.read().decode("utf-8", "replace")
for dirpath, _dirs, files in os.walk(transcripts):
    for name in files:
        with open(os.path.join(dirpath, name), "rb") as fh:
            data = fh.read()
        out["files"] += 1
        out["found"] += [name for f in forms if f in data]
out["internal_exists"] = os.path.exists(internal)
try:
    with open(review, "rb") as fh:
        out["review"] = len(fh.read())
except OSError as exc:
    out["review_error"] = type(exc).__name__
out["state_listing"] = sorted(os.listdir(os.path.dirname(transcripts)))
print(json.dumps(out))
"""


def _outcome_forms(section: str) -> list[str]:
    """*section* raw and JSON-escaped once and twice: the exact renderings the
    redaction keys on (unify/outcome.py), never words of it."""
    once = json.dumps(section, ensure_ascii=False)[1:-1]
    twice = json.dumps(once, ensure_ascii=False)[1:-1]
    return list(dict.fromkeys((section, once, twice)))


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_a_cell_reads_other_sessions_transcripts_and_no_outcome_in_them(
    outcome_world,
    monkeypatch,
):
    """By the lead's design (8 Oct) a cell can read every session's transcript,
    another session's included; the outcome stays out of all of them."""
    # One internal (review) session that carries the outcome.
    await _task_one(monkeypatch, "standalone")
    state = outcome_world["state"]
    readable, internal = state / "transcripts", state / "internal-transcripts"
    reviews = _review_sessions(internal)
    assert reviews, f"no review session indexed in {internal}"
    review = reviews[0]["path"]
    # Another session, whose transcript went through the real redaction path
    # while a message carried the rendered outcome section.
    note = outcome_mod.render(outcome_mod.normalize(OUTCOME))
    assert outcome_mod.OUTCOME_HEADER in note and MARKER in note
    client = h.new_client("system")
    cfg = type("Cfg", (), {"label": "other", "loop_id": "other"})()
    other = contextvars.copy_context().run(transcripts.attach, client, cfg)
    assert other is not None and other.path.parent == readable
    other.observe([{"role": "user", "content": f"before\n{note}after"}])
    host_text = other.path.read_text()
    assert "before\\n[REDACTED:outcome]after" in host_text
    own = [r for r in _sessions(readable) if r.get("origin") == "CodeActActor.act"]
    assert len(own) == 1 and own[0]["path"] != str(other.path)
    forms = _outcome_forms(note)
    report = await _cell(
        ACROSS.format(
            other=str(other.path),
            transcripts=str(readable),
            internal=str(internal),
            review=review,
            forms=json.dumps(forms),
        ),
    )
    # 1. The other session's transcript reads in full.
    assert report["other"] == host_text
    # 2. Nothing under transcripts/ holds the rendered section, in any form.
    assert report["files"] >= 2 and report["found"] == [], report
    # 3. The internal transcripts do not exist for the cell.
    assert report["internal_exists"] is False, report
    assert report.get("review_error") in ("FileNotFoundError", "PermissionError")
    assert "review" not in report, report
    assert "internal-transcripts" not in report["state_listing"], report


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("review", ["standalone", "fork"])
async def test_the_internal_transcripts_keep_the_review_session(
    unify_home,
    monkeypatch,
    review,
):
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
    task = [r for r in _sessions(readable) if r.get("origin") == "CodeActActor.act"]
    assert len(task) == 1
    assert Path(task[0]["path"]).parent == readable


def _visible(argv: list[str], path: Path) -> bool:
    """Replay the bubblewrap mounts in order: whether *path* shows the host's
    content once every mount is made."""
    visible, i = False, 1
    while argv[i] != "--chdir":
        flag = argv[i]
        if flag in ("--ro-bind", "--bind", "--tmpfs"):
            dest = Path(argv[i + (1 if flag == "--tmpfs" else 2)])
            if path == dest or path.is_relative_to(dest):
                visible = flag != "--tmpfs"
            i += 2 if flag == "--tmpfs" else 3
        elif flag in ("--dev", "--proc"):
            i += 2
        else:
            i += 1
    return visible


@pytest.mark.parametrize("log_dir", ["outside", "inside the workspace"])
def test_the_policy_mounts_nothing_that_holds_internal_transcripts_or_logs(
    unify_home,
    monkeypatch,
    tmp_path,
    log_dir,
):
    """The full mount list, for the default workspace and for UNIFY_HOME itself,
    with the LLM request log outside or inside the workspace."""
    from unify import environment

    home = Path(os.path.realpath(unify_home))
    internal = home / "internal-transcripts"
    for local_root in ("", str(home)):
        monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", local_root)
        workspace = Path(local_root) if local_root else home / "workspace"
        logs = (
            Path(os.path.realpath(tmp_path)) / "llm-logs"
            if log_dir == "outside"
            else workspace / "llm-logs"
        )
        monkeypatch.setenv("UNILLM_LOG_DIR", str(logs))
        monkeypatch.setenv("UNIFY_OTEL_LOG_DIR", str(logs.parent / "otel"))
        policy = sandbox.build_policy(fresh=True)
        hidden = (internal, logs, logs.parent / "otel")
        for path in hidden:
            assert policy.readable_violation(path) is not None, path
            assert policy.readable_violation(path / "x.jsonl") is not None, path
        assert policy.readable_violation(workspace / "data.txt") is None
        assert policy.readable_violation(home / "transcripts" / "s.jsonl") is None
        if not sys.platform.startswith("linux") or sandbox.bwrap_path() is None:
            continue
        if log_dir == "inside the workspace":
            # A workspace that holds a log directory is refused as a whole
            # (the workspace kind, _workspace_refusal), before anything runs.
            with pytest.raises(sandbox.SandboxRefusal) as raised:
                sandbox.wrap_argv(["true"], policy)
            assert raised.value.rule == "root-allowlist"
            continue
        argv = sandbox.wrap_argv(
            ["true"],
            policy,
            writable=[environment.environment_dir(), environment.installer_cache()],
        )
        for path in hidden:
            assert not _visible(argv, path), (local_root, path, argv)
        assert _visible(argv, workspace / "data.txt"), argv
        assert _visible(argv, home / "transcripts"), argv


def test_every_transcript_line_drops_the_outcome_section_the_harness_built(
    unify_home,
    monkeypatch,
):
    """Keyed on the note the harness rendered, never on words in model text."""
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
