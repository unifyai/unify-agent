"""The episode journal: the request's transcript kept turn by turn, durably, until its episode is recorded.

A request run journals every line of its episode's transcript as the loop publishes it (``unify.transcripts``' tap),
through the redaction the episode record uses, in the harness-only state directory. A request that ends normally
records its episode as before and removes the journal; one that ends unfinished (``abort``) records nothing and
keeps it with why it stopped (a timeout, or an error's class name); a process killed mid-request leaves it; the next
request's start records either as an incomplete episode with every turn that ran, the one that stopped it included,
whole.

The runs are real (``RequestRun.begin``, the real trajectory, episode writer and evidence store) on a seeded
``UNIFY_HOME``; the turns go through the loop's message dispatcher, as the actor's do. No model is called: nothing
here runs a consolidation pass. The killed request runs in a child process that kills itself with SIGKILL once its
last turn is published.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.memory.integration.test_checkout import _seed
from unify import sandbox, transcripts
from unify.memory.blobs import BlobStore
from unify.memory.episodes import episode_dir, load_episode
from unify.memory.evidence import EvidenceStore
from unify.memory.gitio import Repo
from unify.memory.integration import journal as journal_mod
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.settings import SETTINGS

ROOT = Path(__file__).resolve().parents[3]
REQUEST = "Add up 6 and 7, then multiply by 6."
FAKE_KEY = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret
#: A runaway reply: 22 MiB of text in one model reply, which the record keeps whole.
RUNAWAY_BYTES = 22 * 1024 * 1024


def _call(cid: str, code: str) -> dict:
    function = {"name": "execute_code", "arguments": json.dumps({"code": code})}
    return {"id": cid, "type": "function", "function": function}


#: The request's turns: the request, a reply that runs a cell, the cell's result (it prints a key-shaped string no
#: environment variable holds, which only the record's redaction removes), and the reply after it.
TURNS = [
    {"role": "user", "content": REQUEST},
    {"role": "assistant", "content": None, "tool_calls": [_call("c0", "print((6 + 7) * 6)")]},
    {"role": "tool", "tool_call_id": "c0", "content": f"78\nfound {FAKE_KEY}\n"},
    {"role": "assistant", "content": "The answer is 78."},
]


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A ``UNIFY_HOME`` with a seeded memory and the switch on; a context of the run's own."""
    root = tmp_path / "unify"
    root.mkdir()
    monkeypatch.setenv("UNIFY_HOME", str(root))
    monkeypatch.delenv("UNIFY_STORE_PATH", raising=False)
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    _seed(root)
    return SimpleNamespace(root=root, paths=Paths.under(root), ctx=contextvars.copy_context())


class _Timer:
    def reset(self) -> None:
        pass


class _Client:
    system_message = "You are the actor."

    def __init__(self) -> None:
        self.messages: list = []

    def append_messages(self, msgs: list) -> None:
        self.messages += msgs


def _publish(env, turns: list[dict], after=None) -> None:
    """The turns, published one at a time as the actor's loop publishes them (its message dispatcher, whose
    transcript session continues the episode), with *after* called after each."""
    from unify.common._async_tool.loop_config import LoopConfig
    from unify.common._async_tool.message_dispatcher import LoopMessageDispatcher

    async def main() -> None:
        d = LoopMessageDispatcher(_Client(), LoopConfig("Actor", None, []), _Timer())
        for turn in turns:
            await d.append_msgs([dict(turn)])
            if after is not None:
                after()

    with asyncio.Runner() as runner:
        runner.run(main(), context=env.ctx)


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _journals(paths: Paths) -> list[Path]:
    return sorted(paths.journal.glob("*.jsonl")) if paths.journal.is_dir() else []


def _recorded(paths: Paths, eid: str):
    repo = Repo(paths.episodes)
    names = repo.run("ls-tree", "-r", "--name-only", "main").split()
    (meta,) = [n for n in names if n.endswith(f"/{eid}/meta.json")]
    rel = meta.rsplit("/", 1)[0]
    return repo, rel, load_episode(repo, "main", rel, BlobStore(paths.blobs))


def test_the_journal_lives_where_no_cell_or_box_reads(tmp_path):
    paths = Paths.under(tmp_path)
    assert paths.journal == paths.state_dir / "episode-journal"
    assert paths.state_dir in paths.harness_only()


def test_each_turn_is_journaled_as_published_and_a_finished_episode_is_recorded_as_before(home):
    run = home.ctx.run(request_mod.RequestRun.begin, REQUEST)
    try:
        path = home.paths.journal / f"{run.episode_id}.jsonl"
        header = _lines(path)
        assert len(header) == 1 and header[0]["type"] == journal_mod.HEADER
        assert header[0]["episode_id"] == run.episode_id and header[0]["request"] == REQUEST
        assert header[0]["memory_main"] == run.pin and header[0]["started_at"] == run.started_at
        transcript = transcripts.transcripts_dir() / f"{run.episode_id}.jsonl"
        seen: list[int] = []
        # each turn is on disk in the journal as soon as it is published
        _publish(home, TURNS, after=lambda: seen.append(len(_lines(path)) - 1))
        assert seen == [n + 2 for n in range(len(TURNS))]  # the session's start line, then one per turn
        journaled = _lines(path)[1:]
        written = _lines(transcript)
        assert [r["seq"] for r in journaled] == [r["seq"] for r in written]
        # the transcript keeps the key-shaped string; the journal has it redacted, as the record will
        assert FAKE_KEY in transcript.read_text() and FAKE_KEY not in path.read_text()
        assert "<redacted:key-shaped>" in path.read_text()
        progress: list[str] = []
        with asyncio.Runner() as runner:
            runner.run(run.finish(None, progress=progress.append, consolidate=False), context=home.ctx)
    finally:
        home.ctx.run(run.abort)
    # recorded as before: no mark, the transcript as the episode record always held it; the journal is gone
    assert not path.exists() and _journals(home.paths) == []
    repo, rel, ep = _recorded(home.paths, run.episode_id)
    assert ep.incomplete is None
    assert "incomplete" not in json.loads(repo.show("main", f"{rel}/meta.json"))
    recorded = _lines_of(repo.show("main", f"{rel}/transcript.jsonl"))
    assert recorded == journaled  # the journal held exactly the record's lines
    assert ep.request == [REQUEST] and [c.output for c in ep.cells] == ["78\nfound <redacted:key-shaped>\n"]
    assert not home.paths.errors.exists(), home.paths.errors.read_text()


def _lines_of(data: bytes) -> list[dict]:
    return [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]


@pytest.mark.parametrize(
    "stopped, recorded",
    [
        ({"reason": "timeout"}, {"reason": "timeout"}),
        ({"reason": "error", "error": "RuntimeError"}, {"reason": "error", "error": "RuntimeError"}),
        ({"reason": "error", "error": "not a class name: some text"}, {"reason": "error"}),
        (None, {"reason": journal_mod.UNKNOWN_REASON}),
    ],
)
def test_an_aborted_request_keeps_its_journal_and_the_next_start_records_it_incomplete(home, stopped, recorded):
    run = home.ctx.run(request_mod.RequestRun.begin, REQUEST)
    head = Repo(home.paths.episodes).head()
    _publish(home, TURNS)
    if stopped is None:
        home.ctx.run(run.abort)
    else:
        home.ctx.run(lambda: run.abort(stopped=stopped))
    # nothing is recorded now; the journal keeps every turn, and why the request stopped
    assert Repo(home.paths.episodes).head() == head
    (left,) = _journals(home.paths)
    assert left.name == f"{run.episode_id}.jsonl"
    assert "some text" not in left.read_text()
    again = home.ctx.run(request_mod.RequestRun.begin, "the next request")
    home.ctx.run(again.abort)
    assert again.recovered == [run.episode_id]
    _, _, ep = _recorded(home.paths, run.episode_id)
    assert {k: v for k, v in ep.incomplete.items() if k != "recovered_at"} == recorded
    assert [r["message"]["role"] for r in ep.transcript if r.get("type") == "message"] == [t["role"] for t in TURNS]
    assert not [r for r in ep.transcript if r.get("type") == journal_mod.STOPPED]  # the stop line is meta only
    assert [c.output for c in ep.cells] == ["78\nfound <redacted:key-shaped>\n"]


#: A request killed mid-episode: it publishes the turns, the last one a runaway reply, then kills itself.
_KILLED = textwrap.dedent(
    """
    import asyncio, json, os, signal, sys
    from unify.settings import SETTINGS
    SETTINGS.UNIFY_LOCAL_ROOT = ""
    SETTINGS.UNIFY_MEMORY = "on"
    from unify.common._async_tool.loop_config import LoopConfig
    from unify.common._async_tool.message_dispatcher import LoopMessageDispatcher
    from unify.memory.integration import request as request_mod

    class Timer:
        def reset(self):
            pass

    class Client:
        system_message = "You are the actor."
        def __init__(self):
            self.messages = []
        def append_messages(self, msgs):
            self.messages += msgs

    turns = json.loads(sys.argv[1])
    runaway = {"role": "assistant", "content": "ab" * (int(sys.argv[2]) // 2),
               "tool_calls": [{"id": "c1", "type": "function",
                               "function": {"name": "execute_code", "arguments": json.dumps({"code": "x = 1"})}}]}
    run = request_mod.RequestRun.begin(turns[0]["content"])

    async def main():
        d = LoopMessageDispatcher(Client(), LoopConfig("Actor", None, []), Timer())
        for turn in turns + [runaway]:
            await d.append_msgs([turn])

    asyncio.run(main())
    print(json.dumps({"episode_id": run.episode_id, "started_at": run.started_at, "lines": run.journal.lines}),
          flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
    """,
)


@pytest.mark.timeout(600)
def test_a_request_killed_mid_episode_is_recorded_incomplete_with_every_turn_at_the_next_start(home):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    child = subprocess.run(
        [sys.executable, "-c", _KILLED, json.dumps(TURNS[:3]), str(RUNAWAY_BYTES)],
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == -signal.SIGKILL, child.stderr[-3000:]
    dead = json.loads(child.stdout.strip().splitlines()[-1])
    eid = dead["episode_id"]
    (left,) = _journals(home.paths)
    assert left.name == f"{eid}.jsonl"
    # the session's start line and every turn, the runaway reply included, were on disk when the process died
    assert dead["lines"] == 1 + len(TURNS[:3]) + 1
    episodes_before = Repo(home.paths.episodes).head()

    run = home.ctx.run(request_mod.RequestRun.begin, "the next request")
    try:
        assert run.recovered == [eid]
        assert _journals(home.paths) == [home.paths.journal / f"{run.episode_id}.jsonl"]  # only the new one
    finally:
        home.ctx.run(run.abort)
    assert _journals(home.paths) == [home.paths.journal / f"{run.episode_id}.jsonl"]  # the aborted one keeps its own
    repo, rel, ep = _recorded(home.paths, eid)
    assert repo.head() != episodes_before
    # marked incomplete, the reason unknown; its meta as its process opened it
    assert ep.incomplete["reason"] == journal_mod.UNKNOWN_REASON and ep.incomplete["recovered_at"]
    assert json.loads(repo.show("main", f"{rel}/meta.json"))["incomplete"] == ep.incomplete
    assert ep.started_at == dead["started_at"] and rel == episode_dir(ep)
    assert ep.request == [REQUEST]
    # every turn that ran, in order, the one that stopped it whole
    messages = [r["message"] for r in ep.transcript if r.get("type") == "message"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[3]["content"] == "ab" * (RUNAWAY_BYTES // 2)
    assert FAKE_KEY not in json.dumps(ep.transcript) and "<redacted:key-shaped>" in messages[2]["content"]
    # the cell that finished is a cell; the runaway reply's call never got a result
    assert [c.output for c in ep.cells] == ["78\nfound <redacted:key-shaped>\n"]
    sha = repo.head()
    assert EvidenceStore(home.paths.evidence).episode_ref(eid)[0] == sha


def test_a_journal_without_its_header_is_kept_aside_and_never_retried(home):
    home.paths.journal.mkdir(parents=True)
    torn = home.paths.journal / "20261014T000000-deadbeef.jsonl"
    torn.write_text('{"type": "mess')
    run = home.ctx.run(request_mod.RequestRun.begin, REQUEST)
    home.ctx.run(run.abort)
    assert run.recovered == [] and not torn.exists()
    assert (home.paths.journal / (torn.name + journal_mod.UNRECORDED)).read_text() == '{"type": "mess'


def test_an_episode_that_could_not_be_recorded_keeps_its_journal_for_the_next_start(home, monkeypatch):
    from unify.memory import episodes

    real = episodes.EpisodeWriter.write
    failed: list[str] = []

    def write_once_broken(self, ep):
        if not failed:
            failed.append(ep.episode_id)
            raise RuntimeError("the episodes repo is unavailable")
        return real(self, ep)

    monkeypatch.setattr(episodes.EpisodeWriter, "write", write_once_broken)
    run = home.ctx.run(request_mod.RequestRun.begin, REQUEST)
    try:
        _publish(home, TURNS)
        with asyncio.Runner() as runner:
            runner.run(run.finish(None, progress=lambda _t: None, consolidate=False), context=home.ctx)
    finally:
        home.ctx.run(run.abort)  # after finish: changes nothing
    assert failed == [run.episode_id]
    assert [p.name for p in _journals(home.paths)] == [f"{run.episode_id}.jsonl"]
    assert [json.loads(x)["stage"] for x in home.paths.errors.read_text().splitlines()] == ["episode"]

    again = home.ctx.run(request_mod.RequestRun.begin, "the next request")
    home.ctx.run(again.abort)
    assert again.recovered == [run.episode_id]
    assert _journals(home.paths) == [home.paths.journal / f"{again.episode_id}.jsonl"]  # the aborted one's own
    _, _, ep = _recorded(home.paths, run.episode_id)
    assert ep.incomplete["reason"] == journal_mod.UNKNOWN_REASON
    assert [r["message"]["role"] for r in ep.transcript if r.get("type") == "message"] == [
        t["role"] for t in TURNS
    ]
