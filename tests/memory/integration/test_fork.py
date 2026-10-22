"""The episode-end fork. It runs whenever memory is on (rev 2 has no fork switch), after the episode is recorded,
through the key proxy's fork lane, started by the run record's relay in a box whose only writable mount is the
episode's staging dir. Nothing flows back, and the worker stages only the allowed files, from text and tool-call
replies, executing nothing.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory.integration import fork as fork_mod
from unify.memory.integration import fork_worker as fw
from tests.memory.integration.test_request import (  # noqa: F401
    _begin,
    _finish,
    _transcript,
    mv2,
)

BASE = "http://127.0.0.1:18080/fork/v1"
TOKEN = "fork-token-0123456789abcdef"


def _fresh_lane_state(monkeypatch):
    from unify.memory.integration import switch as sw

    # the fork's lane read afresh in this test (a lane's token is read once per process)
    monkeypatch.setattr(sw, "_FORK", sw._Lane(sw.LIBRARY_FORK_BASE_URL, sw.LIBRARY_FORK_TOKEN_FD, sw.LIBRARY_FORK_TOKEN,
                                              "fork", "no fork lane"))
    return sw


def _token_fd(text=TOKEN + "\n", inheritable=True):
    r, w = os.pipe()
    os.write(w, text.encode())
    os.close(w)
    os.set_inheritable(r, inheritable)
    return r


@pytest.fixture
def lane(monkeypatch):
    """The fork lane as the controller gets it: the base URL and an inherited pipe with the token,
    settled once; both names leave the environment."""
    sw = _fresh_lane_state(monkeypatch)
    fd = _token_fd()
    env = {sw.LIBRARY_FORK_BASE_URL: BASE, sw.LIBRARY_FORK_TOKEN_FD: str(fd)}
    assert sw.settle_fork_route_env(env, SimpleNamespace(**env)) is None
    assert env == {}
    return TOKEN


CODE = 'def double(x):\n    """Twice x."""\n    return 2 * x\n'


def _fence(rel: str, body: str) -> str:
    return f"```python\npath: {rel}\n{body}```\n"


def _source(messages=None):
    client = SimpleNamespace(
        endpoint="openai/gpt-6-luna@openrouter",
        reasoning_effort="low",
    )
    sent = messages or [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "double 21"},
        {"role": "assistant", "content": "42"},
    ]
    return {
        "client": client,
        "messages": copy.deepcopy(sent),
        "sent_messages": sent,
        "tools": [
            {
                "type": "function",
                "function": {"name": "execute_code", "parameters": {}},
            },
        ],
        "tool_choice": "auto",
    }


class _Pipe(io.BytesIO):
    def close(self):  # keep the bytes readable after the writer closes the pipe
        self.closed_by_writer = True


class _Proc:
    """A fake fork relay: on wait, it does what the relay does with its start file (the box's argv and the fork's
    request), running the real worker logic in-process against a scripted proxy on the box's staging bind."""

    def __init__(self, argv, replies, seen, **kw):
        self.argv, self.kw, self.replies, self.seen = argv, kw, list(replies), seen
        self.stdin, self.pid, self.returncode = _Pipe(), 4242, None

    def wait(self, timeout=None):
        self.spec = json.loads(self.stdin.getvalue().decode())
        box = self.spec["argv"]
        host = Path(box[box.index("--bind") + 1])
        req = self.spec["doc"]

        def post(url, headers, body):
            self.seen.append((url, headers, body))
            return self.replies.pop(0)

        fw.run(req, host, post=post)
        self.returncode = 0
        return 0


def _popen(replies, seen, made):
    def popen(argv, **kw):
        p = _Proc(argv, replies, seen, **kw)
        made.append(p)
        return p

    return popen


def _reply(content=None, calls=None, cost="0.0012"):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {
        "choices": [{"message": msg}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "cost": cost,
            "prompt_tokens_details": {"cached_tokens": 900},
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }


# --- the switch ----------------------------------------------------------------------------------------------


# --- the route (fork-lane design): the key proxy's fork lane only ----------------------------------------------


def test_without_the_fork_lane_there_is_no_fork(monkeypatch):
    _fresh_lane_state(monkeypatch)
    assert fork_mod.route() == (None, "no fork lane")


def test_the_route_is_the_fork_lane_with_its_token(lane):
    assert fork_mod.route() == ((BASE + "/chat/completions", TOKEN), None)


def test_the_token_descriptor_is_read_once_and_closed(monkeypatch):
    sw = _fresh_lane_state(monkeypatch)
    fd = _token_fd()
    env = {
        sw.LIBRARY_FORK_BASE_URL: BASE,
        "unify_memory_library_fork_token_fd": str(fd),
    }
    settings = SimpleNamespace(
        **{sw.LIBRARY_FORK_BASE_URL: BASE, sw.LIBRARY_FORK_TOKEN_FD: str(fd)},
    )
    assert sw.settle_fork_route_env(env, settings) is None
    assert env == {}  # every case variant leaves the environment
    with pytest.raises(OSError):
        os.fstat(fd)  # closed: no later child inherits it
    assert sw.settle_fork_route_env({}, settings) is None and sw.fork_lane_route()[
        0
    ] == (BASE + "/chat/completions", TOKEN)


@pytest.mark.parametrize(
    "kind",
    ["half_set", "not_inherited", "bad_base"],
)
def test_a_bad_fork_route_is_refused_never_used(monkeypatch, kind):
    sw = _fresh_lane_state(monkeypatch)
    if kind == "half_set":
        settings = SimpleNamespace(
            **{sw.LIBRARY_FORK_BASE_URL: BASE, sw.LIBRARY_FORK_TOKEN_FD: ""},
        )
    elif kind == "not_inherited":
        settings = SimpleNamespace(
            **{
                sw.LIBRARY_FORK_BASE_URL: BASE,
                sw.LIBRARY_FORK_TOKEN_FD: str(_token_fd(inheritable=False)),
            },
        )
    else:
        settings = SimpleNamespace(
            **{
                sw.LIBRARY_FORK_BASE_URL: "https://openrouter.ai/api/v1",
                sw.LIBRARY_FORK_TOKEN_FD: str(_token_fd()),
            },
        )
    why = sw.settle_fork_route_env({}, settings)
    assert why and TOKEN not in why
    assert sw.fork_lane_route() == (None, why)


def test_the_fork_route_names_never_reach_a_box():
    """Both names are harness-only for the actor's sandbox and for memory's own boxes."""
    from unify import sandbox
    from unify.memory import sandbox_run

    for name in (
        "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD",
    ):
        assert name in sandbox.HARNESS_ONLY_ENV and name in sandbox_run.HARNESS_ONLY_ENV
        assert sandbox._is_secret_env(name, "x")


def test_c5_fork_on_without_a_lane_is_recorded_refused(mv2, monkeypatch):  # noqa: F811
    run, eid, src, seen, made, popen = _started(mv2, monkeypatch, [])
    _fresh_lane_state(monkeypatch)
    assert fork_mod.start(mv2.paths, SimpleNamespace(), eid, popen=popen) is None
    assert made == []
    doc = json.loads((fork_mod.staging_dir(mv2.paths, eid) / "fork.json").read_text())
    assert doc["status"] == "refused: no fork lane"


# --- the worker: what it stages, from either reply shape, executing nothing ------------------------------------


def _files(blocks):
    """(parsed files by path, refused blocks as {path, why, from}) from parser blocks."""
    files = {b["path"]: b["content"] for b in blocks if b["status"] == "parsed"}
    refused = [
        {"path": b["path"], "why": b["refusal"], "from": b["source"]}
        for b in blocks
        if b["status"] == "refused"
    ]
    return files, refused


def _parse(text):
    return _files(fw.parse_reply(text)[0])


# The shapes recorded fork replies used; each one stages the same file.
_SHAPES = {
    "inside": "```python\npath: candidates/double.py\n" + CODE + "```\n",
    "info": "```path: candidates/double.py\n" + CODE + "```\n",
    "line_before": "Here it is.\npath: candidates/double.py\n```python\n"
    + CODE
    + "```\n",
    "block_before": "```text\npath: candidates/double.py\n```\n```python\n"
    + CODE
    + "```\n",
}


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_every_path_shape_the_fork_wrote_is_staged(shape):
    files, refused = _parse(_SHAPES[shape])
    assert files == {"candidates/double.py": CODE} and refused == []


def test_a_path_shape_outside_the_allow_list_is_still_refused():
    files, refused = _parse(
        "```path: ../escape.py\nx\n```\npath: other.txt\n```\nx\n```\n",
    )
    assert files == {}
    assert {r["path"] for r in refused} == {"../escape.py", "other.txt"}


def test_a_path_line_is_only_taken_by_the_fence_right_after_it():
    # a blank line between, or a path-only block followed by prose, names no block; each is recorded as dropped
    blocks, record = fw.parse_reply(
        "path: notes.md\n\n```\nlost\n```\n"
        + "```text\npath: cases/x.json\n```\nprose\n```json\n{}\n```\n",
    )
    assert _files(blocks) == ({}, [])
    assert [(b["path"], b["reason"]) for b in blocks] == [
        ("notes.md", "path_line_without_block"),
        (None, "no_path"),
        ("cases/x.json", "path_block_then_text_then_block"),
        (None, "no_path"),
    ]
    assert record["counts"] == {"blocks": 4, "parsed": 0, "refused": 0, "dropped": 4}


def test_a_fence_info_path_wins_and_a_repeated_inner_path_line_is_dropped():
    files, _ = _parse("```path: notes.md\npath: notes.md\nA lesson.\n```\n")
    assert files == {"notes.md": "A lesson.\n"}


def test_the_worker_answers_tool_calls_without_running_them_and_writes_fork_json(
    tmp_path,
):
    seen = []
    replies = [
        _reply(
            calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": json.dumps(
                            {
                                "code": _fence("candidates/double.py", CODE)
                                + _fence(
                                    "cases/double.json",
                                    '[{"in": 21, "out": 42}]\n',
                                ),
                            },
                        ),
                    },
                },
            ],
        ),
        _reply(content=_fence("notes.md", "Doubling by multiplication worked.\n")),
    ]
    req = {
        "episode": "e1",
        "url": "http://127.0.0.1:1/v1/chat/completions",
        "headers": {"x-unify-session": "fork.e1"},
        "body": {"model": "m", "messages": [{"role": "user", "content": "x"}]},
    }
    doc = fw.run(
        req,
        tmp_path,
        post=lambda u, h, b: (seen.append((h, b)), replies.pop(0))[1],
    )
    assert doc["status"] == "ok" and doc["turns"] == 2
    assert {f["path"]: f["from"] for f in doc["files"]} == {
        "candidates/double.py": "tool_call:execute_code",
        "cases/double.json": "tool_call:execute_code",
        "notes.md": "text",
    }
    assert (tmp_path / "candidates/double.py").read_text() == CODE
    assert doc["usage"] == {
        "prompt_tokens": 2000,
        "cached_tokens": 1800,
        "completion_tokens": 100,
        "reasoning_tokens": 20,
    }
    assert doc["usd"] == "0.0024"
    second = seen[1][1]["messages"]
    assert second[-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": fw.TOOL_REPLY,
    }
    assert [h["x-unify-request"] for h, _ in seen] == ["0", "1"]
    assert json.loads((tmp_path / "fork.json").read_text()) == doc


def test_the_operational_quota_is_reported(tmp_path):
    many = "".join(
        _fence(f"cases/c{i}.json", "{}\n") for i in range(fw.QUOTA_FILES + 5)
    )
    doc = fw.run(
        {
            "episode": "e1",
            "url": "u",
            "headers": {},
            "body": {"model": "m", "messages": []},
        },
        tmp_path,
        post=lambda u, h, b: _reply(content=many),
    )
    assert doc["status"] == "ok: operational quota reached"
    assert len(doc["files"]) == fw.QUOTA_FILES
    assert (
        sum(r["why"] == "operational quota reached" for r in doc["refused_files"]) == 5
    )


def test_the_worker_imports_only_the_standard_library():
    import ast
    import sys

    tree = ast.parse(Path(fw.__file__).read_text())
    names = {
        a.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for a in n.names
    }
    names |= {
        n.module.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module
    }
    assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree))
    assert names <= set(sys.stdlib_module_names) | {"__future__"}


# --- the box ---------------------------------------------------------------------------------------------------


def _started(mv2, monkeypatch, replies, source=None):  # noqa: F811
    run = _begin(mv2, "Double 21.")
    _transcript(run)
    with monkeypatch.context() as m:
        # recorded with no fork (rev 2 starts one at every finish): the test starts it itself
        m.setattr(fork_mod, "reserve", lambda *a, **k: None)
        m.setattr(fork_mod, "start", lambda *a, **k: None)
        _finish(mv2, run)
    assert not fork_mod.staging_dir(mv2.paths, run.episode_id).exists()
    eid = run.episode_id
    import unify.actor.code_act_actor as caa

    src = source or _source()
    monkeypatch.setattr(caa, "_session_fork_source", lambda inner, actor: (src, None))
    seen, made = [], []
    return run, eid, src, seen, made, _popen(replies, seen, made)


def _home_bytes(home: Path) -> dict:
    return {
        str(p.relative_to(home)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(home.rglob("*"))
        if p.is_file()
        and "memory-staging" not in p.parts
        and p.name != "fork-worker.log"
    }


def test_f1_the_request_goes_through_a_pipe_with_a_bare_environment(
    mv2,
    monkeypatch,
    lane,
):  # noqa: F811
    run, eid, src, seen, made, popen = _started(
        mv2,
        monkeypatch,
        [_reply(content=_fence("notes.md", "n\n"))],
    )
    f = fork_mod.start(
        mv2.paths,
        SimpleNamespace(),
        eid,
        popen=popen,
    )
    fork_mod.finish(f)
    (p,) = made
    # the relay: run by path with -I, a fixed PATH, its own empty HOME and the locale names only
    assert p.argv == fork_mod.relay_argv()
    home = Path(p.kw["env"]["HOME"])
    assert home.name == fork_mod.RELAY_HOME and Path(mv2.paths.state_dir) in home.parents
    assert p.kw["env"]["PATH"] == "/usr/bin:/bin"
    assert set(p.kw["env"]) <= {"PATH", "HOME", *fork_mod.RELAY_ENV_NAMES}
    assert p.kw["close_fds"] and p.kw["start_new_session"]
    assert (
        len(p.kw["pass_fds"]) == 1
    )  # only the staging dir's lock, never a token descriptor
    blob = " ".join([*p.argv, *p.kw["env"].values(), *p.spec["argv"]])
    assert TOKEN not in blob and "18080" not in blob and "Double 21" not in blob
    assert p.spec["argv"][-1] == "--comms"  # the relay answers the box's comms calls from the run record
    url, headers, body = seen[0]
    assert url == BASE + "/chat/completions"
    assert (
        headers["Authorization"] == "Bearer " + TOKEN
    )  # the token reaches the worker on its stdin only
    assert (
        headers["x-unify-session"] == f"fork.{eid}"
        and headers["x-unify-call-kind"] == "actor_fork"
    )
    assert body["messages"][:-1] == src["sent_messages"]
    # the instruction, then the fork's lines of the run record (rev 2: every agent is on the record)
    assert body["messages"][-1]["content"].startswith(fork_mod.INSTRUCTION + "\n\n")
    assert body["tools"] == src["tools"] and body["tool_choice"] == "auto"
    assert body["model"] == "openai/gpt-6-luna" and body["reasoning_effort"] == "low"
    assert (
        json.loads((fork_mod.staging_dir(mv2.paths, eid) / "fork.json").read_text())[
            "status"
        ]
        == "ok"
    )


def test_f3_nothing_flows_back_to_the_record_the_transcript_or_the_client(
    mv2,
    monkeypatch,
    lane,
):  # noqa: F811
    # The writer service a request starts is a process of its own that joins the roster when it is up, before or after
    # the snapshot below depending on how fast it starts; it is kept out of this home, so what changes is the fork's
    # doing alone.
    from unify.memory import writer_service

    monkeypatch.setattr(writer_service, "ensure_service", lambda *a, **k: writer_service.ServiceState("off"))
    run, eid, src, seen, made, popen = _started(
        mv2,
        monkeypatch,
        [
            _reply(
                content=_fence("candidates/double.py", CODE)
                + _fence("notes.md", "n\n"),
            ),
        ],
    )
    before = _home_bytes(mv2.home)
    client_before = json.dumps(src["sent_messages"])
    f = fork_mod.start(
        mv2.paths,
        SimpleNamespace(),
        eid,
        popen=popen,
    )
    fork_mod.finish(f)
    after = _home_bytes(mv2.home)
    # rev 2: the fork joins the run record (its roster, the roster's lock and the record's log); nothing else
    # outside its staging dir changes
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changed == {
        "memory-state/agents/roster.json",
        "memory-state/agents/roster.lock",
        "memory-state/agents/run-record.jsonl",
    }, changed
    (p,) = made
    assert p.spec["name"].startswith("fork-") and p.spec["name"] in (mv2.home / "memory-state/agents/roster.json").read_text()
    assert json.dumps(src["sent_messages"]) == client_before
    staged = fork_mod.staging_dir(mv2.paths, eid)
    assert sorted(p.name for p in staged.rglob("*") if p.is_file()) == [
        "double.py",
        "fork-blocks.json",
        "fork-reply.json",
        "fork.json",
        "fork.lock",
        "notes.md",
    ]


def test_finish_starts_the_fork_after_the_episode_is_recorded(
    mv2,
    monkeypatch,
):  # noqa: F811
    calls, reserved = [], []

    def start(paths, handle, eid, **kw):
        calls.append((eid, handle, (Path(paths.home) / "episodes.git").exists()))
        res = kw.get("reservation")
        reserved.append((res.eid, res.staging.is_dir()) if res is not None else None)
        return None

    monkeypatch.setattr(fork_mod, "start", start)
    run = _begin(mv2, "Double 21.")
    _transcript(run)
    _finish(mv2, run)
    assert [(e, recorded) for e, _h, recorded in calls] == [(run.episode_id, True)]
    # the staging dir and its lock were reserved before the episode was recorded
    assert reserved == [(run.episode_id, True)]


def test_finish_gives_the_fork_the_recorded_episodes_part_list(
    mv2,
    monkeypatch,
):  # noqa: F811
    given = []
    monkeypatch.setattr(
        fork_mod,
        "start",
        lambda paths, handle, eid, **kw: given.append(kw.get("parts")),
    )
    run = _begin(mv2, "Double 21.")
    _transcript(run)
    _finish(mv2, run)
    (parts,) = given
    lines = parts.splitlines()
    # the list the fork reads names the recorded parts, the request first with its first line
    assert lines[0] == fork_mod.PARTS_HEAD
    assert lines[1].startswith("- `request`: ") and "Double 21." in lines[1]


def test_a_session_that_cannot_be_forked_is_recorded_never_forced(
    mv2,
    monkeypatch,
    lane,
):  # noqa: F811
    run, eid, src, seen, made, popen = _started(mv2, monkeypatch, [])
    import unify.actor.code_act_actor as caa

    monkeypatch.setattr(
        caa,
        "_session_fork_source",
        lambda i, a: (None, "the session's history was compressed"),
    )
    assert (
        fork_mod.start(
            mv2.paths,
            SimpleNamespace(),
            eid,
            popen=popen,
        )
        is None
    )
    assert made == []
    doc = json.loads((fork_mod.staging_dir(mv2.paths, eid) / "fork.json").read_text())
    assert doc["status"] == "refused: the session's history was compressed"
