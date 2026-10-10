"""Design r5 §2-§3: the episode-end fork. It runs only under UNIFY_MEMORY_V21_FORK=on, after the episode is recorded,
through the actor's own proxy route, in a box whose only writable mount is the episode's staging dir. Nothing flows
back (F3), off is off (F6), and the worker stages only the allowed files, from text and tool-call replies, executing
nothing."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory_v2.integration import fork as fork_mod
from unify.memory_v2.integration import fork_worker as fw
from unify.memory_v2.integration import switch
from unify.settings import SETTINGS
from tests.memory_v2.integration.test_request import (  # noqa: F401
    _begin,
    _finish,
    _transcript,
    mv2,
)

ROUTE = {
    "UNILLM_LLM_GATEWAY_URL": "http://127.0.0.1:18080/v1",
    "UNILLM_LLM_GATEWAY_KEY": "arc-proxy",
}
ON = SimpleNamespace(
    UNIFY_MEMORY_V21="on",
    UNIFY_MEMORY_V21_FORK="on",
    UNIFY_MEMORY_V21_WAIT_SLOT="on",
)
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
    """A fake boxed worker: on wait, it runs the real worker logic in-process against a scripted proxy."""

    def __init__(self, argv, replies, seen, **kw):
        self.argv, self.kw, self.replies, self.seen = argv, kw, list(replies), seen
        self.stdin, self.pid, self.returncode = _Pipe(), 4242, None

    def wait(self, timeout=None):
        host = Path(self.argv[self.argv.index("--bind") + 1])
        req = json.loads(self.stdin.getvalue().decode())

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


def test_the_switch_is_off_by_default_and_takes_only_on_or_off():
    assert switch.parse_v21_fork("") == "off" and switch.parse_v21_fork("on") == "on"
    assert SETTINGS.UNIFY_MEMORY_V21_FORK == "off"
    with pytest.raises(ValueError, match="UNIFY_MEMORY_V21_FORK"):
        switch.parse_v21_fork("maybe")


# --- the route (F5/L5: the actor's own proxy, never a provider) -----------------------------------------------


@pytest.mark.parametrize(
    "env, why",
    [
        ({}, "no proxy route"),
        (
            {**ROUTE, "UNILLM_LLM_GATEWAY_URL": "https://openrouter.ai/api/v1"},
            "the gateway is not a local proxy",
        ),
        (
            {**ROUTE, "UNILLM_LLM_GATEWAY_KEY": "sk-or-v1-abc"},
            "the gateway key is not a proxy placeholder",
        ),
    ],
)
def test_without_the_actors_proxy_route_there_is_no_fork(env, why):
    assert fork_mod.route(env) == (None, why)


def test_the_route_is_the_actors_gateway_with_its_placeholder():
    assert fork_mod.route(ROUTE) == (
        ("http://127.0.0.1:18080/v1/chat/completions", "arc-proxy"),
        None,
    )


# --- the worker: what it stages, from either reply shape, executing nothing ------------------------------------


def test_parse_reply_reads_text_and_tool_call_blocks_and_refuses_other_paths(tmp_path):
    marker = tmp_path / "ran"
    code = f"open({str(marker)!r}, 'w').write('x')\n"  # would create the marker if anything ran it
    msg = {
        "content": _fence("notes.md", "A lesson.\n")
        + _fence("../escape.py", "x\n")
        + _fence("other.txt", "x\n"),
        "tool_calls": [
            {
                "id": "c1",
                "function": {
                    "name": "execute_code",
                    "arguments": json.dumps(
                        {"code": code + "\n" + _fence("candidates/double.py", CODE)},
                    ),
                },
            },
        ],
    }
    files, refused = fw.parse_reply(msg)
    assert files == {"notes.md": "A lesson.\n", "candidates/double.py": CODE}
    assert {(r["path"], r["from"]) for r in refused} == {
        ("../escape.py", "text"),
        ("other.txt", "text"),
    }
    assert not marker.exists()


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


# --- the box (F1, F2) ------------------------------------------------------------------------------------------


def test_the_box_mounts_only_the_worker_and_the_staging_dir_writable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(fork_mod.shutil, "which", lambda n: f"/usr/bin/{n}")
    argv = fork_mod.box_argv(tmp_path / "staging", die_with_parent=True)
    binds = [argv[i + 1 : i + 3] for i, a in enumerate(argv) if a == "--bind"]
    assert binds == [[str(tmp_path / "staging"), "/staging"]]
    ro = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
    assert str(fork_mod.WORKER_FILE) in ro and not any(
        str(Path.home()) in p for p in ro if p != str(fork_mod.WORKER_FILE)
    )
    assert {"--unshare-all", "--share-net", "--clearenv", "--die-with-parent"} <= set(
        argv,
    )
    assert argv[-4:-2] == ["-I", fork_mod.BOX_WORKER] and fork_mod.FORK_MARK in argv


def _started(mv2, monkeypatch, replies, source=None):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", "on")
    run = _begin(mv2, "Double 21.")
    _transcript(run)
    _finish(mv2, run)  # recorded with the fork off
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
        ON,
        environ=ROUTE,
        popen=popen,
    )
    fork_mod.finish(f, ON)
    (p,) = made
    assert (
        p.kw["env"] == {"PATH": "/usr/bin:/bin"}
        and p.kw["close_fds"]
        and p.kw["start_new_session"]
    )
    assert "pass_fds" not in p.kw
    blob = " ".join(p.argv)
    assert "arc-proxy" not in blob and "18080" not in blob and "Double 21" not in blob
    url, headers, body = seen[0]
    assert url == "http://127.0.0.1:18080/v1/chat/completions"
    assert (
        headers["x-unify-session"] == f"fork.{eid}"
        and headers["x-unify-call-kind"] == "actor_fork"
    )
    assert (
        body["messages"][:-1] == src["sent_messages"]
        and body["messages"][-1]["content"] == fork_mod.INSTRUCTION
    )
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
):  # noqa: F811
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
        ON,
        environ=ROUTE,
        popen=popen,
    )
    fork_mod.finish(f, ON)
    assert _home_bytes(mv2.home) == before
    assert json.dumps(src["sent_messages"]) == client_before
    staged = fork_mod.staging_dir(mv2.paths, eid)
    assert sorted(p.name for p in staged.rglob("*") if p.is_file()) == [
        "double.py",
        "fork.json",
        "notes.md",
    ]


def test_finish_starts_the_fork_after_the_episode_is_recorded(
    mv2,
    monkeypatch,
):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21_FORK", "on")
    calls = []

    def start(paths, handle, eid, settings, **kw):
        calls.append((eid, handle, (Path(paths.home) / "episodes.git").exists()))
        return None

    monkeypatch.setattr(fork_mod, "start", start)
    run = _begin(mv2, "Double 21.")
    _transcript(run)
    _finish(mv2, run)
    assert [(e, recorded) for e, _h, recorded in calls] == [(run.episode_id, True)]


def test_f6_off_is_off_no_process_no_staging_dir(mv2, monkeypatch):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21_FORK", "off")
    spawned = []
    real_start = (
        fork_mod.start
    )  # the spy below replaces it for the request; the real one is checked after
    monkeypatch.setattr(fork_mod, "start", lambda *a, **k: spawned.append(a))
    run = _begin(mv2, "Double 21.")
    _transcript(run)
    _finish(mv2, run)
    assert spawned == []
    assert not (Path(mv2.paths.state_dir) / "memory-staging").exists()
    off = SimpleNamespace(UNIFY_MEMORY_V21="on", UNIFY_MEMORY_V21_FORK="off")
    assert (
        real_start(
            mv2.paths,
            None,
            "e",
            off,
            environ=ROUTE,
            popen=lambda *a, **k: spawned.append(a),
        )
        is None
    )
    assert spawned == [] and not (Path(mv2.paths.state_dir) / "memory-staging").exists()


def test_a_session_that_cannot_be_forked_is_recorded_never_forced(
    mv2,
    monkeypatch,
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
            ON,
            environ=ROUTE,
            popen=popen,
        )
        is None
    )
    assert made == []
    doc = json.loads((fork_mod.staging_dir(mv2.paths, eid) / "fork.json").read_text())
    assert doc["status"] == "refused: the session's history was compressed"
