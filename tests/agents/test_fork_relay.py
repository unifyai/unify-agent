"""The fork on the run record: its relay (a host-side process between its box and the record) answers its comms calls,
so a fork can tell the writer where its summary is and what it was unsure of, and answer the writer's question while
it runs (the redesign's A1 acceptance: a fork -> writer message, a writer -> fork question answered).

The fork's model is a stand-in HTTP server on this machine; the worker runs by path as in its box (the box itself,
bubblewrap, is not under test here). Nothing leaves the machine and no key exists.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from unify.agents import run_record as rr
from unify.common import tool_registry as R
from unify.memory.integration import fork, fork_worker
from unify.memory.integration.paths import Paths
from unify.settings import SETTINGS

TOKEN = "fork-lane-token-for-a-test-0042"  # pragma: allowlist secret


def _call(code: str, cid: str) -> dict:
    return {
        "id": cid,
        "type": "function",
        "function": {
            "name": "execute_code",
            "arguments": json.dumps({"thought": "t", "code": code}),
        },
    }


class Model:
    """A stand-in chat-completions endpoint: one scripted reply per request, in order; every request body kept."""

    def __init__(self, replies: list[dict]):
        self.replies, self.bodies = replies, []
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.bodies.append({"body": body, "headers": dict(self.headers)})
                reply = owner.replies[min(len(owner.bodies), len(owner.replies)) - 1]
                data = json.dumps(
                    {
                        "choices": [{"message": reply}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "cost": 0.001,
                        },
                    },
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = (
            f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions"
        )

    def stop(self):
        self.server.shutdown()


#: Names that carry a route, a token or a key (the live audit's names-only check, mirrored): none may be in the
#: relay's environment.
SECRET_NAMES = re.compile(r"BASE_URL|TOKEN|KEY|SECRET|PASSWORD|PASSWD|AUTH|CREDENTIAL|COOKIE", re.IGNORECASE)
PLANTED = {
    "UNIFY_MEMORY_WRITER_BASE_URL": "http://127.0.0.1:1/writer/v1",
    "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL": "http://127.0.0.1:1/fork/v1",
    "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD": "7",
    "OPENROUTER_API_KEY": "not-a-real-key",  # pragma: allowlist secret
    "UNIFY_SOME_SECRET": "x",  # pragma: allowlist secret
}


def _plant(monkeypatch):
    """The controller's environment as the relay's parent has it: route URLs, token and key names."""
    for k, v in PLANTED.items():
        monkeypatch.setenv(k, v)


def _plant_key_files(tmp_path, monkeypatch):
    """A parent whose HOME holds a cloud automation config and whose current folder holds a .env with key names: the
    files the model client's settings read and fetch provider keys with."""
    home, cwd = tmp_path / "parent-home", tmp_path / "parent-cwd"
    (home / ".config" / "gcloud").mkdir(parents=True)
    (home / ".config" / "gcloud" / "automation.json").write_text('{"project": "planted"}')
    (home / ".env").write_text("OPENROUTER_API_KEY=planted\n")  # pragma: allowlist secret
    cwd.mkdir()
    (cwd / ".env").write_text("OPENAI_API_KEY=planted\n")  # pragma: allowlist secret
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    return home, cwd


def test_the_relay_imports_no_module_that_reads_provider_settings_or_keys(tmp_path, monkeypatch):
    """The relay's start file, run as fork.start runs it, imports everything the relay's code paths use and nothing
    that reads settings, .env files or keys (the model client, the settings, the logger, dotenv, provider SDKs); the
    guard refuses each of those."""
    from unify.agents import fork_relay, relay_boot

    _plant(monkeypatch)
    _plant_key_files(tmp_path, monkeypatch)
    folder = fork.relay_folder(SimpleNamespace(state_dir=tmp_path / "state"), "fork-3f9a")
    out = subprocess.run(
        [*fork.relay_argv(), "--modules"],
        cwd=str(folder),
        env=fork.relay_env(folder / fork.RELAY_HOME),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = set(out.stdout.split())
    assert {"unify.agents.run_record", "unify.agents.run_record_tools", "unify.common.tool_registry"} <= loaded
    bad = [m for m in loaded if any(m == f or m.startswith(f + ".") for f in relay_boot.FORBIDDEN)]
    assert not bad and "refused import" not in out.stderr, (bad, out.stderr[-2000:])
    assert fork_relay.RELAY_HOME == fork.RELAY_HOME
    # the guard refuses such an import outright
    probe = subprocess.run(
        [sys.executable, "-I", "-c",
         "import runpy; runpy.run_path(%r, run_name='boot')['boot'](); import unillm" % str(fork.RELAY_BOOT)],  # fmt: skip
        cwd=str(folder),
        env=fork.relay_env(folder / fork.RELAY_HOME),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode != 0 and "refused import unillm" in probe.stderr


def _until(check, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = check()
        if got:
            return got
        time.sleep(0.1)
    raise AssertionError("timed out")


def test_a_fork_tells_the_writer_where_its_summary_is_and_answers_its_question(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    eid = "20261020T210311-3f9a0c1d"
    staging = tmp_path / "staging"
    staging.mkdir()
    rec = rr.join_run_record(home)
    rec.join("writer", "writer", pass_id="ab" * 8)  # a writer pass running now
    name = rec.unique_name("fork", eid)
    rec.join(
        name, "fork", episode=eid
    )  # the request puts the fork on the roster; its relay takes it over
    model = Model(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call(
                        'tools.send("@writer my summary is summary.md; I was unsure what cell:0 did")\n'
                        "print(tools.wait(timeout=60))",
                        "c1",
                    ),
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call(
                        'tools.send("@writer cell:0 read the first line as the size")',
                        "c2",
                    )
                ],
            },
            {
                "role": "assistant",
                "content": "```path: summary.md\n## Asked\nA size.\n```\n",
            },
        ],
    )
    lock = os.open(tmp_path / "fork.lock", os.O_RDWR | os.O_CREAT, 0o600)
    os.set_inheritable(lock, True)
    spec = {
        "home": str(home),
        "name": name,
        "argv": [
            sys.executable,
            "-I",
            str(fork.WORKER_FILE),
            "--staging",
            str(staging),
            "--comms",
        ],
        "doc": {
            "episode": eid,
            "url": model.url,
            "headers": {
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
            },
            "body": {
                "model": "m",
                "messages": [{"role": "user", "content": "the request"}],
                "tools": [],
            },
        },
        "lock_fd": lock,
        "log": str(tmp_path / "fork-worker.log"),
    }
    _plant(monkeypatch)  # the parent's environment carries route URLs and key names; the relay's must not
    parent_home, parent_cwd = _plant_key_files(tmp_path, monkeypatch)
    folder = fork.relay_folder(SimpleNamespace(state_dir=tmp_path / "state"), name)
    relay_log = tmp_path / "relay.log"
    with open(relay_log, "ab") as err:
        relay = subprocess.Popen(  # exactly as fork.start launches it
            fork.relay_argv(),
            stdin=subprocess.PIPE,
            stderr=err,
            cwd=str(folder),
            pass_fds=(lock,),
            start_new_session=True,
            env=fork.relay_env(folder / fork.RELAY_HOME),
        )
    relay.stdin.write(json.dumps(spec).encode())
    relay.stdin.close()
    os.close(lock)
    try:
        # the running relay's own environment, as the live audit reads it, and its folder
        env = dict(
            e.split(b"=", 1)
            for e in Path(f"/proc/{relay.pid}/environ").read_bytes().split(b"\0")
            if e
        )
        names = [k.decode() for k in env]
        assert names and not [n for n in names if SECRET_NAMES.search(n)], names
        assert set(names) <= {"PATH", "HOME", *fork.RELAY_ENV_NAMES}, names
        assert env[b"HOME"].decode() == str(folder / fork.RELAY_HOME) != str(parent_home)
        assert os.readlink(f"/proc/{relay.pid}/cwd") == str(folder) != str(parent_cwd)
        told = _until(lambda: [r for r in rr.read_entries(home) if r["author"] == name])
        assert (
            told[0]["mentions"] == ["writer"]
            and told[0]["role"] == "fork"
            and told[0]["episode"] == eid
        )
        assert f"{eid}/cell:0" in told[0]["refs"]
        # the writer asks the running fork (after the fork's one-second post interval has passed, so its answer is
        # never refused as too fast)
        time.sleep(1.1)
        rec.append("writer", f"@{name} what did cell:0 do?")
        answers = _until(
            lambda: [
                r
                for r in rr.read_entries(home)
                if r["author"] == name and "first line" in r["text"]
            ]
        )
        assert answers[0]["mentions"] == ["writer"]
        assert relay.wait(timeout=60) == 0
    finally:
        if relay.poll() is None:
            relay.kill()
        model.stop()
    # the fork's second request carried the writer's question as its wait's answer
    second = model.bodies[1]["body"]["messages"]
    tool_result = next(m for m in second if m.get("role") == "tool")
    assert (
        "what did cell:0 do?" in tool_result["content"]
        and "sent to writer" in tool_result["content"]
    )
    assert json.loads((staging / fork_worker.FORK_JSON).read_text())["status"] == "ok"
    # it never imported a module that reads provider settings or keys (each would be refused and logged), and its
    # empty HOME went with it
    assert "refused import" not in relay_log.read_text(), relay_log.read_text()
    assert not (folder / fork.RELAY_HOME).exists()
    # the fork left the record when it settled; the route token never reached it
    roster = {r["name"]: r["state"] for r in rr.read_roster(home)["agents"]}
    assert roster[name] == "ended"
    assert (
        TOKEN not in rr.record_path(home).read_text()
        and TOKEN not in rr.deliveries_path(home).read_text()
    )


def test_a_call_that_is_not_only_comms_runs_nothing_and_says_so(tmp_path):
    class Dead:
        def call(self, op, args):  # pragma: no cover - never reached
            raise AssertionError("relayed")

    assert (
        fork_worker.comms_reply(_call("import os\nos.listdir('/')", "x"), Dead())
        == fork_worker.TOOL_REPLY_RECORD
    )
    assert (
        fork_worker.comms_reply(
            {"function": {"name": "read", "arguments": "{}"}}, Dead()
        )
        == fork_worker.TOOL_REPLY_RECORD
    )


def test_a_silent_relay_makes_comms_calls_say_the_record_is_not_reachable(
    tmp_path, monkeypatch
):
    r, w = os.pipe()
    out_r, out_w = os.pipe()
    relay = fork_worker.Relay(
        os.fdopen(r, "rb", buffering=0), os.fdopen(out_w, "wb", buffering=0)
    )
    monkeypatch.setattr(fork_worker, "RECORD_CALL_S", 0.5)
    try:
        assert relay.call("send", {"text": "@writer x"}) == fork_worker.RECORD_DOWN
        assert relay.down and relay.call("who", {}) == fork_worker.RECORD_DOWN
    finally:
        os.close(w)
        os.close(out_r)


def test_with_the_run_record_on_the_fork_starts_through_its_relay_with_the_common_section(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS_RUN_RECORD", "on")
    settings = SimpleNamespace(
        UNIFY_MEMORY="on",
        UNIFY_MEMORY_LIBRARY="on",
        UNIFY_MEMORY_LIBRARY_FORK="on",
        UNIFY_AGENTS_RUN_RECORD="on",
    )
    paths = Paths.under(tmp_path / "home")
    eid = "20261020T210311-3f9a0c1d"
    _plant(monkeypatch)
    monkeypatch.setattr(
        fork, "route", lambda: (("http://127.0.0.1:9/fork/v1", TOKEN), None)
    )
    # the box itself is not under test: its argv as box_argv would end it
    monkeypatch.setattr(
        fork,
        "box_argv",
        lambda staging, die_with_parent: [
            "bwrap",
            "--",
            "/usr/bin/python3",
            "-I",
            fork.BOX_WORKER,
            "--staging",
            "/staging",
        ],
    )
    source = {
        "client": SimpleNamespace(
            endpoint="openai/gpt-6-luna@openrouter", reasoning_effort=None
        ),
        "sent_messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "the request"},
        ],
        "tools": R.json_tools("actor"),
    }
    monkeypatch.setattr(
        fork,
        "request_doc",
        # extra keywords (a segment fork's instruction) pass through as the real request_doc passes them
        lambda handle, e, url, key, parts=None, **kw: fork.request_from_source(
            e, source, url, key, parts=parts, **kw
        ),
    )
    started = []

    class Proc:
        pid = 4242

    def popen(argv, **kw):
        started.append((argv, kw))
        p = Proc()
        p.stdin = SimpleNamespace(
            write=lambda data: started.append(data), close=lambda: None
        )
        return p

    got = fork.start(
        paths, object(), eid, settings, popen=popen, parts="- `request`: the request"
    )
    assert got is not None
    got.writer.join(5)
    (argv, kw), data = started[0], started[1]
    assert argv == fork.relay_argv() and argv[1] == "-I" and kw["start_new_session"] is True
    # least privilege: the relay's environment is the short allow-list, with no route, token or key name, its HOME
    # an empty folder of its own and its current folder that folder's parent (no .env there)
    assert set(kw["env"]) <= {"PATH", "HOME", *fork.RELAY_ENV_NAMES} and "PYTHONPATH" not in kw["env"]
    assert not [n for n in kw["env"] if SECRET_NAMES.search(n)] and not set(PLANTED) & set(kw["env"])
    folder = Path(paths.state_dir) / "fork-relays" / "fork-3f9a"
    assert kw["cwd"] == str(folder) and kw["env"]["HOME"] == str(folder / fork.RELAY_HOME)
    assert not any((folder / fork.RELAY_HOME).iterdir()) and not (folder / ".env").exists()
    spec = json.loads(data)
    assert spec["argv"][-1] == "--comms" and spec["home"] == str(paths.home)
    assert spec["name"] == "fork-3f9a" and spec["lock_fd"] in kw["pass_fds"]
    last = spec["doc"]["body"]["messages"][-1]["content"]
    assert last.count(R.COMMON_SECTION) == 1 and "you are `fork-3f9a` (fork)" in last
    assert fork.RECORD_CALLS_ONLY in last
    # the cached prefix (the actor's messages and tools) is unchanged
    assert (
        spec["doc"]["body"]["messages"][:2] == source["sent_messages"]
        and spec["doc"]["body"]["tools"] == source["tools"]
    )
    roster = {r["name"]: r for r in rr.read_roster(paths.home)["agents"]}
    assert (
        roster["fork-3f9a"]["role"] == "fork" and roster["fork-3f9a"]["episode"] == eid
    )


def test_a_segment_fork_is_named_for_its_segment_and_its_calls_carry_episode_and_segment(
    tmp_path,
    monkeypatch,
):
    """A segment's fork (in-episode memory) is ``fork-<hex>-s<n>`` on the roster with its episode, and under metering its
    calls carry ``X-Unify-Episode`` and ``X-Unify-Segment`` (ids only)."""
    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS_RUN_RECORD", "on")
    assert fork.split_segment("20261020T210311-3f9a0c1d.s2") == (
        "20261020T210311-3f9a0c1d",
        2,
    )
    assert fork.split_segment("20261020T210311-3f9a0c1d") == (
        "20261020T210311-3f9a0c1d",
        None,
    )
    settings = SimpleNamespace(
        UNIFY_MEMORY="on",
        UNIFY_MEMORY_LIBRARY="on",
        UNIFY_MEMORY_LIBRARY_FORK="on",
        UNIFY_AGENTS_RUN_RECORD="on",
        UNIFY_MEMORY_METERING="on",
    )
    paths = Paths.under(tmp_path / "home")
    sid = "20261020T210311-3f9a0c1d.s2"
    monkeypatch.setattr(
        fork, "route", lambda: (("http://127.0.0.1:9/fork/v1", TOKEN), None)
    )
    monkeypatch.setattr(
        fork,
        "box_argv",
        lambda staging, die_with_parent: [
            "bwrap",
            "--",
            "/usr/bin/python3",
            "-I",
            fork.BOX_WORKER,
        ],
    )
    source = {
        "client": SimpleNamespace(
            endpoint="openai/gpt-6-luna@openrouter", reasoning_effort=None
        ),
        "sent_messages": [{"role": "user", "content": "the request"}],
        "tools": R.json_tools("actor"),
    }
    monkeypatch.setattr(
        fork,
        "request_doc",
        # extra keywords (a segment fork's instruction) pass through as the real request_doc passes them
        lambda handle, e, url, key, parts=None, **kw: fork.request_from_source(
            e, source, url, key, parts=parts, **kw
        ),
    )
    started = []

    class Proc:
        pid = 4243

    def popen(argv, **kw):
        started.append((argv, kw))
        p = Proc()
        p.stdin = SimpleNamespace(
            write=lambda data: started.append(data), close=lambda: None
        )
        return p

    got = fork.start(paths, object(), sid, settings, popen=popen)
    assert got is not None
    got.writer.join(5)
    spec = json.loads(started[1])
    assert spec["name"] == "fork-3f9a-s2"
    assert spec["doc"]["headers"]["x-unify-episode"] == "20261020T210311-3f9a0c1d"
    assert spec["doc"]["headers"]["x-unify-segment"] == "2"
    row = next(
        r for r in rr.read_roster(paths.home)["agents"] if r["name"] == "fork-3f9a-s2"
    )
    assert row["episode"] == "20261020T210311-3f9a0c1d" and row["role"] == "fork"


IGNORES_TERM = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "sys.stdin.readline()\n"
    "time.sleep(600)\n"
)


def test_a_box_that_ignores_sigterm_at_the_end_of_its_wait_is_killed(tmp_path, monkeypatch):
    """Review: when the fork's wall clock (its comms wait included) ends, the relay sends SIGTERM and, if the box is
    still running after KILL_AFTER_S, SIGKILL; the relay never waits on such a box for good."""
    import asyncio

    from unify.agents import fork_relay

    monkeypatch.setattr(fork_worker, "WALL_GUARD_S", 0.5)
    monkeypatch.setattr(fork_relay, "_MARGIN_S", 0.0)
    monkeypatch.setattr(fork_relay, "KILL_AFTER_S", 0.5)
    home = tmp_path / "home"
    rec = rr.join_run_record(home)
    rec.join("fork-3f9a", "fork", episode="ep-1")
    spec = {
        "home": str(home),
        "name": "fork-3f9a",
        "argv": [sys.executable, "-I", "-c", IGNORES_TERM],
        "doc": {"headers": {}},
    }
    started = time.monotonic()
    code = asyncio.run(asyncio.wait_for(fork_relay.relay(spec), 30))
    assert code == -9 and time.monotonic() - started < 15
    roster = {r["name"]: r["state"] for r in rr.read_roster(home)["agents"]}
    assert roster["fork-3f9a"] == "ended"
