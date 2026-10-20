"""The runner-owned writer service (UNIFY_MEMORY_WRITER_SERVICE_OWNER=runner): the runner starts one service per cell in
a sandbox of its own that outlives every session, a request never starts it, and the runner stops it (verified) at the
cell's end before the key proxy (MAIN's option 3).

A "session" here is a request process in a pid namespace of its own that dies with its parent (``bwrap --unshare-pid
--die-with-parent``), as the controller's sandbox is: ending it ends everything inside it. The service is the real
service loop with a scripted pass (tests/memory/service_driver.py). No model, network or key.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.memory import test_writer_service as T
from unify.agents import run_record as rr
from unify.memory import service_owner as so
from unify.memory import writer_service as ws

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")

RUNNER = {**T.SWITCHES, "UNIFY_MEMORY_WRITER_SERVICE_OWNER": "runner"}
SESSION = r"""
import json, sys, time
from types import SimpleNamespace
from unify.memory import writer_service as ws
settings = SimpleNamespace(**json.loads(sys.argv[2]))
got = ws.ensure_service(sys.argv[1], settings=settings, argv_prefix=[sys.executable, sys.argv[3]], wait_s=30)
print(got.status, flush=True)
time.sleep(600)
"""


@pytest.fixture
def owned(tmp_path, monkeypatch):
    """A memory home with its stores, the runner's switches, and every service started here stopped at the end."""
    home, out = tmp_path / "home", tmp_path / "out"
    for k, v in RUNNER.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DRIVER_OUT", str(out))
    from unify.memory.integration.consolidate import open_stores
    from unify.memory.integration.paths import Paths

    open_stores(Paths.under(home))
    procs: list = []
    yield type("Owned", (), {"home": home, "out": out, "procs": procs, "tmp": tmp_path})
    for proc in procs:
        if proc.poll() is None:
            so.stop_owned(home, proc, by="operator", reason="the test ended", grace_s=0.5, term_s=2)


def _start(owned, **driver_env) -> subprocess.Popen:
    """The runner's start: the service in the foreground, with the owned environment, in its own folder."""
    env = so.owned_env(owned.home)
    env.update({"DRIVER_OUT": str(owned.out), **{k: str(v) for k, v in driver_env.items()}})
    proc = subprocess.Popen(
        [sys.executable, T.DRIVER, "--home", str(owned.home), "--foreground"],
        env=env,
        cwd=str(so.owned_cwd(owned.home)),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=open(owned.tmp / "service.log", "ab"),
    )
    owned.procs.append(proc)
    T._until(lambda: (ws.read_state(owned.home) or {}).get("pid") == proc.pid and ws.owns(ws.read_state(owned.home)))
    return proc


def _session(owned) -> subprocess.Popen:
    """A request in a session's sandbox (its own pid namespace, dying with its parent): it calls ensure_service."""
    bwrap = shutil.which("bwrap")
    proc = subprocess.Popen(
        [bwrap, "--dev-bind", "/", "/", "--unshare-pid", "--die-with-parent", "--proc", "/proc",
         sys.executable, "-c", SESSION, str(owned.home), json.dumps(RUNNER), T.DRIVER],  # fmt: skip
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ},
        start_new_session=True,
    )
    return proc


def _end_session(proc: subprocess.Popen) -> None:
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=30)


def _events(home, kind):
    return [r for r in ws.events(home) if r.get("event") == kind]


def test_a_request_never_starts_the_runners_service_and_none_yet_is_absent(owned):
    got = ws.ensure_service(owned.home, settings=T._settings(UNIFY_MEMORY_WRITER_SERVICE_OWNER="runner"))
    assert got.status == "absent" and not _events(owned.home, "start")


def test_the_service_survives_sessions(owned):
    """The runner's service runs across sessions: a session's request finds it running (it starts nothing), and the
    session's end leaves it alive; after three sessions it is the same process, with one start and no takeover."""
    proc = _start(owned)
    doc = ws.read_state(owned.home)
    for _ in range(3):
        session = _session(owned)
        assert session.stdout.readline().strip() == "running"
        _end_session(session)
        time.sleep(0.5)
        assert proc.poll() is None and ws.owns(doc)
    assert ws.read_state(owned.home)["pid"] == proc.pid and ws.read_state(owned.home)["owner"] == "runner"
    assert len(_events(owned.home, "start")) == 1 and not _events(owned.home, "takeover")


class Slow:
    """A stand-in model endpoint that answers each call after *delay_s*; it counts calls begun, answered and cut."""

    def __init__(self, delay_s: float):
        self.begun = threading.Event()
        self.answered = 0
        self.cut = 0
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                owner.begun.set()
                time.sleep(delay_s)
                data = b'{"choices": [{"message": {"role": "assistant", "content": "ok"}}]}'
                try:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    owner.answered += 1
                except (BrokenPipeError, ConnectionResetError):
                    owner.cut += 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions"

    def stop(self):
        self.server.shutdown()


def test_a_writer_call_in_flight_completes_across_a_session_end(owned):
    """The pass's model call is in flight when the session that last called ensure_service ends: it completes, and no
    call fails."""
    slow = Slow(3.0)
    try:
        T._open_entry(owned.home)
        _start(owned, DRIVER_HTTP=slow.url, DRIVER_TURNS=0)
        assert slow.begun.wait(60)  # the pass's call is in flight
        session = _session(owned)
        assert session.stdout.readline().strip() == "running"
        _end_session(session)  # the session ends with the call in flight
        (result,) = T._until(lambda: [json.loads(p.read_text()) for p in owned.out.glob("http-*.json")] or None)
        assert result["ok"] and result["status"] == 200, result
        assert slow.answered == 1 and slow.cut == 0
    finally:
        slow.stop()


def test_the_runner_stops_the_service_verified_before_the_proxy(owned):
    """At the cell's end the runner stops its service (the marker, the pass in flight runs to its end, then exit) and
    only then stops the key proxy: the pass's call was answered through the proxy before the stop returned verified."""
    proxy = Slow(2.0)
    T._open_entry(owned.home)
    proc = _start(owned, DRIVER_HTTP=proxy.url, DRIVER_TURNS=0)
    assert proxy.begun.wait(60)
    done = so.stop_owned(owned.home, proc, by="drain", reason="the cell ended", grace_s=30, term_s=5)
    answered_before_proxy_stop = proxy.answered
    proxy.stop()
    assert done.verified and done.how == "clean" and proc.poll() is not None and so.lock_free(owned.home)
    assert answered_before_proxy_stop == 1 and proxy.cut == 0
    (result,) = [json.loads(p.read_text()) for p in owned.out.glob("http-*.json")]
    assert result["ok"]
    assert ws.read_state(owned.home)["state"] == "stopped" and _events(owned.home, "stop")[-1]["why"] == "drain"
    # a later request does not start it again
    assert ws.ensure_service(owned.home, settings=T._settings(UNIFY_MEMORY_WRITER_SERVICE_OWNER="runner")).status == "refused"


def test_a_service_that_will_not_stop_is_terminated_and_verified(owned):
    proc = _start(owned)
    # a service that never looks at its marker (stopped, as a hung process would be)
    os.kill(proc.pid, signal.SIGSTOP)
    try:
        done = so.stop_owned(owned.home, proc, by="operator", reason="test", grace_s=0.5, term_s=1, settle_s=5)
    finally:
        try:
            os.kill(proc.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    assert done.verified and done.how in ("term", "kill") and proc.poll() is not None


def test_the_drain_from_a_session_writes_the_marker_and_verifies_by_the_lock(owned):
    """The build's drain run inside a session's pid namespace cannot see or signal the runner's service: it writes the
    marker and reports verified once the service's lock is free (the service exits at the marker)."""
    proc = _start(owned)
    bwrap = shutil.which("bwrap")
    code = (
        "import sys; from unify.memory import writer_service as ws; "
        "t = ws.stop_service(sys.argv[1], 'drain', 'the cell ended', grace_s=30); print(t.verified, t.how)"
    )
    out = subprocess.run(
        [bwrap, "--dev-bind", "/", "/", "--unshare-pid", "--die-with-parent", "--proc", "/proc",
         sys.executable, "-c", code, str(owned.home)],  # fmt: skip
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.stdout.split()[-2:] == ["True", "clean"], out.stdout + out.stderr[-2000:]
    proc.wait(timeout=30)
    assert so.lock_free(owned.home)


def test_no_key_is_reachable_from_the_runners_service(owned, monkeypatch):
    """The runner's environment holds key, token and secret names and its HOME and folder hold a .env and a cloud
    automation config; the service started with owned_env holds none of those names, runs with an empty HOME of its
    own in its own folder, and its model client holds no key."""
    parent_home = owned.tmp / "runner-home"
    (parent_home / ".config" / "gcloud").mkdir(parents=True)
    (parent_home / ".config" / "gcloud" / "automation.json").write_text('{"project": "planted"}')
    (parent_home / ".env").write_text("OPENROUTER_API_KEY=planted\n")  # pragma: allowlist secret
    monkeypatch.setenv("HOME", str(parent_home))
    monkeypatch.chdir(parent_home)
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "UNIFY_KEY", "UNIFY_MEMORY_WRITER_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "planted-value-for-a-test")  # pragma: allowlist secret
    proc = _start(owned, DRIVER_LIST_HELD=1)
    env = dict(
        e.split(b"=", 1) for e in Path(f"/proc/{proc.pid}/environ").read_bytes().split(b"\0") if e
    )
    names = [k.decode() for k in env]
    secret = [n for n in names if so.SECRET_NAME.search(n) and not n.upper().endswith("_TOKEN_FD")]
    assert not secret, secret
    assert b"planted-value-for-a-test" not in b"".join(env.values())
    assert env[b"HOME"].decode() == str(so.owned_cwd(owned.home) / "home") != str(parent_home)
    assert os.readlink(f"/proc/{proc.pid}/cwd") == str(so.owned_cwd(owned.home))
    assert not any((so.owned_cwd(owned.home) / "home").iterdir())
    keys = T._until(lambda: json.loads((owned.out / "keys.json").read_text()) if (owned.out / "keys.json").exists() else None)
    assert keys == {"held": []}


def test_no_box_of_the_runners_service_can_read_the_run_record(owned):
    """A cell in a pass's confined box (the service's own box builder) opens the run record, its delivery log and its
    roster by their host paths: every attempt fails inside the box."""
    rec = rr.join_run_record(owned.home)
    rec.join("actor-3f9a", "actor")
    rec.join("h1", "helper")
    rr.RunRecordView(rec, "actor-3f9a", "actor").post("actor-3f9a", "@h1 something about item:x")
    targets = [rr.record_path(owned.home), rr.deliveries_path(owned.home), rr.roster_path(owned.home)]
    T._open_entry(owned.home)
    _start(owned, DRIVER_BOX_READ=":".join(str(t) for t in targets), DRIVER_TURNS=0)
    seen = T._until(lambda: (owned.out / "box-read.txt").read_text() if (owned.out / "box-read.txt").exists() else None)
    assert seen.count("REFUSED") == 2 * len(targets) and "OPENED" not in seen, seen
