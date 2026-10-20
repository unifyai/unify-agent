"""The runner-owned service through the command lines a runner that cannot import the build uses (C's learner runs
each in the controller's sandbox): the real service started with ``--foreground --owner runner --until``, ``--ready``,
and ``async_pass drain`` run from another pid namespace. The service is the real loop, idle (nothing open), so no model
is called; no key exists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.memory import test_owned_service as O
from tests.memory import test_writer_service as T
from unify.memory import service_owner as so
from unify.memory import writer_service as ws

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
owned = O.owned  # the fixture


def _env(owned, **extra) -> dict:
    """The runner's child environment: its settings, the owner, no route token (no route here), plus *extra*."""
    env = {k: v for k, v in os.environ.items() if not so.secret_name(k)}
    env.update(O.RUNNER)
    env.update(extra)
    return env


def _bwrap(*argv: str) -> list[str]:
    return [shutil.which("bwrap"), "--dev-bind", "/", "/", "--unshare-pid", "--die-with-parent", "--proc", "/proc",
            *argv]  # fmt: skip


def test_a_start_with_a_credential_name_in_its_environment_is_refused(owned):
    out = subprocess.run(
        [sys.executable, "-m", ws.MODULE, "--home", str(owned.home), "--foreground", "--owner", "runner"],
        env=_env(owned, OPENROUTER_API_KEY="planted", SOME_TOKEN="planted"),  # pragma: allowlist secret
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 3, out.stderr[-2000:]
    line = json.loads(out.stdout.strip().splitlines()[-1])
    assert "OPENROUTER_API_KEY" in line["refused"] and "SOME_TOKEN" in line["refused"]
    assert "planted" not in out.stdout
    assert not ws.events(owned.home) or not [e for e in ws.events(owned.home) if e.get("event") == "start"]


def test_the_runners_command_lines_start_check_and_drain_the_service(owned):
    """C's argv: start (its own HOME and folder), ready, survives a session, then the drain from another pid namespace
    verified (marker, lock free, no signal), and the started process exits."""
    so.write_stop(owned.home, "drain", "the previous stage ended")  # stage 1's end
    until = time.time() + 600
    proc = subprocess.Popen(
        _bwrap(sys.executable, "-B", "-m", ws.MODULE, "--home", str(owned.home), "--foreground", "--owner", "runner",
               "--until", str(until)),  # fmt: skip
        env=_env(owned),
        stdout=subprocess.PIPE,
        stderr=open(owned.tmp / "service.log", "ab"),
        text=True,
        start_new_session=True,
    )
    owned.procs.append(proc)
    try:
        ready = subprocess.run(
            _bwrap(sys.executable, "-B", "-m", ws.MODULE, "--home", str(owned.home), "--ready", "--wait", "60"),
            env=_env(owned), capture_output=True, text=True, timeout=120,
        )  # fmt: skip
        assert ready.returncode == 0, ready.stdout + ready.stderr[-2000:]
        report = json.loads(ready.stdout.strip().splitlines()[-1])
        assert report["ready"] is True and report["owner"] == "runner"
        cleared = [e for e in ws.events(owned.home) if e.get("event") == "stop-cleared"]
        assert cleared and cleared[0]["previous_by"] == "drain"
        doc = ws.read_state(owned.home)
        assert doc["owner"] == "runner" and doc["deadline"] <= ws.stamp(ws.datetime.fromtimestamp(until, ws.timezone.utc))
        # a session comes and goes: the service is untouched
        session = O._session(owned)
        assert session.stdout.readline().strip() == "running"
        O._end_session(session)
        assert proc.poll() is None
        # the drain, from another pid namespace: it cannot see the service, so it signals nothing and verifies by
        # the lock once the service has stopped at the marker
        drain = subprocess.run(
            _bwrap(sys.executable, "-B", "-m", "unify.memory.integration.async_pass", "drain", "--home", str(owned.home)),
            env=_env(owned), capture_output=True, text=True, timeout=180,
        )  # fmt: skip
        assert drain.returncode == 0, drain.stdout + drain.stderr[-2000:]
        out = json.loads(drain.stdout.strip().splitlines()[-1])
        assert out["service"]["verified"] is True and out["service"]["how"] == "clean"
        proc.wait(timeout=30)
        assert so.lock_free(owned.home) and ws.read_state(owned.home)["state"] == "stopped"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_drain_from_another_namespace_never_signals_and_reports_a_service_that_does_not_stop(owned):
    """A runner-owned service that does not stop at the marker (stopped with SIGSTOP): the drain from another pid
    namespace signals nothing and reports it not verified (exit 2); the runner's own kill of its sandbox is the
    backstop."""
    proc = O._start(owned)
    os.kill(proc.pid, 19)  # SIGSTOP
    try:
        code = (
            "import sys, json; from unify.memory import writer_service as ws; "
            "t = ws.stop_service(sys.argv[1], 'drain', 'stage-end', grace_s=1); print(json.dumps([t.verified, t.how]))"
        )
        out = subprocess.run(_bwrap(sys.executable, "-c", code, str(owned.home)), env=_env(owned),
                             capture_output=True, text=True, timeout=120)  # fmt: skip
        assert json.loads(out.stdout.strip().splitlines()[-1]) == [False, "external"], out.stdout + out.stderr[-2000:]
        assert ws.proc_start(proc.pid) is not None  # nothing signalled it
    finally:
        os.kill(proc.pid, 18)  # SIGCONT
    done = so.stop_owned(owned.home, proc, by="drain", reason="stage-end", grace_s=1, term_s=5)
    assert done.verified
