"""The lane relay as a process (unify/lanes/relay.py starting unify/lanes/relay_server.py): the key reaches it on
stdin only and is found nowhere else (argv, environment, journals, its log, replies), the process cannot be read by
another, it stops verifiably (lifeline, SIGTERM, a call in flight cut and journaled), and it dies with its owner.

The provider is a stand-in on this machine (tests/lanes/upstream.py) and the key is a fake.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.lanes.upstream import FAKE_KEY, MODEL, Upstream
from unify.lanes import relay as lane_relay
from unify.lanes.relay import LaneRelay, LaneRelayError

WORST = {MODEL: "0.65625"}
LANES = {"writer": [MODEL], "helper": [MODEL], "fork": [MODEL]}
BODY = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}


def _start(tmp_path: Path, upstream: str, **kw) -> LaneRelay:
    return LaneRelay.start(
        key=FAKE_KEY,
        lanes=LANES,
        cap_usd="2.00",
        worst_case_usd=WORST,
        journal_dir=tmp_path / "lanes",
        upstream=upstream,
        log_path=tmp_path / "lanes" / "relay.log",
        **kw,
    )


def _post(relay: LaneRelay, lane: str, token: str | None = None, timeout: float = 60):
    req = urllib.request.Request(
        relay.base_url(lane) + "/chat/completions",
        data=json.dumps(BODY).encode(),
        headers={
            "Authorization": f"Bearer {token or relay.token(lane)}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _gone(pid: int, timeout: float = 15.0) -> bool:
    """Whether *pid* ends (or is left only as a zombie) within *timeout*."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.1)
    return False


def _files_text(folder: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file())


@pytest.fixture
def upstream():
    up = Upstream(echo=FAKE_KEY.encode())
    yield up
    up.close()


def test_the_key_is_in_the_relays_memory_only(tmp_path, upstream):
    relay = _start(tmp_path, upstream.base)
    try:
        tokens = [relay.token(lane) for lane in LANES]
        status, body = _post(relay, "writer")
        assert status == 200 and FAKE_KEY.encode() not in body and b"[redacted]" in body
        cmdline = Path(f"/proc/{relay.pid}/cmdline").read_bytes()
        assert FAKE_KEY.encode() not in cmdline and not any(
            t.encode() in cmdline for t in tokens
        )
        assert cmdline.split(b"\0")[1:3] == [b"-I", b"-S"]  # isolated, no site packages
        # not dumpable: no other process of this user may read its environment or memory, nor trace it
        with pytest.raises(PermissionError):
            Path(f"/proc/{relay.pid}/environ").read_bytes()
        with pytest.raises(PermissionError):
            Path(f"/proc/{relay.pid}/mem").open("rb").read(1)
        assert lane_relay.RELAY_ENV == {"PATH": "/usr/bin:/bin"}  # all it was given
    finally:
        stopped = relay.stop()
    assert stopped.verified
    written = _files_text(tmp_path / "lanes")
    assert (
        b"lane relay: serving 3 lane(s)" in written
    )  # its log is there, and holds no secret
    assert FAKE_KEY.encode() not in written and not any(
        t.encode() in written for t in tokens
    )


def test_every_lane_gets_its_own_fresh_token_each_run(tmp_path, upstream):
    first = _start(tmp_path, upstream.base)
    try:
        a = {lane: first.token(lane) for lane in LANES}
        status, body = _post(first, "fork", token=a["helper"])
        assert (
            status == 403 and json.loads(body)["error"]["reason"] == "other_lane_token"
        )
    finally:
        first.stop()
    second = _start(tmp_path, upstream.base)
    try:
        b = {lane: second.token(lane) for lane in LANES}
        assert (
            _post(second, "writer", token=a["writer"])[0] == 403
        )  # last run's token is dead
    finally:
        second.stop()
    assert len(set(a.values()) | set(b.values())) == 6
    assert all(len(t) >= 43 for t in [*a.values(), *b.values()])


def test_stop_closes_the_lifeline_and_verifies_the_end(tmp_path, upstream):
    relay = _start(tmp_path, upstream.base)
    pid, port = relay.pid, relay.port
    stopped = relay.stop()
    assert (stopped.verified, stopped.how, stopped.returncode) == (True, "lifeline", 0)
    assert relay.stop() is stopped  # idempotent
    assert _gone(pid, 1)
    with pytest.raises(OSError):
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/writer/v1/chat/completions", data=b"{}", timeout=5
        )
    log = (tmp_path / "lanes" / "relay.log").read_text()
    assert "stopping (lifeline closed)" in log and "stopped; 0 call(s) cut" in log


def test_sigterm_stops_it_the_same_way(tmp_path, upstream):
    relay = _start(tmp_path, upstream.base)
    os.kill(relay.pid, signal.SIGTERM)
    assert _gone(relay.pid)
    assert relay.stop().verified
    assert "stopping (terminated)" in (tmp_path / "lanes" / "relay.log").read_text()


@pytest.mark.timeout(180)
def test_a_call_still_in_flight_at_the_stop_is_cut_and_journaled_cancelled(tmp_path):
    up = Upstream("hold")
    relay = _start(tmp_path, up.base)
    got: dict = {}
    t = threading.Thread(
        target=lambda: got.update(r=_try(lambda: _post(relay, "fork", timeout=120)))
    )
    t.start()
    try:
        while not up.requests:
            time.sleep(0.05)
        started = time.monotonic()
        stopped = relay.stop()
        took = time.monotonic() - started
        t.join(60)
    finally:
        up.close()
    assert stopped.verified and stopped.how == "lifeline"
    from unify.lanes.relay_server import DRAIN_S

    assert DRAIN_S <= took < DRAIN_S + 10
    assert isinstance(got["r"], Exception) or got["r"][0] != 200
    rows = [
        json.loads(x)
        for x in (tmp_path / "lanes" / "fork.jsonl").read_text().splitlines()
    ]
    assert [r["origin"] for r in rows] == ["request_started", "response"]
    assert (rows[1]["status"], rows[1]["error_type"], rows[1]["account_charge"]) == (
        "cancelled",
        "stopping",
        None,
    )
    assert "1 call(s) cut" in (tmp_path / "lanes" / "relay.log").read_text()


def _try(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return exc


OWNER = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
from unify.lanes.relay import LaneRelay
relay = LaneRelay.start(key=sys.argv[2], lanes={"fork": [sys.argv[5]]}, cap_usd="1", worst_case_usd={sys.argv[5]: "0.65625"}, journal_dir=sys.argv[3], upstream=sys.argv[4])
print(json.dumps({"pid": relay.pid}), flush=True)
time.sleep(600)
"""


@pytest.mark.parametrize("how", [signal.SIGKILL, signal.SIGTERM])
def test_the_relay_dies_with_its_owner(tmp_path, upstream, how):
    root = Path(lane_relay.__file__).resolve().parents[2]
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            OWNER,
            str(root),
            FAKE_KEY,
            str(tmp_path / "lanes"),
            upstream.base,
            MODEL,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        pid = json.loads(owner.stdout.readline())["pid"]
        assert not _gone(pid, 0.5)
        owner.send_signal(how)
        owner.wait(30)
        assert _gone(pid), "the relay outlived its owner"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(30)


def test_a_refused_start_names_its_rule_and_leaves_no_process(tmp_path, upstream):
    with pytest.raises(LaneRelayError) as err:
        LaneRelay.start(
            key=FAKE_KEY,
            lanes={"writer": ["openai/gpt-6-sol"]},  # no worst case for it
            cap_usd="2.00",
            worst_case_usd=WORST,
            journal_dir=tmp_path / "lanes",
            upstream=upstream.base,
        )
    assert (
        str(err.value)
        == "the lane relay refused to start: every lane model needs a worst case per call"
    )
    assert FAKE_KEY not in str(err.value)
    out = subprocess.run(
        ["pgrep", "-f", "relay_server.py"], capture_output=True, text=True
    )
    mine = [p for p in out.stdout.split() if _cwd_is(int(p), tmp_path / "lanes")]
    assert mine == []


def _cwd_is(pid: int, folder: Path) -> bool:
    try:
        return os.readlink(f"/proc/{pid}/cwd") == str(folder.resolve())
    except OSError:
        return False
