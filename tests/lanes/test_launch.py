"""The ``unify`` command's side of the lanes (unify/lanes/launch.py): when it serves the lanes and when it leaves the
command as it is, how each lane's route reaches the controller (an inherited descriptor per token, as the lane
contract says, read by the controller's real settings code), and that the relay ends with the run, with a kill of the
command, and with a signal passed on to the controller.

The controller here is a stand-in script that imports the real settings and switch code, reports what it got
(booleans and names, never a token), and calls each lane through the relay. The provider is a stand-in and the key a
fake.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import unify
from tests.lanes.upstream import FAKE_KEY, MODEL, Upstream
from tests.lanes.test_relay_process import _gone
from unify.lanes import launch
from unify.lanes.relay import LaneRelay

ROOT = Path(unify.__file__).resolve().parents[1]


def _settings(**over) -> SimpleNamespace:
    base = {
        "UNIFY_MEMORY": "on",
        "UNIFY_MODEL": f"{MODEL}@openrouter",
        "UNIFY_MEMORY_WRITER_MODEL": "openai/gpt-6-sol",
        "UNIFY_MEMORY_WRITER_SERVICE_CAP_USD": "2.00",
    }
    base.update(over)
    return SimpleNamespace(**base)


ENV = {"OPENROUTER_API_KEY": FAKE_KEY}


def test_with_memory_on_every_lane_is_planned_on_its_model_under_one_cap():
    plan, why = launch.plan(_settings(), ENV)
    assert why is None
    assert plan.lanes == {
        "writer": ("openai/gpt-6-sol",),
        "helper": (MODEL,),
        "fork": (MODEL,),
        "embed": (launch.EMBED_MODEL,),
    }
    assert plan.worst[launch.EMBED_MODEL] == launch.EMBED_WORST_CASE_USD
    assert plan.cap == Decimal("2.00")
    assert plan.worst[MODEL] == Decimal("0.65625") and plan.worst[
        "openai/gpt-6-sol"
    ] == Decimal("13.125")
    assert (
        len(plan.worst) > 100
    )  # the whole pinned table: an old call of another model is booked at its own bound
    pinned = launch.plan(
        _settings(
            UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION='{"input": "1.25", "output": "10"}'
        ),
        ENV,
    )[0]
    assert pinned.prices == {
        "openai/gpt-6-sol": {"input": Decimal("1.25"), "output": Decimal("10")},
        launch.EMBED_MODEL: {"input": Decimal("0.02"), "output": Decimal("0")},
    }


@pytest.mark.parametrize(
    "settings, environ",
    [
        (_settings(UNIFY_MEMORY=""), ENV),  # memory off
        (
            _settings(UNIFY_MEMORY_FROZEN="on"),
            ENV,
        ),  # frozen: no writer, fork or service runs
        (
            _settings(UNIFY_MEMORY_LIBRARY_FORK_BASE_URL="http://127.0.0.1:9/fork/v1"),
            ENV,
        ),  # given (settings)
        (
            _settings(),
            {**ENV, "unify_memory_writer_token_fd": "5"},
        ),  # given (environment, any case)
    ],
)
def test_no_lane_is_served_when_memory_wants_none_or_they_are_given(settings, environ):
    assert launch.plan(settings, environ) == (None, None)


@pytest.mark.parametrize(
    "settings, environ, rule",
    [
        (_settings(), {}, "OPENROUTER_API_KEY is not set"),
        (
            _settings(UNIFY_MEMORY_WRITER_SERVICE_CAP_USD=""),
            ENV,
            "(the memory spend cap) is not set",
        ),
        (
            _settings(
                UNIFY_MODEL="claude-x@anthropic",
                UNIFY_MEMORY_WRITER_MODEL="m@anthropic",
            ),
            ENV,
            "no lane's model is an OpenRouter model",
        ),
    ],
)
def test_lanes_that_cannot_be_served_leave_the_command_as_it_is_with_a_note(
    settings, environ, rule
):
    plan, why = launch.plan(settings, environ)
    assert plan is None and rule in why and FAKE_KEY not in why


def test_a_lane_whose_model_cannot_be_served_is_left_out_by_name():
    plan, why = launch.plan(_settings(UNIFY_MODEL="claude-x@anthropic"), ENV)
    assert set(plan.lanes) == {"writer", "embed"} and plan.lanes["writer"] == (
        "openai/gpt-6-sol",
    )
    assert (
        why
        == "memory: no helper, fork lane (its model is not an OpenRouter model the worst-case table bounds)"
    )


def test_the_handover_is_one_fresh_inherited_pipe_per_lane_token(tmp_path):
    up = Upstream()
    relay = LaneRelay.start(
        key=FAKE_KEY,
        lanes={"writer": [MODEL], "helper": [MODEL], "fork": [MODEL]},
        cap_usd="2",
        worst_case_usd={MODEL: "0.65625"},
        journal_dir=tmp_path / "lanes",
        upstream=up.base,
    )
    try:
        env, fds = launch.handover(
            relay, {"PATH": "/usr/bin", "UNIFY_MEMORY_WRITER_TOKEN": "stale"}
        )
        try:
            assert "UNIFY_MEMORY_WRITER_TOKEN" not in env
            assert env["UNIFY_MEMORY_WRITER_JOURNAL"] == str(relay.journal("writer"))
            for lane, (base_name, fd_name) in launch.LANE_ENV.items():
                assert env[base_name] == f"http://127.0.0.1:{relay.port}/{lane}/v1"
                fd = int(env[fd_name])
                assert fd in fds and fd > 2 and os.get_inheritable(fd)
                assert os.read(fd, 4097) == (relay.token(lane) + "\n").encode()
                assert (
                    os.read(fd, 1) == b""
                )  # its write end is closed: the reader reaches its end
            assert not any(
                relay.token(lane) in v for lane in launch.LANE_ENV for v in env.values()
            )
        finally:
            for fd in fds:
                os.close(fd)
    finally:
        relay.stop()
        up.close()


# --- the command as a process ---------------------------------------------------------------------------------------

CONTROLLER = r"""
import json, os, signal, sys, time, urllib.request, urllib.error
if sys.argv[1:] == ["sleep"]:
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(43))  # before it reports, so a SIGTERM after finds it
watch = ("BASE_URL", "TOKEN", "JOURNAL")
before = {k: v for k, v in os.environ.items() if k.startswith("UNIFY_") and any(w in k for w in watch)}
fds = [int(v) for k, v in before.items() if k.endswith("_TOKEN_FD")]
import unify.settings  # reads each lane's descriptor once, as the controller does
from unify.settings import SETTINGS
from unify.memory.integration import switch
fork, fork_why = switch.fork_lane_route()
helper, helper_why = switch.helper_lane_route()
writer = switch.writer_route(SETTINGS.UNIFY_MEMORY_WRITER_BASE_URL, switch.writer_token(SETTINGS))
routes = {
    "fork": (fork[0].rsplit("/chat/completions", 1)[0], fork[1]),
    "helper": (helper[0], helper[1].get_secret_value()),
    "writer": (writer[0], writer[1].get_secret_value()),
}
def post(base, token):
    req = urllib.request.Request(base + "/chat/completions", method="POST",
        data=json.dumps({"model": os.environ["UNIFY_MEMORY_WRITER_MODEL"], "messages": []}).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
def is_open(fd):
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False
from unify.lanes import embed_token
etok = embed_token.token()
def post_embed(url, token):
    req = urllib.request.Request(url, method="POST",
        data=json.dumps({"model": "openai/text-embedding-3-small", "input": ["x"]}).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
tokens = [t for _, t in routes.values()] + [etok]
print(json.dumps({
    "pid": os.getpid(),
    "names_before": sorted(before),
    "names_after": sorted(k for k in os.environ if k.startswith("UNIFY_") and any(w in k for w in watch)),
    "status": {**{lane: post(*routes[lane]) for lane in sorted(routes)}, "embed": post_embed(SETTINGS.UNIFY_EMBED_URL, etok)},
    "crossed": post(routes["writer"][0], routes["fork"][1]),
    "token_in_env": any(t in v for t in tokens for v in os.environ.values()),
    "distinct": len(set(tokens)) == 4,
    "fds_open": [is_open(fd) for fd in fds],
    "journal": SETTINGS.UNIFY_MEMORY_WRITER_JOURNAL,
    "embed_url": SETTINGS.UNIFY_EMBED_URL,
    "bases": {lane: routes[lane][0] for lane in routes},
}), flush=True)
if sys.argv[1:] == ["sleep"]:
    time.sleep(600)
sys.exit(7)
"""

LAUNCHER = r"""
import json, sys
from pathlib import Path
from unify.lanes import launch
from unify.lanes.relay import LaneRelay
class Recording(LaneRelay):
    @classmethod
    def start(cls, **kw):
        relay = super().start(**kw)
        Path(sys.argv[3]).write_text(json.dumps({"pid": relay.pid, "port": relay.port}))
        return relay
launch.LaneRelay = Recording
code = launch.run_controller(sys.argv[5:], Path(sys.argv[1]), controller=[sys.executable, sys.argv[2]], upstream=sys.argv[4])
sys.exit(99 if code is None else code)
"""


@pytest.fixture
def command(tmp_path):
    """Start the command (the launcher) with memory on, a fake key and a stand-in provider; yields a starter."""
    up = Upstream()
    (tmp_path / "controller.py").write_text(CONTROLLER)
    (tmp_path / "launcher.py").write_text(LAUNCHER)
    home = tmp_path / "home"
    home.mkdir()
    env = {
        k: v for k, v in os.environ.items() if not k.upper().startswith("UNIFY_MEMORY")
    }
    env.update(
        {
            "UNIFY_MEMORY": "on",
            "UNIFY_MEMORY_WRITER_SERVICE_CAP_USD": "2.00",
            "UNIFY_MODEL": f"{MODEL}@openrouter",
            "UNIFY_MEMORY_WRITER_MODEL": MODEL,
            "OPENROUTER_API_KEY": FAKE_KEY,
            "UNIFY_HOME": str(home),
            "PYTHONPATH": str(ROOT),
        }
    )
    started: list[subprocess.Popen] = []

    def start(*args: str) -> tuple[subprocess.Popen, Path]:
        pidfile = tmp_path / "relay.json"
        proc = subprocess.Popen(
            [
                sys.executable,
                str(tmp_path / "launcher.py"),
                str(home),
                str(tmp_path / "controller.py"),
                str(pidfile),
                up.base,
                *args,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        started.append(proc)
        return proc, pidfile

    yield SimpleNamespace(start=start, home=home, upstream=up)
    for proc in started:
        if proc.poll() is None:
            proc.kill()
            proc.wait(30)
    up.close()


@pytest.mark.timeout(300)
def test_the_controller_gets_each_lane_on_its_descriptor_and_the_relay_ends_with_the_run(
    command,
):
    proc, pidfile = command.start()
    out, err = proc.communicate(timeout=240)
    assert proc.returncode == 7, err[-3000:]
    report = json.loads(out.strip().splitlines()[-1])
    relay = json.loads(pidfile.read_text())
    lanes = command.home / "memory-state" / "lanes"
    assert report["names_before"] == sorted(
        [
            *(name for pair in launch.LANE_ENV.values() for name in pair),
            "UNIFY_MEMORY_WRITER_JOURNAL",
            "UNIFY_EMBED_TOKEN_FD",
        ]
    )
    # the settings took every token off its descriptor, closed it, and left no token or descriptor name behind
    assert report["names_after"] == [
        "UNIFY_MEMORY_WRITER_BASE_URL",
        "UNIFY_MEMORY_WRITER_JOURNAL",
    ]
    assert (
        report["fds_open"] == [False, False, False, False]
        and report["token_in_env"] is False
        and report["distinct"]
    )
    assert report["bases"] == {
        lane: f"http://127.0.0.1:{relay['port']}/{lane}/v1" for lane in launch.LANE_ENV
    }
    assert (
        report["status"] == {"fork": 200, "helper": 200, "writer": 200, "embed": 200}
        and report["crossed"] == 403
    )
    assert report["journal"] == str(lanes / "writer.jsonl")
    assert _gone(relay["pid"])
    assert (
        report["embed_url"] == f"http://127.0.0.1:{relay['port']}/embed/v1/embeddings"
    )
    for lane in [*launch.LANE_ENV, "embed"]:
        rows = [
            json.loads(x) for x in (lanes / f"{lane}.jsonl").read_text().splitlines()
        ]
        assert [r["status"] for r in rows] == ["started", "completed"]
    assert FAKE_KEY not in out + err
    assert FAKE_KEY.encode() not in b"".join(
        p.read_bytes() for p in lanes.iterdir() if p.is_file()
    )


@pytest.mark.timeout(300)
def test_killing_the_command_ends_the_relay(command):
    proc, pidfile = command.start("sleep")
    report = json.loads(proc.stdout.readline())
    relay = json.loads(pidfile.read_text())
    try:
        proc.send_signal(signal.SIGKILL)
        proc.wait(30)
        assert _gone(relay["pid"]), "the relay outlived the command"
    finally:
        try:
            os.kill(report["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.timeout(300)
def test_a_sigterm_to_the_command_reaches_the_controller_and_the_relay_then_ends(
    command,
):
    proc, pidfile = command.start("sleep")
    json.loads(proc.stdout.readline())
    relay = json.loads(pidfile.read_text())
    proc.send_signal(signal.SIGTERM)
    proc.wait(120)
    assert proc.returncode == 43  # the controller's own exit, passed back
    assert _gone(relay["pid"])


def test_with_memory_off_the_command_runs_in_this_process_and_starts_nothing(tmp_path):
    notes: list[str] = []
    got = launch.run_controller(
        ["act", "hi"],
        tmp_path,
        settings=_settings(UNIFY_MEMORY=""),
        environ=dict(ENV),
        note=notes.append,
    )
    assert got is None and notes == [] and not (tmp_path / "memory-state").exists()


def test_a_relay_that_cannot_start_leaves_the_command_as_it_is(tmp_path, monkeypatch):
    notes: list[str] = []
    monkeypatch.setattr(
        launch,
        "_bounds",
        lambda: {
            "openai/gpt-6-sol": Decimal("13.125"),
            MODEL: Decimal("0.65625"),
            "not a model id": Decimal("1"),  # the relay refuses its configuration
        },
    )
    got = launch.run_controller(
        ["act", "hi"],
        tmp_path,
        settings=_settings(),
        environ=dict(ENV),
        note=notes.append,
    )
    assert got is None
    assert notes == [
        "memory has no model lanes: the lane relay refused to start: a worst case names a malformed model"
    ]


def _stand_in_controller(code: int) -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


def test_the_writer_service_is_stopped_while_the_relay_still_serves_then_the_relay(
    tmp_path,
):
    up = Upstream()
    seen: dict = {}

    def drain(home):
        journal = home / "memory-state" / "lanes" / "writer.jsonl"
        seen["home"], seen["journal"] = home, journal.exists()
        seen["log"] = (journal.parent / "relay.log").read_text()

    try:
        got = launch.run_controller(
            [],
            tmp_path,
            settings=_settings(),
            environ=dict(ENV),
            controller=_stand_in_controller(5),
            upstream=up.base,
            drain=drain,
            note=lambda _t: None,
        )
    finally:
        up.close()
    assert got == 5  # the controller's own exit code
    assert (
        seen["home"] == tmp_path and seen["journal"] and "stopping" not in seen["log"]
    )  # drained before the stop
    log = (tmp_path / "memory-state" / "lanes" / "relay.log").read_text()
    assert "stopping (lifeline closed)" in log and "stopped; 0 call(s) cut" in log


def test_an_interrupt_while_the_writer_service_stops_skips_the_wait(tmp_path):
    up = Upstream()
    notes: list[str] = []

    def slow_drain(home):
        os.kill(
            os.getpid(), signal.SIGINT
        )  # the user's Ctrl-C while the writer's pass in flight finishes
        time.sleep(60)

    started = time.monotonic()
    try:
        got = launch.run_controller(
            [],
            tmp_path,
            settings=_settings(),
            environ=dict(ENV),
            controller=_stand_in_controller(0),
            upstream=up.base,
            drain=slow_drain,
            note=notes.append,
        )
    finally:
        up.close()
    assert got == 0 and time.monotonic() - started < 30
    assert notes == [
        "memory: not waiting; the writer stops after its pass, and the next run tidies up"
    ]
    assert (
        "stopped; 0 call(s) cut"
        in (tmp_path / "memory-state" / "lanes" / "relay.log").read_text()
    )


def _service_left(home: Path, *, state="running", marker=False) -> None:
    folder = home / "memory-state" / "memory-service"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "service.json").write_text(json.dumps({"state": state}))
    if marker:
        (folder / "service.stop").write_text(json.dumps({"by": "suspend"}))


def _recording_drain(home: Path, calls: list):
    def drain(at):
        calls.append((home / "memory-state" / "lanes" / "relay.log").exists())

    return drain


@pytest.mark.parametrize(
    "state, marker", [("running", False), ("idle", False), ("stopped", True)]
)
def test_a_run_alone_on_the_home_first_stops_a_service_an_ended_run_left(
    tmp_path, state, marker
):
    """A service still running (or a stop cut short) when no other run holds the home holds dead lanes: it is
    stopped before this run's relay and controller start, and again at this run's end.
    """
    _service_left(tmp_path, state=state, marker=marker)
    up = Upstream()
    calls: list = []
    try:
        got = launch.run_controller(
            [],
            tmp_path,
            settings=_settings(),
            environ=dict(ENV),
            controller=_stand_in_controller(0),
            upstream=up.base,
            drain=_recording_drain(tmp_path, calls),
            note=lambda _t: None,
        )
    finally:
        up.close()
    assert got == 0
    assert calls == [False, True]  # before the relay started, then at the end


def test_a_service_stopped_cleanly_is_left_alone_at_the_start(tmp_path):
    _service_left(tmp_path, state="stopped")
    up = Upstream()
    calls: list = []
    try:
        launch.run_controller(
            [],
            tmp_path,
            settings=_settings(),
            environ=dict(ENV),
            controller=_stand_in_controller(0),
            upstream=up.base,
            drain=_recording_drain(tmp_path, calls),
            note=lambda _t: None,
        )
    finally:
        up.close()
    assert calls == [True]  # at the end only


def test_a_service_while_another_run_holds_the_home_is_left_to_that_run(tmp_path):
    import fcntl

    _service_left(tmp_path)
    folder = tmp_path / "memory-state" / "lanes"
    folder.mkdir(parents=True)
    other = os.open(folder / "runs.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(other, fcntl.LOCK_SH)  # another run, alive
    up = Upstream()
    calls: list = []
    try:
        launch.run_controller(
            [],
            tmp_path,
            settings=_settings(),
            environ=dict(ENV),
            controller=_stand_in_controller(0),
            upstream=up.base,
            drain=_recording_drain(tmp_path, calls),
            note=lambda _t: None,
        )
    finally:
        os.close(other)
        up.close()
    assert calls == [True]  # this run's own end only


def test_the_end_of_a_run_gives_the_writers_pass_at_most_five_minutes_then_suspends_it(
    tmp_path, monkeypatch
):
    from unify.memory.integration import async_pass

    seen: list = []
    monkeypatch.setattr(
        async_pass, "drain", lambda paths, **kw: seen.append((paths.home, kw))
    )
    launch._drain_writer(tmp_path)
    assert launch.EXIT_WAIT_S == 300.0
    assert seen == [
        (tmp_path, {"end": "suspend", "wait_s": 300.0, "service_wait_s": 300.0})
    ]


@pytest.mark.parametrize(
    "settings, environ",
    [
        (_settings(UNIFY_EMBED_URL="http://127.0.0.1:9/mine"), ENV),
        (_settings(), {**ENV, "UNIFY_EMBED_URL": "http://127.0.0.1:9/mine"}),
        (_settings(), {**ENV, "UNIFY_EMBED_TOKEN_FD": "7"}),
        (_settings(UNIFY_LOCAL_EMBEDDINGS=True), ENV),
    ],
)
def test_an_embedding_route_the_user_set_is_never_replaced(settings, environ):
    plan, _why = launch.plan(settings, environ)
    assert "embed" not in plan.lanes and set(plan.lanes) == {"writer", "helper", "fork"}
