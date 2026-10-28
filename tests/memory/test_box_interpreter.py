"""The persistent confined interpreter of a memory agent (unify/common/box_interpreter.py): the worker child in the
memory box, its restart and kill rules, the staged ``tools`` module, the ``sh`` helper and the open log. Keyless;
the box tests need bubblewrap and prlimit, as the other box tests do."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import pytest

from tests.memory.box_channel_standin import StandInChannel, fd_listing_code
from unify.common import box_interpreter as bi
from unify.common.box_interpreter import Binds, BoxInterpreter, MemoryProfile
from unify.common.box_shell_source import SOURCE as SHELL_SOURCE
from unify.memory import sandbox_run as sr

needs_box = pytest.mark.skipif(
    shutil.which("bwrap") is None or not os.access(sr.PRLIMIT, os.X_OK),
    reason="bubblewrap and prlimit required",
)

LOST = "the interpreter was restarted: variables are gone, files remain"


async def _start(tmp_path: Path, *, channel=None, stub=None, limit: float = 30.0, **profile_kw) -> BoxInterpreter:
    inputs, memory = tmp_path / "inputs", tmp_path / "memory"
    inputs.mkdir(exist_ok=True)
    memory.mkdir(exist_ok=True)
    (inputs / "seed.txt").write_text("seed\n")
    profile_kw.setdefault("opens_path", tmp_path / "record" / "box-opens.jsonl")
    profile = MemoryProfile(pass_id="pass-1", **profile_kw)
    return await BoxInterpreter.start(
        agent="writer",
        profile=profile,
        binds=Binds(ro={inputs: "/inputs"}, rw={memory: "/memory"}),
        env={},
        channel=channel,
        stub_source=stub,
        cell_limit_s=limit,
    )


# --- without a box ---------------------------------------------------------------------------------------------


def test_stage_read_only_is_new_and_never_through_a_link(tmp_path):
    path = bi.stage_read_only(tmp_path, "tools.py", b"VALUE = 1\n")
    assert path.read_bytes() == b"VALUE = 1\n"
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):  # created new, never over an existing file
        bi.stage_read_only(tmp_path, "tools.py", b"other")
    (tmp_path / "decoy.py").write_text("decoy")
    (tmp_path / "linked.py").symlink_to(tmp_path / "decoy.py")
    with pytest.raises(FileExistsError):  # never through a link
        bi.stage_read_only(tmp_path, "linked.py", b"other")
    assert (tmp_path / "decoy.py").read_text() == "decoy"


def test_open_log_from_audit_records():
    audit = {
        "records": [
            {"event": "open", "path": "/inputs/a.txt", "mode": "rb", "tid": 1, "cell_thread": True},
            {"event": "open", "path": "/memory/b.txt", "mode": "a+", "tid": 1, "cell_thread": True},
            {"event": "open", "path": "/memory/c.txt", "mode": "w", "tid": 1, "cell_thread": True},
            {"event": "open", "path": "/memory/d.txt", "mode": "r+", "tid": 1, "cell_thread": True},
            {"event": "open", "path": "/work/e.txt", "mode": "w", "tid": 1, "cell_thread": True},
            {"event": "open", "path": "/memory/../etc/passwd", "mode": "r", "tid": 1},
            {"event": "open", "path": 7, "mode": "r"},
            {"event": "os.listdir", "path": "/memory"},
        ],
        "dropped": 2,
        "failed": 0,
        "bytes": 10,
    }
    log = bi.open_log(audit, "on")
    assert log["rows"] == [
        {"event": "open", "path": "/inputs/a.txt", "mode": "r"},
        {"event": "open", "path": "/memory/b.txt", "mode": "a"},
        {"event": "open", "path": "/memory/c.txt", "mode": "w"},
        {"event": "open", "path": "/memory/d.txt", "mode": "w"},
    ]
    assert (log["dropped"], log["dead"], log["reason"]) == (4, False, "ok")
    assert bi.open_log(audit, "ImportError: no") == {"rows": [], "dropped": 0, "dead": True, "reason": "no_start"}
    assert bi.open_log(None, "on")["reason"] == "reader_failed"
    assert sr.box_open_fields(log) == {
        "box_opens": 4,
        "box_opens_dropped": 4,
        "box_hook_dead": False,
        "box_hook_reason": "ok",
    }


def test_cell_text_puts_restart_notes_around_the_output():
    res = bi.CellResult(output="", error="Traceback ...", notes_before=["(a)"], notes_after=["(b)"])
    assert res.text() == "(a)\nTraceback ...\n(b)"
    assert bi.CellResult(output="").text() == "(no output)"


def test_the_cell_wrapper_runs_code_at_top_level():
    src = bi.wrap_cell("x = 1\nx + 1", "<cell 1>", last_value=True)
    ns: dict = {"__builtins__": {"__import__": __import__}}
    exec(compile(src, "<string>", "exec"), ns)
    value = asyncio.run(ns["__exec_wrapper"]())
    assert value == 2 and ns["x"] == 1


async def _later(value):
    await asyncio.sleep(0)
    return value


def _run_wrapped(code: str, ns: dict, *, last_value: bool):
    exec(compile(bi.wrap_cell(code, "<cell 1>", last_value=last_value), "<string>", "exec"), ns)
    return asyncio.run(ns["__exec_wrapper"]())


@pytest.mark.parametrize("last_value", [True, False])
def test_a_top_level_await_runs_and_its_names_persist(last_value):
    ns: dict = {"__builtins__": {"__import__": __import__}, "later": _later}
    value = _run_wrapped("x = await later(5)\ny = x + 1\n", ns, last_value=last_value)
    assert value is None and ns["x"] == 5 and ns["y"] == 6
    ns = {"__builtins__": __builtins__, "later": _later}
    _run_wrapped(
        "async def count():\n    for i in range(3):\n        yield await later(i)\nseen = [i async for i in count()]\n",
        ns,
        last_value=last_value,
    )
    assert ns["seen"] == [0, 1, 2]


def test_a_last_value_await_returns_its_value():
    ns: dict = {"__builtins__": {"__import__": __import__}, "later": _later}
    assert _run_wrapped("x = 2\nawait later(x * 21)", ns, last_value=True) == 42
    assert ns["x"] == 2


def test_a_cell_without_await_is_unchanged_and_a_coroutine_value_is_returned_not_awaited():
    ns: dict = {"__builtins__": {"__import__": __import__}, "make": lambda: _later(7)}
    assert _run_wrapped("a = 1\nb = a + 1\nb * 10", ns, last_value=True) == 20 and ns["b"] == 2
    value = _run_wrapped("make()", ns, last_value=True)
    assert asyncio.iscoroutine(value)
    value.close()


@pytest.mark.parametrize("last_value", [True, False])
def test_a_syntax_error_still_reports_the_cell_line(last_value):
    with pytest.raises(SyntaxError) as raised:
        _run_wrapped("x = 1\ny = (2,\n", {"__builtins__": {"__import__": __import__}}, last_value=last_value)
    assert raised.value.filename == "<cell 1>" and raised.value.text is not None


# --- in the box -----------------------------------------------------------------------------------------------


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_state_persists_across_cells(tmp_path):
    interp = await _start(tmp_path)
    try:
        r1 = await interp.run_cell("x = 41\ndef bump(v):\n    return v + 1\n")
        assert r1.error is None and not r1.restarted, r1.text()
        r2 = await interp.run_cell("print(bump(x))")
        assert r2.output.strip() == "42" and r2.text() == "42"
        r3 = await interp.run_cell("raise ValueError('boom')")
        assert r3.error is not None and "ValueError: boom" in r3.error
        assert "__exec_wrapper" not in r3.error and '<cell 3>", line 1' in r3.error
        r4 = await interp.run_cell("print(x)")
        assert r4.output.strip() == "41" and not r4.restarted
    finally:
        await interp.stop()
    assert not interp.box_alive()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_top_level_await_runs_in_the_box_and_its_names_persist(tmp_path):
    interp = await _start(tmp_path)
    try:
        r1 = await interp.run_cell("import asyncio\nv = await asyncio.sleep(0, 41)\nprint('AWAITED', v)")
        assert r1.error is None and r1.output.strip() == "AWAITED 41", r1.text()
        r2 = await interp.run_cell("print(v + 1)")
        assert r2.output.strip() == "42" and not r2.restarted
        r3 = await interp.run_cell("await asyncio.sleep(0)\nraise ValueError('after the await')")
        assert r3.error is not None and "ValueError: after the await" in r3.error
        assert "__exec_wrapper" not in r3.error and '<cell 3>", line 2' in r3.error
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_timed_out_cell_restarts_the_interpreter_and_says_so(tmp_path):
    interp = await _start(tmp_path, limit=2.0)
    try:
        await interp.run_cell("y = 1\nopen('/memory/kept.txt', 'w').write('kept')")
        started = time.monotonic()
        r = await interp.run_cell("import time\ntime.sleep(60)")
        assert time.monotonic() - started < 30
        assert r.timed_out and r.restarted
        assert r.notes_after == [f"(the cell timed out after 2 s; {LOST})"]
        assert r.text().endswith(f"(the cell timed out after 2 s; {LOST})")
        assert not interp.box_alive()
        r2 = await interp.run_cell("print(globals().get('y'), open('/memory/kept.txt').read())")
        assert r2.output.strip() == "None kept"
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "code, what",
    [
        ("import threading, time\nthreading.Thread(target=time.sleep, args=(60,), daemon=True).start()", "a thread"),
        ("import subprocess\nsubprocess.Popen(['sleep', '60'])", "a process"),
    ],
)
async def test_something_left_running_forces_a_restart(tmp_path, code, what):
    interp = await _start(tmp_path)
    try:
        await interp.run_cell("z = 3")
        r = await interp.run_cell(code)
        assert r.restarted and not r.timed_out
        assert r.notes_after == [f"({what} this cell started was still running after it ended; {LOST})"]
        assert not interp.box_alive()
        r2 = await interp.run_cell("print(globals().get('z'))")
        assert r2.output.strip() == "None"
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_finished_threads_and_processes_do_not_restart_it(tmp_path):
    interp = await _start(tmp_path)
    try:
        r = await interp.run_cell(
            "import subprocess, threading\n"
            "t = threading.Thread(target=print, args=('from a thread',))\n"
            "t.start(); t.join()\n"
            "print(subprocess.run(['echo', 'from a process'], capture_output=True, text=True).stdout.strip())\n"
            "kept = 5\n",
        )
        assert not r.restarted, r.text()
        assert r.output.split("\n")[:2] == ["from a thread", "from a process"]
        assert (await interp.run_cell("print(kept)")).output.strip() == "5"
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_kill_is_verified(tmp_path, monkeypatch):
    interp = await _start(tmp_path)
    pid = interp._backend.host_pid
    assert sr._descendants(pid)
    monkeypatch.setattr(sr, "_KILL_GRACE_S", 0.5)
    # a survivor that SIGKILL did not remove (the kill is shared: unify/common/process_tree.py)
    monkeypatch.setattr(sr.process_tree, "alive", lambda p, start: True)
    with pytest.raises(RuntimeError, match="survived SIGKILL"):
        await interp.stop()
    monkeypatch.undo()
    deadline = time.monotonic() + 10
    while sr._descendants(pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert not sr._descendants(pid)  # the SIGKILL itself was sent


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_no_harness_object_is_proxied_to_a_memory_agent(tmp_path):
    interp = await _start(tmp_path)
    try:
        r = await interp.run_cell(
            "import gc\n"
            "w = [o for o in gc.get_objects() if type(o).__name__ == 'Worker' and hasattr(o, 'request_sync')][0]\n"
            "for op in ('call', 'describe', 'dir', 'doc', 'fn_begin'):\n"
            "    try:\n"
            "        w.request_sync(op, target={'root': 'functions', 'path': []}, args=[], kwargs={})\n"
            "        print(op, 'answered')\n"
            "    except Exception as exc:\n"
            "        print(op, type(exc).__name__, exc)\n",
        )
        lines = r.output.strip().split("\n")
        assert [ln.split(" ", 1)[0] for ln in lines] == ["call", "describe", "dir", "doc", "fn_begin"]
        for line in lines:
            assert line.split(" ", 2)[1:] == ["BoundaryRefusal", bi._NO_PROXY], line
        assert not r.restarted
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_code_is_never_in_an_argument_vector(tmp_path):
    interp = await _start(tmp_path)
    try:
        marker = "cell-code-marker-5d1c"
        task = asyncio.create_task(interp.run_cell(f"import time\nm = {marker!r}\ntime.sleep(2)"))
        await asyncio.sleep(1.0)
        pid = interp._backend.host_pid
        cmdlines = []
        for p in [pid, *(q for q, _ in sr._descendants(pid))]:
            try:
                cmdlines.append(Path(f"/proc/{p}/cmdline").read_bytes())
            except OSError:
                pass
        assert any(b"worker_child.py" in c and b"\0-I\0-S\0" in c for c in cmdlines), cmdlines
        assert all(marker.encode() not in c for c in cmdlines)
        r = await task
        assert r.error is None
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_stub_is_staged_read_only_and_imported_as_tools(tmp_path):
    interp = await _start(tmp_path, stub="VALUE = 7\n")
    try:
        r = await interp.run_cell(
            "import os, stat, sys\n"
            "print(tools.VALUE, tools.__file__, oct(stat.S_IMODE(os.stat(tools.__file__).st_mode)),"
            " sys.modules['tools'] is tools)\n"
            "for target in (tools.__file__, '/tools/other.py'):\n"
            "    try:\n"
            "        open(target, 'a').close()\n"
            "        print('writable')\n"
            "    except OSError:\n"
            "        print('read-only')\n",
        )
        assert r.output.strip().split("\n") == ["7 /tools/tools.py 0o444 True", "read-only", "read-only"], r.text()
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_sh_keeps_the_working_directory_and_exports(tmp_path):
    interp = await _start(tmp_path, stub=SHELL_SOURCE)
    try:
        r1 = await interp.run_cell(
            "r = tools.sh('mkdir -p /memory/sub && cd /memory/sub && export GREETING=hello; echo started')\n"
            "print(r.strip(), r.returncode, type(r).__name__, isinstance(r, str))\n",
        )
        assert r1.output.strip() == "started 0 ShellResult True", r1.text()
        assert not r1.restarted  # the shell ends with its call
        r2 = await interp.run_cell(
            "r = tools.sh('pwd; echo $GREETING; echo oops >&2; exit 3')\nprint(repr(str(r)), r.returncode)",
        )
        assert r2.output.strip() == "'/memory/sub\\nhello\\noops\\n' 3", r2.text()
        r3 = await interp.run_cell("print(tools.sh('pwd').strip())")
        assert r3.output.strip() == "/memory/sub"
        r4 = await interp.run_cell("r = tools.sh('sleep 30', timeout=1)\nprint(r.returncode, 'timed out' in r)")
        assert r4.output.strip() == "None True" and not r4.restarted, r4.text()
        r5 = await interp.run_cell("print(len(tools.sh('yes | head -c 600000')))")
        assert int(r5.output.strip()) < 300 * 1024
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_check_examples_runs_a_function_of_the_session_in_the_box(tmp_path):
    """The example check runs the candidate where the agent's cells run: a function defined in an earlier cell, on
    a path only the box has, with no channel at all (a host call would fail)."""
    from unify.common import tool_registry as R

    stub = R.render_stub(R.allow_list("writer", frozenset()), "writer")
    interp = await _start(tmp_path, stub=stub)
    try:
        r1 = await interp.run_cell("def first_line(path):\n    return open(path).read().splitlines()[0]\n")
        assert r1.error is None, r1.text()
        r2 = await interp.run_cell(
            "r = tools.check_examples(first_line, [('/inputs/seed.txt', 'seed'), {'input': '/inputs/seed.txt',"
            " 'expected': 'other'}, ('/inputs/none.txt', 'x')])\n"
            "print(r.splitlines()[0], r.passed, r.total)\n"
            "print([row['passed'] for row in r.results], r.results[2]['error'].split(':')[0])\n",
        )
        assert r2.output.strip().split("\n") == ["passed 1 of 3 1 3", "[True, False, False] FileNotFoundError"], (
            r2.text()
        )
        assert not r2.restarted
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_open_log_fields_are_filled_per_cell(tmp_path):
    interp = await _start(tmp_path)
    try:
        r = await interp.run_cell(
            "print(open('/inputs/seed.txt').read().strip())\n"
            "open('/memory/new.txt', 'w').write('x')\n"
            "open('/tmp/elsewhere.txt', 'w').write('y')\n",
            call=5,
        )
        assert r.output.strip() == "seed"
        assert sr.box_open_fields(r.opens) == {
            "box_opens": 2,
            "box_opens_dropped": 0,
            "box_hook_dead": False,
            "box_hook_reason": "ok",
        }
        r2 = await interp.run_cell("print(open('/memory/new.txt').read())")
        assert sr.box_open_fields(r2.opens)["box_opens"] == 1
        rows = [json.loads(ln) for ln in (tmp_path / "record" / "box-opens.jsonl").read_text().splitlines()]
        stamp = {"agent": "writer", "pass_id": "pass-1", "event": "open"}
        assert rows == [
            {**stamp, "cell": 1, "call": 5, "path": "/inputs/seed.txt", "mode": "r"},
            {**stamp, "cell": 1, "call": 5, "path": "/memory/new.txt", "mode": "w"},
            {**stamp, "cell": 2, "path": "/memory/new.txt", "mode": "r"},
        ]
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_closed_channel_restarts_it_with_a_new_channel(tmp_path):
    opened: list[StandInChannel] = []

    def open_channel() -> StandInChannel:
        opened.append(StandInChannel())
        return opened[-1]

    interp = await _start(tmp_path, channel=open_channel)
    try:
        await interp.run_cell("w = 1")
        assert opened[0].events[:3] == [("after_spawn",), ("cell_started", 1), ("cell_ended", 1)]
        opened[0].closed, opened[0].close_reason = True, "frame_between_cells"
        r = await interp.run_cell("print(globals().get('w'), __import__('os').environ['TOOLS_CHANNEL'])")
        assert r.notes_before == [f"(tool channel closed: frame_between_cells; {LOST})"] and r.restarted
        assert len(opened) == 2 and r.output.strip() == f"None {opened[1].env_value()}"
    finally:
        await interp.stop()
    assert opened[1].closed


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_quota_breach_kills_the_box(tmp_path):
    breach: list = [None]
    interp = await _start(tmp_path, measure=lambda: breach[0])
    try:
        r1 = await interp.run_cell("q = 1")
        assert r1.quota is None and interp.box_alive()
        breach[0] = "over quota: more than 3 bytes under /memory"
        r2 = await interp.run_cell("open('/memory/big.txt', 'w').write('xxxx')")
        assert r2.quota == breach[0] and r2.restarted
        assert r2.notes_after == [f"({breach[0]}; {LOST})"]
        assert not interp.box_alive()
        breach[0] = None
        assert await interp.measure_quota() is None
    finally:
        await interp.stop()


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_an_idle_interpreter_is_stopped_and_restarted_lazily(tmp_path):
    interp = await _start(tmp_path, idle_restart_s=0.5)
    try:
        await interp.run_cell("v = 1")
        await asyncio.sleep(2.0)
        assert not interp.box_alive()
        r = await interp.run_cell("print(globals().get('v'))")
        assert r.output.strip() == "None" and r.restarted
        assert r.notes_before == [
            "(the interpreter was idle for more than 0.5 s and was restarted: variables are gone, files remain)",
        ]
    finally:
        await interp.stop()


# The descriptors a memory agent's interpreter holds, by what they are. Nothing else may be open: no route token, no
# host file, no other pipe.
_OUTPUT_FILE = re.compile(r"/tmp/unify-worker-output-\w+ \(deleted\)")


def fds_match(fds: dict) -> bool:
    """Whether the box's descriptors are exactly :data:`EXPECTED_FDS`, the random-number device at most once (it is
    open or not depending on how the box's Python seeds itself)."""
    got = Counter(_kind(t) for t in fds.values())
    return got.get("urandom", 0) <= 1 and +(got - Counter({"urandom": 1})) == +(EXPECTED_FDS - Counter({"urandom": 1}))


def _kind(target: str) -> str:
    if target == "/dev/null":
        return "null"
    if _OUTPUT_FILE.fullmatch(target):
        return "output"
    if target.startswith("pipe:"):
        return "pipe"
    if target.startswith("socket:"):
        return "socket"
    if target == "anon_inode:[eventpoll]":
        return "epoll"
    if target == "/dev/urandom":
        return "urandom"
    return target


EXPECTED_FDS = Counter(
    {
        # 0 (the worker's stdin, replaced so a cell cannot read the protocol) and the descriptor it was opened on
        "null": 2,
        # 1 and 2 (whatever a subprocess or C code writes) and the worker's own handle on that private file
        "output": 3,
        # the protocol's two pipes (moved off 0 and 1) and the tool channel's two box ends
        "pipe": 4,
        # the event loop: its epoll descriptor and its wake-up socket pair
        "epoll": 1,
        "socket": 2,
        # the random-number device (found open in both the memory box and the actor box: a device, no host data);
        # a Python that reads its seed with getrandom() keeps none open, so it is counted apart (fds_match)
        "urandom": 1,
    },
)


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_box_holds_exactly_the_expected_descriptors(tmp_path):
    ch = StandInChannel()
    interp = await _start(tmp_path, channel=ch, stub="VALUE = 1\n")
    try:
        r = await interp.run_cell(fd_listing_code())
        got = json.loads(r.output.strip().splitlines()[-1])
        fds = {int(k): v for k, v in got["fds"].items()}
        assert got["channel"] == ch.env_value()
        req, rep = ch.box_fds()
        assert fds[req].startswith("pipe:") and fds[rep].startswith("pipe:")
        assert fds[0] == "/dev/null"
        assert _kind(fds[1]) == "output" and _kind(fds[2]) == "output"
        assert fds_match(fds), fds
    finally:
        await interp.stop()
