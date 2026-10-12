"""The box's open log: the observe-only audit hook staged as /inputs/sitecustomize.py, its pipe to the host, the
host-stamped box-opens.jsonl rows, the execute_code tool-event fields and the box interpreter's flags.
"""

import asyncio
import json
import os
import shutil
import time
from types import SimpleNamespace

import pytest

from unify.memory import helpers
from unify.memory import sandbox_run as sr
from unify.memory.trigger import PassRequest
from tests.memory.test_writer_pass import Turns, _call, _writer
from tests.memory.test_writer_library_tools import EPS

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)


def _inputs(tmp_path):
    inputs = tmp_path / "in"
    inputs.mkdir()
    sr.stage_open_hook(inputs)
    (inputs / "x.txt").write_text("x")
    return inputs


def test_the_staged_hook_is_the_module_and_read_only(tmp_path):
    inputs = _inputs(tmp_path)
    hook = inputs / "sitecustomize.py"
    assert hook.read_bytes() == sr.OPEN_HOOK_SOURCE.read_bytes()
    assert hook.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):  # created new, never over an existing file
        sr.stage_open_hook(inputs)


def test_host_parsing_keeps_only_hook_fields():
    row = sr._open_row(
        b'{"event": "open", "path": "/inputs/a", "mode": "r", "pass_id": "x", "call": 9}',
    )
    assert row == {"event": "open", "path": "/inputs/a", "mode": "r"}
    for bad in (
        b'{"event": "open", "path": "/etc/passwd", "mode": "r"}',
        b'{"event": "open", "path": "/inputs/../etc/passwd", "mode": "r"}',
        b'{"event": "open", "path": "/inputsx/a", "mode": "r"}',
        b'{"event": "open", "path": "/memory/a", "mode": "q"}',
        b'{"event": "exec", "path": "/memory/a", "mode": "r"}',
        b"not json",
    ):
        assert sr._open_row(bad) is None, bad


def test_only_pipes_enter_the_box(tmp_path):
    f = tmp_path / "f"
    f.write_text("")
    with open(f) as fh, pytest.raises(ValueError, match="only pipe descriptors"):
        sr.run_confined(["/bin/true"], pass_fds=(fh.fileno(),))


def test_no_box_means_no_measurement():
    assert sr.box_open_fields(None) == {
        "box_opens": 0,
        "box_opens_dropped": 0,
        "box_hook_dead": None,
        "box_hook_reason": None,
    }


@needs_bwrap
def test_a_flood_beyond_the_cap_is_counted_as_dropped_without_stalling(tmp_path):
    inputs = _inputs(tmp_path)
    code = "for _ in range(20000):\n    open('/inputs/x.txt').close()\nprint('done')\n"
    t0 = time.monotonic()
    with sr.BoxOpenLog(cap_bytes=2000) as log:
        r = sr.run_confined(
            [str(sr.PYTHON), "-s", "-c", code],
            ro={inputs: "/inputs"},
            env={"PYTHONPATH": "/inputs", **log.env},
            pass_fds=log.pass_fds,
            timeout_s=60,
        )
    res = log.result
    assert r.returncode == 0 and not r.timed_out, r.stderr
    assert r.stdout.split() == ["done"]
    assert time.monotonic() - t0 < 45
    assert res["dropped"] > 0 and not res["dead"]
    assert 0 < len(res["rows"]) and sum(len(json.dumps(o)) for o in res["rows"]) <= 2000
    assert all(
        o == {"event": "open", "path": "/inputs/x.txt", "mode": "r"}
        for o in res["rows"]
    )


@needs_bwrap
def test_a_cell_that_opens_nothing_in_the_areas_is_measured_not_dead(tmp_path):
    inputs = _inputs(tmp_path)
    with sr.BoxOpenLog() as log:
        r = sr.run_confined(
            [str(sr.PYTHON), "-s", "-c", "open('/etc/passwd').close()"],
            ro={inputs: "/inputs"},
            env={"PYTHONPATH": "/inputs", **log.env},
            pass_fds=log.pass_fds,
            timeout_s=60,
        )
    assert r.returncode == 0, r.stderr
    # the hook reported its start, so no rows means no opens, not a blind cell
    assert log.result == {"rows": [], "dropped": 0, "dead": False, "reason": "ok"}


@needs_bwrap
def test_a_writer_cell_open_is_recorded_and_stamped_by_the_host(tmp_path):
    code = (
        "import os, sys\n"
        "open('/inputs/request.json').read()\n"
        "fd = int(os.environ['MEMORY_BOX_OPEN_LOG'][3:])\n"
        'os.write(fd, b\'{"event": "open", "path": "/inputs/claimed", "mode": "r", '
        '"pass_id": "forged", "call": 99, "agent": "actor"}\\n\')\n'
        "for name in ('box-opens.jsonl', 'tool-events.jsonl'):\n"
        "    with open('/memory/' + name, 'w') as fh:\n"
        '        fh.write(\'{"pass_id": "forged"}\\n\')\n'
        "print('no_user_site', sys.flags.no_user_site)\n"
    )
    model = Turns(
        [
            _call("a", "read", {"path": "/inputs/batch_map.json"}),
            _call("b", "execute_code", {"code": code}),
        ],
    )
    out = tmp_path / "out"
    _, _, writer = _writer(
        tmp_path,
        model,
        library_mode=True,
        max_calls=4,
        pass_record_root=str(out),
    )
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    assert "no_user_site 1" in model.outputs["b"], model.outputs["b"]
    rows = writer.tool_events
    read_row = next(r for r in rows if r["tool"] == "read")
    cell_row = next(r for r in rows if r["tool"] == "execute_code")
    assert "box_opens" not in read_row
    assert cell_row["box_hook_dead"] is False and cell_row["box_opens_dropped"] == 0
    written = [
        json.loads(line)
        for line in (out / "p1" / "box-opens.jsonl").read_text().splitlines()
    ]
    assert written == writer.box_opens and cell_row["box_opens"] == len(written)
    for o in written:
        assert set(o) == {"agent", "pass_id", "call", "event", "path", "mode"}
        assert (o["agent"], o["pass_id"], o["call"]) == (
            "writer",
            "p1",
            cell_row["call"],
        )
    paths = {(o["path"], o["mode"]) for o in written}
    assert ("/inputs/request.json", "r") in paths
    assert (
        "/memory/box-opens.jsonl",
        "w",
    ) in paths  # the write through /memory is an observed open, nothing more
    assert (
        "/inputs/claimed",
        "r",
    ) in paths  # a line from the box keeps only its event, path and mode
    assert "forged" not in (out / "p1" / "box-opens.jsonl").read_text()
    assert "forged" not in (out / "p1" / "tool-events.jsonl").read_text()


@needs_bwrap
def test_a_helper_cell_open_is_recorded_beside_the_pass_events(tmp_path):
    hin, lib, work, cells = (
        tmp_path / n for n in ("inputs", "memory", "work", "cells")
    )
    for d in (hin, lib, work, cells):
        d.mkdir()
    sr.stage_open_hook(hin)
    (hin / "x.txt").write_text("x")
    audit: dict = {}
    out = helpers.box_cell(
        "import sys\nopen('/inputs/x.txt').read()\nprint(sys.flags.no_user_site)\n",
        hin,
        lib,
        work,
        cells,
        60.0,
        audit,
    )
    assert out.strip() == "1", out
    events = tmp_path / "out" / "p1" / "tool-events.jsonl"
    ctx = SimpleNamespace(pass_id="p1", events=events, box_opens=[])
    row = {"helper_call": "h-1", "call": 2, "tool": "execute_code"}
    helpers.record_box_opens(ctx, row, audit)
    assert (
        row["box_opens"] == 1
        and row["box_opens_dropped"] == 0
        and row["box_hook_dead"] is False
    )
    want = [
        {
            "agent": "helper",
            "pass_id": "p1",
            "helper_call": "h-1",
            "call": 2,
            "event": "open",
            "path": "/inputs/x.txt",
            "mode": "r",
        },
    ]
    assert ctx.box_opens == want
    assert [
        json.loads(line)
        for line in (events.parent / "box-opens.jsonl").read_text().splitlines()
    ] == want


def test_the_open_rows_file_is_never_written_through_a_link(tmp_path):
    target = tmp_path / "elsewhere"
    target.write_text("")
    link = tmp_path / "p1" / "box-opens.jsonl"
    link.parent.mkdir()
    os.symlink(target, link)
    assert sr.append_open_rows(link, [{"event": "open"}]) is False
    assert target.read_text() == ""
    hard = tmp_path / "p2" / "box-opens.jsonl"
    hard.parent.mkdir()
    os.link(target, hard)
    assert sr.append_open_rows(hard, [{"event": "open"}]) is False
    assert target.read_text() == ""


@needs_bwrap
def test_a_box_without_the_hook_is_dead_with_its_reason(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with sr.BoxOpenLog() as log:
        r = sr.run_confined(
            [str(sr.PYTHON), "-s", "-c", "print(1)"],
            ro={empty: "/inputs"},
            env={"PYTHONPATH": "/inputs", **log.env},
            pass_fds=log.pass_fds,
            timeout_s=60,
        )
    assert r.returncode == 0, r.stderr
    assert log.result["dead"] is True and log.result["reason"] == "no_start"


@needs_bwrap
def test_a_timed_out_cell_keeps_the_opens_it_made_and_is_not_dead(tmp_path):
    inputs = _inputs(tmp_path)
    code = "open('/inputs/x.txt').close()\nimport time\ntime.sleep(30)\n"
    with sr.BoxOpenLog() as log:
        r = sr.run_confined(
            [str(sr.PYTHON), "-s", "-c", code],
            ro={inputs: "/inputs"},
            env={"PYTHONPATH": "/inputs", **log.env},
            pass_fds=log.pass_fds,
            timeout_s=2,
        )
    assert r.timed_out
    assert log.result["dead"] is False and log.result["reason"] == "ok"
    assert log.result["rows"] == [
        {"event": "open", "path": "/inputs/x.txt", "mode": "r"},
    ]


@needs_bwrap
def test_concurrent_cells_each_have_their_own_complete_record(tmp_path):
    import threading

    inputs = _inputs(tmp_path)
    results = {}

    def cell(k):
        with sr.BoxOpenLog() as log:
            sr.run_confined(
                [str(sr.PYTHON), "-s", "-c", "open('/inputs/x.txt').close()"],
                ro={inputs: "/inputs"},
                env={"PYTHONPATH": "/inputs", **log.env},
                pass_fds=log.pass_fds,
                timeout_s=60,
            )
        results[k] = log.result

    threads = [threading.Thread(target=cell, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 4
    for res in results.values():
        assert res["dead"] is False and res["reason"] == "ok" and len(res["rows"]) == 1


@needs_bwrap
def test_the_host_drops_its_write_end_at_spawn_so_the_pipe_ends_when_the_box_exits(
    tmp_path,
):
    inputs = _inputs(tmp_path)
    log = sr.BoxOpenLog()
    held = {}

    def spawned():
        log.close_write()
        held["host_write_end"] = log._w

    sr.run_confined(
        [str(sr.PYTHON), "-s", "-c", "open('/inputs/x.txt').close()"],
        ro={inputs: "/inputs"},
        env={"PYTHONPATH": "/inputs", **log.env},
        pass_fds=log.pass_fds,
        on_spawn=spawned,
    )
    # the call has returned and finish() has not run: the drain reached end of file when the box exited
    log._thread.join(5)
    assert held["host_write_end"] is None and not log._thread.is_alive()
    res = log.finish()
    assert res["reason"] == "ok" and res["rows"] == [
        {"event": "open", "path": "/inputs/x.txt", "mode": "r"},
    ]


@needs_bwrap
def test_lines_the_box_writes_after_the_host_dropped_its_end_are_kept(tmp_path):
    inputs = _inputs(tmp_path)
    code = "import time\ntime.sleep(0.5)\nopen('/inputs/x.txt').close()\n"
    with sr.BoxOpenLog() as log:
        sr.run_confined(
            [str(sr.PYTHON), "-s", "-c", code],
            ro={inputs: "/inputs"},
            env={"PYTHONPATH": "/inputs", **log.env},
            pass_fds=log.pass_fds,
            on_spawn=log.close_write,
        )
    assert log.result["rows"] == [
        {"event": "open", "path": "/inputs/x.txt", "mode": "r"},
    ]
    assert log.result["dropped"] == 0


def test_closing_the_write_end_is_idempotent_and_finish_still_reports():
    log = sr.BoxOpenLog()
    log.close_write()
    log.close_write()
    assert log.pass_fds == () and log.finish()["reason"] == "no_start"
