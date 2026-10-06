"""Symbolic: starting, ending and stopping helpers; replies go only to the spawner."""

import asyncio

import pytest

from unify.agents.options import Options
from unify.agents.pool import Pool
from unify.agents.record import Record

TASK = "Summarise a.csv: rows, columns and missing values."


def _pool(start, **opts):
    return Pool(
        Record(None, options=Options(min_post_interval_s=0, **opts)),
        start_helper=start,
        spawn_allowed=True,
    )


def _replies(text):
    async def start(name, spawner, task, task_seq):
        await asyncio.sleep(0.01)
        return text

    return start


@pytest.mark.asyncio
async def test_spawn_posts_the_task_and_the_reply_comes_back_to_the_spawner():
    pool = _pool(_replies('{"rows": 3}'))
    name = await pool.spawn("root", TASK)
    assert name == "h1"
    task_entry = pool.record.entries[0]
    assert (task_entry.author, task_entry.text, task_entry.mentions) == (
        "root",
        f"@h1 {TASK}",
        ("h1",),
    )
    got = await pool.record.wait_for("root", timeout=5)
    assert [(e.author, e.kind, e.text, e.mentions) for e in got] == [
        ("h1", "reply", '{"rows": 3}', ("root",)),
    ]
    assert pool.list("root") == [
        {"name": "h1", "state": "replied", "task_seq": 1, "last_seq": 2},
    ]


@pytest.mark.asyncio
async def test_only_the_main_agent_spawns_and_only_when_allowed():
    pool = _pool(_replies("x"))
    await pool.spawn("root", TASK)
    with pytest.raises(PermissionError, match="only the main agent"):
        await pool.spawn("h1", TASK)
    refused = Pool(
        Record(None, options=Options()),
        start_helper=_replies("x"),
        spawn_allowed=False,
        spawn_refusal="needs the worker sandbox",
    )
    with pytest.raises(PermissionError, match="worker sandbox"):
        await refused.spawn("root", TASK)
    off = _pool(_replies("x"), max_total=0)
    with pytest.raises(PermissionError, match="off"):
        await off.spawn("root", TASK)
    await pool.close("test over")


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["", "No-op", "N/A", "none", "   do it   "])
async def test_empty_or_placeholder_tasks_are_refused(task):
    pool = _pool(_replies("x"))
    with pytest.raises(ValueError, match="task"):
        await pool.spawn("root", task)


@pytest.mark.asyncio
async def test_names_are_checked_and_limits_hold():
    gate = asyncio.Event()

    async def start(name, spawner, task, task_seq):
        await gate.wait()
        return "ok"

    pool = _pool(start, max_live=2, max_total=3)
    assert await pool.spawn("root", TASK, name="csv-stats") == "csv-stats"
    for bad in ("user", "all", "root", "Bad Name", "csv-stats"):
        with pytest.raises(ValueError):
            await pool.spawn("root", TASK, name=bad)
    await pool.spawn("root", TASK)
    with pytest.raises(RuntimeError, match="2 helpers are running"):
        await pool.spawn("root", TASK)
    gate.set()
    await asyncio.sleep(0.05)
    await pool.spawn("root", TASK)
    with pytest.raises(RuntimeError, match="3 helpers"):
        await pool.spawn("root", TASK)
    await pool.close("test over")


@pytest.mark.asyncio
async def test_a_failed_helper_leaves_a_notice_for_its_spawner():
    async def start(name, spawner, task, task_seq):
        raise RuntimeError("step limit reached")

    pool = _pool(start)
    await pool.spawn("root", TASK)
    got = await pool.record.wait_for("root", timeout=5)
    assert got[0].kind == "system" and "h1 ended without replying" in got[0].text
    assert "step limit reached" in got[0].text and got[0].mentions == ("root",)
    assert pool.list("root")[0]["state"] == "failed"


@pytest.mark.asyncio
async def test_a_long_reply_is_cut_with_a_note():
    pool = _pool(_replies("é" * 20_000))
    await pool.spawn("root", TASK)
    got = await pool.record.wait_for("root", timeout=5)
    assert got[0].kind == "reply" and len(got[0].text.encode()) <= 16 * 1024
    assert "cut" in got[0].text and "transcript" in got[0].text


@pytest.mark.asyncio
async def test_stop_posts_a_cancel_and_ends_the_helper_at_once():
    started = asyncio.Event()

    async def start(name, spawner, task, task_seq):
        started.set()
        await asyncio.sleep(3600)

    pool = _pool(start)
    await pool.spawn("root", TASK)
    await started.wait()
    pool.stop("root", "h1", "found it elsewhere")
    await asyncio.sleep(0.05)
    cancel = pool.record.entries[-1]
    assert (cancel.author, cancel.kind, cancel.mentions) == ("root", "cancel", ("h1",))
    assert pool.list("root")[0]["state"] == "stopped"
    assert pool._tasks["h1"].done()
    with pytest.raises(PermissionError):
        pool.stop("h1", "h1")


@pytest.mark.asyncio
async def test_close_stops_running_helpers_and_verifies():
    async def start(name, spawner, task, task_seq):
        await asyncio.sleep(3600)

    pool = _pool(start)
    await pool.spawn("root", TASK)
    await pool.spawn("root", TASK)
    report = await pool.finish_request("the answer")
    assert report == {"h1": "ended", "h2": "ended"}
    kinds = [(e.author, e.kind) for e in pool.record.entries]
    assert ("root", "reply") in kinds and kinds.count(("harness", "system")) == 2
    assert pool.record.entries[2].mentions == ("user",)
