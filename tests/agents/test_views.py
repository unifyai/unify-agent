"""Symbolic: what model code sees; the author always comes from the binding."""

import inspect

import pytest

from unify.agents.options import Options
from unify.agents.pool import Pool
from unify.agents.record import Record
from unify.agents.views import AgentsView, RecordView


def _views(user_reads=False):
    rec = Record(None, options=Options(min_post_interval_s=0), user_reads=user_reads)
    rec.add_agent("h1", spawner="root")
    return rec, RecordView(rec, "root"), RecordView(rec, "h1")


def test_post_returns_plain_data_and_reports_unknown_names():
    rec, root, _ = _views()
    out = root.post("@h1 and @nobody")
    assert out == {"seq": 1, "mentions": ["h1"], "unresolved": ["nobody"]}
    assert rec.entries[0].author == "root"


def test_no_api_takes_an_author_except_read_as_a_filter():
    for cls in (RecordView, AgentsView):
        for name, fn in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_") and name != "read":
                assert "author" not in inspect.signature(fn).parameters


def test_posting_to_the_user_with_nobody_reading_warns_at_once():
    _, _, helper = _views(user_reads=False)
    assert helper.post("@user which year?")["warnings"] == [
        "nobody reads @user in this run",
    ]
    _, root2, _ = _views(user_reads=True)
    assert "warnings" not in root2.post("@user which year?")


def test_mentioning_a_finished_helper_warns():
    rec, root, _ = _views()
    rec.participants["h1"].state = "replied"
    assert root.post("@h1 one more thing")["warnings"] == ["h1 has finished"]


def test_read_returns_dicts_and_marks_nothing():
    rec, root, helper = _views()
    helper.post("@root done")
    assert root.read(mentions_me=True) == [rec.entries[0].to_dict()]
    assert rec.participants["root"].returned == set()


@pytest.mark.asyncio
async def test_wait_returns_dicts():
    rec, root, helper = _views()
    helper.post("@root done")
    assert await root.wait(timeout=1) == [rec.entries[0].to_dict()]


@pytest.mark.asyncio
async def test_agents_view_spawns_as_its_owner():
    async def start(name, spawner, task, task_seq):
        return "ok"

    rec = Record(None, options=Options(min_post_interval_s=0))
    pool = Pool(rec, start_helper=start, spawn_allowed=True)
    agents = AgentsView(pool, "root")
    name = await agents.spawn("Summarise b.csv: rows and columns please.")
    assert rec.entries[0].author == "root" and name == "h1"
    assert agents.list()[0]["name"] == "h1"
    await pool.close("test over")
