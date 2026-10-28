"""Symbolic: no name or id collisions — names survive a reload, run files never mix."""

import asyncio

import pytest

from unify.agents.log import RecordLog
from unify.agents.options import Options
from unify.agents.pool import Pool
from unify.agents.record import Record

REQUEST = "Summarise a.csv: rows, columns and missing values."


async def _done(name, spawner, request, request_seq):
    return f"{name} done"


@pytest.mark.asyncio
async def test_helper_names_and_numbering_survive_a_reload(tmp_path):
    # rev 2: a pool runs on the run record (helpers are named from its roster), so the reload reopens that record
    from tests.agents.records import run_record
    from unify.agents import run_record as rr

    home = tmp_path / "home"
    first = Pool(run_record(home=home), start_helper=_done)
    await first.spawn("root", REQUEST)
    await first.spawn("root", REQUEST, name="csv-stats")
    await asyncio.sleep(0.05)
    last = first.record.last_seq

    record = rr.open_run_record(home, create=False, options=Options(min_post_interval_s=0))
    record.root = "root"
    # the reopened record knows the earlier agents from the run's roster: a helper that replied has left (its name
    # stays taken), and who started it is kept
    rows = {r["name"]: r for r in record.roster_rows(include_ended=True)}
    assert rows["h1"]["state"] == "ended"
    assert rows["csv-stats"]["spawner"] == "root"
    again = Pool(record, start_helper=_done)
    assert await again.spawn("root", REQUEST) == "h2"
    with pytest.raises(ValueError):
        await again.spawn("root", REQUEST, name="csv-stats")
    assert record.entries[-1].seq == last + 1
    await again.close("test over")


def test_a_helper_running_at_a_crash_reloads_as_lost(tmp_path):
    path = tmp_path / "run.jsonl"
    rec = Record(RecordLog(path, create=True), options=Options())
    rec.add_agent("h1", spawner="root")
    rec.append("root", f"@h1 {REQUEST}")
    reloaded = Record(RecordLog(path, create=False), options=Options())
    assert reloaded.participants["h1"].state == "lost"
    assert reloaded.participants["h1"].request_seq == 1


def test_a_new_record_refuses_an_existing_file_and_entries_never_mix(tmp_path):
    path = tmp_path / "same-id.jsonl"
    rec = Record(RecordLog(path, create=True), options=Options())
    rec.append("user", "first run")
    with pytest.raises(FileExistsError):
        RecordLog(path, create=True)
    assert [e.text for e in RecordLog(path, create=False).load()[0]] == ["first run"]


def test_resume_never_creates_a_file_by_accident(tmp_path):
    with pytest.raises(FileNotFoundError):
        RecordLog(tmp_path / "missing.jsonl", create=False)


