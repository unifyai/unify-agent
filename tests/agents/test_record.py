"""Symbolic: one writer, ordering, mentions, waiting and the boundary cursor."""

import asyncio

import pytest

from unify.agents.log import RecordLog
from unify.agents.options import Options
from unify.agents.record import HARNESS, USER, PostRefused, Record


def _record(tmp_path=None, **opts):
    log = RecordLog(tmp_path / "run.jsonl", create=True) if tmp_path else None
    ticks = iter(range(10_000))
    rec = Record(
        log,
        options=Options(**opts),
        clock=lambda: "2026-10-06T11:00:00.000Z",
        monotonic=lambda: float(next(ticks)),
    )
    rec.add_agent("h1", spawner="root")
    rec.add_agent("h2", spawner="root")
    return rec


def test_sequence_numbers_are_global_and_the_file_matches_memory(tmp_path):
    rec = _record(tmp_path)
    for i in range(5):
        rec.append("root" if i % 2 else "h1", f"n{i}")
    assert [e.seq for e in rec.entries] == [1, 2, 3, 4, 5]
    reloaded = Record(
        RecordLog(tmp_path / "run.jsonl", create=False),
        options=Options(),
    )
    assert reloaded.entries == rec.entries


def test_mentions_resolve_to_participants_and_unknown_names_are_reported():
    rec = _record()
    entry, unresolved = rec.append("root", "@h1 and @h3 and @user, cc @harness")
    assert entry.mentions == ("h1", "user") and unresolved == ["h3", "harness"]


def test_all_means_every_live_agent_but_the_author():
    rec = _record()
    entry, _ = rec.append("h1", "@all found it")
    assert entry.mentions == ("root", "h2")


def test_oversized_text_and_unknown_kind_are_refused():
    rec = _record()
    with pytest.raises(PostRefused, match="16384"):
        rec.append("root", "x" * 16_385)
    with pytest.raises(PostRefused, match="kind"):
        rec.append("root", "x", kind="shout")


def test_the_record_refuses_entries_past_its_limit():
    rec = _record(max_entries=2)
    rec.append("root", "a")
    rec.append("root", "b")
    with pytest.raises(PostRefused, match="full"):
        rec.append("root", "c")


def test_posting_too_fast_is_refused_with_the_wait():
    rec = Record(
        None,
        options=Options(min_post_interval_s=5.0),
        monotonic=lambda: 100.0,
    )
    rec.check_rate("root")
    with pytest.raises(PostRefused, match="retry after"):
        rec.check_rate("root")


def test_read_filters_and_keeps_the_newest_within_the_limit():
    rec = _record()
    for i in range(6):
        rec.append("h1", f"@root n{i}" if i % 2 else f"n{i}")
    got = rec.read(since=2, mentions="root", limit=2)
    assert [e.text for e in got] == ["@root n3", "@root n5"]
    assert [e.seq for e in rec.read(author="h1", limit=50)] == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        rec.read(limit=0)


@pytest.mark.asyncio
async def test_wait_returns_when_an_entry_for_me_arrives():
    rec = _record()

    async def later():
        await asyncio.sleep(0.05)
        rec.append("h2", "not for root")
        await asyncio.sleep(0.05)
        rec.append("h1", "@root done")

    asyncio.get_running_loop().create_task(later())
    got = await rec.wait_for("root", timeout=5)
    assert [e.text for e in got] == ["@root done"]
    assert rec.participants["root"].returned == {2}


@pytest.mark.asyncio
async def test_wait_times_out_with_nothing():
    rec = _record()
    assert await rec.wait_for("root", timeout=0.05) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [None, 0, -1, "soon"])
async def test_wait_timeout_is_required_positive_and_capped(bad, monkeypatch):
    rec = _record()
    with pytest.raises(ValueError, match="timeout"):
        await rec.wait_for("root", timeout=bad)
    seen = {}

    async def fake_wait_for(aw, timeout):
        seen["timeout"] = timeout
        aw.cancel()
        raise asyncio.TimeoutError

    passed = iter([False, True])  # not yet on the first look; passed after one wait
    monkeypatch.setattr("unify.agents.record._wait_for", fake_wait_for)
    monkeypatch.setattr(
        "unify.agents.record._deadline_passed",
        lambda *a: next(passed),
    )
    assert await rec.wait_for("root", timeout=1e9) == []
    assert seen["timeout"] <= 600.0


def test_user_entries_are_for_the_root_and_cancel_entries_for_their_target():
    rec = _record()
    u, _ = rec.append(USER, "use 2024")
    c, _ = rec.append("root", "@h1 stop", kind="cancel", mentions=["h1"])
    assert rec.is_for(u, "root") and not rec.is_for(u, "h1")
    assert rec.is_for(c, "h1") and not rec.is_for(c, "h2")


def test_delivery_all_gives_everyone_everything_but_their_own():
    rec = _record(delivery="all")
    e, _ = rec.append("h1", "a finding")
    assert rec.is_for(e, "root") and rec.is_for(e, "h2") and not rec.is_for(e, "h1")


def test_take_block_advances_the_cursor_and_logs_it(tmp_path):
    rec = _record(tmp_path)
    rec.append("h1", "@root first")
    block = rec.take_block("root")
    assert "#1 h1: @root first" in block
    assert rec.take_block("root") is None
    assert RecordLog(tmp_path / "run.jsonl", create=False).load()[1] == {"root": 1}


def test_entries_not_for_me_wait_and_are_counted_in_the_next_block():
    rec = _record()
    rec.append("h1", "@h2 between helpers")
    assert rec.take_block("root") is None
    assert rec.participants["root"].cursor == 0
    rec.append(USER, "and now this")
    block = rec.take_block("root")
    assert "#2 user: and now this" in block and "+1 other entries" in block


def test_listeners_see_each_entry_once():
    rec = _record()
    seen = []
    rec.add_listener(seen.append)
    rec.append(HARNESS, "notice", kind="system", mentions=["root"])
    assert [e.seq for e in seen] == [1]
