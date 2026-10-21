"""Symbolic: one act() is bound as the main agent or as a helper; spawning needs the worker sandbox."""

import re

import pytest

from unify.agents import binding
from unify.agents.run_record import record_path


@pytest.fixture
def record_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(binding, "_run_home", lambda: tmp_path / "home")
    return tmp_path / "home"


def test_the_first_act_is_the_main_agent_and_has_seen_the_request(record_mode):
    b = binding.bind_for_act(request="Count the rows in a.csv", user_reads=True)
    rec = b.pool.record
    # rev 2: the main agent is ``actor-<hex>`` on the memory home's one run record
    assert re.fullmatch(r"actor-[0-9a-f]{4}", b.name) and rec.root == b.name
    assert rec.entries[0].author == "user"
    assert rec.participants[b.name].cursor == 1
    assert rec.path == record_path(record_mode) and rec.path.exists()
    assert set(b.globals()) == {"record", "agents"}
    assert binding.current_root_pool() is b.pool


@pytest.mark.asyncio
async def test_a_helpers_act_binds_as_that_helper(record_mode):
    root = binding.bind_for_act(request="Count rows", user_reads=False)
    root.pool.record.add_agent("h1", spawner=root.name)
    token = binding._CURRENT.set((root.pool, "h1"))
    try:
        helper = binding.bind_for_act(request="ignored", user_reads=False)
    finally:
        binding._CURRENT.reset(token)
    assert helper.name == "h1" and helper.pool is root.pool
    assert len(root.pool.record.entries) == 1


@pytest.mark.asyncio
async def test_the_boundary_callback_returns_the_block(record_mode):
    b = binding.bind_for_act(request="Count rows", user_reads=False)
    assert await b.on_turn_boundary() is None
    # rev 2: a user line is addressed to the main agent (cli_bridge.user_message mentions the root)
    b.pool.record.append("user", "use 2024", mentions=[b.name])
    assert "#2 user: use 2024" in await b.on_turn_boundary()


def test_a_long_request_still_binds(record_mode):
    b = binding.bind_for_act(request="ä" * 30_000, user_reads=False)
    entry = b.pool.record.entries[0]
    assert entry.author == "user" and len(entry.text.encode()) <= 16 * 1024
    assert "cut" in entry.text


def test_the_main_agents_first_cursor_is_written(record_mode):
    from unify.agents.log import RecordLog

    b = binding.bind_for_act(request="Count rows", user_reads=False)
    assert RecordLog(b.pool.record.path, create=False).load()[1] == {b.name: 1}
