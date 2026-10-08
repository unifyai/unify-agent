"""Symbolic: one act() is bound as the main agent or as a helper; spawning needs the worker sandbox."""

import pytest

from unify.agents import binding


@pytest.fixture
def record_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(binding, "records_dir", lambda: tmp_path / "records")


def test_the_first_act_is_the_main_agent_and_has_seen_the_request(record_mode):
    b = binding.bind_for_act(request="Count the rows in a.csv", user_reads=True)
    rec = b.pool.record
    assert b.name == "root" and rec.entries[0].author == "user"
    assert rec.participants["root"].cursor == 1
    assert rec.path.parent.parent.name == "records" and rec.path.exists()
    assert rec.path.name == "record.jsonl"  # one folder per run
    assert set(b.globals()) == {"record", "agents"}
    assert binding.current_root_pool() is b.pool


@pytest.mark.asyncio
async def test_a_helpers_act_binds_as_that_helper(record_mode):
    root = binding.bind_for_act(request="Count rows", user_reads=False)
    root.pool.record.add_agent("h1", spawner="root")
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
    b.pool.record.append("user", "use 2024")
    assert "#2 user: use 2024" in await b.on_turn_boundary()


def test_spawning_is_permitted_with_python_in_the_worker():
    assert binding.spawn_permitted() == (True, "")


def test_spawning_needs_the_worker_sandbox(python_in_process):
    allowed, why = binding.spawn_permitted()
    assert allowed is False and "sandboxed worker" in why


def test_the_prompt_section_states_the_rules():
    text = binding.PROMPT_SECTION
    for words in (
        "await agents.spawn",
        "record.post",
        "await record.wait",
        "never while you work",
        "Entries from `user` are your instructions",
        "only you can answer them",
    ):
        assert words in text
    assert "example" not in text.lower()


def test_a_long_request_still_binds(record_mode):
    b = binding.bind_for_act(request="ä" * 30_000, user_reads=False)
    entry = b.pool.record.entries[0]
    assert entry.author == "user" and len(entry.text.encode()) <= 16 * 1024
    assert "cut" in entry.text


def test_the_main_agents_first_cursor_is_written(record_mode):
    from unify.agents.log import RecordLog

    b = binding.bind_for_act(request="Count rows", user_reads=False)
    assert RecordLog(b.pool.record.path, create=False).load()[1] == {"root": 1}
