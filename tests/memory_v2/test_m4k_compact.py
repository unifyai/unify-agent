"""UNIFY_MEMORY_V21_COMPACT (MAIN 11 Oct; prepared, off until the lead decides): when the writer's conversation passes
a size, its oldest tool results move out to /outputs/compacted/ behind a pointer to re-open them; Sol's own messages
(reasoning, decisions, calls), the request and the last messages are never touched; off, nothing changes.
"""

import json

from unify.memory_v2 import sol_pass
from unify.memory_v2.integration import switch
from unify.memory_v2.sol_pass import PassConfig
from unify.settings import SETTINGS
from tests.memory_v2.test_sol_pass import _call
from tests.memory_v2.test_sol_v21_tools import _pass


def _conv(n_tools, size):
    msgs = [
        {"role": "system", "content": "brief"},
        {"role": "user", "content": "Pass p1: {}"},
    ]
    for i in range(n_tools):
        msgs.append(
            {
                "role": "assistant",
                "content": f"thinking {i}",
                "reasoning_content": f"why {i}",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    },
                ],
            },
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "content": f"result {i} " + "x" * size,
            },
        )
    return msgs  # fmt: skip


def test_a_small_conversation_is_never_changed(tmp_path):
    msgs = _conv(3, 100)
    before = json.dumps(msgs)
    assert (
        sol_pass._compact(msgs, tmp_path, at=10_000) == 0 and json.dumps(msgs) == before
    )


def test_old_results_move_out_behind_a_pointer_and_everything_else_stays(tmp_path):
    msgs = _conv(10, 3_000)
    keep = json.dumps(msgs[-sol_pass.COMPACT_KEEP_LAST :])
    asst = [json.dumps(m) for m in msgs if m["role"] != "tool"]
    original = msgs[3]["content"]
    moved = sol_pass._compact(msgs, tmp_path, at=20_000)
    assert moved > 0
    assert len(json.dumps(msgs)) <= 10_000 or all(
        m["content"].startswith("[compacted: ")
        for m in msgs[: -sol_pass.COMPACT_KEEP_LAST]
        if m["role"] == "tool"
    )
    assert [
        json.dumps(m) for m in msgs if m["role"] != "tool"
    ] == asst  # reasoning, decisions, calls kept
    assert json.dumps(msgs[-sol_pass.COMPACT_KEEP_LAST :]) == keep
    stub = msgs[3]["content"]
    assert (
        stub.startswith("[compacted: this result (")
        and 'read(path="/outputs/compacted/' in stub
    )
    name = stub.split("/outputs/compacted/")[1].split('"')[0]
    assert (
        tmp_path / "compacted" / name
    ).read_text() == original  # re-openable, byte for byte
    assert (
        sol_pass._compact(msgs, tmp_path, at=20_000) == 0
        or len(json.dumps(msgs)) > 20_000
    )  # no churn under the size


def test_the_trigger_is_size_alone(tmp_path):
    many_small = _conv(60, 10)  # many turns, small: never compacted
    assert sol_pass._compact(many_small, tmp_path, at=50_000) == 0
    few_big = _conv(
        8,
        30_000,
    )  # few turns, big: compacted (the last messages are never touched)
    assert sol_pass._compact(few_big, tmp_path, at=50_000) > 0


def test_a_pass_compacts_only_with_the_switch_on(tmp_path, monkeypatch):
    monkeypatch.setattr(sol_pass, "COMPACT_AT_CHARS", 20_000)
    turns = [
        _call(f"r{i}", "read_episode", {"episode": "e1", "part": "cell:0", "offset": 0})
        for i in range(10)
    ]
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    _, on = _pass(tmp_path / "on", turns, max_calls=14, compact=True)
    _, off = _pass(tmp_path / "off", turns, max_calls=14)
    stubbed = lambda model: sum(  # noqa: E731
        1
        for m in model.seen
        if m.get("role") == "tool"
        and str(m.get("content", "")).startswith("[compacted: ")
    )
    assert stubbed(on) > 0 and stubbed(off) == 0
    assert [m for m in on.seen if m.get("role") == "assistant"] == [
        m for m in off.seen if m.get("role") == "assistant"
    ]


def test_the_switch_is_off_by_default_and_takes_only_on_or_off():
    assert SETTINGS.UNIFY_MEMORY_V21_COMPACT == "off" and PassConfig().compact is False
    assert (
        switch.parse_v21_compact("") == "off" and switch.parse_v21_compact("on") == "on"
    )
    assert switch.PARSERS[switch.V21_COMPACT] is switch.parse_v21_compact
    try:
        switch.parse_v21_compact("maybe")
    except ValueError as exc:
        assert "UNIFY_MEMORY_V21_COMPACT" in str(exc)
    else:
        raise AssertionError("an unknown value was accepted")
