"""Compaction measures the prompt text, never the provider's
opaque fields, and compacts only for a real drop, so it never fires call after call; each miner reference in the batch
map carries a gist of its code and its instance count."""

import json
from pathlib import Path

from unify.memory_v2 import writer_pass

SIZES = Path(__file__).parent / "golden" / "writer_conversation_sizes.json"


def _message(role, content, args, psf, reasoning, i):
    m = {"role": role, "content": "x" * content}
    if role == "system":
        m["content"] = (
            "s" * 16_000
        )  # the brief (about 4,060 tokens); the transcript names it only
    if role == "assistant":
        m["content"] = None
        m["tool_calls"] = [
            {
                "id": f"c{i}",
                "type": "function",
                "function": {"name": "f", "arguments": "a" * args},
            },
        ]
        m["provider_specific_fields"] = {"blob": "y" * psf}
        m["reasoning_content"] = "z" * reasoning
    if role == "tool":
        m["tool_call_id"] = f"c{i - 1}"
    return m


def test_the_rehearsal_pass_that_thrashed_never_compacts_call_after_call(tmp_path):
    """A rehearsal pass's real message sizes, blobs included: under the earlier serialized measure the conversation was
    "over size" from call ~50 on, so compaction fired on every call; now it fires only for a real drop.
    """
    rows = json.loads(SIZES.read_text())["rows"]
    for at in (
        240_000,
        writer_pass.COMPACT_AT_CHARS,
    ):  # the earlier recorded value, and today's
        msgs, events, old_over, calls = [], [], 0, 0
        (tmp_path / str(at)).mkdir()
        for i, (role, content, args, psf, reasoning) in enumerate(rows):
            if role == "assistant":
                calls += 1
                if sum(len(json.dumps(m)) for m in msgs) > at:
                    old_over += 1  # the earlier measure: a compaction on this call
                if writer_pass._compact(msgs, tmp_path / str(at), at):
                    events.append(calls)
            msgs.append(_message(role, content, args, psf, reasoning, i))
        assert old_over >= 10  # the scenario that thrashed
        assert len(events) <= 2 and all(b - a > 5 for a, b in zip(events, events[1:]))
        assert sum(writer_pass._prompt_chars(m) for m in msgs) <= at + 2 * max(
            r[1] + r[2] for r in rows
        )


def test_provider_fields_and_reasoning_are_not_prompt_text():
    m = {
        "role": "assistant", "content": None, "reasoning_content": "z" * 5000,
        "provider_specific_fields": {"blob": "y" * 9000},
        "tool_calls": [{"id": "c", "type": "function", "function": {"name": "read", "arguments": '{"path": "/x"}'}}],
    }  # fmt: skip
    assert writer_pass._prompt_chars(m) == len("read") + len('{"path": "/x"}')
    assert writer_pass._prompt_chars({"role": "tool", "content": "abc"}) == 3


def test_no_compaction_without_a_real_drop(tmp_path):
    """The writer's own messages alone over the size, with little old tool output: nothing moves (a stub would only
    break the cache)."""
    big_calls = [{
        "role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "execute_code", "arguments": "a" * 30_000}},
        ],
    }
        for i in range(10)]  # fmt: skip
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i, a in enumerate(big_calls):
        msgs += [a, {"role": "tool", "tool_call_id": f"c{i}", "content": "ok " * 100}]
    before = json.dumps(msgs)
    assert (
        writer_pass._compact(msgs, tmp_path, at=100_000) == 0
        and json.dumps(msgs) == before
    )
