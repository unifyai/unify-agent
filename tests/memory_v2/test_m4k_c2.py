"""m4k-c2 (MAIN 11 Oct, the rehearsal-2 diagnosis ab734c1e): compaction measures the prompt text, never the provider's
opaque fields, and compacts only for a real drop, so it never fires call after call; each S0 reference in the batch
map carries a gist of its code and its instance count."""

import json
from pathlib import Path

from unify.memory_v2 import sol_pass

SIZES = Path(__file__).parent / "golden" / "dr2_med_a_pass1_sizes.json"


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
    """dr2-med-A pass 1's real message sizes, blobs included: under m4k-c's serialized measure the conversation was
    "over size" from call ~50 on, so compaction fired on every call; now it fires only for a real drop.
    """
    rows = json.loads(SIZES.read_text())["rows"]
    for at in (
        240_000,
        sol_pass.COMPACT_AT_CHARS,
    ):  # m4k-c's recorded value, and today's
        msgs, events, old_over, calls = [], [], 0, 0
        (tmp_path / str(at)).mkdir()
        for i, (role, content, args, psf, reasoning) in enumerate(rows):
            if role == "assistant":
                calls += 1
                if sum(len(json.dumps(m)) for m in msgs) > at:
                    old_over += 1  # m4k-c's measure: a compaction on this call
                if sol_pass._compact(msgs, tmp_path / str(at), at):
                    events.append(calls)
            msgs.append(_message(role, content, args, psf, reasoning, i))
        assert old_over >= 10  # the scenario that thrashed
        assert len(events) <= 2 and all(b - a > 5 for a, b in zip(events, events[1:]))
        assert sum(sol_pass._prompt_chars(m) for m in msgs) <= at + 2 * max(
            r[1] + r[2] for r in rows
        )


def test_provider_fields_and_reasoning_are_not_prompt_text():
    m = {
        "role": "assistant", "content": None, "reasoning_content": "z" * 5000,
        "provider_specific_fields": {"blob": "y" * 9000},
        "tool_calls": [{"id": "c", "type": "function", "function": {"name": "read", "arguments": '{"path": "/x"}'}}],
    }  # fmt: skip
    assert sol_pass._prompt_chars(m) == len("read") + len('{"path": "/x"}')
    assert sol_pass._prompt_chars({"role": "tool", "content": "abc"}) == 3


def test_no_compaction_without_a_real_drop(tmp_path):
    """Sol's own messages alone over the size, with little old tool output: nothing moves (a stub would only break
    the cache)."""
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
        sol_pass._compact(msgs, tmp_path, at=100_000) == 0
        and json.dumps(msgs) == before
    )


def test_each_s0_reference_carries_a_gist_and_its_instances():
    tier_a = [{
        "key": "k" * 64, "template": "\nfor r in range(__q0):\n    row = g[r]\n    out.append(row)\n",
        "at": [{"episode": "e1", "cell": 0}, {"episode": "e1", "cell": 2}, {"episode": "e2", "cell": 0}],
    }]  # fmt: skip
    tier_b = {
        "units": {"u1": {"episode": "e3", "cell": 1, "src": "def fill(g):\n    return g\n"}},
        "clusters": [{"units": ["u1"], "akeys": ["x"]}],
    }  # fmt: skip
    refs = sol_pass._s0_by_episode(tier_a, tier_b)
    a = refs["e1"][0]
    assert (
        a["gist"] == "for r in range(__q0):\n    row = g[r]\n    out.append(row)"
        and a["instances"] == 3
    )
    b = refs["e3"][0]
    assert b["gist"] == "def fill(g):\n    return g" and b["instances"] == 1
    assert sol_pass._gist(None) == "" and sol_pass._gist("x\n" * 9).count("\n") == 2
