"""The memory agents (the writer and each helper) compact their conversations as the agent loop does (rev 2: always,
with no switch): the same trigger (a call's prompt at COMPACTION_THRESHOLD of the model's context window, or the
fixed COMPACTION_AT_TOKENS, whichever comes first), the same summary request (the last request, unchanged, plus the
agent loop's instruction) and the same restart (the system message and the summary).
The summary call goes through the agent's own model turn (its route, its model) and is booked like any other call;
its input is message content only; the transcript keeps what the compaction took out.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace


from unify.common._async_tool import cache_discipline, context_compression
from unify.memory import helpers, transcripts, writer_pass
from unify.memory.trigger import PassRequest
from tests.memory.test_writer_library_tools import EPS
from tests.memory.test_writer_pass import _call, _writer

INSTRUCTION = cache_discipline.COMPRESSION_FORK_INSTRUCTION
ROUTE_TOKEN = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret
ENV_TOKEN = "sk-or-v1-" + "cd" * 32  # pragma: allowlist secret
SUMMARY = "SUMMARY: read e1 cell 0"


class Compacting:
    """A model turn with a known context window: each call reports *prompt* prompt tokens (*cached* of them read
    from the cache); a request ending with the agent loop's summary instruction is answered with a summary.
    """

    def __init__(self, turns, window=1_000, prompt=900, cached=600):
        self.queue, self.prompt, self.cached = list(turns), prompt, cached
        self.requests: list[list[dict]] = []
        self.seen: list[dict] = []
        if window is not None:
            self.max_input_tokens = window
        # the lane's configuration, as a real turn holds it: never part of what is sent
        self.route = SimpleNamespace(
            base_url="https://lane.invalid/v1",
            token=ROUTE_TOKEN,
        )

    def _usage(self):
        return {
            "prompt_tokens": self.prompt,
            "prompt_tokens_details": {"cached_tokens": self.cached},
        }

    async def __call__(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        self.seen = messages
        if messages and messages[-1].get("content") == INSTRUCTION:
            return {
                "role": "assistant",
                "content": SUMMARY,
                "usage": self._usage(),
            }, "0.002"
        msg = (
            self.queue.pop(0)
            if self.queue
            else _call("fin", "finish", {"summary": "done"})
        )
        return {**msg, "usage": self._usage()}, "0.001"


def _reads(n):
    return [
        _call(f"r{i}", "read_episode", {"episode": "e1", "part": "cell:0", "offset": 0})
        for i in range(n)
    ]


def _run(tmp_path, model, **cfg):
    (tmp_path / "rec").mkdir(parents=True, exist_ok=True)
    _, _, writer = _writer(
        tmp_path,
        model,
        max_calls=12,
        pass_record_root=str(tmp_path / "rec"),
        **cfg,
    )
    writer.load = EPS.__getitem__
    out = asyncio.run(
        writer.run(PassRequest("incremental", "svc", ["e1", "e2"], False), "p1"),
    )
    return out, writer


def _events(tmp_path):
    path = tmp_path / "rec" / "p1" / "tool-events.jsonl"
    return (
        [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    )


def _summaries(model):
    return [r for r in model.requests if r and r[-1].get("content") == INSTRUCTION]


def test_the_summary_call_goes_through_the_agents_own_turn_and_is_booked(tmp_path):
    model = Compacting(_reads(3))
    out, writer = _run(tmp_path, model)
    summaries = _summaries(model)
    assert summaries
    first = model.requests.index(summaries[0])
    before, after = model.requests[first - 1], model.requests[first + 1]
    # the summary request is the last request sent, unchanged, plus the instruction (the cached prefix holds)
    assert summaries[0][: len(before)] == before
    # the next call starts from the system message and the summary, as the agent loop restarts
    assert after[0]["role"] == "system" and after[0] == before[0]
    assert after[1]["content"] == context_compression.restart_text(
        SUMMARY,
        writer_pass.RESTART_POINTER,
    )
    # booked like any other call: counted, its charge in the pass's spend, its usage on a compaction row
    normal = len(model.requests) - len(summaries)
    assert out.calls == len(model.requests)
    assert Decimal(out.usd) == Decimal("0.001") * normal + Decimal("0.002") * len(
        summaries,
    )
    rows = [r for r in _events(tmp_path) if r.get("event") == "compaction"]
    assert len(rows) == len(summaries) and rows[0]["usage"]["usd"] == "0.002"
    assert rows[0]["usage"]["input"] == 900 and rows[0]["restarted"] is True
    assert (
        out.compactions == len(summaries)
        and out.messages_moved > 0
        and out.chars_moved > 0
    )
    assert out.input_tokens == 900 * len(model.requests)
    assert out.cache_read_tokens == 600 * len(model.requests)
    # the headroom: the largest prompt a call reported, beside the window the trigger read
    assert out.max_prompt_tokens == 900 and out.context_window == 1_000
    # the transcript keeps what the compaction took out, and marks the summary call
    lines = [json.loads(x) for x in writer.transcript("p1")]
    msgs = [x["message"] for x in lines if "message" in x]
    assert any(m.get("tool_calls") and m["tool_calls"][0]["id"] == "r0" for m in msgs)
    marked = [m for m in msgs if m.get("compaction_call")]
    assert [m["content"] for m in marked[:2]] == [INSTRUCTION, SUMMARY]


def test_no_compaction_with_an_unknown_window_below_the_fixed_point(tmp_path):
    model = Compacting(_reads(3), window=None)
    out, _ = _run(tmp_path / "unknown", model)
    assert not _summaries(model)
    assert out.compactions == 0 and out.messages_moved == 0
    # an unknown window is recorded as unknown, never as a number
    assert out.context_window is None


def test_with_no_known_window_the_fixed_point_compacts(tmp_path):
    fixed = context_compression.COMPACTION_AT_TOKENS
    model = Compacting(_reads(3), window=None, prompt=fixed)
    out, _ = _run(tmp_path / "at", model)
    assert out.compactions >= 1 and _summaries(model) and out.context_window is None
    model = Compacting(_reads(3), window=None, prompt=fixed - 1)
    out, _ = _run(tmp_path / "below", model)
    assert out.compactions == 0 and not _summaries(model)


def test_under_the_threshold_nothing_is_compacted(tmp_path):
    below = int(1_000 * context_compression.COMPACTION_THRESHOLD) - 1
    model = Compacting(_reads(3), window=1_000, prompt=below)
    out, _ = _run(tmp_path, model)
    assert out.compactions == 0 and not _summaries(model)


def test_no_credential_reaches_the_summary_input_or_the_records(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFY_MEMORY_WRITER_TOKEN", ENV_TOKEN)
    model = Compacting(_reads(2))
    out, writer = _run(tmp_path, model)
    assert out.compactions >= 1
    sent = json.dumps(_summaries(model))
    assert (
        ROUTE_TOKEN not in sent and ENV_TOKEN not in sent and "lane.invalid" not in sent
    )
    # message content only: no transport field of any kind
    transport = (
        "api_key",
        "api_base",
        "base_url",
        "headers",
        "extra_headers",
        "authorization",
    )
    assert not any(k in m for r in _summaries(model) for m in r for k in transport)
    kept = "\n".join(writer.transcript("p1")) + json.dumps(_events(tmp_path))
    assert ROUTE_TOKEN not in kept and ENV_TOKEN not in kept


def test_the_restart_is_the_system_message_and_the_summary():
    async def turn(messages, tools):
        return {"role": "assistant", "content": "the summary"}, "0.001"

    msgs = [
        {"role": "system", "content": "brief"},
        {"role": "user", "content": "Pass p1"},
        _call("a", "read", {"path": "/inputs"}),
        {"role": "tool", "tool_call_id": "a", "content": "x" * 100},
    ]
    # rev 2 keeps no request prefix (UNIFY_COMPACTION_KEEP_PREFIX is gone): the system message, then the summary
    comp = asyncio.run(writer_pass.compact_by_summary(msgs, turn, []))
    assert (
        comp.restart[0] is msgs[0]
        and len(comp.restart) == 2
        and comp.restart[1]["role"] == "user"
    )
    assert comp.restart[1]["content"] == context_compression.restart_text(
        "the summary",
        writer_pass.RESTART_POINTER,
    )


def test_a_reply_without_a_summary_leaves_the_conversation():
    async def turn(messages, tools):
        return _call("x", "read", {"path": "/inputs"}), "0.001"

    msgs = [
        {"role": "system", "content": "brief"},
        {"role": "user", "content": "Pass p1"},
    ]
    comp = asyncio.run(writer_pass.compact_by_summary(msgs, turn, []))
    assert comp.restart is None and comp.usd == "0.001"


def test_a_helper_compacts_on_its_own_turn_and_books_the_call(tmp_path):
    calls = []

    async def turn(messages, tools):
        calls.append(copy.deepcopy(messages))
        return {
            "role": "assistant",
            "content": "helper summary",
            "usage": {"prompt_tokens": 50},
        }, "0.003"

    ctx = SimpleNamespace(
        pass_record=SimpleNamespace(calls_in_flight=set()),
        deadline=time.monotonic() + 30,
        spend=SimpleNamespace(usd=Decimal("0"), unknown=0),
        archived={},
        events=tmp_path / "tool-events.jsonl",
        pass_id="p1",
    )
    doc = {"turns": 3, "_usd": Decimal("0"), "unknown_cost_calls": 0}
    msgs = [
        {"role": "system", "content": "helper brief"},
        {"role": "user", "content": "Task"},
        _call("a", "overview", {"episode": "e1"}),
        {"role": "tool", "tool_call_id": "a", "content": "y" * 500},
    ]
    original = list(msgs)
    asyncio.run(helpers._compact_helper(ctx, "call-0", turn, msgs, [], doc))
    assert (
        calls and calls[0][-1]["content"] == INSTRUCTION and calls[0][:-1] == original
    )
    assert (
        doc["turns"] == 4
        and doc["_usd"] == Decimal("0.003")
        and ctx.spend.usd == Decimal("0.003")
    )
    assert (
        msgs[0] is original[0]
        and len(msgs) == 2
        and "helper summary" in msgs[1]["content"]
    )
    assert ctx.archived["call-0"][: len(original)] == original
    (row,) = [json.loads(x) for x in ctx.events.read_text().splitlines()]
    assert (
        row["event"] == "compaction"
        and row["helper"] == "call-0"
        and row["usage"]["usd"] == "0.003"
    )
    # the helper's stored transcript keeps all of it
    m = transcripts.record_transcript(
        tmp_path,
        "p1",
        "call-0",
        [*ctx.archived["call-0"], *msgs],
    )
    assert m["lines"] == 1 + len(ctx.archived["call-0"]) + len(msgs)


# --- a recorded pass's sizes, replayed through the trigger (a measurement, no model) -----------------------------

ROWS = json.loads(
    (Path(__file__).parent / "golden" / "writer_compaction_rereads.json").read_text(),
)["rows"]
#: the recorded pass's tool JSON in characters, and characters per token measured over recorded writer
#: conversations: each call's prompt tokens are estimated from its characters
TOOLS_CHARS, CHARS_PER_TOKEN = 10_147, 1.48
KEEP_TURNS = 8


def _replay(window: int) -> dict:
    """The recorded pass (it compacted three times under the earlier file-moving policy, its writer re-reading each
    time what had just been moved) replayed through :func:`.writer_pass.compaction_due`, each compaction leaving
    the brief and a summary. Returns the compactions (by turn), the re-reads, within KEEP_TURNS turns, of results a
    compaction took out, and the cache-hit rate (the share of each call's prompt its previous call sent).
    """
    convo: list[tuple[int, str | None]] = []  # chars, call key of a tool result
    events, rereads, turn = [], [], 0
    taken: dict[str, int] = {}
    keys: dict[str, str] = {}
    prev, hits, sent = 0, 0, 0
    for row in ROWS:
        role, n = row[0], row[1]
        if role == "assistant":
            prompt = int((sum(c for c, _ in convo) + TOOLS_CHARS) / CHARS_PER_TOKEN)
            hits, sent, prev = hits + min(prev, prompt), sent + prompt, prompt
            if writer_pass.compaction_due(prompt, window):
                events.append(turn + 1)
                taken.update({k: turn for _, k in convo if k})
                convo = [convo[0], (2_000, None)]  # the brief and a summary
                prev = int((convo[0][0] + TOOLS_CHARS) / CHARS_PER_TOKEN)
            turn += 1
            for cid, _name, key, _reread in row[2]:
                keys[cid] = key
                if key in taken and turn - taken[key] <= KEEP_TURNS:
                    rereads.append((key, taken[key], turn))
            convo.append((n, None))
        elif role == "tool":
            convo.append((n, keys.get(row[2])))
        else:
            convo.append((n, None))
    return {
        "events": events,
        "rereads": rereads,
        "cache_hit_rate": round(hits / sent, 4),
    }


#: the window unillm registers for the configured writer and helper models; a run reads it from unillm at run time
REPLAY_WINDOW = 1_050_000


def test_the_recorded_pass_stays_under_the_shared_point_and_its_cache_rate_is_recorded():
    """Under the shared rule (85% of the window or the fixed point, whichever comes first) and the models' window
    (1,050,000), the recorded pass, whose prompt grows to about 113k tokens, stays under the fixed 150,000-token point:
    it does not compact, and nothing is re-read."""
    r = _replay(REPLAY_WINDOW)
    print("replay", json.dumps(r))  # the measurement, in the test log
    assert (
        context_compression.compaction_point(REPLAY_WINDOW)
        == context_compression.COMPACTION_AT_TOKENS
        == 150_000
    )
    assert r["events"] == [], r
    assert r["rereads"] == [], r
    assert 0.0 < r["cache_hit_rate"] <= 1.0


def test_every_agent_reads_one_compaction_point():
    """The agent loop and the memory agents decide with the same constants and function."""
    cc = context_compression
    # a large window: the fixed point comes first
    assert cc.compaction_point(1_000_000) == 150_000
    assert writer_pass.compaction_due(
        150_000,
        1_000_000,
    ) and not writer_pass.compaction_due(149_999, 1_000_000)
    assert cc.context_over_threshold(150_000, cc.COMPACTION_THRESHOLD, 1_000_000)
    # a smaller window: 85% of it comes first
    assert cc.compaction_point(40_000) == 34_000
    assert writer_pass.compaction_due(
        34_000,
        40_000,
    ) and not writer_pass.compaction_due(33_999, 40_000)
    # no known window: the fixed point alone
    for unknown in (None, 0, "x"):
        assert cc.compaction_point(unknown) == 150_000
        assert writer_pass.compaction_due(
            150_000,
            unknown,
        ) and not writer_pass.compaction_due(149_999, unknown)
    # an unknown prompt size never triggers
    assert not writer_pass.compaction_due(None, 1_000)


def test_the_fixed_point_is_one_constant():
    """Rev 2: the fixed compaction point is a constant (UNIFY_COMPACTION_AT_TOKENS is gone), used with no known window
    and whenever it comes before the window's threshold."""
    assert context_compression.COMPACTION_AT_TOKENS == 150_000
    assert context_compression.compaction_point(None) == 150_000
    assert context_compression.compaction_point(1_000_000) == 150_000
