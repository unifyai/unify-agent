"""Symbolic: ``UNIFY_SHORTLIST_RELATED`` lists possibly related entries by statement, apart from the gated list.

The request-gated shortlist finds a stored entry when the new request
shares the rare words of the request it was stored for. In the retrieval
study SEMANTIC-V2 (5 Oct) that found the entry for 88-96% of benchmark
repeats as logged, but for 10-13% once the task sentence was reworded and
3% of day-to-day requests in other words, and no threshold of any matcher
told a reworded repeat from a near-miss. With the switch the gated list
stays the only one that asserts a match; under its own header follow at
most two more entries closest in meaning to the request by their "use this
when" statement, each with that statement and the request it was stored
for, so the model judges the intent. Embeddings here are a fake over a few
concepts (the real embedder's numbers are in the research artifact
``overhaul-lanes/tier2-related-v1``); requests are captured at unillm's
transport, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import re

import numpy as np
import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import library_shortlist as ls
from unify.actor import related_shortlist as rs
from unify.function_manager import task_origin
from unify.function_manager.function_manager import FunctionManager
from unify.settings import ProductionSettings, SETTINGS

GATE = "similar_request:0.175"
RELATED = "statement:2:0.5"
PREAMBLE = "You are my personal assistant. Keep replies short."


def _ask(task: str) -> str:
    return f"{PREAMBLE}\n{task}"


STORED_FOR = _ask(
    "Email the Q3 sales report (reports/q3_sales.xlsx) to my manager with a "
    "quick note saying it covers July to September.",
)
# The same task in other words: almost no word in common with STORED_FOR.
REWORDED = _ask(
    "Can you get the latest revenue summary over to my boss? It lives in "
    "reports/sept_revenue.pdf.",
)
# Similar words, another intent: listed, but only as possibly related.
NEAR_MISS = _ask(
    "Forward my boss's latest email about the revenue summary to the team.",
)
UNRELATED = _ask(
    "Plan a week of vegetarian dinners for two and write the shopping list.",
)
OTHERS = [
    (
        "book_table",
        "Book a table at a restaurant for a party.",
        _ask("Book a table for four at an Italian restaurant on Friday evening."),
    ),
    (
        "chart_expenses",
        "Turn a CSV of expenses into a monthly bar chart.",
        _ask("Convert the attached CSV of expenses into a monthly bar chart."),
    ),
    (
        "passport_reminder",
        "Set a reminder to renew a passport.",
        _ask("Remind me to renew my passport in March."),
    ),
]
SEND_DOC = "Attach a report file and email it to the user's line manager."

# The fake embedder's concepts: words of one group point the same way.
CONCEPTS = [
    {"report", "reports", "summary", "revenue", "sales"},
    {"manager", "boss", "boss's"},
    {"email", "send", "forward", "attach", "over"},
    {"dinner", "dinners", "vegetarian", "shopping", "meal"},
    {"restaurant", "table", "book"},
    {"chart", "csv", "expenses"},
    {"passport", "renew", "remind", "reminder"},
]


def _vector(text: str) -> np.ndarray:
    words = re.findall(r"[a-z']+", text.lower())
    v = np.array(
        [0.05] + [sum(w in group for w in words) for group in CONCEPTS],
        dtype=np.float32,
    )
    return v / np.linalg.norm(v)


@pytest.fixture
def embed_calls(monkeypatch):
    """Every embed() call, answered by the concept fake; nothing reaches a model."""
    calls: list[list[str]] = []

    def fake(texts):
        calls.append(list(texts))
        return np.stack([_vector(t) for t in texts])

    import unify.common.embeddings as embeddings
    import unify.common.semantic_search as semantic_search

    monkeypatch.setattr(embeddings, "embed", fake)
    monkeypatch.setattr(semantic_search, "embed", fake)
    return calls


@pytest.fixture
def switches(monkeypatch):
    def set_(*, related=RELATED, gate=GATE, guidance=False):
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_GATE", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_RELATED", related)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_CORPUS", "stream")
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_IDENTIFIERS", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_ORIGIN", guidance)
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SNAPSHOT", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


def _source(name: str, doc: str, params: str = "x: str") -> str:
    return f'def {name}({params}) -> str:\n    """{doc}"""\n    return x\n'


def _seed(actor):
    fm = actor.function_manager
    _in_task(
        STORED_FOR,
        lambda: fm.add_functions(
            implementations=_source("send_report_to_manager", SEND_DOC),
        ),
    )
    for name, doc, request in OTHERS:
        _in_task(request, lambda: fm.add_functions(implementations=_source(name, doc)))


async def _act(task, *, seed=_seed, embed_calls=None, outer_task=None):
    actor = caa.CodeActActor()
    if seed is not None:
        seed(actor)
    token = task_origin.enter(outer_task) if outer_task else None
    try:
        with h.scripted([lambda: h.completion(content="done")] * 8) as provider:
            before = len(embed_calls) if embed_calls is not None else 0
            handle = await actor.act(task, persist=False)
            at_start = list((embed_calls or [])[before:])
            await asyncio.wait_for(handle.result(), 60)
    finally:
        task_origin.leave(token)
        await actor.close()
    return h.session_requests(provider.requests), at_start


def _first_user(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


def _tier1(text: str) -> list[str]:
    for header in (ls._GATED_HEADER_CALL, ls._GATED_HEADER):
        if header in text:
            block = text[text.index(header) :].split("\n\n", 1)[0]
            return ls.shortlisted_names(block)["functions"]
    return []


def _tier2(text: str) -> list[str]:
    return rs.related_names(text)["functions"]


def _tier2_lines(text: str) -> list[str]:
    if rs.HEADER not in text:
        return []
    return text[text.index(rs.HEADER) :].split("\n\n", 1)[0].splitlines()[1:]


# ── through the actor ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_reworded_request_gets_the_entry_under_tier_two_not_tier_one(
    switches,
    embed_calls,
):
    switches()
    requests, at_start = await _act(REWORDED, embed_calls=embed_calls)
    first = _first_user(requests[0])
    assert _tier1(first) == []
    assert _tier2(first)[0] == "send_report_to_manager", first
    line = _tier2_lines(first)[0]
    # Its statement (a template: nobody wrote one), labelled so, and the
    # request it was stored for without the stream's shared preamble.
    assert line.startswith(
        "- function `send_report_to_manager(x: str) -> str`: Use this when you "
        "need to send report to manager. " + SEND_DOC,
    )
    assert rs.TEMPLATE_LABEL in line
    assert line.endswith(
        ' · stored for: "Email the Q3 sales report (reports/q3_sales.xlsx) to my '
        'manager with a quick note saying it covers July to September."',
    )
    assert PREAMBLE not in line
    # One embedding call at act() start: the request's distinct lines first.
    assert len(at_start) == 1
    assert at_start[0][0] == REWORDED.split("\n", 1)[1]
    assert first.endswith(f"\n\n---\n\n{REWORDED}")
    assert requests[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_unrelated_request_gets_nothing(switches, embed_calls):
    switches()
    requests, at_start = await _act(UNRELATED, embed_calls=embed_calls)
    first = _first_user(requests[0])
    assert rs.HEADER not in first and ls._GATED_HEADER not in first
    assert first.endswith(UNRELATED)
    # Ranked (one call), and nothing reached the floor.
    assert len(at_start) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_near_miss_may_be_listed_but_only_as_possibly_related(
    switches,
    embed_calls,
):
    switches()
    requests, _ = await _act(NEAR_MISS, embed_calls=embed_calls)
    first = _first_user(requests[0])
    assert _tier1(first) == []
    assert "send_report_to_manager" in _tier2(first)
    assert first.index(rs.HEADER) < first.index("send_report_to_manager")
    assert "judge whether the intent matches" in rs.HEADER


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_entry_the_gate_lists_is_not_listed_again(switches, embed_calls):
    switches()
    requests, at_start = await _act(STORED_FOR, embed_calls=embed_calls)
    first = _first_user(requests[0])
    assert _tier1(first) == ["send_report_to_manager"]
    assert "send_report_to_manager" not in _tier2(first)
    # The gated list comes first, unchanged.
    assert first.startswith(ls._GATED_HEADER)
    assert len(at_start) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_sub_agent_gets_no_list_and_embeds_nothing(switches, embed_calls):
    switches()
    requests, at_start = await _act(
        REWORDED,
        embed_calls=embed_calls,
        outer_task="the caller's task",
    )
    assert rs.HEADER not in _first_user(requests[0])
    assert at_start == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_the_first_request_is_the_gated_one_byte_for_byte(
    switches,
    embed_calls,
):
    from unify import db

    firsts = {}
    for related in ("", RELATED):
        switches(related=related)
        db.clear()
        task_origin.request_log_path().unlink(missing_ok=True)
        requests, at_start = await _act(STORED_FOR, embed_calls=embed_calls)
        firsts[related] = (requests[0], at_start)
    (off, off_calls), (on, _) = firsts[""], firsts[RELATED]
    assert off_calls == []
    # Nothing passes the floor besides the gated entry: the request is the same.
    assert _tier2(_first_user(on)) == []
    assert on["messages"] == off["messages"] and on["tools"] == off["tools"]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_an_actor_refuses_the_tier_without_the_gate(switches):
    switches(gate="")
    actor = caa.CodeActActor()
    try:
        with pytest.raises(ValueError, match="UNIFY_SHORTLIST_GATE"):
            await actor.act(REWORDED, persist=False)
    finally:
        await actor.close()


# ── the core surface does not bind tier-two functions ───────────────────


class _Library:
    def __init__(self):
        self.rows = [
            {
                "function_id": 1,
                "name": "send_report_to_manager",
                "argspec": "(x: str) -> str",
                "docstring": SEND_DOC,
                "metadata": {},
            },
            {
                "function_id": 2,
                "name": "email_summary",
                "argspec": "(x: str) -> str",
                "docstring": "Email a revenue summary to a manager.",
                "metadata": {},
            },
        ]

    def _gated_shortlist_rows(self, threshold, k):
        return [{**self.rows[0], "similar_request": 0.4, "usage_calls": 1}]

    def _related_candidates(self):
        return [dict(r) for r in self.rows]


def test_only_the_gated_functions_are_bound(switches, embed_calls):
    switches()
    lib = _Library()
    asked: list = []

    def bind(names):
        asked.append(list(names))
        return {name: False for name in names}

    text = ls.shortlist_block(lib, None, REWORDED, gate=0.175, bind=bind)
    assert asked == [["send_report_to_manager"]]
    assert ls.CALL_FORM in text.split("\n\n")[0]
    assert _tier2(text) == ["email_summary"]
    assert "not loaded until read" in rs.HEADER
    assert ls.CALL_FORM not in text.split("\n\n")[1]


# ── statements ──────────────────────────────────────────────────────────


def test_a_written_statement_is_used_and_a_template_is_labelled():
    row = {"name": "send_report_to_manager", "docstring": SEND_DOC, "metadata": {}}
    assert rs.statement_of("function", row) == (
        f"Use this when you need to send report to manager. {SEND_DOC}",
        False,
    )
    written = {**row, "metadata": {"use_when": "Use this when a report goes up."}}
    assert rs.statement_of("function", written) == (
        "Use this when a report goes up.",
        True,
    )
    shown = rs.render(
        [
            {**row, "kind": "function", "statement": "S1.", "written": True},
            {**row, "kind": "function", "statement": "S2.", "written": False},
        ],
    ).splitlines()
    assert rs.TEMPLATE_LABEL not in shown[1] and rs.TEMPLATE_LABEL in shown[2]


def test_the_review_line_is_parsed_and_kept_only_for_an_entry_of_this_request(
    switches,
    embed_calls,
):
    switches()
    fm = FunctionManager()
    _in_task(
        STORED_FOR,
        lambda: fm.add_functions(
            implementations=_source("send_report_to_manager", SEND_DOC),
        ),
    )
    _in_task(
        OTHERS[0][2],
        lambda: fm.add_functions(implementations=_source("book_table", "Book.")),
    )
    reply = (
        "Stored one function.\n"
        "use_when send_report_to_manager: Use this when a report file should be "
        "emailed to the requester's manager. A second sentence is dropped.\n"
        "use_when book_table: Use this when booking a table.\n"
        "use_when no_such_function: Use this when nothing.\n"
        '{"answer_outcome": "unknown"}'
    )
    outcomes = _in_task(STORED_FOR, lambda: rs.record_statements(reply, fm, None))
    assert outcomes == {
        ("function", "send_report_to_manager"): "kept",
        ("function", "book_table"): "not stored under this request",
        ("function", "no_such_function"): "not stored under this request",
    }
    kept = fm._get_log_by_function_id(
        function_id=fm.list_functions()["send_report_to_manager"]["function_id"],
    )
    assert kept["metadata"]["use_when"] == (
        "Use this when a report file should be emailed to the requester's manager."
    )
    # The kept statement was embedded once, at store time.
    assert embed_calls[-1] == [kept["metadata"]["use_when"]]


def test_a_statement_naming_this_task_instance_is_not_kept(switches, embed_calls):
    switches()
    fm = FunctionManager()
    request = _ask(
        'Email the "Q3 Board Pack Final" report to my manager. Ref: req-7f3a91c2.',
    )
    _in_task(
        request,
        lambda: fm.add_functions(
            implementations=_source("send_report_to_manager", SEND_DOC),
        ),
    )
    for statement in (
        'Use this when the "Q3 Board Pack Final" report must be emailed.',
        "Use this when handling req-7f3a91c2.",
    ):
        outcomes = _in_task(
            request,
            lambda: rs.record_statements(
                f"use_when send_report_to_manager: {statement}",
                fm,
                None,
            ),
        )
        assert outcomes == {("function", "send_report_to_manager"): "instance"}
    row = fm._get_log_by_function_id(
        function_id=fm.list_functions()["send_report_to_manager"]["function_id"],
    )
    assert "use_when" not in row["metadata"]
    assert embed_calls == []


def test_off_no_statement_is_kept(switches, embed_calls):
    switches(related="")
    fm = FunctionManager()
    _in_task(
        STORED_FOR,
        lambda: fm.add_functions(
            implementations=_source("send_report_to_manager", "S."),
        ),
    )
    line = "use_when send_report_to_manager: Use this when a report goes up."
    assert _in_task(STORED_FOR, lambda: rs.record_statements(line, fm, None)) == {}


def test_guidance_with_an_origin_keeps_its_statement_hidden_from_reads(
    switches,
    embed_calls,
):
    from unify.guidance_manager.guidance_manager import GuidanceManager

    switches(guidance=True)
    gm = GuidanceManager()
    out = _in_task(
        STORED_FOR,
        lambda: gm.add_guidance(title="Sending reports", content="Attach, then email."),
    )
    gid = out["details"]["guidance_id"]
    line = f"use_when guidance {gid}: Use this when a report must reach a manager."
    assert _in_task(STORED_FOR, lambda: rs.record_statements(line, None, gm)) == {
        ("guidance", str(gid)): "kept",
    }
    (row,) = gm._origin_rows()
    assert row["metadata"]["use_when"] == "Use this when a report must reach a manager."
    assert "use_when" not in json.dumps(gm.get_guidance(guidance_id=gid), default=str)
    # Another request did not store it: nothing is kept.
    assert _in_task(UNRELATED, lambda: rs.record_statements(line, None, gm)) == {
        ("guidance", str(gid)): "not stored under this request",
    }


def test_the_review_lines_are_read_however_they_are_decorated():
    text = (
        "- `use_when double: Use this when a number is doubled.`\n"
        "  * use_when guidance #12: Use this when tidying a list.\n"
        "USE_WHEN halve : Use this when halving.\n"
        "use_when double: Use this when a value is doubled.\n"
    )
    assert rs.parse_statements(text) == {
        ("function", "double"): "Use this when a value is doubled.",
        ("guidance", "12"): "Use this when tidying a list.",
        ("function", "halve"): "Use this when halving.",
    }


# ── the review asks for statements only with the switch ─────────────────


async def _review(review_reply: str):
    """A persistent session over STORED_FOR, its end and its scripted review."""
    from unify.actor.code_act_actor import (
        SESSION_ENDED,
        CodeActActor,
        _StorageCheckHandle,
    )
    from unify.common.async_tool_loop import start_async_tool_loop

    actor = CodeActActor()
    token = task_origin.enter(STORED_FOR)
    try:
        actor.function_manager.add_functions(
            implementations=_source("send_report_to_manager", SEND_DOC),
        )
        replies = [
            lambda: h.completion(content="Sent."),
            lambda: h.completion(content=review_reply),
        ]
        with h.scripted(replies) as provider:
            inner = start_async_tool_loop(
                h.new_client("You are a scripted actor."),
                STORED_FOR,
                h.session_tools(actor),
                loop_id="CodeActActor.act",
                log_steps=False,
                timeout=60,
                persist=True,
            )
            handle = _StorageCheckHandle(inner=inner, actor=actor)
            while True:
                note = await asyncio.wait_for(handle.next_notification(), 30)
                if isinstance(note, dict) and note.get("type") == "response":
                    break
            await handle.stop(SESSION_ENDED)
            while True:
                note = await asyncio.wait_for(handle.next_notification(), 30)
                if isinstance(note, dict) and note.get("type") in (
                    "storage_review_complete",
                    "storage_review_skipped",
                ):
                    break
            await asyncio.wait_for(handle._lifecycle_task, 30)
        meta = actor.function_manager._get_log_by_function_id(
            function_id=actor.function_manager.list_functions()[
                "send_report_to_manager"
            ]["function_id"],
        )["metadata"]
    finally:
        task_origin.leave(token)
        await actor.close()
    return note, provider.requests, meta


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("related", ["", RELATED])
async def test_the_review_is_asked_for_statements_and_its_line_is_kept(
    switches,
    embed_calls,
    monkeypatch,
    related,
):
    switches(related=related)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", False)
    note, requests, meta = await _review(
        "Nothing new to store.\nuse_when send_report_to_manager: Use this when "
        "a report file should be emailed to the requester's manager.",
    )
    assert note["type"] == "storage_review_complete"
    asked = rs.REVIEW_SECTION.split("\n")[0] in json.dumps(requests[-1]["messages"])
    assert asked is bool(related)
    if related:
        assert meta["use_when"].startswith("Use this when a report file")
    else:
        assert "use_when" not in meta


# ── the texts and the setting ───────────────────────────────────────────


def test_the_texts_ask_nothing_and_name_no_benchmark():
    from tests.actor.code_act.test_prompt_generality import BENCHMARK_WORDS

    words = re.findall(r"[a-z]+", rs.HEADER.lower())
    for word in ("must", "always", "first", "before", "try", "should"):
        assert word not in words
    for text in (rs.HEADER, rs.REVIEW_SECTION, rs.TEMPLATE_LABEL):
        assert not BENCHMARK_WORDS.search(text)


def test_the_distinct_lines_drop_what_half_the_stream_shares():
    earlier = [frozenset(task_origin.line_keys(_ask(t))) for t in ("a b", "c d", "e f")]
    assert rs.distinct_text(REWORDED, earlier) == (
        REWORDED.split("\n", 1)[1],
        [PREAMBLE],
    )
    # Fewer than two earlier requests: the whole request.
    assert rs.distinct_text(REWORDED, earlier[:1]) == (REWORDED, [])


def test_at_most_k_rows_none_under_the_floor_best_first():
    rows = [
        ("function", {"function_id": i, "name": n, "docstring": d, "metadata": {}})
        for i, (n, d) in enumerate(
            [
                ("send_report_to_manager", SEND_DOC),
                ("email_summary", "Email a revenue summary to a manager."),
                ("forward_report", "Forward a report."),
                ("book_table", "Book a restaurant table."),
            ],
            start=1,
        )
    ]
    calls = []

    def embed(texts):
        calls.append(texts)
        return np.stack([_vector(t) for t in texts])

    kept = rs.related_rows(rows, REWORDED, [], k=2, floor=0.5, embed=embed)
    assert len(calls) == 1 and len(kept) == 2
    assert kept[0]["_cosine"] >= kept[1]["_cosine"] >= 0.5
    assert "book_table" not in [r["name"] for r in kept]
    assert rs.related_rows(rows, UNRELATED, [], k=2, floor=0.5, embed=embed) == []
    assert rs.related_rows(rows, REWORDED, [], k=2, floor=0.99, embed=embed) == []


@pytest.mark.parametrize(
    "value, stored, parsed",
    [
        ("", "", None),
        ("statement:2", "statement:2", (2, None)),
        (" Statement:1 ", "statement:1", (1, None)),
        ("statement:2:0.65", "statement:2:0.65", (2, 0.65)),
    ],
)
def test_the_setting_parses(value, stored, parsed):
    settings = ProductionSettings(UNIFY_SHORTLIST_RELATED=value)
    assert settings.UNIFY_SHORTLIST_RELATED == stored
    assert settings.shortlist_related() == parsed


@pytest.mark.parametrize(
    "value",
    [
        "2",
        "statement",
        "statement:0",
        "statement:3",
        "statement:2:1",
        "cosine:2",
        "statement:x",
    ],
)
def test_the_setting_refuses_other_values(value):
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_SHORTLIST_RELATED=value)


def test_the_floor_follows_the_embedder():
    from unify.common import embeddings

    assert (
        rs.floor_for(embeddings.LOCAL.model, None)
        == rs.DEFAULT_FLOORS["BAAI/bge-small-en-v1.5"]
    )
    assert (
        rs.floor_for(embeddings.OPENROUTER.model, None)
        == rs.DEFAULT_FLOORS["openai/text-embedding-3-small"]
    )
    assert rs.floor_for(embeddings.OPENROUTER.model, 0.4) == 0.4


# ── in the core sandbox: a tier-two function is not callable until read ─


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_under_the_core_surface_only_the_gated_function_is_callable(
    core_world,
    switches,
    embed_calls,
    monkeypatch,
):
    from unify import db

    switches()
    monkeypatch.setattr(SETTINGS, "UNIFY_CORE_BIND_LISTED", True)

    def gated(self, threshold, k, *args, **kwargs):
        row = db.query_one(
            "SELECT function_id, name, argspec, docstring FROM functions "
            "WHERE name = 'send_report_to_manager'",
        )
        return [{**row, "similar_request": 0.4, "usage_calls": 0}] if row else []

    monkeypatch.setattr(FunctionManager, "_gated_shortlist_rows", gated)
    actor = new_actor(can_store=False)
    actor.function_manager.add_functions(
        implementations=[
            _source("send_report_to_manager", SEND_DOC),
            _source("email_summary", "Email a revenue summary to a manager."),
        ],
    )
    code = (
        "try:\n"
        "    print('send_report_to_manager', send_report_to_manager('ok'))\n"
        "except NameError:\n"
        "    print('send_report_to_manager', 'NameError')\n"
        "try:\n"
        "    print('email_summary', email_summary('ok'))\n"
        "except NameError:\n"
        "    print('email_summary', 'NameError')\n"
    )
    replies = (
        lambda: h.completion(
            calls=[("execute_code", {"thought": "Try both.", "code": code})],
        ),
        lambda: h.completion(content="done"),
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act(REWORDED, persist=False)
            await asyncio.wait_for(handle.result(), 120)
    finally:
        await actor.close()
    requests = h.session_requests(provider.requests)
    first = _first_user(requests[0])
    assert _tier1(first) == ["send_report_to_manager"]
    assert _tier2(first) == ["email_summary"]
    reply = json.dumps(
        [m["content"] for m in requests[-1]["messages"] if m.get("role") == "tool"],
    )
    assert "send_report_to_manager ok" in reply, reply
    assert "email_summary NameError" in reply, reply
