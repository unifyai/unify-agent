"""Symbolic: ``UNIFY_EVIDENCE_LIST_MATCHER=judge``, a model decides whether a stored entry does the request's job.

The matching bake-off (research artifact matching-bakeoff-v1) found a floor
on one similarity score cannot both find reworded repeats and stay silent on
near-misses; an embedding shortlist with no floor, read by one small model
call that may answer "none", did both (balanced accuracy 0.98 on a fresh,
preregistered set, against 0.63 for the floor). These tests pin the pieces:
the prompt is the bake-off's, the cards carry what the bake-off's cards
carried, the pool puts key matches first, a reply picks one card or none,
and a failed call picks nothing. The model is a scripted coroutine: nothing
leaves the process.
"""

from __future__ import annotations

import asyncio
import hashlib
import re

import numpy as np
import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act import test_evidence_list as tel
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import evidence_judge as ej
from unify.actor import evidence_list as ev
from unify.actor import related_shortlist
from unify.function_manager import task_origin
from unify.settings import SETTINGS

SPEND = "Find the five largest card payments last month and total them."
SPEND_AGAIN = "Total my three smallest card payments from last week."
REFUND = "Refund the largest card payment from last month."
#: sha256 of the bake-off's ``JUDGE_B`` template (matching-bakeoff-v1, scripts/mb_prompts.py).
JUDGE_B_SHA256 = "1e057e15e6c561af3ee2296ccbf725cf768451ebf237cced2ff11eb7f6d67d1a"  # pragma: allowlist secret


def _function(fid: int, name: str, doc: str, origin: str = "", **meta) -> dict:
    metadata = dict(meta)
    if origin:
        metadata[task_origin.REQUESTS_FIELD] = [origin]
    return {
        "function_id": fid,
        "name": name,
        "argspec": "(n, period, largest=True)",
        "docstring": doc,
        "metadata": metadata,
    }


def _note(gid: int, title: str, content: str, origin: str = "") -> dict:
    metadata = {task_origin.REQUESTS_FIELD: [origin]} if origin else {}
    return {
        "guidance_id": gid,
        "title": title,
        "content": content,
        "metadata": metadata,
    }


TOTAL = _function(
    1,
    "total_extreme_payments",
    "Total the n largest or smallest card payments in a period.",
    SPEND,
)
REFUNDER = _function(2, "refund_payment", "Refund one card payment.", REFUND)
NOTE = _note(
    7,
    "Card payments",
    "Payments are listed newest first; amounts are negative.",
    SPEND,
)
ENTRIES = {
    ("function", "total_extreme_payments"): TOTAL,
    ("function", "refund_payment"): REFUNDER,
    ("guidance", "7"): NOTE,
}


def test_the_prompt_is_the_bakeoff_judge_verbatim():
    """The prompt the bake-off measured (scripts/mb_prompts.py JUDGE_B), byte for byte."""
    assert hashlib.sha256(ej.PROMPT.encode()).hexdigest() == JUDGE_B_SHA256
    assert "the setting of a condition or direction" in ej.PROMPT
    assert 'Answer "none" if no entry does the same job' in ej.PROMPT
    text = ej.prompt(SPEND_AGAIN, ["E1: a", "E2: b"])
    assert (
        SPEND_AGAIN in text and "E1: a\nE2: b" in text and "{" in text.splitlines()[-1]
    )


def test_a_card_carries_name_description_job_and_origin():
    card = ej.judge_card("E1", "function", TOTAL)
    assert card.splitlines()[0] == "E1: total_extreme_payments"
    assert "what it does: Total the n largest" in card
    assert "written for (job): (unknown)" in card  # no review-written statement
    assert f"original request: {SPEND}" in card
    stated = dict(
        TOTAL,
        metadata={
            **TOTAL["metadata"],
            related_shortlist.STATEMENT: "Use this when totalling extreme payments.",
        },
    )
    assert (
        "written for (job): Use this when totalling extreme payments."
        in ej.judge_card("E1", "function", stated)
    )
    note = ej.judge_card("E2", "guidance", NOTE)
    assert note.startswith("E2: Card payments\n   what it does: Payments are listed")


def test_a_long_origin_is_left_off_the_card_and_shared_identifiers_are_named():
    long_origin = "x" * (ej.ORIGIN_CHARS + 1)
    row = _function(3, "f", "d", long_origin)
    assert "original request" not in ej.judge_card("E1", "function", row)
    assert "its request also named: INV-2041, ledger.csv" in ej.judge_card(
        "E1",
        "function",
        row,
        ["INV-2041", "ledger.csv"],
    )


def test_the_embedded_card_holds_name_signature_description_and_origin():
    text = ej.card_text("function", TOTAL)
    assert text.splitlines()[:2] == [
        "total_extreme_payments",
        "(n, period, largest=True)",
    ]
    assert text.endswith(f"Written for: {SPEND}")
    assert ej.card_text("guidance", _note(9, "T", "C")) == "T\n\nC"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"choice": "E2", "confidence": 80}', (1, 80.0)),
        ('Sure: {"choice": "e1", "confidence": "90"}', (0, 90.0)),
        ('{"choice": "none", "confidence": 95}', (None, 95.0)),
        ('{"choice": "E9", "confidence": 50}', (None, 50.0)),  # out of range
        ('{"choice": "2"}', (None, None)),  # not an entry id
        ("no json here", (None, None)),
        ("{not json}", (None, None)),
    ],
)
def test_a_reply_picks_one_card_or_none(raw, expected):
    assert ej.parse_choice(raw, 3) == expected


def test_the_pool_puts_key_matches_first_then_the_closest_cards():
    scores = {
        ("f", "a"): 0.9,
        ("f", "b"): 0.8,
        ("f", "c"): 0.7,
        ("f", "d"): 0.6,
        ("f", "e"): 0.5,
        ("f", "g"): 0.4,
    }
    assert ej.pool([], scores) == [
        ("f", "a"),
        ("f", "b"),
        ("f", "c"),
        ("f", "d"),
        ("f", "e"),
    ]
    assert ej.pool([("f", "g")], scores)[:2] == [("f", "g"), ("f", "a")]
    assert len(ej.pool([("f", "g")], scores)) == ej.K
    assert ej.pool([("f", "g")], scores, include_keyed=False)[0] == ("f", "a")
    assert ("f", "a") not in ej.pool([], scores, exclude=[("f", "a")])


def _scripted(reply):
    prompts = []

    async def generate(text):
        prompts.append(text)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return generate, prompts


def test_the_judge_picks_the_entry_that_does_the_same_job():
    generate, prompts = _scripted('{"choice": "E1", "confidence": 88}')
    keys = [("function", "total_extreme_payments"), ("function", "refund_payment")]
    verdict = asyncio.run(ej.decide(SPEND_AGAIN, ENTRIES, keys, generate=generate))
    assert verdict.choice == ("function", "total_extreme_payments")
    assert verdict.confidence == 88.0 and not verdict.failed
    assert len(prompts) == 1 and "E1: total_extreme_payments" in prompts[0]
    assert "E2: refund_payment" in prompts[0]


def test_none_and_a_failed_call_pick_nothing():
    keys = [("function", "refund_payment")]
    generate, _ = _scripted('{"choice": "none", "confidence": 97}')
    assert (
        asyncio.run(ej.decide(SPEND_AGAIN, ENTRIES, keys, generate=generate)).choice
        is None
    )
    generate, _ = _scripted(RuntimeError("provider down"))
    failed = asyncio.run(ej.decide(SPEND_AGAIN, ENTRIES, keys, generate=generate))
    assert failed.choice is None and failed.failed


def test_no_candidates_means_no_call():
    generate, prompts = _scripted('{"choice": "E1"}')
    verdict = asyncio.run(
        ej.decide(SPEND_AGAIN, ENTRIES, [("function", "gone")], generate=generate),
    )
    assert verdict.choice is None and prompts == []


def test_the_texts_name_no_benchmark_and_ask_for_no_example_check():
    text = ej.PROMPT.lower()
    for word in ("arc", "appworld", "scienceworld", "puzzle", "grid", "demo"):
        assert f" {word}" not in text
    for phrase in (
        "example pair",
        "expected output",
        "check against",
        "verify against",
    ):
        assert phrase not in text


# ── the list with the judge (evidence_list.aselect / ablock, the actor) ──────


LIB = ev.build_library([TOTAL, REFUNDER], [NOTE], [(1, 7)])


def _embed(texts):
    """Payments-ish texts close together, anything else apart (a stand-in embedder)."""
    out = []
    for t in texts:
        low = t.lower()
        v = np.array(
            [0.1, "payment" in low, "total" in low, "refund" in low],
            dtype=np.float32,
        )
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


def _aselect(request, generate, logged=()):
    return asyncio.run(
        ev.aselect(
            LIB,
            request,
            request,
            None,
            list(logged),
            {},
            threshold=0.175,
            embed=_embed,
            generate=generate,
        ),
    )


def test_an_exact_repeat_is_listed_without_asking_the_model():
    generate, prompts = _scripted('{"choice": "E1"}')
    listing = _aselect(SPEND, generate)
    assert [card.key() for card, _ in listing.seen] == [
        ("function", "total_extreme_payments"),
    ]
    assert listing.seen[0][1].rank == 3 and prompts == [] and listing.verdict is None


def test_the_judged_pick_heads_a_seen_before_card_that_says_so():
    generate, prompts = _scripted('{"choice": "E1", "confidence": 90}')
    listing = _aselect(SPEND_AGAIN, generate)
    assert len(prompts) == 1 and listing.related == []
    ((card, match),) = listing.seen
    assert match.how == "judged" and card.key() == listing.verdict.choice
    text = ev.render(listing.seen, listing.related, {})
    assert text.startswith(ev.SEEN_HEADER)
    assert "a model judged that it does the same job as this request" in text


def test_none_from_the_model_means_no_list():
    generate, _ = _scripted('{"choice": "none", "confidence": 99}')
    listing = _aselect("Translate the onboarding guide into Spanish.", generate)
    assert listing.seen == [] and listing.related == []
    assert ev.render(listing.seen, listing.related, {}) is None


def test_a_failed_judge_call_lists_nothing_beyond_the_same_request():
    generate, _ = _scripted(RuntimeError("timeout"))
    listing = _aselect(SPEND_AGAIN, generate)
    assert listing.seen == [] and listing.verdict.failed


def test_a_shared_identifier_puts_the_entry_first_and_is_named_on_its_card():
    tagged = _function(
        5,
        "reconcile_invoice",
        "Reconcile one invoice against the ledger.",
        "Reconcile invoice INV-20417 with ledger.csv",
    )
    lib = ev.build_library([TOTAL, tagged], [], [])
    request = "Email the vendor about invoice INV-20417 being late."
    logged = [
        SPEND,
        "Reconcile invoice INV-20417 with ledger.csv",
        "Plan the offsite.",
        request,
    ]
    generate, prompts = _scripted('{"choice": "none"}')
    asyncio.run(
        ev.aselect(
            lib,
            request,
            request,
            None,
            logged,
            {},
            threshold=0.175,
            embed=_embed,
            generate=generate,
        ),
    )
    assert "E1: reconcile_invoice" in prompts[0]
    assert "its request also named: INV-20417" in prompts[0]


@pytest.fixture
def judge_switches(monkeypatch):
    def set_(*, evidence="on", matcher="judge", record=True):
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_EVIDENCE_LIST", evidence)
        monkeypatch.setattr(SETTINGS, "UNIFY_EVIDENCE_LIST_MATCHER", matcher)
        monkeypatch.setattr(SETTINGS, "UNIFY_ENTRY_RECORD", record)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_LIFT", "")

    return set_


def test_the_matcher_setting_parses_and_refuses_bad_values():
    from unify.settings import ProductionSettings

    assert (
        ProductionSettings(
            UNIFY_EVIDENCE_LIST_MATCHER=" Judge ",
        ).UNIFY_EVIDENCE_LIST_MATCHER
        == "judge"
    )
    assert (
        ProductionSettings(UNIFY_EVIDENCE_LIST_MATCHER="").UNIFY_EVIDENCE_LIST_MATCHER
        == ""
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_EVIDENCE_LIST_MATCHER="cosine")


def test_the_judge_refuses_to_start_without_the_list(judge_switches):
    judge_switches(evidence="")
    with pytest.raises(
        ValueError,
        match="UNIFY_EVIDENCE_LIST_MATCHER needs UNIFY_EVIDENCE_LIST",
    ):
        ev.require_prerequisites()
    judge_switches()
    ev.require_prerequisites()
    assert ev.judged()
    judge_switches(matcher="keys")
    assert not ev.judged()


# ── through the actor: one judge call at the task start, its pick in the first message ──


embed_calls = tel.embed_calls  # the concept fake through the real embedding cache


def _pick(name):
    """A judge reply naming the card *name* heads in the prompt it answers (``none`` if absent)."""

    def reply():
        prompt_text = h._ACTIVE_PROVIDER[0].requests[-1]["messages"][-1]["content"]
        found = re.search(r"^(E\d): " + re.escape(name) + "$", prompt_text, re.M)
        choice = found.group(1) if found else "none"
        return h.completion(content=f'{{"choice": "{choice}", "confidence": 85}}')

    return reply


async def _act_judged(task, judge_reply):
    actor = caa.CodeActActor()
    tel._seed(actor)
    first_reply = (
        judge_reply
        if callable(judge_reply)
        else (lambda: h.completion(content=judge_reply))
    )
    replies = [first_reply] + [
        lambda: h.completion(content="done"),
    ] * 8
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act(task, persist=False)
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    judge = [r for r in provider.requests if "SAME JOB" in str(r["messages"])]
    actor_requests = h.session_requests(
        [r for r in provider.requests if r not in judge],
    )
    first = next(
        m["content"] for m in actor_requests[0]["messages"] if m["role"] == "user"
    )
    return judge, first


@pytest.fixture
def actor_switches(monkeypatch):
    def set_(*, matcher="judge"):
        for name, value in {
            "UNIFY_LIBRARY_SHORTLIST": True,
            "UNIFY_EVIDENCE_LIST": "on",
            "UNIFY_EVIDENCE_LIST_MATCHER": matcher,
            "UNIFY_ENTRY_RECORD": True,
            "UNIFY_SHORTLIST_LIFT": "",
            "UNIFY_SHORTLIST_GATE": "",
            "UNIFY_SHORTLIST_RELATED": "",
            "UNIFY_TASK_ORIGIN": True,
            "UNIFY_TRY_FIRST": False,
            "UNIFY_SIMILAR_REQUEST_IDENTIFIERS": True,
            "UNIFY_SIMILAR_REQUEST_CORPUS": "stream",
            "UNIFY_DISCOVERY_GATE": False,
            "UNIFY_LIBRARY_SNAPSHOT": True,
            "UNIFY_BUILTIN_GUIDANCE": False,
            "UNIFY_STORE_CHECK": "",
            "UNIFY_STORE_VERIFY": "",
        }.items():
            monkeypatch.setattr(SETTINGS, name, value)

    return set_


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_the_actor_asks_the_judge_once_and_lists_its_pick(
    actor_switches,
    embed_calls,
):
    actor_switches()
    judge, first = await _act_judged(tel.NEW_PUZZLE, _pick("rotate_table"))
    assert len(judge) == 1
    prompt_text = judge[0]["messages"][-1]["content"]
    assert re.search(r"^E\d: rotate_table$", prompt_text, re.M)
    assert tel.NEW_PUZZLE.strip() in prompt_text
    block = tel._tier(first, ev.SEEN_HEADER)
    assert "rotate_table" in block
    assert "a model judged that it does the same job as this request" in block
    assert ev.RELATED_HEADER not in first


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_the_actor_lists_nothing_when_the_judge_says_none(
    actor_switches,
    embed_calls,
):
    actor_switches()
    judge, first = await _act_judged(
        tel.UNRELATED,
        '{"choice": "none", "confidence": 99}',
    )
    assert len(judge) == 1
    assert ev.SEEN_HEADER[:20] not in first and ev.RELATED_HEADER not in first
