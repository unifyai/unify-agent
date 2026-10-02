"""Symbolic: the actor's prompt and the storage rulebook name no benchmark.

The harness is general-purpose; benchmarks test whether its behaviour
transfers, so their names and actions must not reach what the model reads.
Text has been written from benchmark evidence before: the reply-protocol
note (``UNIFY_REPLY_PROTOCOL_NOTE``) came from ARC episodes whose actor
delegated ``request_demos`` to sub-actors that called
``request_demonstration``, and it checks only its own words. This lint
renders the actor's system prompt and tool schemas, and the storage review's
rulebook constants, notes and sent requests, with every prompt-affecting
switch off, on and mixed, and fails on any benchmark or dataset name, or a
benchmark's own vocabulary, as a whole word.

Text that already uses one of these words for its general meaning is
allowed by its exact wording in ``ALLOWED``, so the lint flags new text, not
the shipped rulebook.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import prompt_builders as pb
from unify.settings import SETTINGS

# Benchmark and dataset names (with or without a separator), and words that
# belong to one benchmark's protocol: ARC's grids and demonstrations.
BENCHMARK_WORDS = re.compile(
    r"\b("
    r"arc|arc[\s_-]?agi|appworld|app[\s_-]world|science[\s_-]?world|"
    r"alf[\s_-]?world|travel[\s_-]?planner|web[\s_-]?arena|swe|swe[\s_-]?bench|"
    r"grids?|demos?|demonstrations?|demonstrate[sd]?"
    r")\b",
    re.IGNORECASE,
)

# Existing text that uses a listed word for its general meaning, by exact
# wording. Each entry is removed before the scan, so the same word anywhere
# else still fails.
ALLOWED = {
    # The guidance store's rule on deliverables a requester will want again:
    # "demonstrate" is any request to show how, not an ARC demonstration.
    "_STORAGE_TWO_STORES": "Framing such as 'demonstrate', 'walk through' or 'example'",
}

# Switches whose value changes the actor's prompt or the storage rulebook.
SWITCH_SETS = {
    "off": {},
    "on": {
        "UNIFY_REVIEW_FRAMING": "unified",
        "UNIFY_CURATION_DOCTRINE": "compose",
        "UNIFY_REPLY_PROTOCOL_NOTE": True,
        "UNIFY_FUNCTION_PATCH": True,
        "UNIFY_STORE_CHECK": "resolve",
    },
    "framing only": {"UNIFY_REVIEW_FRAMING": "unified"},
    "doctrine and note": {
        "UNIFY_CURATION_DOCTRINE": "compose",
        "UNIFY_REPLY_PROTOCOL_NOTE": True,
    },
}
SWITCH_OFF = {
    "UNIFY_REVIEW_FRAMING": "",
    "UNIFY_CURATION_DOCTRINE": "",
    "UNIFY_REPLY_PROTOCOL_NOTE": False,
    "UNIFY_FUNCTION_PATCH": False,
    "UNIFY_STORE_CHECK": "",
}

PROMPT_MODES = {
    "act": {"can_store": True, "discovery_first_policy": True},
    "persist": {"can_store": True, "persist": True},
    "read only": {"can_store": True, "library_read_only": True},
    "no store": {},
}


@pytest.fixture(params=sorted(SWITCH_SETS))
def switches(request, monkeypatch):
    for name, value in {**SWITCH_OFF, **SWITCH_SETS[request.param]}.items():
        assert hasattr(SETTINGS, name), name
        monkeypatch.setattr(SETTINGS, name, value)
    return request.param


def _without_allowed(text: str) -> str:
    for wording in ALLOWED.values():
        text = text.replace(wording, "")
    return text


def _benchmark_words(texts: dict[str, str]) -> list[str]:
    found = []
    for source, text in texts.items():
        text = _without_allowed(text)
        for match in BENCHMARK_WORDS.finditer(text):
            context = text[max(0, match.start() - 50) : match.end() + 50]
            found.append(f"{source}: {match.group(0)!r} in {context!r}")
    return found


def _rulebook() -> dict[str, str]:
    texts = {
        name: value
        for name, value in vars(caa).items()
        if name.startswith(("_STORAGE_", "_REVIEW_")) and isinstance(value, str)
    }
    for name, fn in vars(caa).items():
        if not (name.startswith("_storage_") and name.endswith("_note")):
            continue
        params = inspect.signature(fn).parameters.values()
        if all(p.default is not p.empty for p in params):
            texts[f"{name}()"] = fn()
    texts["_storage_review_outcome_note(lessons=True)"] = (
        caa._storage_review_outcome_note(lessons=True)
    )
    texts["_storage_base_instructions()"] = caa._storage_base_instructions()
    texts["_review_fork_role()"] = caa._review_fork_role()
    return texts


# ── the lint itself ──────────────────────────────────────────────────────


def test_the_pattern_catches_names_and_actions_as_whole_words():
    caught = [
        "ARC",
        "arc-agi",
        "AppWorld",
        "ScienceWorld",
        "science world",
        "ALFWorld",
        "TravelPlanner",
        "WebArena",
        "SWE-bench",
        "grid",
        "demos",
        "demonstration",
    ]
    for word in caught:
        assert BENCHMARK_WORDS.search(f"solve the {word} task"), word
    for text in ("search", "archive", "gridlock", "demonstrably", "answer"):
        assert not BENCHMARK_WORDS.search(text), text


def test_each_allowance_is_still_the_shipped_text():
    for source, wording in ALLOWED.items():
        assert wording in getattr(caa, source), source
        assert BENCHMARK_WORDS.search(wording), source


# ── the actor ────────────────────────────────────────────────────────────


@pytest.fixture
def actor_tools():
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return actor, dict(actor.get_tools("act"))


@pytest.mark.parametrize("mode", sorted(PROMPT_MODES))
def test_the_actors_prompt_names_no_benchmark(switches, actor_tools, mode):
    actor, tools = actor_tools
    prompt = pb.build_code_act_prompt(
        environments=actor.environments,
        tools=tools,
        **PROMPT_MODES[mode],
    )
    assert "### Execution Rules" in prompt
    assert _benchmark_words({f"prompt ({switches}, {mode})": prompt}) == []


def test_the_actors_tool_schemas_name_no_benchmark(actor_tools):
    from unify.common.llm_helpers import method_to_schema

    _actor, tools = actor_tools
    schemas = {
        name: json.dumps(method_to_schema(getattr(tool, "fn", tool), name))
        for name, tool in tools.items()
    }
    assert "execute_code" in schemas
    assert _benchmark_words(schemas) == []


# ── the storage review ───────────────────────────────────────────────────


def test_the_rulebook_names_no_benchmark(switches):
    texts = _rulebook()
    assert {"_STORAGE_TWO_STORES", "_REVIEW_FORK_ROLE"} <= set(texts)
    assert _benchmark_words(texts) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fork", [False, True], ids=["standalone", "fork"])
async def test_the_sent_review_names_no_benchmark(switches, monkeypatch, fork):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", fork)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", fork)
    _summary, _, requests = await h.scenario_review()
    review = requests[2]["messages"]
    # The rulebook arrives as the system prompt, or as the fork's last message.
    sent = review[-1]["content"] if fork else review[0]["content"]
    assert "## Instructions" in sent
    assert _benchmark_words({f"review ({switches})": sent}) == []
