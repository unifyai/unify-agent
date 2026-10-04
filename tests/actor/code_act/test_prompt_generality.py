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

The same texts must not tell the agent to check its work against example
inputs and outputs it was given (4 Oct): that discusses the task under test
with the agent, and the code-first, try-first, inline-curation and
compose-doctrine texts each did. Checking is mechanical instead:
``UNIFY_FUNCTION_CASES`` records real calls of a stored function and replays
them on every change. Upstream's shipped sentences of that kind are allowed
by their exact wording in ``UPSTREAM_EXAMPLE_CHECKS``.
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

# Instructions to check, verify or reproduce given examples or input/output
# pairs. A sentence fails when it names examples or known input/output values
# and a check in the same sentence, or uses one of the fixed phrasings.
EXAMPLE_CHECK_PHRASES = re.compile(
    r"\b("
    r"every example|(worked|given|provided|sample) examples?|"
    r"inputs?\s+(and|/|->|→)\s+(the\s+)?(exact\s+)?outputs?|"
    r"(input|output)[\s/-]*pairs?|expected outputs?|known results?"
    r")\b",
    re.IGNORECASE,
)
EXAMPLE_WORDS = re.compile(
    r"\b(examples?|(inputs?|outputs?)\b.{0,40}\bpairs)\b",
    re.IGNORECASE,
)
CHECK_WORDS = re.compile(
    r"\b(check(s|ed|ing)?|verif(y|ies|ied|ying)|reproduc(e|es|ed|ing)|"
    r"match(es|ed|ing)?|compare[sd]?)\b",
    re.IGNORECASE,
)

# Upstream's shipped sentences that ask for observed input/output pairs, by
# exact wording, removed before the scan. Our switches add none.
UPSTREAM_EXAMPLE_CHECKS = {
    # The shipped rulebook's docstring rule (``UNIFY_*`` switches leave it as
    # is): record the trajectory's input/output pairs in a pure function's
    # docstring so a later change can be checked against them.
    "_STORAGE_WHAT_CAN_BE_STORED": (
        "Where the trajectory contains concrete inputs and the exact output "
        "a pure function reproduces, record those pairs in the docstring as "
        "well, so a later change can be checked against them."
    ),
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
        "UNIFY_TRY_FIRST": True,
        "UNIFY_CODE_FIRST": True,
        "UNIFY_FUNCTION_CASES": True,
    },
    "framing only": {"UNIFY_REVIEW_FRAMING": "unified"},
    "doctrine and note": {
        "UNIFY_CURATION_DOCTRINE": "compose",
        "UNIFY_REPLY_PROTOCOL_NOTE": True,
    },
    "try first and code first": {
        "UNIFY_TRY_FIRST": True,
        "UNIFY_CODE_FIRST": True,
    },
}
SWITCH_OFF = {
    "UNIFY_REVIEW_FRAMING": "",
    "UNIFY_CURATION_DOCTRINE": "",
    "UNIFY_REPLY_PROTOCOL_NOTE": False,
    "UNIFY_FUNCTION_PATCH": False,
    "UNIFY_STORE_CHECK": "",
    "UNIFY_TRY_FIRST": False,
    "UNIFY_CODE_FIRST": False,
    "UNIFY_FUNCTION_CASES": False,
}

PROMPT_MODES = {
    "act": {"can_store": True, "discovery_first_policy": True},
    "persist": {"can_store": True, "persist": True},
    "read only": {"can_store": True, "library_read_only": True},
    "no store": {},
    # UNIFY_PROMPT_CLOCK=message / UNIFY_CACHE_AFFINITY_SCOPE=static
    "static": {"can_store": True, "session_sections": False},
    # UNIFY_INLINE_CURATION
    "inline on": {"can_store": True, "inline_curation": "on"},
    "inline only": {"inline_curation": "only"},
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


def _sentences(text: str) -> list[str]:
    """``text`` in sentences, with line wrapping undone."""
    flat = " ".join(text.split())
    return [s for s in re.split(r"(?<=[.!?;:])\s+|\s+-\s+", flat) if s]


def _checks_against_examples(sentence: str) -> bool:
    return bool(
        EXAMPLE_CHECK_PHRASES.search(sentence)
        or (EXAMPLE_WORDS.search(sentence) and CHECK_WORDS.search(sentence)),
    )


def _example_checks(texts: dict[str, str]) -> list[str]:
    found = []
    for source, text in texts.items():
        text = " ".join(text.split())
        for wording in UPSTREAM_EXAMPLE_CHECKS.values():
            text = text.replace(wording, "")
        for sentence in _sentences(text):
            if _checks_against_examples(sentence):
                found.append(f"{source}: {sentence!r}")
    return found


def _findings(texts: dict[str, str]) -> list[str]:
    """Benchmark words, and instructions to check against given examples."""
    return _benchmark_words(texts) + _example_checks(texts)


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
    texts["_INLINE_ONLY_REASON"] = caa._INLINE_ONLY_REASON
    return texts


def _session_texts(tools: dict) -> dict[str, str]:
    """What the model reads outside the system prompt and the rulebook."""
    from unify.common._async_tool.repeat_guard import RepeatGuard
    from unify.function_manager import inline_curation, store_cases, store_verify

    guard = RepeatGuard()
    guard.surfaced("the same reply")
    guard.requester_said("That is not right.")
    case = store_cases.Case(
        case_id=1,
        function_id=1,
        kind="pass",
        status="active",
        source_hash="",
        call={"args": [], "kwargs": {"x": 1}},
        args_shown="x=1",
        result={"shown": "2"},
        error=None,
        trace=(),
        trace_complete=True,
        session=None,
        outcome=None,
        retired_why=None,
        recorded_at="",
    )
    replays = [
        store_cases.Replay(case, status, "detail")
        for status in (
            store_cases.DIVERGED,
            store_cases.INCONCLUSIVE,
            store_cases.NOW_PASSES,
            store_cases.STILL_FAILS,
            store_cases.PRESERVED,
        )
    ]
    try:
        inline_curation.check_names(["def tmp():\n    return 1\n"])
    except ValueError as exc:
        naming = str(exc)
    from unify.function_manager import instance_lint

    tokens = instance_lint.tokens_of(
        'Task id: case-7a4cf12e, titled "Weekly Sales Summary".',
    )
    lint_texts = {
        "instance refusal (name)": instance_lint.refusal(
            "mirror_7a4cf12e",
            instance_lint.name_problem("mirror_7a4cf12e", tokens),
        ),
        "instance refusal (task id)": instance_lint.refusal(
            "task_7a4cf12e_copy",
            instance_lint.name_problem("task_7a4cf12e_copy", tokens),
        ),
        "instance warning": instance_lint.text_warning(
            "its docstring",
            "Learned on case-7a4cf12e.",
            tokens,
        ),
    }
    texts = {
        "library snapshot": caa._library_snapshot_line(
            (3, 0),
            has_fm_tools=True,
            has_gm_tools=True,
            discovery_gate=True,
        ),
        "repeat guard": guard.check("the same reply"),
        "case refusal": store_cases.refusal("f", replays),
        "case report": store_cases.report("f", replays),
        "naming refusal": naming,
        "_TRY_FIRST_NOTE": pb._TRY_FIRST_NOTE,
        "_CODE_FIRST": pb._CODE_FIRST,
        "_INLINE_ONLY_BULLET": pb._INLINE_ONLY_BULLET,
        "store_verify.doctrine()": store_verify.doctrine(),
        "_inline_function_bullet()": pb._inline_function_bullet(tools),
        **lint_texts,
    }
    assert all(texts.values()), [k for k, v in texts.items() if not v]
    assert not any("because None" in v for v in texts.values())
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


def test_the_pattern_catches_instructions_to_check_against_examples():
    caught = [
        # What the switches said before 4 Oct.
        "write a program that produces it, check it against every example "
        "or known result you have, and answer with its output.",
        "When a check fails, fix the program's logic and re-run it (never "
        "hard-code the expected outputs).",
        "Run a stored function on inputs you already have, and check its "
        "output against the evidence you have (worked examples, expected "
        "formats, earlier feedback).",
        "Record an observed input and output in the docstring.",
        # Other phrasings.
        "Verify the function on the examples before storing it.",
        "Make sure the program reproduces each example.",
        "Compare the result with the given input/output pairs.",
        "Test it on the provided examples.",
    ]
    for text in caught:
        assert _example_checks({"t": text}), text
    for text in (
        "Confirm the outcome from real evidence (return values, a re-read).",
        "For example, store `compute_summary(week)`; check that it returns.",
        "Store code that ran and worked.",
        "Each write is refused unless its names resolve and it loads.",
    ):
        assert not _example_checks({"t": text}), text


def test_each_upstream_example_check_is_still_the_shipped_text():
    for source, wording in UPSTREAM_EXAMPLE_CHECKS.items():
        assert wording in " ".join(getattr(caa, source).split()), source
        assert _checks_against_examples(wording), source


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
    assert _findings({f"prompt ({switches}, {mode})": prompt}) == []


def test_the_actors_tool_schemas_name_no_benchmark(switches, actor_tools):
    from unify.common.llm_helpers import method_to_schema

    _actor, tools = actor_tools
    schemas = {
        name: json.dumps(method_to_schema(getattr(tool, "fn", tool), name))
        for name, tool in tools.items()
    }
    assert "execute_code" in schemas
    assert _findings(schemas) == []


def test_what_the_session_adds_to_its_messages_names_no_benchmark(actor_tools):
    _actor, tools = actor_tools
    tools = {**tools, "FunctionManager_retire_case": None}
    assert _findings(_session_texts(tools)) == []


def test_the_switched_texts_are_in_the_prompt_they_lint(monkeypatch, actor_tools):
    """The prompt scanned under "on" carries every switched section."""
    for name, value in {**SWITCH_OFF, **SWITCH_SETS["on"]}.items():
        monkeypatch.setattr(SETTINGS, name, value)
    actor, tools = actor_tools
    prompt = pb.build_code_act_prompt(
        environments=actor.environments,
        tools=tools,
        can_store=True,
        inline_curation="on",
    )
    for text in (pb._TRY_FIRST_NOTE, pb._CODE_FIRST, "**Functions, during the task**"):
        assert text in prompt


# ── the storage review ───────────────────────────────────────────────────


def test_the_rulebook_names_no_benchmark(switches):
    texts = _rulebook()
    assert {"_STORAGE_TWO_STORES", "_REVIEW_FORK_ROLE"} <= set(texts)
    assert _findings(texts) == []


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
    assert _findings({f"review ({switches})": sent}) == []
