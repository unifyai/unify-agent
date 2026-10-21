"""Symbolic: the actor's prompt and the storage rulebook name no benchmark.

The harness is general-purpose; benchmarks test whether its behaviour
transfers, so their names and actions must not reach what the model reads.
Text has been written from benchmark evidence before: the reply-protocol
note (``UNIFY_REPLY_PROTOCOL_NOTE``) came from ARC episodes whose actor
delegated ``request_demos`` to sub-actors that called
``request_demonstration``, and it checks only its own words. This lint
renders the actor's system prompt and tool schemas, and the storage review's
rulebook constants, notes and sent requests, under the defaults and with
the switches still in progress on, and fails on any benchmark or dataset name, or a
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

from unify.actor.core_surface import PromptSurface
import inspect
import json
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import prompt_builders as pb
from unify.actor import library_shortlist
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

# The switches still in progress (WIP_SWITCHES.md) that change what the
# model reads; every other prompt-affecting switch was baked at the freeze.
SWITCH_SETS = {
    "default": {},
    "wip on": {
        "UNIFY_REPLY_CHANNEL": "code+text",
        "UNIFY_BIND_REQUEST": "on",
        "UNIFY_VARIABLE_INVENTORY": "on",
    },
}
SWITCH_OFF = {
    "UNIFY_REPLY_CHANNEL": "",
    "UNIFY_BIND_REQUEST": "",
    "UNIFY_VARIABLE_INVENTORY": "",
}

PROMPT_MODES = {
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
    texts["_storage_base_instructions()"] = caa._storage_base_instructions()
    texts["_review_fork_role()"] = caa._review_fork_role()
    return texts


def _cell_messages(code: str, output: str) -> list[dict]:
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "execute_code", "arguments": json.dumps({"code": code})},
    }
    return [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": output},
    ]


def _session_texts(tools: dict) -> dict[str, str]:
    """What the model reads outside the system prompt and the rulebook."""
    from unify.common._async_tool import loop_stop
    from unify.function_manager import store_cases, store_verify

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
    # UNIFY_ESCAPE_DRIFT_CHECK: the refusal, with one and with many literals.
    from unify.function_manager import escape_drift

    hits = escape_drift.drifted(
        'def f(xs):\n    return "".join(x + "\\\\n" for x in xs) + r"\\\\bE\\\\b"\n',
        escape_drift.cell_literals(["s = '\\n'\nm = r'\\bE\\b'\n"]),
    )
    assert len(hits) == 2, hits
    lint_texts["escape drift refusal"] = escape_drift.refusal("f", hits[:1])
    lint_texts["escape drift refusal (many)"] = escape_drift.refusal("f", hits * 4)
    texts = {
        "library snapshot": caa._library_snapshot_line(
            (3, 0),
            has_fm_tools=True,
            has_gm_tools=True,
        ),
        # UNIFY_LOOP_STOP: the notice before the last word, and the reply.
        "loop stop notice": loop_stop.Stop(k=10, last_word=True).notice,
        "loop stop reply": loop_stop.Stop(k=10, last_word=True).headline,
        "loop stop cancel": loop_stop.Stop(k=10, last_word=True).cancelled,
        "case refusal": store_cases.refusal("f", replays),
        "case report": store_cases.report("f", replays),
        # The library shortlist
        "library shortlist header": library_shortlist._HEADER,
        "store_verify.doctrine()": store_verify.doctrine(),
        # UNIFY_VARIABLE_INVENTORY: the harness's words around the model's
        # own names (every kind of description, and the overflow).
        "variable inventory": _inventory_line(),
        **lint_texts,
    }
    assert all(texts.values()), [k for k, v in texts.items() if not v]
    assert not any("because None" in v for v in texts.values())
    return texts


def _inventory_line() -> str:
    from unify.actor.execution.worker_child import (
        describe_value,
        render_inventory,
    )

    ns = {"__name__": "__sandbox_lint__"}
    exec("def f(a, *b, **c):\n    pass\nclass K:\n    pass\n", ns)
    values = [
        3,
        2.5,
        True,
        None,
        "short",
        "x" * 100,
        b"xy",
        [1, 2],
        [[1, 2], [3, 4]],
        (1,),
        {"a": 1},
        {1},
        frozenset(),
        10**100,
        ns["f"],
        ns["K"],
        ns["K"](),
    ]
    entries = [(f"v{i}", describe_value(v)) for i, v in enumerate(values)]
    assert all(text for _, text in entries)
    # Every description, and the line with its overflow.
    return "; ".join(text for _, text in entries) + "\n" + render_inventory(entries)


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
        **PROMPT_MODES[mode],
        core=PromptSurface(),
    )
    assert "### Execution" in prompt
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
    """The prompt scanned under "wip on" carries every switched section."""
    for name, value in {**SWITCH_OFF, **SWITCH_SETS["wip on"]}.items():
        monkeypatch.setattr(SETTINGS, name, value)
    actor, tools = actor_tools
    prompt = pb.build_code_act_prompt(
        environments=actor.environments,
        can_store=True,
        core=PromptSurface(),
    )
    assert pb._BIND_REQUEST_LINE in prompt


# ── the storage review ───────────────────────────────────────────────────


def test_the_review_gate_names_no_benchmark():
    from unify.actor import review_gate

    texts = {
        "GATE_SYSTEM_PROMPT": review_gate.GATE_SYSTEM_PROMPT,
        "gate user message": review_gate.build_user_message(
            trajectory=[{"role": "user", "content": "the request"}],
            final_result="the reply",
        ),
    }
    assert _findings(texts) == []


def test_the_rulebook_names_no_benchmark(switches):
    texts = _rulebook()
    assert {"_STORAGE_TWO_STORES", "_REVIEW_FORK_ROLE_UNIFIED"} <= set(texts)
    assert _findings(texts) == []


@pytest.mark.asyncio
async def test_the_sent_review_names_no_benchmark(switches):
    _summary, _, requests = await h.scenario_review()
    review = requests[2]["messages"]
    # The rulebook arrives as the fork's last message.
    sent = review[-1]["content"]
    assert "## Instructions" in sent
    assert _findings({f"review ({switches})": sent}) == []
