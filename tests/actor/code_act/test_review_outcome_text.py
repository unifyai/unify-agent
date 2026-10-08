"""Symbolic: the storage review sees only the checker's verdict, and its writes
never carry the outcome section it read.

A checker's reasons and summary can name the expected answer, and the
review's writes (the function and guidance libraries, whose store every later
cell reads) outlive the session. So the review's prompt carries only the
verdict -- solved or not, the score and how many checks passed -- and any
library write that holds the exact outcome section the harness rendered, raw
or JSON-escaped, is refused. The review's own words about whether the task
passed are its lessons and are stored as usual.

The model is the scripted transport (tests/cache_discipline_helpers.py);
the core review's cells run in the real sandboxed worker, skipped where
bubblewrap is missing.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_outcome_channel import FAILED, _persistent_review
from tests.helpers import _handle_project
from unify import outcome as outcome_mod
from unify.actor import code_act_actor as caa
from unify.actor import core_surface
from unify.settings import SETTINGS

# Fresh per process, so no earlier run's text can match it.
SUMMARY = f"the expected answer was {uuid.uuid4().hex[:12]}"
REASON = "no email to Kim"
OUTCOME = {**FAILED, "summary": SUMMARY}
LESSON = (
    "The test failed on the email step: the task was not solved because no "
    "email reached Kim. Check the recipient before sending."
)


def _section(outcome=OUTCOME) -> str:
    return outcome_mod.render(outcome_mod.normalize(outcome))


def _function_with(text: str) -> str:
    """A stored function whose docstring is *text* and which returns it."""
    return (
        "def checker_note() -> str:\n"
        f"    {json.dumps(text)}\n"
        f"    return {json.dumps(text)}\n"
    )


@pytest.fixture
def no_admission(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", "")


# ── (b) the prompt: the verdict, never the checker's reasons ────────────


def test_the_section_carries_the_verdict_only():
    text = _section()
    assert text.startswith(outcome_mod.OUTCOME_HEADER + "\n")
    assert "- Solved: no" in text
    assert "- Score: 0.5" in text
    assert "- Checks: 1 of 2 passed" in text
    assert "source: `grader`" in text
    assert "not from the agent" in text
    # neither the reasons, nor the summary, nor the checks' own names
    for detail in (REASON, SUMMARY, "Checker's summary", "email_sent"):
        assert detail not in text, detail
    assert outcome_mod.render(None) == ""


def test_the_structured_verdict_is_kept_whole_on_the_handle():
    # The normalized outcome the session holds still has every field; only
    # the rendering leaves them out.
    held = outcome_mod.normalize(OUTCOME)
    assert held["solved"] is False and held["checks_passed"] == 1
    assert held["checks"][0]["reason"] == REASON
    assert held["summary"] == SUMMARY


@pytest.mark.asyncio
@pytest.mark.parametrize("review", ["standalone", "fork"])
async def test_the_reviews_prompt_has_the_verdict_and_not_the_reasons(
    no_admission,
    monkeypatch,
    review,
):
    if review == "standalone":
        monkeypatch.setattr(
            caa,
            "_review_fork_source",
            lambda inner, actor: (None, "the test refuses the fork"),
        )
    _note, requests, handle, _ = await _persistent_review(outcome=OUTCOME)
    assert handle._outcome["summary"] == SUMMARY
    sent = json.dumps(requests[3], default=str)
    assert outcome_mod.OUTCOME_HEADER in sent
    assert "- Solved: no" in sent and "- Checks: 1 of 2 passed" in sent
    for detail in (REASON, SUMMARY, "email_sent"):
        assert detail not in sent, detail


@pytest.mark.asyncio
async def test_the_gate_reads_the_verdict_only(monkeypatch):
    from unify.actor import review_gate

    seen: list[str] = []

    async def decide(**kwargs):
        seen.append(kwargs["outcome_note"])
        return review_gate.GateDecision(review=False, decided=True, reason="no")

    monkeypatch.setattr(review_gate, "decide", decide)
    monkeypatch.setattr(review_gate, "library_is_empty", lambda counts: False)
    await _persistent_review(outcome=OUTCOME)
    assert seen and "- Solved: no" in seen[0]
    assert REASON not in seen[0] and SUMMARY not in seen[0]


# ── (c) the writes: the rendered section is refused, lessons are not ────


@pytest.fixture
def managers(unify_home):
    from unify import db
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    db.reset_store()
    yield FunctionManager(include_primitives=False), GuidanceManager()
    db.reset_store()


def _forms(section: str) -> dict[str, str]:
    once = json.dumps(section)[1:-1]
    return {"raw": section, "json": once, "json twice": json.dumps(once)[1:-1]}


@pytest.mark.parametrize("form", ["raw", "json", "json twice"])
def test_a_function_carrying_the_section_is_refused(managers, form):
    fm, _gm = managers
    text = _forms(_section())[form]
    # The section as it stands in the source: in its docstring, verbatim.
    source = f'def checker_note() -> int:\n    """Note.\n{text}"""\n    return 1\n'
    with pytest.raises(outcome_mod.OutcomeCarried, match="outcome section"):
        fm.add_functions(implementations=[source])
    assert "checker_note" not in fm.list_functions()


def test_a_patch_that_writes_the_section_in_is_refused(managers):
    fm, _gm = managers
    fm.add_functions(implementations=[_function_with("Note.")])
    section = _section()
    out = fm.patch_function(
        name="checker_note",
        old='return "Note."',
        new=f"return {json.dumps(section)}",
        why="keep the verdict",
    )
    assert "outcome section" in out["error"]
    out = fm.patch_function(
        name="checker_note",
        old='return "Note."',
        new='return "Unsolved."',
        why=section,
    )
    assert "outcome section" in out["error"]
    assert '"Unsolved."' not in fm.list_functions(include_implementations=True)[
        "checker_note"
    ].get("implementation", "")


@pytest.mark.parametrize("write", ["add", "update", "patch"])
def test_guidance_carrying_the_section_is_refused(managers, write):
    _fm, gm = managers
    section = _section()
    gid = gm.add_guidance(title="Email", content="Send it.")["details"]["guidance_id"]
    with pytest.raises(outcome_mod.OutcomeCarried, match="outcome section"):
        if write == "add":
            gm.add_guidance(title="Verdict", content=f"Learned:\n{section}")
        elif write == "update":
            gm.update_guidance(guidance_id=gid, content=section)
        else:
            gm.patch_guidance(id_or_title=gid, old="Send it.", new=section, why="x")
    rows = gm.filter()
    assert all(outcome_mod.OUTCOME_HEADER not in str(r) for r in rows)


def test_ordinary_lessons_and_the_models_own_words_are_stored(managers):
    fm, gm = managers
    section = _section()
    # The model's own account of the verdict, a line or two of the section,
    # and the header on its own are not the section.
    own = (
        f"{LESSON}\nSolved: no. Score 0.5; 1 of 2 checks passed.\n"
        f"{outcome_mod.OUTCOME_HEADER}\n- Solved: no\n"
    )
    assert section not in own
    out = gm.add_guidance(title="Email lessons", content=own)
    assert out["outcome"] == "guidance created successfully"
    assert fm.add_functions(implementations=[_function_with(LESSON)]) == {
        "checker_note": "added",
    }


def test_a_section_this_process_never_rendered_is_not_a_key(managers):
    fm, _gm = managers
    # A verdict for an outcome no session received: rendered text is the key,
    # and this text was never rendered.
    foreign = _section().replace("- Score: 0.5", "- Score: 0.25")
    assert fm.add_functions(implementations=[_function_with(foreign)]) == {
        "checker_note": "added",
    }


# ── (c) end to end: a scripted review that copies the section ───────────


def _review_copies_the_section():
    section = _section()
    return (
        lambda: h.completion(
            calls=[
                (
                    "FunctionManager_add_functions",
                    {"implementations": [_function_with(section)]},
                ),
            ],
        ),
        lambda: h.completion(
            calls=[
                (
                    "GuidanceManager_add_guidance",
                    {"title": "Verdict", "content": section},
                ),
            ],
        ),
        lambda: h.completion(
            calls=[
                (
                    "GuidanceManager_add_guidance",
                    {"title": "Email lessons", "content": LESSON},
                ),
            ],
        ),
        lambda: h.completion(content="Stored the lesson."),
    )


def _is_review(request: dict) -> bool:
    return any(
        m.get("role") == "user"
        and str(m.get("content", "")).startswith("## Curating The Library")
        for m in request["messages"]
    )


def _tool_results(request: dict) -> list[str]:
    return [
        str(m.get("content")) for m in request["messages"] if m.get("role") == "tool"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("review", ["standalone", "fork"])
async def test_a_review_that_copies_the_section_is_refused(
    no_admission,
    monkeypatch,
    review,
):
    if review == "standalone":
        monkeypatch.setattr(
            caa,
            "_review_fork_source",
            lambda inner, actor: (None, "the test refuses the fork"),
        )
    from unify.actor.code_act_actor import CodeActActor

    stored: dict = {}
    real_close = CodeActActor.close

    async def close(self):
        stored["functions"] = self.function_manager.list_functions()
        stored["guidance"] = [g.title for g in self.guidance_manager.filter()]
        return await real_close(self)

    monkeypatch.setattr(CodeActActor, "close", close)
    note, requests, _handle, _ = await _persistent_review(
        _review_copies_the_section(),
        outcome=OUTCOME,
    )
    assert note["message"] == "Stored the lesson."
    results = _tool_results(requests[-1])[-3:]
    assert "outcome section" in results[0] and "outcome section" in results[1]
    assert "guidance created successfully" in results[2]
    assert "checker_note" not in stored["functions"]
    assert "Email lessons" in stored["guidance"]
    assert "Verdict" not in stored["guidance"]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_core_reviews_cell_that_copies_the_section_is_refused(
    core_world,  # noqa: F811
):
    section = core_surface.python_names(_section())
    code = (
        "for write in (\n"
        f"    functions.add(implementations=[{_function_with(section)!r}]),\n"
        f"    guidance.add(title='Verdict', content={section!r}),\n"
        f"    guidance.add(title='Email lessons', content={LESSON!r}),\n"
        "):\n"
        "    try:\n"
        "        print('ok', await write)\n"
        "    except Exception as exc:\n"
        "        print('refused', type(exc).__name__, exc)\n"
    )

    def _cell(source: str):
        return lambda: h.completion(
            calls=[("execute_code", {"thought": "Next.", "code": source})],
        )

    session = [_cell("print(4 * 2)"), lambda: h.completion(content="8")]
    review = [_cell(code), lambda: h.completion(content="Stored the lesson.")]
    actor = new_actor()
    try:
        with h.scripted([]) as provider:

            def _reply():
                queue = review if _is_review(provider.requests[-1]) else session
                return queue.pop(0)()

            provider.replies = [_reply] * 10
            handle = await actor.act("Double four.", persist=False)
            outcome_mod.post(handle.outcome_session_id, OUTCOME)
            assert await asyncio.wait_for(handle.result(), 120) == "8"
            await asyncio.wait_for(handle._completion_event.wait(), 120)
        functions = actor.function_manager.list_functions()
        guidance = [g.title for g in actor.guidance_manager.filter()]
    finally:
        await actor.close()
    reviews = [r for r in provider.requests if _is_review(r)]
    rulebook = reviews[0]["messages"][-1]["content"]
    assert "- Solved: no" in rulebook and REASON not in rulebook
    out = _tool_results(reviews[-1])[-1]
    assert out.count("refused OutcomeCarried") == 2, out
    assert "guidance created successfully" in out
    assert "checker_note" not in functions
    assert guidance == ["Email lessons"]
