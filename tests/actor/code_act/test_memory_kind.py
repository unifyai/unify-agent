"""Symbolic: ``UNIFY_MEMORY_KIND``, keep what a session learned in one form only.

``functions``: the storage review is offered no guidance write tool.
``notes``: it is offered no function write tool, with no failure framing.
``examples``: no review runs; the harness keeps the session as one verbatim
"Worked example" guidance entry. A scripted persistent session runs the real
review path; no model or network is called.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.actor.code_act import test_outcome_channel as toc
from tests.actor.code_act.test_outcome_channel import switches  # noqa: F401 (fixture)
from unify import outcome as outcome_mod
from unify.actor import code_act_actor as caa
from unify.actor import memory_kind
from unify.settings import ProductionSettings, SETTINGS


def _review_request(requests: list) -> dict:
    """The storage review's request: the one offered the library's write tools."""
    return next(
        r
        for r in requests
        if "GuidanceManager_add_guidance"
        in {t["function"]["name"] for t in r.get("tools") or []}
    )


def _review_tools(monkeypatch, kind: str):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_KIND", kind)
    captured: dict = {}
    real = caa.start_async_tool_loop

    def spy(*args, **kwargs):
        if kwargs.get("loop_id") == "StorageCheck(CodeActActor.act)":
            captured["tools"] = sorted(kwargs["tools"])
        return real(*args, **kwargs)

    return captured, spy


def test_the_switch_is_off_by_default_and_parses():
    assert ProductionSettings.model_fields["UNIFY_MEMORY_KIND"].default == ""
    assert ProductionSettings(UNIFY_MEMORY_KIND=" Notes ").UNIFY_MEMORY_KIND == "notes"
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_MEMORY_KIND="skills")


@pytest.mark.asyncio
async def test_functions_offers_the_review_no_guidance_writes(monkeypatch, switches):
    captured, spy = _review_tools(monkeypatch, "functions")
    with patch.object(caa, "start_async_tool_loop", spy):
        await toc._persistent_review()
    assert "FunctionManager_add_functions" in captured["tools"]
    for name in memory_kind.GUIDANCE_WRITE_TOOLS:
        assert name not in captured["tools"]
    assert "GuidanceManager_search" in captured["tools"]


@pytest.mark.asyncio
async def test_notes_offers_the_review_no_function_writes_and_no_failure_framing(
    monkeypatch,
    switches,
):
    captured, spy = _review_tools(monkeypatch, "notes")
    with patch.object(caa, "start_async_tool_loop", spy):
        _note, requests, _handle, _ = await toc._persistent_review()
    for name in outcome_mod.LESSON_REFUSED_TOOLS:
        assert name not in captured["tools"]
    assert "GuidanceManager_add_guidance" in captured["tools"]
    review = _review_request(requests)
    assert outcome_mod.LESSONS_HEADER not in toc._review_text(review)
    assert memory_kind.RULES["notes"] not in toc._review_text(review)


@pytest.mark.asyncio
async def test_examples_runs_no_review_and_keeps_one_worked_example(
    monkeypatch,
    switches,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_KIND", "examples")
    started: list = []
    kept: list = []
    real_loop = caa.start_async_tool_loop
    real_store = memory_kind.store_example

    def spy(*args, **kwargs):
        started.append(kwargs.get("loop_id"))
        return real_loop(*args, **kwargs)

    def store(*args, **kwargs):
        out = real_store(*args, **kwargs)
        kept.append(out)
        return out

    with (
        patch.object(caa, "start_async_tool_loop", spy),
        patch.object(
            memory_kind,
            "store_example",
            store,
        ),
    ):
        note, _requests, _handle, _ = await toc._persistent_review()
    assert "StorageCheck(CodeActActor.act)" not in started
    assert note["type"] == "storage_review_skipped"
    assert "UNIFY_MEMORY_KIND=examples" in note["message"]
    (stored,) = kept
    assert stored is not None and stored["title"].startswith(memory_kind.TITLE)


def test_an_example_is_kept_verbatim_with_its_outcome():
    title, content = memory_kind.example_text(
        request="Total the meals spend for Q3.",
        code="print(sum(rows))",
        answer="4323.59",
        outcome={"solved": True},
    )
    assert title == "Worked example: Total the meals spend for Q3."
    assert "```python\nprint(sum(rows))\n```" in content
    assert "Answer given:\n4323.59" in content
    assert "checker accepted" in content
    _, unknown = memory_kind.example_text(
        request="Tidy the folder.",
        code=None,
        answer="Done.",
        outcome=None,
    )
    assert "No code cell's output repeats the answer." in unknown
    assert "Outcome: unknown" in unknown


def test_the_answer_is_the_last_reply_never_a_stop_notice(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_KIND", "examples")
    added: list = []

    class FakeGuidance:
        def add_guidance(self, *, title, content):
            added.append((title, content))

    class FakeActor:
        guidance_manager = FakeGuidance()

    trajectory = [
        {"role": "user", "content": "Tidy the downloads folder."},
        {"role": "assistant", "content": "I moved 12 files into Archive/."},
    ]
    out = memory_kind.store_example(
        FakeActor(),
        request="Tidy the downloads folder.",
        trajectory=trajectory,
    )
    assert out == {
        "title": "Worked example: Tidy the downloads folder.",
        "has_code": False,
    }
    assert "Answer given:\nI moved 12 files into Archive/." in added[0][1]


def test_a_kind_refuses_the_actors_own_mid_task_writes(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_KIND", "notes")
    monkeypatch.setattr(SETTINGS, "UNIFY_INLINE_CURATION", "on")
    with pytest.raises(RuntimeError, match="UNIFY_INLINE_CURATION"):
        memory_kind.require_prerequisites()
    monkeypatch.setattr(SETTINGS, "UNIFY_INLINE_CURATION", "")
    memory_kind.require_prerequisites()
