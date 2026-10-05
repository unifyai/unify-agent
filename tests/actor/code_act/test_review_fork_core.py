"""Symbolic: ``UNIFY_REVIEW_FORK_CORE`` forks the storage review of a core-surface session.

Under ``UNIFY_TOOL_SURFACE=core`` the session's only JSON tool is
``execute_code``, so the forked review (``UNIFY_REVIEW_FORK``), which stores
through the library tools of the list it reuses, fell back to the standalone
librarian: a new conversation whose first call read 0% from the provider's
cache (12k-65k tokens, mean 26k on ARC LOW; 8.6% of the core arm's ARC LOW
USD, 16% on ScienceWorld). With the switch on the review is a fork again --
the session's last request, unchanged, plus one user message -- and stores
through the list's own ``execute_code``: its cells run in a sandbox of their
own, in the confined worker, holding only ``functions`` and ``guidance``.

The model is the scripted transport (tests/cache_discipline_helpers.py);
cells run in the real sandboxed worker, skipped where bubblewrap is missing.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor as _actor,
    tool_names as _tool_names,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import core_surface
from unify.common._async_tool import cache_discipline as cd
from unify.settings import ProductionSettings, SETTINGS

DOUBLE = "def double(x: int) -> int:\n    return x * 2\n"
_REVIEW_OPENING = "## Storage Review\n\n"


def _cell(code: str):
    return lambda: h.completion(
        calls=[("execute_code", {"thought": "Next step.", "code": code})],
    )


def _dumps(messages: list[dict]) -> list[str]:
    return [json.dumps(m, default=str) for m in messages]


def _is_review(request: dict) -> bool:
    return any(
        m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(_REVIEW_OPENING)
        for m in request["messages"]
    )


def _tool_texts(request: dict) -> list[str]:
    return [
        json.dumps(m["content"]) for m in request["messages"] if m.get("role") == "tool"
    ]


@pytest.fixture
def fork_core(core_world, monkeypatch):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK_CORE", True)
    return core_world


async def _session_and_review(session_replies, review_replies):
    """A core session through ``act`` and the review after it."""
    actor = _actor()
    closes: list = []
    real_close = core_surface.ReviewSandbox.close

    async def spy_close(self):
        closes.append(self._session is not None)
        return await real_close(self)

    session = list(session_replies)
    review = list(review_replies)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(core_surface.ReviewSandbox, "close", spy_close)
            with h.scripted([]) as provider:

                def _reply():
                    queue = review if _is_review(provider.requests[-1]) else session
                    return queue.pop(0)()

                provider.replies = [_reply] * 20
                handle = await actor.act("Double four.", persist=False)
                result = await asyncio.wait_for(handle.result(), 120)
                await asyncio.wait_for(handle._completion_event.wait(), 120)
        stored = actor.function_manager.list_functions()
    finally:
        await actor.close()
    return result, provider.requests, stored, closes


# ── the fork ─────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_the_review_forks_the_core_session_and_stores_through_python(fork_core):
    session = (
        _cell("task_value = 41\nprint(task_value + 1)"),
        lambda: h.completion(content="8"),
    )
    review = (
        _cell(
            f"print(await functions.add(implementations=[{DOUBLE!r}]))\n"
            "print(sorted(await functions.list()))",
        ),
        _cell("print(task_value)"),
        _cell("print(await functions.run('double', x=2))"),
        lambda: h.completion(content="Stored double."),
    )
    result, requests, stored, closes = await _session_and_review(session, review)
    assert result == "8"
    session_requests = [r for r in requests if not _is_review(r)]
    reviews = [r for r in requests if _is_review(r)]
    assert len(session_requests) == 2 and len(reviews) == 4

    # The session's last request, byte for byte, then its reply, then the
    # rulebook: the same execute_code-only tool list and tool choice.
    last, first_review = session_requests[-1], reviews[0]
    n = len(last["messages"])
    assert _dumps(first_review["messages"])[:n] == _dumps(last["messages"])
    assert len(first_review["messages"]) == n + 2
    reply = first_review["messages"][n]
    assert (reply["role"], reply["content"]) == ("assistant", "8")
    assert h.request_bytes(first_review)["tools"] == h.request_bytes(last)["tools"]
    assert _tool_names(first_review) == ["execute_code"]
    assert first_review["tool_choice"] == last["tool_choice"]
    for later in reviews[1:]:
        assert h.request_bytes(later)["tools"] == h.request_bytes(last)["tools"]
        sent = _dumps(first_review["messages"])
        assert _dumps(later["messages"])[: len(sent)] == sent

    # The rulebook names the libraries as the sandbox does.
    rulebook = first_review["messages"][-1]["content"]
    assert rulebook.startswith(_REVIEW_OPENING)
    assert caa._REVIEW_FORK_TOOLS_CORE in rulebook
    assert caa._REVIEW_FORK_TOOLS not in rulebook
    assert "FunctionManager_" not in rulebook and "GuidanceManager_" not in rulebook
    assert "functions.add" in rulebook
    assert rulebook.endswith("## Final Result\n\n8")

    # The review stored through python; its sandbox had none of the task's
    # variables; it ran no stored function; its sandbox was closed.
    assert "double" in stored
    outputs = _tool_texts({"messages": reviews[-1]["messages"][n + 2 :]})
    assert "double" in outputs[0]
    assert "NameError" in outputs[1] and "task_value" in outputs[1]
    assert core_surface.REVIEW_RUN_REFUSAL in outputs[2]
    assert closes == [True]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_review_that_runs_no_cell_starts_no_worker(fork_core):
    session = (lambda: h.completion(content="nothing to do"),)
    review = (lambda: h.completion(content="Nothing worth storing."),)
    _result, requests, stored, closes = await _session_and_review(session, review)
    assert [_is_review(r) for r in requests] == [False, True]
    assert closes == [False]
    assert "double" not in stored


@pytest.mark.asyncio
async def test_a_bash_cell_is_refused_in_the_review():
    box = core_surface.ReviewSandbox(object(), core_surface.review_policy())
    out = await box.execute_code("List files.", "ls", language="bash")
    assert "Python cells only" in out["error"]
    assert box._session is None


# ── what the review may do ─────────────────────────────────────────────────


def test_the_review_policy_writes_and_refuses_a_lessons_reviews_function_writes():
    from unify import outcome as outcome_mod

    full = core_surface.review_policy()
    assert full.review is True
    for method in ("functions.add", "functions.delete", "guidance.add"):
        assert full.refusal(method) is None, method
    lessons = core_surface.review_policy(
        lesson_refusals={
            name: outcome_mod.LESSON_MASK_RULE
            for name in outcome_mod.LESSON_REFUSED_TOOLS
        },
    )
    assert outcome_mod.LESSON_MASK_RULE in lessons.refusal("functions.add")
    assert outcome_mod.LESSON_MASK_RULE in lessons.refusal("functions.delete")
    assert lessons.refusal("guidance.add") is None
    # A session's policy is unchanged.
    assert core_surface.WritePolicy().review is False
    assert core_surface.WritePolicy().withheld == ()


@pytest.mark.parametrize("doctrine", ["", "compose", "minimal"])
@pytest.mark.parametrize("framing", ["", "unified"])
def test_the_rulebook_names_no_json_library_tool_after_translation(
    monkeypatch,
    doctrine,
    framing,
):
    from unify import outcome as outcome_mod

    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", doctrine)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FRAMING", framing)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    text = core_surface.python_names(
        caa._review_fork_role(core=True)
        + caa._storage_doctrine_sections()
        + caa._storage_base_instructions()
        + outcome_mod.render(None, lessons=True),
    )
    assert caa._REVIEW_FORK_TOOLS_CORE in text
    for name in ("FunctionManager_", "GuidanceManager_", "install_python_packages"):
        assert name not in text, name


# ── fallbacks and the switch ───────────────────────────────────────────────


def _recorded_session(tool: str = "execute_code"):
    client = h.new_client("You are a scripted actor.")
    client._messages.append({"role": "user", "content": "task"})
    cd.record_sent_request(
        client,
        list(client.messages),
        {
            "tools": [{"type": "function", "function": {"name": tool}}],
            "tool_choice": "auto",
        },
    )
    client._messages.append({"role": "assistant", "content": "done"})
    inner = SimpleNamespace(_client=client, _compression=SimpleNamespace(count=0))
    return inner, SimpleNamespace(_preprocess_msgs=None)


@pytest.fixture
def core_switches(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "core")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", "sandboxed")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK_CORE", True)


def test_the_fork_source_is_marked_core(core_switches):
    source, why = caa._review_fork_source(*_recorded_session())
    assert why is None and source["core"] is True
    assert cd.schema_names(source["tools"]) == ["execute_code"]


@pytest.mark.parametrize(
    "case, reason",
    [
        ("switch off", "no library tools"),
        ("no worker", "needs UNIFY_WORKSPACE=sandboxed and UNIFY_WORKSPACE_PYTHON"),
        ("store verify", "UNIFY_STORE_VERIFY"),
        ("no execute_code", "no execute_code"),
        ("no discipline", "needs UNIFY_CACHE_DISCIPLINE"),
    ],
)
def test_the_core_review_falls_back_and_says_why(
    monkeypatch,
    core_switches,
    case,
    reason,
):
    if case == "switch off":
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK_CORE", False)
    if case == "no worker":
        monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    if case == "store verify":
        from unify.function_manager import store_verify

        monkeypatch.setattr(store_verify, "enabled", lambda: True)
    if case == "no discipline":
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", False)
    tool = "final_response" if case == "no execute_code" else "execute_code"
    source, why = caa._review_fork_source(*_recorded_session(tool))
    assert source is None and reason in why


def test_without_the_core_surface_the_switch_changes_nothing(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK_CORE", True)
    source, why = caa._review_fork_source(*_recorded_session("FunctionManager_x"))
    assert why is None and "core" not in source


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False)])
def test_the_setting_parses_booleans(value, expected):
    settings = ProductionSettings(UNIFY_REVIEW_FORK_CORE=value)
    assert settings.UNIFY_REVIEW_FORK_CORE is expected


def test_the_default_is_off():
    assert ProductionSettings.model_fields["UNIFY_REVIEW_FORK_CORE"].default is False
