"""Symbolic: ``UNIFY_INLINE_CURATION``: the actor stores working units and repairs failed ones during the task.

As shipped, the actor writes a function only when the user asks; what it
works out itself is left to the review after the task. Hermes ("When you work
out a non-trivial workflow, record it with skill_manage"), OpenClaw (a wrong
skill is repaired in the same turn) and prime-agent let the same agent write
and repair skills during the task as well as after it. Unchecked in-task
writes are risky: on ARC, when Unify's actor could write directly, it stored
functions named ``_unused`` and ``nope``. So with the switch on the prompt
says to store only units that ran and were checked, and every function the
actor itself adds passes a naming rule and the storage check of
``UNIFY_STORE_CHECK=resolve`` (whether or not that is set); with
``UNIFY_FUNCTION_CASES`` an inline patch is replayed like any other. ``on``
keeps the post-task review; ``only`` skips it, since
``UNIFY_STORE_ADMISSION=never`` would withhold the inline writes too. With
any admission value the switch is ignored, with a log line. No model is
called: the tool loop is captured before it starts and the review is mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import unify.actor.code_act_actor as code_act_actor
from tests.helpers import _handle_project
from unify.actor import prompt_builders as pb
from unify.actor.code_act_actor import CodeActActor, _StorageCheckHandle
from unify.common.llm_helpers import method_to_schema
from unify.common.tool_spec import ToolSpec
from unify.function_manager import inline_curation
from unify.function_manager.function_manager import FunctionManager
from unify.settings import ProductionSettings, SETTINGS

SHIPPED_FUNCTIONS_BULLET = "explicit user requests to add/update/delete functions"
INLINE_BULLET = "**Functions, during the task**"


def _fn(tool):
    return tool.fn if isinstance(tool, ToolSpec) else tool


# --------------------------------------------------------------------------- #
#  The switch                                                                  #
# --------------------------------------------------------------------------- #


def test_the_switch_is_off_by_default_and_parses_its_values():
    assert ProductionSettings.model_fields["UNIFY_INLINE_CURATION"].default == ""
    for raw, value in [
        ("", ""),
        ("on", "on"),
        (" ONLY ", "only"),
        ("true", "on"),
        ("0", ""),
    ]:
        assert (
            ProductionSettings(UNIFY_INLINE_CURATION=raw).UNIFY_INLINE_CURATION == value
        )
    with pytest.raises(ValueError, match="UNIFY_INLINE_CURATION"):
        ProductionSettings(UNIFY_INLINE_CURATION="sometimes")


@pytest.mark.parametrize(
    "name",
    [
        "nope",
        "_unused",
        "tmp",
        "test",
        "foo",
        "temp",
        "_helper_fn",
        "foo_bar",
        "do_it",
        "tmp_test",
        "parseDates",
        "Parse_dates",
        "parse__dates",
        "solve",
    ],
)
def test_names_that_do_not_describe_behaviour_are_named_as_such(name):
    assert inline_curation.name_problem(name)


@pytest.mark.parametrize(
    "name",
    [
        "parse_invoice_dates",
        "count_rows_by_status",
        "test_connection",
        "get_user_id",
        "parse_iso8601_dates",
    ],
)
def test_descriptive_names_pass(name):
    assert inline_curation.name_problem(name) is None


# --------------------------------------------------------------------------- #
#  The prompt                                                                  #
# --------------------------------------------------------------------------- #

_TOOLS = {
    "execute_code": None,
    "FunctionManager_search_functions": None,
    "FunctionManager_add_functions": None,
    "GuidanceManager_add_guidance": None,
}


def _prompt(tools=_TOOLS, **kwargs) -> str:
    return pb.build_code_act_prompt(
        environments={},
        tools=tools,
        can_store=True,
        **kwargs,
    )


def test_off_the_prompt_is_as_shipped():
    prompt = _prompt()
    assert prompt == _prompt(inline_curation="")
    assert SHIPPED_FUNCTIONS_BULLET in prompt
    assert INLINE_BULLET not in prompt


def test_on_the_function_bullet_invites_inline_writes_of_working_units():
    prompt = _prompt(inline_curation="on")
    assert SHIPPED_FUNCTIONS_BULLET not in prompt
    assert INLINE_BULLET in prompt
    section = prompt.split("#### Writing to the libraries")[1].split("####")[0]
    for phrase in (
        "once a reusable unit ran and worked",
        "`FunctionManager_add_functions`",
        "keeping its behaviour on the",
        "new function with a new name",
        "a name that says what it does",
        "names resolve and it loads",
        "**Guidance, during the task**",
        "`GuidanceManager_update_guidance`",
    ):
        assert phrase in " ".join(section.split()), phrase
    # The review still follows the task.
    assert "a dedicated review extracts functions" in section
    # Static text: the cached prefix is the same for every session.
    assert prompt == _prompt(inline_curation="on")


def test_the_bullet_names_the_tools_the_session_has():
    tools = {
        **_TOOLS,
        "FunctionManager_patch_function": None,
        "GuidanceManager_patch_guidance": None,
        "FunctionManager_retire_case": None,
    }
    section = " ".join(_prompt(tools, inline_curation="on").split())
    assert "fix it with `FunctionManager_patch_function`" in section
    assert "`GuidanceManager_patch_guidance`" in section
    assert "does what the function did on its recorded calls" in section
    plain = " ".join(_prompt(inline_curation="on").split())
    assert "`FunctionManager_add_functions` with `overwrite=True`" in plain
    assert "recorded calls" not in plain


@pytest.mark.parametrize("framing", ["", "unified"])
def test_only_says_no_review_follows(monkeypatch, framing):
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FRAMING", framing)
    prompt = _prompt(inline_curation="only")
    assert INLINE_BULLET in prompt
    assert "Nothing reviews this trajectory for the libraries" in prompt
    section = prompt.split("#### Writing to the libraries")[1].split("####")[0]
    assert "`store_skills`" not in section


def test_the_inline_text_names_no_benchmark():
    import re

    text = pb._inline_function_bullet(_TOOLS) + pb._INLINE_ONLY_BULLET
    words = set(re.findall(r"[a-z]+", text.lower()))
    for word in ("arc", "grid", "appworld", "alfworld", "scienceworld", "spotify"):
        assert word not in words


# --------------------------------------------------------------------------- #
#  act(): tools, prompt and reviews                                            #
# --------------------------------------------------------------------------- #


async def _start_act(monkeypatch, *, persist: bool = False) -> dict:
    """Run ``act()`` up to its tool loop and capture what the loop was given."""
    captured: dict = {}

    def fake_loop(client, message, tools, **kwargs):
        captured["tools"] = dict(tools)
        captured["policy"] = kwargs.get("tool_policy")
        captured["extra_compression_tools"] = kwargs.get("extra_compression_tools")
        handle = MagicMock()
        handle.result = AsyncMock(return_value="done")
        handle.next_notification = AsyncMock(
            side_effect=lambda: asyncio.Event().wait(),
        )
        handle._client = MagicMock(messages=[])
        return handle

    real_builder = pb.build_code_act_prompt

    def spy_builder(**kwargs):
        captured["prompt_kwargs"] = dict(kwargs)
        captured["prompt"] = real_builder(**kwargs)
        return captured["prompt"]

    monkeypatch.setattr(code_act_actor, "start_async_tool_loop", fake_loop)
    monkeypatch.setattr(code_act_actor, "build_code_act_prompt", spy_builder)
    monkeypatch.setattr(code_act_actor, "_start_storage_check_loop", lambda **kw: None)
    monkeypatch.setattr(code_act_actor, "publish_manager_method_event", AsyncMock())
    actor = CodeActActor(timeout=30)
    captured["actor"] = actor
    try:
        handle = await actor.act("Do something", persist=persist, can_store=True)
        captured["handle"] = handle
        searched = ["FunctionManager_search_functions", "GuidanceManager_search"]
        decision = captured["policy"](5, dict(captured["tools"]), searched)
        captured["visible_later"] = set(decision[1])
    finally:
        try:
            await actor.close()
        except Exception:
            pass
    return captured


_WRITES = {
    "FunctionManager_add_functions",
    "FunctionManager_delete_function",
    "GuidanceManager_add_guidance",
    "GuidanceManager_update_guidance",
}


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_on_the_session_has_guarded_write_tools(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_INLINE_CURATION", "on")
    captured = await _start_act(monkeypatch)
    assert _WRITES | {"store_skills"} <= captured["visible_later"]
    assert captured["prompt_kwargs"]["inline_curation"] == "on"
    assert captured["prompt_kwargs"]["can_store"] is True
    assert INLINE_BULLET in captured["prompt"]
    assert captured["extra_compression_tools"] == ["store_skills"]
    assert captured["handle"]._skip_review is None
    add = _fn(captured["tools"]["FunctionManager_add_functions"])
    fm = captured["actor"].function_manager
    assert add != fm.add_functions and add.__wrapped__ == fm.add_functions
    # The schema the model sees is the shipped one.
    assert method_to_schema(add, "FunctionManager_add_functions") == method_to_schema(
        fm.add_functions,
        "FunctionManager_add_functions",
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_off_the_session_is_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_INLINE_CURATION", "")
    captured = await _start_act(monkeypatch)
    assert "inline_curation" not in captured["prompt_kwargs"]
    assert SHIPPED_FUNCTIONS_BULLET in captured["prompt"]
    add = _fn(captured["tools"]["FunctionManager_add_functions"])
    assert add == captured["actor"].function_manager.add_functions


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_with_admission_set_inline_curation_is_off_and_logged(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_INLINE_CURATION", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", str(tmp_path / "v.json"))
    lines: list[str] = []
    real_info = code_act_actor.logger.info
    monkeypatch.setattr(
        code_act_actor.logger,
        "info",
        lambda msg, *a, **kw: lines.append(str(msg)) or real_info(msg, *a, **kw),
    )
    captured = await _start_act(monkeypatch)
    assert "inline_curation" not in captured["prompt_kwargs"]
    assert INLINE_BULLET not in captured["prompt"]
    assert not _WRITES & captured["visible_later"]
    assert any(
        "UNIFY_INLINE_CURATION=on ignored: UNIFY_STORE_ADMISSION" in line
        for line in lines
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_only_starts_no_review(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_INLINE_CURATION", "only")
    monkeypatch.setattr(SETTINGS, "UNIFY_TURN_STORAGE_REVIEWS", True)
    captured = await _start_act(monkeypatch, persist=True)
    assert _WRITES <= captured["visible_later"]
    assert "store_skills" not in captured["tools"]
    assert captured["extra_compression_tools"] is None
    # No storage notice: it describes the review and store_skills.
    assert captured["prompt_kwargs"]["can_store"] is False
    assert "### Skill Storage" not in captured["prompt"]
    assert "Nothing reviews this trajectory" in captured["prompt"]
    handle = captured["handle"]
    assert isinstance(handle, _StorageCheckHandle)
    assert handle._turn_reviews_enabled is False
    assert "UNIFY_INLINE_CURATION=only" in handle._skip_review


def _inner_handle(result_future: "asyncio.Future[str]") -> MagicMock:
    inner = MagicMock()

    async def _result():
        return await result_future

    inner.result = _result
    inner.next_notification = AsyncMock(side_effect=lambda: asyncio.Event().wait())
    inner._client = MagicMock(messages=[{"role": "user", "content": "do something"}])
    task = MagicMock()
    task.get_ask_tools = MagicMock(return_value={})
    task.get_completed_tool_metadata = MagicMock(return_value={})
    inner._task = task
    return inner


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("skip", [None, code_act_actor._INLINE_ONLY_REASON])
async def test_a_skipped_review_is_not_started_and_says_why(skip):
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    actor = MagicMock(function_manager=None, guidance_manager=None)
    with (
        patch("unify.actor.code_act_actor._start_storage_check_loop") as loop,
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        loop.return_value = None
        handle = _StorageCheckHandle(
            inner=_inner_handle(result_future),
            actor=actor,
            skip_review=skip,
        )
        result_future.set_result("done")
        for _ in range(500):
            if handle.done():
                break
            await asyncio.sleep(0.01)
        assert handle.done()
    notes = []
    while not handle._notification_q.empty():
        notes.append(handle._notification_q.get_nowait())
    skipped = [n for n in notes if n.get("type") == "storage_review_skipped"]
    if skip is None:
        loop.assert_called_once()
        assert not skipped
    else:
        loop.assert_not_called()
        assert [n["message"] for n in skipped] == [skip]


# --------------------------------------------------------------------------- #
#  The guards on the actor's own writes                                        #
# --------------------------------------------------------------------------- #

SUMMARY = (
    "def count_rows_by_status(rows: list) -> dict:\n"
    "    out = {}\n"
    "    for row in rows:\n"
    "        out[row['status']] = out.get(row['status'], 0) + 1\n"
    "    return out\n"
)
UNRESOLVED = (
    "def fetch_open_tickets() -> list:\n" "    return tickets_api.list(status='open')\n"
)


def _guarded(fm: FunctionManager) -> dict:
    tools = {
        "FunctionManager_add_functions": ToolSpec(fn=fm.add_functions),
        "FunctionManager_patch_function": ToolSpec(fn=fm.patch_function),
    }
    return {n: _fn(t) for n, t in code_act_actor._guard_inline_writes(tools).items()}


@_handle_project
@pytest.mark.parametrize("name", ["nope", "_unused", "tmp", "foo_bar"])
def test_a_junk_name_is_refused_with_the_rule(monkeypatch, name):
    fm = FunctionManager(include_primitives=False)
    add = _guarded(fm)["FunctionManager_add_functions"]
    with pytest.raises(ValueError) as refused:
        add(implementations=[f"def {name}(x: int) -> int:\n    return x\n"])
    message = str(refused.value)
    assert message.startswith("Nothing was stored.")
    assert f"`{name}`" in message
    assert "snake_case, at least two words" in message
    assert "Rename the function after what it does" in message
    assert name not in fm.list_functions()


@_handle_project
def test_one_junk_name_refuses_the_whole_batch():
    fm = FunctionManager(include_primitives=False)
    add = _guarded(fm)["FunctionManager_add_functions"]
    with pytest.raises(ValueError, match="`nope`"):
        add(implementations=[SUMMARY, "def nope() -> None:\n    return None\n"])
    assert fm.list_functions() == {}


@_handle_project
def test_a_verified_unit_with_a_descriptive_name_is_stored(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    fm = FunctionManager(include_primitives=False)
    add = _guarded(fm)["FunctionManager_add_functions"]
    assert add(implementations=[SUMMARY]) == {"count_rows_by_status": "added"}
    namespace: dict = {}
    fm.list_functions(_return_callable=True, _namespace=namespace)
    rows = [{"status": "open"}, {"status": "done"}, {"status": "open"}]
    assert namespace["count_rows_by_status"](rows) == {"open": 2, "done": 1}


@_handle_project
def test_the_storage_check_applies_to_inline_writes_only(monkeypatch):
    """With UNIFY_STORE_CHECK off the actor's write is checked; the review's is as shipped."""
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    fm = FunctionManager(include_primitives=False)
    add = _guarded(fm)["FunctionManager_add_functions"]
    out = add(implementations=[UNRESOLVED], raise_on_error=False)
    assert "'fetch_open_tickets' was not stored" in out["fetch_open_tickets"]
    assert "`tickets_api` is not defined" in out["fetch_open_tickets"]
    assert "fetch_open_tickets" not in fm.list_functions()
    # The same write outside the guard (the review's tools) is as shipped.
    assert fm.add_functions(implementations=[UNRESOLVED]) == {
        "fetch_open_tickets": "added",
    }
    assert not inline_curation.store_check_forced()


@_handle_project
def test_an_inline_patch_that_changes_a_recorded_case_is_refused(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    fm = FunctionManager(include_primitives=False)
    tools = _guarded(fm)
    tools["FunctionManager_add_functions"](
        implementations=["def double_value(x: int) -> int:\n    return x * 2\n"],
    )
    namespace: dict = {}
    fm.list_functions(_return_callable=True, _namespace=namespace)
    assert namespace["double_value"](3) == 6
    patch_fn = tools["FunctionManager_patch_function"]
    refused = patch_fn(name="double_value", old="x * 2", new="x * 3", why="triple")
    assert "returned 6 before; now returns 9" in refused["error"]
    kept = patch_fn(name="double_value", old="x * 2", new="2 * x", why="tidy")
    assert kept["status"] == "patched"
    # An overwrite through the guarded add is replayed the same way.
    out = tools["FunctionManager_add_functions"](
        implementations=["def double_value(x: int) -> int:\n    return x + 1\n"],
        overwrite=True,
        raise_on_error=False,
    )
    assert "now returns 4" in out["double_value"]
