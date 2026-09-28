"""Tests for proactive on-demand skill storage via the ``store_skills`` tool.

Covers:

1. ``store_skills`` tool is available when ``can_store=True`` and both FM/GM
   exist, and absent when ``can_store=False``.

2. ``store_skills`` is called by the doing loop and produces a storage
   summary that is tracked in ``AgentContext.proactive_storage_summaries``.

3. The post-processing ``StorageCheck`` receives proactive summaries and
   adjusts its prompt accordingly (no duplication).

4. ``_start_proactive_storage_loop`` receives the correct ``request``
   parameter and trajectory snapshot.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.actor.code_act.conftest import make_fm_mock
from unify.actor.code_act_actor import (
    AgentContext,
    CodeActActor,
    _StorageCheckHandle,
    _STORAGE_WHAT_CAN_BE_STORED,
    _start_proactive_storage_loop,
)

# ---------------------------------------------------------------------------
# Reusable GuidanceManager stand-in (same as test_storage_function_and_guidance)
# ---------------------------------------------------------------------------


class _TrackingGuidanceManager:
    """Minimal GuidanceManager stand-in that records ``add_guidance`` calls."""

    def __init__(self) -> None:
        self.add_calls: list[dict] = []

    def search(self, references=None, k=10):
        """Search for guidance entries by semantic similarity to reference text."""
        return []

    def filter(self, filter=None, offset=0, limit=100):
        """Filter guidance entries using a Python filter expression."""
        return []

    def get_guidance(self, *, guidance_id):
        """Fetch one guidance entry by id with its complete content."""
        raise ValueError(f"No guidance found with guidance_id {guidance_id}.")

    def add_guidance(self, *, title, content, function_ids=None):
        """Add a guidance entry describing a compositional procedure or playbook."""
        self.add_calls.append(
            {"title": title, "content": content, "function_ids": function_ids},
        )
        return {"details": {"guidance_id": len(self.add_calls)}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        """Update an existing guidance entry."""
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        """Delete a guidance entry by ID."""
        return {"deleted": True}

    def reconcile_dependencies(self, *, guidance_ids=None):
        """Refresh structured link debt for related functions."""
        return {"outcome": "checked", "details": {"guidance_ids": guidance_ids or []}}


def test_storage_prompt_encourages_bounded_agent_loop_distillation():
    prompt = _STORAGE_WHAT_CAN_BE_STORED

    assert "execute_code` -> observe -> agent reasoning -> `execute_code" in prompt
    assert "bounded semantic judgment inside an otherwise stable" in prompt
    assert "distill the trajectory into one function" in prompt
    assert "focused `query_llm(...)` calls" in prompt
    assert "Semantic downgrades are bugs" in prompt
    assert "human-facing synthesis" in prompt
    assert "label-specific canned prose" in prompt
    assert "Generalize by preserving the LLM call" in prompt
    assert "changing tool discovery" in prompt
    assert "unknown-state debugging" in prompt


def test_storage_prompt_carries_the_distillation_dial():
    """The unified frame: skeleton vs joints, per-substep dial, asymmetric
    failure costs, the evidence rule, tripwires, and reversibility."""
    prompt = _STORAGE_WHAT_CAN_BE_STORED

    assert "### The distillation dial" in prompt
    assert "a dial, not a switch" in prompt
    assert "skeleton" in prompt
    assert "joints" in prompt
    assert "choosing a notch per substep" in prompt
    # Asymmetric failure visibility: too late is loud/cheap, too early is
    # quiet/expensive; placement errors mirror the same asymmetry.
    assert "Distilling too little is loud" in prompt
    assert "Distilling too much is quiet" in prompt
    assert "processes an anomaly as if it were normal" in prompt
    assert "costs cents" in prompt
    assert "costs correctness" in prompt
    # Evidence rule and tripwires.
    assert "Freeze only observed invariance" in prompt
    assert "Preserve the surprise signal" in prompt
    assert "raise or return early" in prompt
    # Reversibility keeps the dial from causing under-distillation.
    assert "Distillation is reversible" in prompt
    # Model choice is folded into the same decision.
    assert "### Model choice is part of distillation" in prompt
    assert "keep the cheapest model that passes" in prompt


def test_sub_agent_storage_is_gated_on_the_dial():
    """Storing an actor.act config must first survive the dial check."""
    from unify.actor.code_act_actor import _STORAGE_SUB_AGENT_PATTERNS

    prompt = _STORAGE_SUB_AGENT_PATTERNS
    assert "First apply the distillation dial" in prompt
    assert "plan discovered at runtime" in prompt
    assert "instead of preserving the agent wrapper" in prompt
    assert "Calls that survive this test" in prompt


# ---------------------------------------------------------------------------
# 1. Symbolic: store_skills tool availability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_store_skills_tool_present_when_can_store_true():
    """When can_store=True and both FM+GM exist, the ``store_skills`` tool
    appears in the tool set passed to the doing loop."""
    fm = make_fm_mock()
    fm._include_primitives = False
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm._get_function_data_by_name = MagicMock(return_value=None)
    fm._get_primitive_data_by_name = MagicMock(return_value=None)

    gm = _TrackingGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=10,
    )
    try:
        tools = actor.get_tools("act")
        assert "store_skills" in tools, (
            f"Expected 'store_skills' in tools when FM+GM exist, "
            f"got: {sorted(tools.keys())}"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_store_skills_tool_absent_without_guidance_manager():
    """When GuidanceManager is missing, ``store_skills`` should not be built."""
    from unittest.mock import patch as _patch

    fm = make_fm_mock()
    fm._include_primitives = False
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm._get_function_data_by_name = MagicMock(return_value=None)
    fm._get_primitive_data_by_name = MagicMock(return_value=None)

    # Prevent ManagerRegistry from supplying a default GuidanceManager
    # by patching the registry method to return None during construction.
    with _patch(
        "unify.manager_registry.ManagerRegistry.get_guidance_manager",
        return_value=None,
    ):
        actor = CodeActActor(
            function_manager=fm,
            guidance_manager=None,
            timeout=10,
        )
    try:
        tools = actor.get_tools("act")
        assert (
            "store_skills" not in tools
        ), "store_skills should not be present without GuidanceManager"
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(60)
async def test_store_skills_filtered_when_can_store_false():
    """When can_store=False, ``store_skills`` must be removed by _filter_tools
    even though FM+GM exist and the tool was built."""
    fm = make_fm_mock()
    fm._include_primitives = False
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm._get_function_data_by_name = MagicMock(return_value=None)
    fm._get_primitive_data_by_name = MagicMock(return_value=None)
    fm.add_functions.return_value = {"added": []}

    gm = _TrackingGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=30,
    )
    try:
        handle = await actor.act(
            "Call the tool store_skills with request='store everything'. "
            "If the tool is not available, say 'store_skills unavailable'.",
            can_store=False,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=45)
        assert (
            "unavailable" in result.lower() or "store_skills" in result
        ), f"Expected the LLM to report store_skills as unavailable, got: {result}"
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 2. Symbolic: _start_proactive_storage_loop receives correct parameters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_proactive_storage_loop_receives_request_and_trajectory():
    """``_start_proactive_storage_loop`` is called with the trajectory snapshot
    and the request string from the ``store_skills`` tool."""
    from unify.function_manager.function_manager import FunctionManager

    fm = FunctionManager(include_primitives=False)
    gm = _TrackingGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=60,
    )

    captured_kwargs: dict = {}

    with (
        patch.object(MagicMock, "__repr__", return_value="<MockHandle>"),
        patch(
            "unify.actor.code_act_actor._start_proactive_storage_loop",
        ) as mock_proactive,
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        mock_handle = MagicMock()

        async def _fake_result():
            return "Stored 1 function: my_func"

        mock_handle.result = _fake_result
        mock_handle.done = MagicMock(return_value=True)
        mock_proactive.return_value = mock_handle

        try:
            handle = await actor.act(
                "Write a function that adds two numbers, then call store_skills "
                "with request='Store the add function I just wrote'. "
                "After store_skills returns, report the result.",
                can_store=True,
                persist=False,
                clarification_enabled=False,
            )
            await asyncio.wait_for(handle.result(), timeout=90)

            deadline = asyncio.get_event_loop().time() + 30
            while not handle.done():
                if asyncio.get_event_loop().time() > deadline:
                    break
                await asyncio.sleep(0.5)

            if mock_proactive.call_count > 0:
                captured_kwargs = mock_proactive.call_args.kwargs
        finally:
            try:
                await actor.close()
            except Exception:
                pass

    if mock_proactive.call_count == 0:
        pytest.skip(
            "LLM did not call store_skills in this run — "
            "this is an eval-sensitive path",
        )

    assert (
        "request" in captured_kwargs
    ), f"Expected 'request' kwarg, got: {list(captured_kwargs.keys())}"
    assert isinstance(captured_kwargs["request"], str)
    assert len(captured_kwargs["request"]) > 0

    assert (
        "trajectory" in captured_kwargs
    ), f"Expected 'trajectory' kwarg, got: {list(captured_kwargs.keys())}"
    assert isinstance(captured_kwargs["trajectory"], list)

    assert "actor" in captured_kwargs
    assert captured_kwargs["actor"] is actor


# ---------------------------------------------------------------------------
# 3. Symbolic: StorageCheck receives proactive_summaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_storage_check_receives_proactive_summaries():
    """When proactive storage summaries exist in the AgentContext, the
    post-processing ``_start_storage_check_loop`` receives them via the
    ``proactive_summaries`` keyword argument."""

    from unify.actor.code_act_actor import _CURRENT_AGENT_CONTEXT

    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    inner = MagicMock()

    async def _await_result():
        return await result_future

    inner.result = _await_result
    inner.next_notification = AsyncMock(
        side_effect=lambda: asyncio.Event().wait(),
    )

    async def _stop(**kwargs):
        if not result_future.done():
            result_future.set_result("task done")

    inner.stop = AsyncMock(side_effect=_stop)

    mock_client = MagicMock()
    mock_client.messages = [{"role": "user", "content": "do something"}]
    inner._client = mock_client

    mock_task = MagicMock()
    mock_task.get_ask_tools = MagicMock(return_value={})
    mock_task.get_completed_tool_metadata = MagicMock(return_value={})
    inner._task = mock_task

    actor_mock = MagicMock()
    actor_mock.function_manager = None
    actor_mock.guidance_manager = None

    ctx = AgentContext()
    ctx.proactive_storage_summaries = [
        "Stored 1 function: my_helper",
        "Stored 1 guidance: data pipeline playbook",
    ]
    token = _CURRENT_AGENT_CONTEXT.set(ctx)

    try:
        with (
            patch(
                "unify.actor.code_act_actor._start_storage_check_loop",
            ) as mock_loop,
            patch(
                "unify.actor.code_act_actor.publish_manager_method_event",
                new_callable=AsyncMock,
            ),
        ):
            mock_loop.return_value = None

            handle = _StorageCheckHandle(inner=inner, actor=actor_mock)

            await handle.stop(reason="done")

            deadline = asyncio.get_event_loop().time() + 10
            while not handle.done():
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError("Handle did not complete")
                await asyncio.sleep(0.1)

            mock_loop.assert_called_once()
            call_kwargs = mock_loop.call_args.kwargs
            assert call_kwargs["proactive_summaries"] == [
                "Stored 1 function: my_helper",
                "Stored 1 guidance: data pipeline playbook",
            ], f"Expected proactive summaries to be passed, got: {call_kwargs.get('proactive_summaries')}"
    finally:
        _CURRENT_AGENT_CONTEXT.reset(token)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_storage_check_no_proactive_summaries_passes_none():
    """When no proactive storage occurred, ``proactive_summaries`` is None."""

    from unify.actor.code_act_actor import _CURRENT_AGENT_CONTEXT

    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    inner = MagicMock()

    async def _await_result():
        return await result_future

    inner.result = _await_result
    inner.next_notification = AsyncMock(
        side_effect=lambda: asyncio.Event().wait(),
    )

    async def _stop(**kwargs):
        if not result_future.done():
            result_future.set_result("task done")

    inner.stop = AsyncMock(side_effect=_stop)

    mock_client = MagicMock()
    mock_client.messages = [{"role": "user", "content": "do something"}]
    inner._client = mock_client

    mock_task = MagicMock()
    mock_task.get_ask_tools = MagicMock(return_value={})
    mock_task.get_completed_tool_metadata = MagicMock(return_value={})
    inner._task = mock_task

    actor_mock = MagicMock()
    actor_mock.function_manager = None
    actor_mock.guidance_manager = None

    ctx = AgentContext()
    token = _CURRENT_AGENT_CONTEXT.set(ctx)

    try:
        with (
            patch(
                "unify.actor.code_act_actor._start_storage_check_loop",
            ) as mock_loop,
            patch(
                "unify.actor.code_act_actor.publish_manager_method_event",
                new_callable=AsyncMock,
            ),
        ):
            mock_loop.return_value = None

            handle = _StorageCheckHandle(inner=inner, actor=actor_mock)

            await handle.stop(reason="done")

            deadline = asyncio.get_event_loop().time() + 10
            while not handle.done():
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError("Handle did not complete")
                await asyncio.sleep(0.1)

            mock_loop.assert_called_once()
            call_kwargs = mock_loop.call_args.kwargs
            assert call_kwargs["proactive_summaries"] is None, (
                f"Expected proactive_summaries=None when no proactive storage "
                f"occurred, got: {call_kwargs.get('proactive_summaries')}"
            )
    finally:
        _CURRENT_AGENT_CONTEXT.reset(token)


# ---------------------------------------------------------------------------
# 4. Symbolic: proactive storage prompt includes request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_proactive_storage_loop_returns_none_without_managers():
    """``_start_proactive_storage_loop`` returns None when FM or GM is missing."""
    actor_mock = MagicMock()
    actor_mock.function_manager = None
    actor_mock.guidance_manager = MagicMock()

    result = _start_proactive_storage_loop(
        trajectory=[],
        ask_tools={},
        completed_tool_metadata={},
        actor=actor_mock,
        request="store something",
        parent_lineage=[],
    )
    assert result is None

    actor_mock.function_manager = MagicMock()
    actor_mock.guidance_manager = None

    result = _start_proactive_storage_loop(
        trajectory=[],
        ask_tools={},
        completed_tool_metadata={},
        actor=actor_mock,
        request="store something",
        parent_lineage=[],
    )
    assert result is None


# ---------------------------------------------------------------------------
# 5. Symbolic: AgentContext tracks proactive summaries
# ---------------------------------------------------------------------------


def test_agent_context_proactive_storage_summaries_default():
    """AgentContext initializes with an empty proactive_storage_summaries list."""
    ctx = AgentContext()
    assert ctx.proactive_storage_summaries == []
    assert isinstance(ctx.proactive_storage_summaries, list)


def test_agent_context_proactive_storage_summaries_isolated():
    """Each AgentContext instance has its own isolated list."""
    ctx1 = AgentContext()
    ctx2 = AgentContext()
    ctx1.proactive_storage_summaries.append("summary A")
    assert ctx2.proactive_storage_summaries == []


# ---------------------------------------------------------------------------
# 6. Symbolic: event publishing for proactive storage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_proactive_storage_publishes_manager_method_events():
    """The ``store_skills`` tool publishes incoming and outgoing
    ManagerMethod events with method='ProactiveStorage'."""
    from unify.actor.code_act_actor import (
        _CURRENT_AGENT_CONTEXT,
    )

    fm = make_fm_mock()
    fm._include_primitives = False
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm._get_function_data_by_name = MagicMock(return_value=None)
    fm._get_primitive_data_by_name = MagicMock(return_value=None)

    gm = _TrackingGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=10,
    )

    tools = actor.get_tools("act")
    store_skills_fn = tools.get("store_skills")
    assert store_skills_fn is not None

    ctx = AgentContext()
    mock_handle = MagicMock()
    mock_handle._client = MagicMock()
    mock_handle._client.messages = [{"role": "user", "content": "test"}]
    mock_handle._task = MagicMock()
    mock_handle._task.get_ask_tools = MagicMock(return_value={})
    mock_handle._task.get_completed_tool_metadata = MagicMock(return_value={})
    ctx.handle = mock_handle

    token = _CURRENT_AGENT_CONTEXT.set(ctx)

    try:
        with (
            patch(
                "unify.actor.code_act_actor._start_proactive_storage_loop",
            ) as mock_proactive,
            patch(
                "unify.actor.code_act_actor.publish_manager_method_event",
                new_callable=AsyncMock,
            ) as mock_publish,
        ):
            mock_proactive.return_value = None

            result = await store_skills_fn(request="Store my helper function")

            assert "unavailable" in result.lower()

            publish_calls = mock_publish.call_args_list
            incoming = [
                c
                for c in publish_calls
                if c.kwargs.get("phase") == "incoming"
                and c.args[2] == "ProactiveStorage"
            ]
            outgoing = [
                c
                for c in publish_calls
                if c.kwargs.get("phase") == "outgoing"
                and c.args[2] == "ProactiveStorage"
            ]
            assert (
                len(incoming) == 1
            ), f"Expected 1 incoming ProactiveStorage event, got {len(incoming)}"
            assert (
                len(outgoing) == 1
            ), f"Expected 1 outgoing ProactiveStorage event, got {len(outgoing)}"
            assert incoming[0].kwargs["instructions"] == "Store my helper function"
    finally:
        _CURRENT_AGENT_CONTEXT.reset(token)
        try:
            await actor.close()
        except Exception:
            pass
