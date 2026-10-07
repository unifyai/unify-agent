from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from unify.common.async_tool_loop import (
    SteerableToolHandle,
    AsyncToolLoopHandle,
    start_async_tool_loop,
)
from unify.common._async_tool.tools_utils import ToolCallMetadata
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
)
from tests.baked_defaults import as_shipped  # noqa: F401


class CustomArgsHandle(SteerableToolHandle):
    """A handle that records steering calls, including a custom method that
    takes structured args beyond what the core steer() actions can carry.

    stop/pause/resume/interject here intentionally take only what
    steer()'s single string `payload` can express (an optional reason /
    message) — a custom handle can no longer extend those *specific*
    actions' signatures with extra structured kwargs (that capability was
    real pre-steer(), via per-call-id minted tools adopting the override's
    full signature). Extra structured kwargs are still fully supported, but
    only for genuinely custom methods reached via
    steer(action="call", method=..., payload=<JSON object>) — see
    `escalate` below.
    """

    def __init__(self) -> None:
        self._done_ev = asyncio.Event()
        self._result_text: str = "inner-complete"
        self.interject_calls: List[Dict[str, Any]] = []
        self.pause_calls: List[Dict[str, Any]] = []
        self.resume_calls: List[Dict[str, Any]] = []
        self.stop_calls: List[Dict[str, Any]] = []
        self.ask_calls: List[Dict[str, Any]] = []
        self.escalate_calls: List[Dict[str, Any]] = []
        # Mark custom write-only helpers
        self.write_only_methods = ["abort"]

    async def ask(self, question: str) -> "SteerableToolHandle":
        self.ask_calls.append({"question": question})
        return self

    async def interject(self, message: str) -> Optional[str]:
        self.interject_calls.append({"message": message})
        return None

    def stop(self, reason: Optional[str] = None) -> None:
        self.stop_calls.append({"reason": reason})
        self._done_ev.set()

    async def pause(self) -> Optional[str]:
        self.pause_calls.append({})
        return "paused"

    async def resume(self) -> Optional[str]:
        self.resume_calls.append({})
        return "resumed"

    # Custom method with structured args beyond a single string — reachable
    # only via steer(action="call", method="escalate", payload=<JSON object>).
    def escalate(self, level: int, note: str = "") -> str:
        self.escalate_calls.append({"level": level, "note": note})
        return f"escalated:{level}:{note}"

    # Write-only helper: terminate with an "aborted" result. This method is
    # intentionally write-only (no returned value used by the loop); the loop
    # should acknowledge and finish when the nested handle resolves.
    def abort(self, *, reason: Optional[str] = None) -> None:
        self._result_text = "aborted"
        self._done_ev.set()
        return None

    def done(self) -> bool:
        return self._done_ev.is_set()

    async def result(self) -> str:
        await self._done_ev.wait()
        return self._result_text

    # New abstract event APIs – simple stubs for tests
    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


async def spawn_custom_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
    """Return a CustomArgsHandle to exercise dynamic helper schemas/args."""
    return CustomArgsHandle()


@pytest.fixture(scope="function")
def client(llm_config):
    return new_llm_client(**llm_config)


# as_shipped: deleted in step 5 (steerable handles)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_dynamic_helper_args_are_exposed_and_forwarded(client):
    """
    End-to-end: the LLM should discover the custom `escalate` method (from
    the "[steerable ...]" started announcement) and invoke it via
    steer(action="call", method="escalate", payload=<JSON object>) with
    structured args that reach the underlying handle method.
    """

    client.set_system_message(
        "Call `spawn_custom_handle` to start a task that exposes dynamic helpers.",
    )

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_custom_handle": spawn_custom_handle},
        timeout=60,
        max_steps=10,
    )

    # Ensure the spawn tool has been requested so helpers will be exposed
    await _wait_for_tool_request(client, "spawn_custom_handle")

    await outer.interject(
        'Now, call the custom method `escalate` with level=3, note="user_request" '
        'via steer(action="call"). Then respond only with: done',
    )

    final = await outer.result()
    assert final is not None, "Loop should complete with a response"

    msgs = client.messages or []

    def _extract_first_steer_call_payload() -> Dict[str, Any]:
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                if fn.get("name") != "steer":
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    continue
                if args.get("action") == "call" and args.get("method") == "escalate":
                    try:
                        return json.loads(args.get("payload") or "{}")
                    except Exception:
                        return {}
        return {}

    payload = _extract_first_steer_call_payload()

    # The LLM should have passed the custom kwargs as a JSON object payload.
    assert payload.get("level") in {3, "3"}
    assert str(payload.get("note", "")).lower().replace(" ", "_") in {
        "user_request",
    }


# as_shipped: deleted in step 5 (steerable handles)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_custom_abort_finishes_nested(client):
    """
    End-to-end: expose a write-only custom method `abort` on the spawned
    handle, reachable via steer(action="call", method="abort"). The model
    should call it, we acknowledge immediately, and the nested handle should
    resolve with the "aborted" message allowing the outer loop to finish.
    """

    client.set_system_message(
        "Call `spawn_custom_handle` to start a task that exposes dynamic helpers.",
    )

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_custom_handle": spawn_custom_handle},
        timeout=60,
        max_steps=20,
        time_awareness=False,
    )

    # Ensure the spawn tool has been requested so helpers will be exposed
    await _wait_for_tool_request(client, "spawn_custom_handle")

    # Instruct the model to call abort (a custom method reachable via
    # steer(action="call", method="abort")) and then reply with 'done'
    await outer.interject(
        'Now, call the custom method `abort` via steer(action="call") '
        "immediately, then respond only with: done",
    )

    final = await outer.result()
    assert final is not None, "Loop should complete with a response"

    # Verify that a tool message shows the nested handle finished with "aborted"
    msgs = client.messages or []

    def _has_aborted_tool_message(messages: List[Dict[str, Any]]) -> bool:
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content")
            if isinstance(content, str):
                txt = content.strip().strip('"').lower()
                if txt == "aborted":
                    return True
        return False

    assert _has_aborted_tool_message(msgs)


# as_shipped: deleted in step 5 (steerable handles)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_custom_outer_handle_instantiated(client):
    """
    Simple sanity check: the start helper should instantiate the provided
    custom outer handle class, and its extended stop signature should be
    usable immediately (e.g., accepts cancel=...).
    """

    class CustomOuterHandle(AsyncToolLoopHandle):
        async def stop(self, *, cancel: bool | None = None, reason: Optional[str] = None) -> None:  # type: ignore[override]
            # Delegate to base stop for cancellation; accepting `cancel` is the point of this test
            await super().stop(reason=reason)

    # Minimal prompt; we don't need tools for this test – just verify instantiation & signature
    client.set_system_message("Reply briefly.")

    outer = start_async_tool_loop(
        client,
        message="hi",
        tools={},
        timeout=60,
        max_steps=1,
        handle_cls=CustomOuterHandle,
    )

    # Returned handle is our custom class
    assert isinstance(outer, CustomOuterHandle)

    # Its stop signature now accepts `cancel`
    import inspect as _inspect

    params = _inspect.signature(outer.stop).parameters
    assert "cancel" in params

    # Calling stop with cancel should not raise, even with no delegate
    await outer.stop(cancel=True, reason="test")
    # Wait for graceful shutdown of the handle task
    await outer.result()


def test_steer_docstring_is_constant_regardless_of_live_handle_overrides():
    """
    Pre-steer(), a handle overriding pause/resume/interject/ask/stop's
    docstring (or adding extra kwargs) got its own per-call-id minted tool
    whose schema/docstring reflected that override — verified by
    `test_dynamic_helpers_use_base_docstrings` and
    `test_dynamic_helpers_use_overridden_docstrings`. That capability is
    gone by design: steer's docstring is now one frozen string
    (STEER_DOC) so the schema stays byte-identical no matter which handle
    is live — this is the flip side of the same fix that killed issues
    01/05/06. This asserts that invariant directly: two wildly different
    handles (no override vs. fully custom docstrings/signatures) produce
    the exact same generated `steer` docstring.
    """
    from unify.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unify.common._async_tool.tools_data import ToolsData

    class BaseLikeHandle(SteerableToolHandle):
        def __init__(self) -> None:
            self._done = asyncio.Event()

        async def ask(self, question: str) -> "SteerableToolHandle":
            return self

        async def interject(self, message: str):
            return None

        def stop(self, reason: Optional[str] = None):
            pass

        async def pause(self):
            return "paused"

        async def resume(self):
            return "resumed"

        def done(self) -> bool:
            return self._done.is_set()

        async def result(self) -> str:
            await self._done.wait()
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    class OverrideDocHandle(BaseLikeHandle):
        async def ask(self, question: str) -> "SteerableToolHandle":
            """Ask override doc: consult safe cache only."""
            return self

        async def interject(self, message: str, *, importance: int = 1):
            """Interject override doc: only interject if importance >= 1."""
            return None

        def stop(self, reason: Optional[str] = None):
            """Stop override doc: stop only if safe to cancel."""

        async def pause(self, *, gate: Optional[str] = None):
            """Pause override doc: only pause if XYZ precondition holds."""
            return "paused"

        async def resume(self, *, token: Optional[str] = None):
            """Resume override doc: resume with a session token if required."""
            return "resumed"

    def _steer_doc_for(handle) -> str:
        tools_data = ToolsData({}, client=None, logger=None)
        task = object()
        tools_data.info[task] = ToolCallMetadata(
            name="dummy",
            call_id="c1",
            call_dict={"function": {"arguments": "{}"}},
            call_idx=0,
            chat_context=None,
            assistant_msg={},
            is_interjectable=False,
            tool_schema={},
            llm_arguments={},
            raw_arguments_json="{}",
            handle=handle,
        )
        tools_data.pending.add(task)
        factory = DynamicToolFactory(tools_data)
        factory.generate()
        return factory.dynamic_tools["steer"].__doc__

    assert _steer_doc_for(BaseLikeHandle()) == _steer_doc_for(OverrideDocHandle())


def test_custom_call_discovery_preserves_annotations_for_public_methods():
    """
    steer(action="call", method=...) validates its JSON-object payload against
    the target method's real signature (loop.py binds
    `inspect.signature(bound).bind(**parsed)`), so the signature
    `_discover_custom_public_methods` returns must preserve real annotations
    (e.g. int) — this is what makes a wrong-typed payload get caught as a
    schema-error result rather than silently coerced. No per-method minted
    tool exists anymore to inspect via method_to_schema, so this checks the
    discovery function directly, and separately confirms method_to_schema
    still renders an integer type when applied ad hoc to a discovered method
    (still a meaningful check: schema derivation itself hasn't regressed,
    even though no tool is actually minted from it at runtime now).
    """
    import inspect

    from unify.common.async_tool_loop import SteerableToolHandle
    from unify.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unify.common.llm_helpers import method_to_schema

    class _AnnotatedHandle(SteerableToolHandle):
        def __init__(self) -> None:
            pass

        def set_value(self, task_id: int, note: str | None = None) -> str:
            return f"set:{task_id}:{note or ''}"

        async def ask(self, question: str):  # type: ignore[override]
            return self

        async def interject(self, message: str):  # type: ignore[override]
            return None

        def stop(self, reason: str | None = None):  # type: ignore[override]
            pass

        async def pause(self):  # type: ignore[override]
            return "paused"

        async def resume(self):  # type: ignore[override]
            return "resumed"

        def done(self) -> bool:  # type: ignore[override]
            return True

        async def result(self) -> str:  # type: ignore[override]
            return "OK"

        async def next_clarification(self) -> dict:  # type: ignore[override]
            return {}

        async def next_notification(self) -> dict:  # type: ignore[override]
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
            return None

    handle = _AnnotatedHandle()
    custom_methods = DynamicToolFactory._discover_custom_public_methods(handle)
    assert "set_value" in custom_methods, sorted(custom_methods)
    bound = custom_methods["set_value"]

    sig = inspect.signature(bound)
    assert "task_id" in sig.parameters
    ann = sig.parameters["task_id"].annotation
    if ann is not inspect._empty:
        assert (
            (ann is int)
            or (ann == int)
            or (ann == "int")
            or (getattr(ann, "__name__", None) == "int")
        )

    # A correctly-typed payload binds cleanly (mirrors loop.py's validation)...
    sig.bind(task_id=1, note="x")
    # ...a wrong-shaped one is rejected the same way loop.py would reject it.
    with pytest.raises(TypeError):
        sig.bind(not_a_real_param=1)

    schema = method_to_schema(bound, include_class_name=False)
    params = schema["function"]["parameters"]
    assert "task_id" in params["properties"]
    prop = params["properties"]["task_id"]
    is_integer = (prop.get("type") == "integer") or any(
        (d.get("type") == "integer")
        for d in prop.get("anyOf", [])
        if isinstance(prop, dict)
    )
    assert is_integer, f"expected integer type for task_id, got: {prop}"
    assert "task_id" in params.get("required", [])


def test_custom_call_discovery_ignores_internal_introspection_methods():
    """
    Regression test: `_discover_custom_public_methods` — the mechanism
    steer(action="call", method=...) consults at execution time — must not
    surface internal introspection methods (e.g. `get_wrapped_handles`,
    `_get_wrapped_handles`) even when they are public on the handle class,
    while still correctly exposing other public methods.
    """
    from unify.common.async_tool_loop import SteerableToolHandle
    from unify.common._async_tool.dynamic_tools_factory import DynamicToolFactory

    # A custom handle that carries introspection-style plumbing of its own
    class IntrospectiveHandle(SteerableToolHandle):
        def __init__(self):
            self._done_ev = asyncio.Event()
            self._wrapped: list = []

        # Wrapper plumbing - SHOULD NOT be exposed
        def _get_wrapped_handles(self) -> list:
            return list(self._wrapped)

        def _wrap_handle(self, handle) -> None:
            self._wrapped.append(handle)

        # Valid public method - SHOULD be exposed
        def public_action(self, arg: str) -> str:
            return f"echo: {arg}"

        # Internal method - SHOULD NOT be exposed
        def _internal_method(self):
            pass

        # Standard steerable methods
        async def ask(self, q: str, **kw):
            return self

        async def interject(self, m: str, **kw):
            pass

        def stop(self, r=None):
            pass

        async def pause(self):
            return "paused"

        async def resume(self):
            return "resumed"

        def done(self):
            return True

        async def result(self):
            return "ok"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    handle = IntrospectiveHandle()
    custom_methods = DynamicToolFactory._discover_custom_public_methods(handle)

    # Check 1: public_action should be discoverable
    assert "public_action" in custom_methods, sorted(custom_methods)

    # Check 2: private introspection methods should NOT be discoverable.
    assert "_internal_method" not in custom_methods
    assert not any(
        "get_wrapped_handles" in name for name in custom_methods
    ), f"Introspection method leaked: {sorted(custom_methods)}"
