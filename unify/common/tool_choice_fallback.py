"""Retry a forced tool choice as ``auto`` when the provider refuses it.

Unify forces tool calls in several places: the discovery-first policy makes the
first turn a function search and a guidance search, a turn with tools in
flight must call a tool, a turn over the context threshold must compress, and
a structured answer goes through a response tool. Each sends
``tool_choice="required"`` (or a named tool). Some models refuse any forced
choice while they reason, with an HTTP 400 such as::

    invalid_request_error: tool_choice: type "tool" and "any" are not
    supported for this model.

With ``UNIFY_TOOL_CHOICE_FALLBACK`` set, :func:`install_tool_choice_fallback`
wraps a client's ``generate`` so that such a refusal is retried once with
``tool_choice="auto"`` and a short instruction, sent as the request's last
message, to make the required call first. If the reply still makes no
required call it is re-prompted once, with the rejected reply dropped from the
transcript. The model's endpoint is remembered for the rest of the process, so
its later forced calls go straight to ``auto`` plus the instruction. The
instruction and the re-prompt are request-only: the transcript keeps only the
model's accepted reply. While a fallback call is in flight
:func:`forced_tool_choice_in_fallback` returns the original forced choice, so
completion mutators that act only on forced turns keep working.

Any other error, an unforced call, and every call with the switch off behave
exactly as without this module.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import re
from typing import Any, Callable, Iterable, Optional

from unify.logger import LOGGER

# Endpoints that refused a forced tool choice in this process.
_AUTO_ONLY_ENDPOINTS: set[str] = set()

_FORCED_TOOL_CHOICE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "unify_forced_tool_choice_in_fallback",
    default=None,
)

_UNSUPPORTED = re.compile(
    r"tool_choice.{0,200}?(?:not supported|unsupported|not allowed)"
    r"|(?:does not support|doesn't support|do not support).{0,120}?tool_choice",
    re.IGNORECASE | re.DOTALL,
)
_STATUS_400 = re.compile(r"""["']?(?:code|status(?:_code)?)["']?\s*[:=]\s*400\b""")

_MAX_LISTED_TOOLS = 12
_MAX_REJECTED_CHARS = 600

TOOL_CHOICE_FALLBACK_MARK = "_tool_choice_fallback"


def forced_tool_choice_in_fallback() -> Any:
    """The forced tool choice a fallback call stands in for, else ``None``."""
    return _FORCED_TOOL_CHOICE.get()


def is_forced_tool_choice(tool_choice: Any) -> bool:
    """True for ``"required"``/``"any"`` or a choice naming one tool."""
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() in ("required", "any")
    if isinstance(tool_choice, dict):
        return str(tool_choice.get("type") or "").lower() in ("function", "tool", "any")
    return False


def _exception_chain(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_unsupported_tool_choice_error(exc: BaseException) -> bool:
    """True for an HTTP 400 whose message says the tool_choice is not supported."""
    for err in _exception_chain(exc):
        text = f"{type(err).__name__}: {err}"
        status = getattr(err, "status_code", None)
        is_400 = (
            status == 400
            or "BadRequest" in type(err).__name__
            or bool(_STATUS_400.search(text))
        )
        if is_400 and _UNSUPPORTED.search(text):
            return True
    return False


def _named_tool(tool_choice: Any) -> Optional[str]:
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    if tool_choice.get("name"):
        return str(tool_choice["name"])
    return None


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def build_instruction(tool_choice: Any, tools: Any) -> str:
    """The request-only instruction that stands in for a forced tool choice."""
    named = _named_tool(tool_choice)
    if named:
        return (
            f"This turn requires a call to the tool `{named}`. Call `{named}` now, "
            "before anything else; do not reply with text only."
        )
    names = _tool_names(tools)
    listed = ""
    if names and len(names) <= _MAX_LISTED_TOOLS:
        listed = " (" + ", ".join(f"`{n}`" for n in names) + ")"
    return (
        "This turn requires a tool call. Call the tool or tools this turn needs "
        f"from those available{listed} now, before anything else; do not reply "
        "with text only."
    )


def _reprompt(rejected: Optional[str], instruction: str) -> str:
    text = (
        "Your previous reply made no required tool call, so it was not delivered "
        "and had no effect. " + instruction
    )
    rejected = (rejected or "").strip()
    if rejected:
        if len(rejected) > _MAX_REJECTED_CHARS:
            rejected = rejected[:_MAX_REJECTED_CHARS] + " …"
        text += (
            f"\n\nFor reference only, the reply that was not delivered:\n> {rejected}"
        )
    return text


def _called_names(message: Any) -> list[str]:
    calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    names: list[str] = []
    for call in calls or []:
        function = (
            call.get("function")
            if isinstance(call, dict)
            else getattr(call, "function", None)
        )
        name = (
            function.get("name")
            if isinstance(function, dict)
            else getattr(function, "name", None)
        )
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _reply_message(result: Any) -> Any:
    choices = getattr(result, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "message", None)


def _complies(result: Any, tool_choice: Any) -> bool:
    message = _reply_message(result)
    if message is None:
        # A reply shape this module cannot read is accepted as it is.
        return True
    called = _called_names(message)
    named = _named_tool(tool_choice)
    if named:
        return named in called
    return bool(called)


def _reply_text(result: Any) -> Optional[str]:
    message = _reply_message(result)
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )
    if isinstance(content, str):
        return content
    if content is not None:
        try:
            return json.dumps(content)
        except Exception:  # noqa: BLE001
            return str(content)
    return None


def _endpoint(client: Any) -> str:
    return str(
        getattr(client, "endpoint", None) or getattr(client, "_endpoint", "") or "",
    )


class _Transcript:
    """The list ``generate`` reads (``_messages`` on the real client)."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.attr = "_messages" if hasattr(client, "_messages") else "messages"

    def get(self) -> list:
        return list(getattr(self.client, self.attr, None) or [])

    def strip(self, *transient: dict) -> None:
        """Remove the request-only messages (by identity) from the transcript."""
        current = getattr(self.client, self.attr, None)
        if not isinstance(current, list):
            return
        ids = {id(m) for m in transient}
        kept = [m for m in current if id(m) not in ids]
        if len(kept) != len(current):
            current[:] = kept


def _plan(client: Any, args: tuple, kwargs: dict) -> Optional[Any]:
    """The forced tool choice this call carries, or ``None`` when out of scope."""
    if args or kwargs.get("stream"):
        return None
    tool_choice = kwargs.get("tool_choice", getattr(client, "_tool_choice", None))
    return tool_choice if is_forced_tool_choice(tool_choice) else None


def _remember(client: Any, tool_choice: Any, exc: BaseException) -> None:
    endpoint = _endpoint(client)
    if endpoint and endpoint not in _AUTO_ONLY_ENDPOINTS:
        _AUTO_ONLY_ENDPOINTS.add(endpoint)
        LOGGER.warning(
            f"tool_choice {tool_choice!r} refused by {endpoint} ({type(exc).__name__}); "
            "retrying as 'auto' with an instruction to make the required call, "
            "and doing so for this model's later forced calls in this process.",
        )


def install_tool_choice_fallback(client: Any) -> Any:
    """Wrap ``client.generate`` with the fallback (idempotent); returns *client*."""
    if getattr(client, TOOL_CHOICE_FALLBACK_MARK, False):
        return client
    original: Callable[..., Any] = client.generate
    transcript = _Transcript(client)

    def _base(kwargs: dict) -> list:
        """The messages the forced call would have sent (before the system prompt)."""
        if kwargs.get("messages") is not None:
            base = list(kwargs["messages"])
        else:
            base = transcript.get()
        if kwargs.get("user_message") is not None:
            base.append({"role": "user", "content": kwargs["user_message"]})
        return base

    def _fallback_kwargs(kwargs: dict, base: list, extra: list[dict]) -> dict:
        new_kwargs = dict(kwargs)
        new_kwargs.pop("user_message", None)
        new_kwargs["tool_choice"] = "auto"
        new_kwargs["messages"] = base + extra
        return new_kwargs

    def _drop_reply() -> None:
        """Remove the rejected reply a stateful call appended to the transcript."""
        current = getattr(client, transcript.attr, None)
        if (
            isinstance(current, list)
            and current
            and isinstance(current[-1], dict)
            and current[-1].get("role") == "assistant"
        ):
            current.pop()

    def _instruction(tool_choice: Any, kwargs: dict) -> dict:
        return {
            "role": "user",
            "content": build_instruction(tool_choice, kwargs.get("tools")),
        }

    def _nudge(result: Any, instruction: dict) -> dict:
        return {
            "role": "user",
            "content": _reprompt(_reply_text(result), instruction["content"]),
        }

    async def _call_async(kwargs: dict, base: list, extra: list[dict]) -> Any:
        try:
            result = original(**_fallback_kwargs(kwargs, base, extra))
            if inspect.isawaitable(result):
                result = await result
            return result
        finally:
            transcript.strip(*extra)

    def _call_sync(kwargs: dict, base: list, extra: list[dict]) -> Any:
        try:
            return original(**_fallback_kwargs(kwargs, base, extra))
        finally:
            transcript.strip(*extra)

    async def _fallback_async(tool_choice: Any, kwargs: dict) -> Any:
        token = _FORCED_TOOL_CHOICE.set(tool_choice)
        try:
            base = _base(kwargs)
            instruction = _instruction(tool_choice, kwargs)
            result = await _call_async(kwargs, base, [instruction])
            if _complies(result, tool_choice):
                return result
            _drop_reply()
            nudge = _nudge(result, instruction)
            return await _call_async(kwargs, base, [instruction, nudge])
        finally:
            _FORCED_TOOL_CHOICE.reset(token)

    def _fallback_sync(tool_choice: Any, kwargs: dict) -> Any:
        token = _FORCED_TOOL_CHOICE.set(tool_choice)
        try:
            base = _base(kwargs)
            instruction = _instruction(tool_choice, kwargs)
            result = _call_sync(kwargs, base, [instruction])
            if _complies(result, tool_choice):
                return result
            _drop_reply()
            nudge = _nudge(result, instruction)
            return _call_sync(kwargs, base, [instruction, nudge])
        finally:
            _FORCED_TOOL_CHOICE.reset(token)

    if _is_async_client(client, original):

        @functools.wraps(original)
        async def generate(*args: Any, **kwargs: Any) -> Any:
            tool_choice = _plan(client, args, kwargs)
            if tool_choice is None:
                result = original(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result
            if _endpoint(client) in _AUTO_ONLY_ENDPOINTS:
                return await _fallback_async(tool_choice, kwargs)
            try:
                result = original(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result
            except Exception as exc:
                if not is_unsupported_tool_choice_error(exc):
                    raise
                _remember(client, tool_choice, exc)
            return await _fallback_async(tool_choice, kwargs)

    else:

        @functools.wraps(original)
        def generate(*args: Any, **kwargs: Any) -> Any:
            tool_choice = _plan(client, args, kwargs)
            if tool_choice is None:
                return original(*args, **kwargs)
            if _endpoint(client) in _AUTO_ONLY_ENDPOINTS:
                return _fallback_sync(tool_choice, kwargs)
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                if not is_unsupported_tool_choice_error(exc):
                    raise
                _remember(client, tool_choice, exc)
            return _fallback_sync(tool_choice, kwargs)

    client.generate = generate
    setattr(client, TOOL_CHOICE_FALLBACK_MARK, True)
    return client


def _is_async_client(client: Any, generate: Callable[..., Any]) -> bool:
    try:
        import unillm

        if isinstance(client, unillm.AsyncUnify):
            return True
        if isinstance(client, unillm.Unify):
            return False
    except Exception:  # noqa: BLE001
        pass
    return inspect.iscoroutinefunction(generate)


def reset_for_tests() -> None:
    """Forget the endpoints remembered in this process."""
    _AUTO_ONLY_ENDPOINTS.clear()


__all__ = [
    "build_instruction",
    "forced_tool_choice_in_fallback",
    "install_tool_choice_fallback",
    "is_forced_tool_choice",
    "is_unsupported_tool_choice_error",
    "reset_for_tests",
]
