"""The actor: one model loop whose only tool runs Python cells in a sandboxed interpreter.

Each :meth:`CodeActActor.act` call is one request. The model's one tool is
``execute_code(code)``; every other harness tool is a call ``tools.<name>(...)``
from a cell (unify/actor/in_code_tools.py). The request joins the run's shared
record (unify/agents), and, with memory on, reads the memory index at the end of
its system prompt (unify/memory/integration/hooks.py).
"""

import asyncio
import copy
import json
import logging
import types
from typing import TYPE_CHECKING, Any, Callable, Optional, Type

from pydantic import BaseModel

from unify.actor import in_code_tools as _in_code_tools
from unify.actor.prompt_builders import build_code_act_prompt
from unify.common.async_tool_loop import ToolLoopHandle, start_async_tool_loop
from unify.common.llm_client import new_llm_client
from unify.common.llm_meter import current_run_meter, new_run_meter

if TYPE_CHECKING:
    from unify.actor.environments.base import BaseEnvironment


logger = logging.getLogger(__name__)

#: The stop reason of a persistent session its requester ended (``/quit``, end of input).
SESSION_ENDED = "session ended"

#: What a persistent session that starts with no request opens with.
_INTERACTIVE_OPENING = (
    "This is an interactive session. Acknowledge that you are ready and "
    "wait for the user to provide instructions via interjection."
)
#: How long one cell may run, in seconds, unless the actor is built with another limit.
CELL_LIMIT_S = 3600.0
#: How many requests one actor runs at once.
MAX_CONCURRENT_REQUESTS = 20
#: How long a request waits for a free slot before it is refused, in seconds.
SLOT_WAIT_S = 30.0


def _session_fork_source(
    inner: Any,
    actor: "CodeActActor",
    *,
    allow_compressed: bool = False,
) -> tuple[Optional[dict], Optional[str]]:
    """The session's conversation for a fork, or why it cannot be continued.

    The source holds the session's ``client``, its raw ``messages``, the
    ``tools`` and ``tool_choice`` of its last request, and ``sent_messages``:
    its history as that request sent it (system prompt first, preprocessed)
    followed by what came after, so a fork that sends them unchanged starts
    with the last request's bytes.
    """
    from unify.common._async_tool import cache_discipline
    from unify.common._async_tool.messages import find_unreplied_assistant_entries

    client = getattr(inner, "_client", None)
    if client is None:
        return None, "the session has no LLM client"
    # a compacted session is forked only where the caller says so (memory during an episode: its compaction was a
    # pause whose segment fork covered the cells before it, so the fork continues the compacted conversation)
    if getattr(getattr(inner, "_compression", None), "count", 0) and not allow_compressed:
        return None, "the session's history was compressed"
    last = cache_discipline.last_sent_request(client)
    if last is None or not last.get("messages"):
        return None, "the session recorded no request"
    if not last.get("tools"):
        return None, "the session's last request carried no tools"

    raw = copy.deepcopy(list(getattr(client, "messages", None) or []))
    as_sent = raw
    preprocess = getattr(actor, "_preprocess_msgs", None)
    if preprocess is not None:
        try:
            as_sent = preprocess(copy.deepcopy(raw)) or as_sent
        except Exception:
            return None, "the session's message preprocessor failed on its history"
    system = getattr(client, "system_message", None)
    if system and not (as_sent and as_sent[0].get("role") == "system"):
        as_sent = [{"role": "system", "content": system}, *as_sent]
    sent = last["messages"]

    def _bytes(messages: list) -> list[str]:
        return [json.dumps(m, default=str) for m in messages]

    if _bytes(as_sent[: len(sent)]) != _bytes(sent):
        return None, "the session's history changed after its last request"
    if find_unreplied_assistant_entries(types.SimpleNamespace(messages=raw)):
        return None, "the session ended with unanswered tool calls"
    return (
        {
            "client": client,
            "messages": raw,
            "sent_messages": as_sent,
            "tools": last["tools"],
            "tool_choice": last.get("tool_choice"),
        },
        None,
    )


async def _end_agents_request(
    agents_binding,
    *,
    reply: Optional[str] = None,
    reason: str = "",
) -> None:
    """End the main agent's request in its agent record.

    With a reply, the reply is recorded and running helpers are stopped;
    without one, the helpers are stopped with ``reason``. Only the main agent
    ends a request, and a failure here never costs the caller its result.
    """
    if agents_binding is None or not getattr(
        agents_binding,
        "is_main",
        agents_binding.name == "root",
    ):
        return
    try:
        if reply is not None:
            await agents_binding.pool.finish_request(reply)
        else:
            await agents_binding.pool.close(reason)
    except Exception:
        logger.warning("could not end the request in the agent record", exc_info=True)
    finally:
        # on the run record the main agent leaves at its request's end
        leave = getattr(agents_binding, "leave", None)
        if callable(leave):
            leave()


class CodeActActor:
    """An actor that answers a request by writing and running Python cells.

    Args:
        environments: The namespaces bound in every cell, such as the
            ``primitives`` an environment registers (``UNIFY_ENV_NAMESPACES``,
            unify/function_manager/primitives/environment.py); one environment
            per namespace.
        timeout: How long one cell may run, in seconds.
        model: The model endpoint; ``None`` uses ``UNIFY_MODEL``.
        preprocess_msgs: Called on the messages before each model call; it
            returns the messages to send.
    """

    def __init__(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]] = None,
        timeout: float = CELL_LIMIT_S,
        model: Optional[str] = None,
        preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    ) -> None:
        from unify.actor.environments.base import by_namespace

        self.environments: dict[str, "BaseEnvironment"] = by_namespace(
            environments or [],
        )
        self._timeout = float(timeout)
        self._model = model
        self._preprocess_msgs = preprocess_msgs
        self._slots = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def act(
        self,
        request: str,
        *,
        guidelines: Optional[str] = None,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        persist: Optional[bool] = None,
    ) -> ToolLoopHandle:
        """Start work on *request* and return the handle of its loop.

        Args:
            request: What to do, in plain language.
            guidelines: How to work, added to the system prompt as its last
                section.
            clarification_enabled: Whether someone reads the questions the
                actor posts to its requester on the run's record.
            response_format: A Pydantic model the answer must match; the
                answer is then given with ``tools.answer``.
            persist: Keep the session open after the answer: each further
                message is a new request in the same interpreter.
        """
        from unify.actor import core_surface
        from unify.agents.binding import bind_for_act
        from unify.memory.integration import hooks as _memory

        core_surface.require_prerequisites()
        # The run's record: this request joins it, as the main agent or as the
        # helper a pool started. A question to the requester is a record post.
        agents = bind_for_act(
            request=str(request or ""),
            user_reads=bool(clarification_enabled),
        )
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=SLOT_WAIT_S)
        except asyncio.TimeoutError:
            raise RuntimeError(
                "CodeActActor is at capacity (too many concurrent sessions). "
                "Try again later or reduce concurrency.",
            ) from None
        from unify.function_manager.execution_env import create_execution_globals

        # The namespace the sandboxed worker's proxy serves to cells: the base
        # globals and each environment's object under its namespace.
        shadow = create_execution_globals()
        for namespace, env in self.environments.items():
            shadow[namespace] = env.get_instance()

        async def _cleanup() -> None:
            self._slots.release()

        system_prompt = build_code_act_prompt(
            environments=self.environments,
            structured=response_format is not None,
            guidelines=guidelines,
        )
        # With memory on, the memory index ends the cached system prompt.
        system_prompt = _memory.system_prompt(system_prompt)
        client = new_llm_client(
            self._model,
            purpose="planning",
            origin="CodeActActor.act",
        )
        # What opens the session's first user message: with memory on, the
        # size of the library copy, then the record's section.
        first_message_parts: list[str] = []
        if _memory.enabled():
            first_message_parts.append(
                _memory.library_line(actor=agents.name == agents.pool.record.root),
            )
        run_meter = new_run_meter()
        meter_token = current_run_meter.set(run_meter)
        in_code = None
        try:
            first_message_parts.append(agents.prompt_section())
            in_code = _in_code_tools.open_for_act(
                agents=agents,
                shadow=shadow,
                response_format=response_format,
                cell_limit_s=self._timeout,
                request=request,
            )
            # With memory on, the library's answer for this request (its librarian
            # asked once, before the first turn) ends the first user message.
            if _memory.enabled() and agents.name == agents.pool.record.root:
                from unify.memory import librarian as _librarian

                first_message_parts.append(await _librarian.request_start_block(in_code.state))
            # The shared opening comes first, then the actor's own prompt; both
            # are constant for the run.
            opening = in_code.opening()
            client.set_system_message(f"{opening}\n\n{system_prompt}")
            first_message = "\n\n".join(part for part in first_message_parts if part)
            handle = start_async_tool_loop(
                client,
                request or _INTERACTIVE_OPENING,
                {"execute_code": in_code.tool()},
                loop_id="CodeActActor.act",
                log_steps=True,
                response_format=response_format,
                persist=persist,
                preprocess_msgs=self._preprocess_msgs,
                prompt_caching=("system", "tools", "messages"),
                on_turn_boundary=agents.on_turn_boundary,
                first_message_context=first_message or None,
                tools_in_code=True,
            )
        except BaseException:
            if in_code is not None:
                await in_code.close()
            await _cleanup()
            raise
        finally:
            current_run_meter.reset(meter_token)
        handle.run_meter = run_meter  # type: ignore[attr-defined]

        result_of_loop = handle.result
        loop_handle = handle

        async def _result_with_cleanup() -> str:
            try:
                try:
                    result = await result_of_loop()
                except BaseException:
                    await _end_agents_request(
                        agents,
                        reason="the main agent's run failed",
                    )
                    raise
                # The main agent's answer ends its request; helpers still
                # running are stopped and checked to have ended. A stopped run or
                # a finished session has no answer to record.
                stop_event = getattr(loop_handle, "_stop_event", None)
                if persist or (stop_event is not None and stop_event.is_set()):
                    await _end_agents_request(
                        agents,
                        reason="the main agent's session ended or was stopped",
                    )
                else:
                    await _end_agents_request(agents, reply=str(result))
                return result
            finally:
                try:
                    await in_code.close()
                finally:
                    await _cleanup()

        handle.result = _result_with_cleanup  # type: ignore[assignment]
        handle.in_code_tools = in_code  # type: ignore[attr-defined]
        handle.agents_pool = agents.pool  # type: ignore[attr-defined]
        return handle
