"""Scripted slow-brain turns for the conversation manager tests.

The slow brain reaches its model through the scripted transport
(``tests/scripted_model.py``), so nothing leaves the process. Every brain
request offers tools, so the scripted model classifies it as an ``actor``
request; so is a request of the ``ask()`` loop, which offers ``ask_question``
instead of the brain's ``send_unify_message``. A brain turn is one decision
that may make several tool calls together: replying and then waiting is
``brain_turn(say("..."), WAIT)``.

Typical use::

    model = ScriptedModel(actor=[brain_turn(say("Got it."), WAIT)])
    with scripted(model):
        result = await cm.step_until_wait(UnifyMessageReceived(content="Hi"))
    (request,) = brain_requests(model)
    assert "Hi" in rendered_state(request)

A ``wait`` with a ``delay`` schedules a turn of its own outside the step that
made it, so the turns here wait without one.
"""

from __future__ import annotations

from typing import Any

from openai.types.chat import ChatCompletion

from tests.scripted_model import Call, ScriptedModel, reply

#: The brain ends its turn with nothing more to do.
WAIT = ("wait", {})


def say(content: str) -> tuple[str, dict]:
    """A ``send_unify_message`` call: the brain replies *content* in the chat."""
    return ("send_unify_message", {"content": content})


def act(query: str, **args: Any) -> tuple[str, dict]:
    """An ``act`` call: the brain starts an action for *query*."""
    return ("act", {"query": query, **args})


def steer(tool: str, handle_id: int, **args: Any) -> tuple[str, dict]:
    """A call of a steering tool (``stop_action``, ``pause_action``, ...)
    addressed to the in-flight action *handle_id*."""
    return (tool, {"handle_id": handle_id, **args})


def brain_turn(*calls: tuple[str, dict]) -> ChatCompletion:
    """One slow-brain decision: *calls*, made together in a single turn."""
    return reply(calls=list(calls))


def message_text(message: dict) -> str:
    """A request message's text, whether a string or a list of parts."""
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


def brain_requests(model: ScriptedModel) -> list[Call]:
    """The slow brain's requests, in order (they offer ``send_unify_message``)."""
    return [c for c in model.of("actor") if "send_unify_message" in c.tool_names]


def rendered_state(call: Call) -> str:
    """What a brain request shows the model of the conversation: its last
    message, the state rendered for this turn."""
    return message_text(call.messages[-1])
