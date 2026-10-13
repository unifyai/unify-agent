"""The slow brain's decisions, scripted, for the conversation manager's
action tests.

The slow brain makes one model call per turn and executes the tool calls
the reply names. These tests are about what the harness does with those
calls (which action starts, which handle a steer reaches, what the next
turn is shown), so the model is scripted: the transport below unillm is
replaced by :class:`BrainModel`, a :class:`tests.scripted_model.ScriptedModel`
that tells the brain's requests apart from the other requests that share
the ``actor`` kind (a CodeActActor's turns).

The brain is scripted one step at a time, right before the step runs::

    brain = scripted_brain(query_llm=[reply("Two files so far.")])
    brain.next(turn(act("List the CSV files."), say("On it."), wait()))
    result = await cm.step_until_wait(UnifyMessageReceived(content="..."))

A step's first turn always answers the event the step delivers. Events that
land later in the step (an action's result after a stop, say) can open more
turns than the script names; those extra turns are answered with
:func:`idle_turn` (the brain waits) instead of taking a turn meant for the next
step. ``brain.next`` fails when the previous step left a scripted turn
unused, so a script cannot drift silently.

A manager whose event loop runs in the test's own event loop (the
integration tests') also runs brain turns in the background, when an event
reaches it through the broker outside a step (an action's result after the
step ended). Those turns come from the manager's own tasks, not from the
test's: they have the kind ``brain_background`` and are answered with
:func:`idle_turn`, so they never take a step's turn either.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Callable, Iterator, Optional

import pytest

from tests.scripted_model import (
    Call,
    ScriptedModel,
    _text,
    reply,
    scripted,
)

# The slow brain is the only caller that offers both of these tools.
BRAIN_TOOLS = frozenset({"send_unify_message", "wait"})


# ── the brain's tool calls ───────────────────────────────────────────────────


def turn(*calls: tuple[str, dict], content: Optional[str] = None) -> Any:
    """One brain turn: the reply that makes these tool calls, in order."""
    return reply(content, calls=list(calls))


def act(query: str, **args: Any) -> tuple[str, dict]:
    """Start an action (``act``) with *query*."""
    return ("act", {"query": query, **args})


def say(content: str, **args: Any) -> tuple[str, dict]:
    """Send *content* to the user (``send_unify_message``)."""
    return ("send_unify_message", {"content": content, **args})


def wait() -> tuple[str, dict]:
    """End the turn (``wait`` with no delay: no timed follow-up turn)."""
    return ("wait", {})


def steer(operation: str, handle_id: int, **args: Any) -> tuple[str, dict]:
    """Steer an action: ``steer("pause", 0)``, ``steer("ask", 0, question=...)``,
    ``steer("interject", 0, message=...)``, ``steer("stop", 0, reason=...)``."""
    return (f"{operation}_action", {"handle_id": handle_id, **args})


def idle_turn(call: Call) -> Any:
    """The answer to an extra turn opened by an event the step did not
    script for: the brain waits."""
    return turn(wait())


# ── the scripted model ───────────────────────────────────────────────────────


def is_brain_request(request: dict) -> bool:
    """Whether *request* (the transport's keyword arguments) is a brain turn."""
    names = {
        (t.get("function") or {}).get("name") or t.get("name")
        for t in request.get("tools") or []
        if isinstance(t, dict)
    }
    return BRAIN_TOOLS <= names


# Set in the test's own task when the model is built. The test's steps run
# the brain in that task (or in tasks it starts, which copy its context); the
# manager's background turns run in tasks its event loop started earlier.
_TEST_FLOW: contextvars.ContextVar[Optional["BrainModel"]] = contextvars.ContextVar(
    "scripted_brain_test_flow",
    default=None,
)


class BrainModel(ScriptedModel):
    """A :class:`ScriptedModel` whose slow-brain requests have the kind
    ``brain`` and answer from the turns :meth:`next` gives, then from *idle*
    (``None``: an unscripted brain turn fails the test). Brain requests from
    outside the test's flow have the kind ``brain_background`` and are
    answered from *idle* too.

    Build it in the test's own task (the ``scripted_brain`` fixture does)."""

    def __init__(self, *, idle: Any = idle_turn, **scripts: Any) -> None:
        super().__init__(**scripts)
        self.scripts["brain"] = []
        self.idle = idle
        _TEST_FLOW.set(self)

    def next(self, *turns: Any) -> "BrainModel":
        """The brain's turns for the next step, in order."""
        left = self.scripts["brain"]
        assert not left, f"scripted brain turns never requested: {left}"
        self.scripts["brain"] = list(turns)
        return self

    def _next(self, call: Call) -> Any:
        if call.kind == "actor" and is_brain_request(call.request):
            call.kind = "brain" if _TEST_FLOW.get() is self else "brain_background"
            if call.kind == "brain_background" or not self.scripts["brain"]:
                if self.idle is not None:
                    return self.idle
        return super()._next(call)

    # ── what the brain was shown ─────────────────────────────────────────

    def brain_calls(self) -> list[Call]:
        """The brain's requests, in order."""
        return self.of("brain")

    def last_state(self) -> str:
        """The state the brain was shown on its latest turn."""
        calls = self.brain_calls()
        assert calls, f"the brain never ran: {self.kinds()}"
        return state_of(calls[-1])


def state_of(call: Call) -> str:
    """The rendered state a brain request carries: its last user message."""
    users = [m for m in call.messages if m.get("role") == "user"]
    assert users, "a brain request without a state message"
    return _text(users[-1].get("content"))


def user_texts(call: Call) -> list[str]:
    """The text of each user message of a request, in order."""
    return [_text(m.get("content")) for m in call.messages if m.get("role") == "user"]


def prompt_of(call: Call) -> str:
    """The text of a request's last user message (a simulated actor's
    ``ask`` or ``interject`` prompt)."""
    texts = user_texts(call)
    return texts[-1] if texts else ""


@contextlib.contextmanager
def brain_scripted(**kwargs: Any) -> Iterator[BrainModel]:
    """Install a :class:`BrainModel` built from *kwargs* for the block."""
    with scripted(BrainModel(**kwargs)) as model:
        yield model


@pytest.fixture
def scripted_brain() -> Iterator[Callable[..., BrainModel]]:
    """``scripted_brain(query_llm=[...], ...)`` builds a :class:`BrainModel`
    and installs it for the rest of the test."""
    stack = contextlib.ExitStack()

    def build(**kwargs: Any) -> BrainModel:
        return stack.enter_context(brain_scripted(**kwargs))

    with stack:
        yield build
