"""``wait`` and ``ask`` return at once when no message can arrive: every other running agent on the record is itself
blocked in ``wait`` or ``ask`` with nothing unread. The last agent to start waiting releases every waiter; a waiter is
never released while another agent runs and is not waiting. Keyless: every model is a scripted fake; the one pass in a
real box needs bubblewrap and prlimit, as the other box tests do.
"""

from __future__ import annotations

import time

import pytest

from unify.memory import helpers, messaging
from tests.memory.test_helpers import _ctx, _msg, _tc, helper_boxes  # noqa: F401 (fixture)
from tests.memory.test_tool_channel_box import _cell

#: what ``wait`` answers when every other running agent is waiting too
ALL_WAITING = f"{messaging.NO_NEW}; {messaging.ALL_WAITING}"


def _record(*names: str) -> messaging.PassRecord:
    """A writer's record with the helpers *names* running (started on the record, no model behind them)."""
    pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 600, run_home=__import__("tempfile").mkdtemp())
    for name in names:
        pr.add_helper(name, "a task")
        pr.record.add_agent(name, spawner="writer")
    return pr


@pytest.mark.parametrize("similar", [False, True])
@pytest.mark.parametrize("in_code", [False, True])
def test_the_helper_brief_shows_sending_before_reading_replies(similar, in_code):
    brief = helpers.helper_brief(similar)
    role = brief[brief.index("You are a helper") :]
    if in_code:
        calls = ("tools.send(", "tools.ask(", "tools.messages()", "tools.wait()")
    else:
        calls = ("`send(", "`ask(", "`messages`", "`wait`")
    first = [role.index(c) for c in calls]
    assert first == sorted(first), dict(zip(calls, first))


WRITER_CELL = """
print("DELEGATE", tools.delegate("draft an area function", episodes=[]))
print("WAIT", tools.wait()[:300])
"""

REPLY_CELL = 'print("REPLY", tools.wait()[:300])'

HELPER_WAIT_CELL = 'print("HELPER WAIT", tools.wait())'


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def _call_done() -> dict:
    return _cell("h-done", "tools.done('nothing more', [])")
