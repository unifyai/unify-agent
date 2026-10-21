"""
Steerability ConversationManager → CodeActActor integration tests.

These check the brain's steering tools on actions run by the production
actor:
- stop
- pause/resume, interject and ask
- steering isolation when multiple handles are in-flight

The actor's handle has no pause, resume, interject or ask any more: its loop
lost its pause gate and only takes ``stop``, ``submit`` and clarification
answers. A pause, interject or ask steer on an actor's action is therefore
refused, and the refusal is reported in the steering tool's result; resume
reports that the action is not paused. The tests pin that behaviour: the
refusal is reported, and the action is unharmed and still completes.

The slow brain and the actor's model are scripted (``scripted_brain``). An
action stays in flight while its model call waits on a gate the test
releases (``held``).
"""

import asyncio
import os
from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    extract_actor_handle,
    get_actor_started_event,
    held,
    inject_actor_result,
    recent_result,
    scripted_brain_and_actor,
    tool_results,
    wait_for_actor_completion,
    wait_for_condition,
    wait_until_done,
)
from tests.legacy.conversation_manager.actions.scripted_brain import (
    act,
    say,
    state_of,
    steer,
    turn,
    user_texts,
    wait,
)
from tests.scripted_model import cell, reply
from unify.legacy.conversation_manager.events import UnifyMessageReceived
from unify.legacy.conversation_manager.domains.brain_action_tools import (
    get_handle_paused_state,
)
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (steerable handles through the conversation manager)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("as_shipped"),
]


async def _start(cm, model, request: str, query: str, ack: str) -> tuple[int, object]:
    """The user's *request* makes the brain start an action running *query*;
    returns its handle id and handle, once the actor's model call arrived."""
    calls_before = len(model.of("actor"))
    model.next(turn(act(query), say(ack), wait()))
    result = await cm.step_until_wait(UnifyMessageReceived(content=request))
    assert_no_errors(result)
    handle_id = get_actor_started_event(result).handle_id
    await wait_for_condition(
        lambda: len(model.of("actor")) > calls_before,
        timeout=60,
        timeout_message="The actor never made its first model call.",
    )
    return handle_id, extract_actor_handle(cm, handle_id)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_pause_resume_inflight_handle(initialized_cm_codeact, scripted_brain):
    """Pause and resume an in-flight actor action via natural user messages.

    The actor's handle has no pause: the brain's pause steer is refused and
    the refusal is reported in the tool's result, resume reports the action
    is not paused, and the action is unharmed and completes.
    """
    cm = initialized_cm_codeact
    gate = asyncio.Event()
    answer = "Q3 revenue was 4.2M USD, up 8% on Q2, per the finance notes."
    model = scripted_brain_and_actor(scripted_brain, held(gate, reply(answer)))

    handle_id, handle = await _start(
        cm,
        model,
        "Search my notes files for anything about the quarterly revenue figures.",
        "Search the user's notes files for the quarterly revenue figures.",
        "Searching your notes for the revenue figures.",
    )

    # User asks to pause.
    model.next(turn(steer("pause", handle_id), say("Pausing the search."), wait()))
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Hold on, pause that search for a moment. I need to think about what I actually need.",
        ),
    )
    paused = recent_result(cm, "pause_action")
    assert '"result": "Error:' in paused and "pause" in paused, paused
    assert get_handle_paused_state(handle) is None
    assert not handle.done() and handle_id in cm.cm.in_flight_actions

    # User asks to resume.
    model.next(turn(steer("resume", handle_id), say("Continuing."), wait()))
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Okay, go ahead and continue that revenue search.",
        ),
    )
    assert f"Action {handle_id} is not paused." in recent_result(cm, "resume_action")

    gate.set()
    final = await wait_for_actor_completion(cm, handle_id, timeout=120)
    assert final == answer
    await wait_until_done(handle)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_stop_inflight_handle(initialized_cm_codeact, scripted_brain):
    """Stop an in-flight actor handle via a natural user message: the stop
    reaches the actor, ends its model call, and the action completes as
    stopped."""
    cm = initialized_cm_codeact
    gate = asyncio.Event()
    model = scripted_brain_and_actor(
        scripted_brain,
        held(gate, reply("The launch moved to May.")),
    )

    try:
        handle_id, handle = await _start(
            cm,
            model,
            "Look through my recent notes files for mentions of the product launch timeline.",
            "Find mentions of the product launch timeline in the user's recent notes files.",
            "Looking through your notes for the launch timeline.",
        )

        # User asks to cancel.
        reason = "The user already found what they needed."
        model.next(
            turn(steer("stop", handle_id, reason=reason), say("Cancelled."), wait()),
        )
        await cm.step_until_wait(
            UnifyMessageReceived(
                content="Never mind, cancel that search. I already found what I needed.",
            ),
        )

        actor_result = await wait_for_actor_completion(cm, handle_id, timeout=120)
        assert actor_result.startswith(f"Task stopped as requested. Reason: {reason}")
        # The stop ended the actor's model call: it never got an answer.
        assert len(model.of("actor")) == 1 and model.of("actor")[0].answer is None

        # Inject the ActorResult to trigger CM bookkeeping
        # (CMStepDriver patches the event broker and doesn't auto-forward background events).
        await inject_actor_result(
            cm,
            handle_id=handle_id,
            result=actor_result,
            success=True,
        )

        assert handle_id not in cm.cm.in_flight_actions
        assert handle_id in cm.cm.completed_actions
        await wait_until_done(handle)
    finally:
        gate.set()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_interject_midflight_constraints(initialized_cm_codeact, scripted_brain):
    """Interject constraints mid-flight via a natural user message.

    The actor's handle has no interject: the brain's interject steer is
    refused and the refusal is reported in the tool's result. The constraint
    never reaches the actor, and the action is unharmed and completes.
    """
    cm = initialized_cm_codeact
    gate = asyncio.Event()
    answer = "The marketing notes cover the spring campaign launch and its channels."
    model = scripted_brain_and_actor(scripted_brain, held(gate, reply(answer)))

    handle_id, handle = await _start(
        cm,
        model,
        "Summarize my recent notes files about the marketing campaign.",
        "Summarize the user's recent notes files about the marketing campaign.",
        "Summarizing the marketing campaign notes.",
    )

    # User adds a constraint mid-flight.
    constraint = "Only include items that mention a dollar amount or budget figure."
    model.next(
        turn(
            steer("interject", handle_id, message=constraint),
            say("Got it, only items with a dollar figure."),
            wait(),
        ),
    )
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Actually, only include items that mention a specific dollar amount or budget figure.",
        ),
    )
    interjected = recent_result(cm, "interject_action")
    assert '"result": "Error:' in interjected and "interject" in interjected, interjected
    assert not handle.done() and handle_id in cm.cm.in_flight_actions

    gate.set()
    final = await wait_for_actor_completion(cm, handle_id, timeout=120)
    assert final == answer
    assert not any(
        constraint in text for call in model.of("actor") for text in user_texts(call)
    )
    await wait_until_done(handle)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_two_concurrent_handles_pause_one_other_completes(
    initialized_cm_codeact,
    scripted_brain,
):
    """
    Concurrency: CM tracks two in-flight actor handles and steers them independently.

    Scenario:
    - User requests a research task (A).
    - User asks to pause it while they think: the actor's handle has no
      pause, so the steer is refused (reported in the tool's result) and A
      keeps running.
    - User sends a second, unrelated request (B).
    - B completes while A is still in flight, untouched.
    - User resumes A (reported as not paused); A completes.
    """
    cm = initialized_cm_codeact
    gate_a = asyncio.Event()
    answer_a = "The budget review settled on a 5% cut to travel and a hiring freeze."
    answer_b = "Alice's phone number is +1 555 0101."
    # A's model call arrives first and waits; B's is answered at once.
    model = scripted_brain_and_actor(
        scripted_brain,
        held(gate_a, reply(answer_a)),
        reply(answer_b),
    )

    # Action A: research task.
    handle_id_a, handle_a = await _start(
        cm,
        model,
        (
            "Search my notes files for anything about the quarterly budget review "
            "and summarize the key discussion points."
        ),
        "Summarize the key discussion points about the quarterly budget review in the user's notes files.",
        "Looking into the budget review notes.",
    )

    # User asks to pause A.
    model.next(turn(steer("pause", handle_id_a), say("Pausing that."), wait()))
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Hold on, pause that budget search for now. I want to do something else first.",
        ),
    )
    assert '"result": "Error:' in recent_result(cm, "pause_action")
    assert not handle_a.done()

    # Action B: unrelated request while A waits.
    handle_id_b, handle_b = await _start(
        cm,
        model,
        "Look up Alice's phone number for me.",
        "Look up Alice's phone number.",
        "Looking up Alice's number.",
    )

    # CM tracks both handles (in-flight or completed).
    all_actions = {**cm.cm.in_flight_actions, **cm.cm.completed_actions}
    assert handle_id_a in all_actions
    assert handle_id_b in all_actions
    assert handle_id_a != handle_id_b
    assert handle_a is not handle_b

    # B completes while A is still in flight (steering isolation; no cross-talk).
    final_b = await wait_for_actor_completion(cm, handle_id_b, timeout=120)
    assert final_b == answer_b
    assert not handle_a.done()
    assert handle_id_a in cm.cm.in_flight_actions

    # User resumes A.
    model.next(turn(steer("resume", handle_id_a), say("Continuing the budget search."), wait()))
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Okay, you can continue that budget search now.",
        ),
    )
    assert f"Action {handle_id_a} is not paused." in recent_result(cm, "resume_action")

    gate_a.set()
    final_a = await wait_for_actor_completion(cm, handle_id_a, timeout=120)
    assert final_a == answer_a
    await wait_until_done(handle_a)
    await wait_until_done(handle_b)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_ask_completed_action_about_trajectory(
    initialized_cm_codeact,
    scripted_brain,
):
    """
    User asks about a completed action's trajectory.

    This validates the completed_actions registry: the action and its handle
    persist after completion, with the handle's trajectory (its model turns
    and cell results). The actor's handle has no ask: the brain's ask steer
    on the completed action is answered with the refusal, which is recorded
    on the action and shown to the brain on another turn.

    Flow:
    1. User requests an action (triggers act); the actor lists the files
    2. Action completes and moves to completed_actions
    3. User asks about how the action was performed; the brain asks it
    """
    cm = initialized_cm_codeact

    from unify.workspace import get_local_root

    (Path(get_local_root()) / "notes.txt").write_text("Launch checklist.\n")
    answer = "Your workspace has one file: notes.txt (18 bytes)."
    model = scripted_brain_and_actor(
        scripted_brain,
        reply(
            calls=[
                cell(
                    "import os\n"
                    "print([(p, os.path.getsize(p)) for p in sorted(os.listdir('.'))"
                    " if os.path.isfile(p)])",
                ),
            ],
        ),
        reply(answer),
    )

    # Step 1: Start an action that will complete
    model.next(
        turn(
            act("List every file in the user's workspace with its size."),
            say("Listing your files."),
            wait(),
        ),
    )
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="List every file in my workspace with its size.",
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id

    # Wait for action to complete
    actor_result = await wait_for_actor_completion(cm, handle_id, timeout=120)
    assert actor_result == answer
    assert "('notes.txt', 18)" in tool_results(model.of("actor")[1])[-1]

    # Inject the ActorResult event to trigger CM's event handler
    # (test driver patches event broker and doesn't forward background events)
    await inject_actor_result(
        cm,
        handle_id=handle_id,
        result=actor_result,
        success=True,
    )

    # Verify handle moved to completed_actions
    assert (
        handle_id not in cm.cm.in_flight_actions
    ), f"Handle {handle_id} should be removed from in_flight_actions after completion"
    assert (
        handle_id in cm.cm.completed_actions
    ), f"Handle {handle_id} should be in completed_actions after completion"

    # Verify the completed action data is preserved
    completed_data = cm.cm.completed_actions[handle_id]
    assert "handle" in completed_data, "Completed action should preserve handle"
    assert "query" in completed_data, "Completed action should preserve query"
    assert completed_data["query"], "Completed action query should not be empty"

    # The handle keeps its trajectory: the cell it ran and what the cell printed.
    handle = completed_data["handle"]
    assert handle is not None, "Completed action handle should not be None"
    await wait_until_done(handle)
    history = handle.get_history()
    assert any(
        tc.get("function", {}).get("name") == "execute_code"
        for m in history
        for tc in (m.get("tool_calls") or [])
    )
    assert any(
        m.get("role") == "tool" and "notes.txt" in str(m.get("content"))
        for m in history
    )

    # Step 3: the user asks how it went; the brain asks the completed action.
    question = "What files did you find, and how did you list them?"
    model.next(
        turn(steer("ask", handle_id, question=question), wait()),
        turn(say("I listed the workspace files: notes.txt (18 bytes)."), wait()),
    )
    asked = await cm.step_until_wait(
        UnifyMessageReceived(content="How did you find those files?"),
    )
    (ask_event,) = [
        a
        for a in cm.cm.completed_actions[handle_id]["handle_actions"]
        if a["action_name"] == f"ask_{handle_id}"
    ]
    assert ask_event["query"] == question and ask_event["status"] == "completed"
    assert ask_event["response"].startswith("Error:") and "ask" in ask_event["response"]
    # The answer opened another brain turn, which was shown it.
    assert asked.llm_step_count == 2
    assert ask_event["response"] in state_of(model.of("brain")[-1])

    assert_no_errors(result)
