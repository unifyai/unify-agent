"""
tests/conversation_manager/test_steer_action.py
====================================================

Tests that verify ConversationManager's steering tools reach the in-flight
actions started via `act`.

These tests follow a pattern:
1. User sends a request; the brain starts an action with `act`
2. User sends distractor small talk; the brain only replies
3. User sends a steering command; the brain steers the action (ask, stop,
   pause, resume, interject)
4. We verify the steer reached the action's handle, and what the brain is
   shown about the action afterwards

The brain's model is scripted (``scripted_brain``): which steer a live model
picks for a phrasing is a question about the model. The actor is the
SimulatedActor, whose ``ask`` and ``interject`` make a model call of their
own (the ``query_llm`` kind), scripted here too.
"""

import functools

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.scripted_brain import (
    act,
    prompt_of,
    say,
    state_of,
    steer,
    turn,
    user_texts,
    wait,
)
from tests.legacy.conversation_manager.cm_helpers import (
    assert_act_triggered,
    assert_efficient,
    assert_steering_called,
    build_cm_context,
    filter_events_by_type,
    get_in_flight_action_count,
    has_steering_tool_call,
)
from tests.assertion_helpers import assertion_failed
from tests.scripted_model import reply
from unify.legacy.conversation_manager.events import (
    ActorHandleStarted,
    UnifyMessageReceived,
    UnifyMessageSent,
)
from unify.legacy.handle_state import get_handle_paused_state
from tests.baked_defaults import as_shipped  # noqa: F401

# Actions stay in-flight indefinitely with steps=None, duration=None.
# Tests check what the steering tools did to the action; actions are
# completed via trigger_completion() in test cleanup.
# as_shipped: deleted in step 5 (steering)
pytestmark = [pytest.mark.usefixtures("as_shipped")]


async def _start_action(cm, brain, request: str, query: str, ack: str):
    """The user's *request* makes the brain start an action running *query*."""
    brain.next(turn(act(query), say(ack), wait()))
    result = await cm.step_until_wait(UnifyMessageReceived(content=request))
    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Initial request should trigger act",
        cm=cm,
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result),
    )
    return result


async def _chat(cm, brain, message: str, answer: str):
    """Small talk: the brain only replies."""
    brain.next(turn(say(answer), wait()))
    return await cm.step_until_wait(UnifyMessageReceived(content=message))


def _action(cm, handle_id: int = 0) -> dict:
    """The action's record, in flight or completed."""
    data = cm.cm.in_flight_actions.get(handle_id) or cm.cm.completed_actions.get(
        handle_id,
    )
    assert data is not None, f"no action {handle_id}"
    return data


def _history(cm, handle_id: int = 0) -> list[str]:
    """The names of the events in the action's history, in order."""
    return [a["action_name"] for a in _action(cm, handle_id)["handle_actions"]]


def _sent(result) -> list[str]:
    return [e.content for e in filter_events_by_type(result.output_events, UnifyMessageSent)]


def _in_flight_render(cm) -> str:
    """The in-flight pane the brain is shown on its next turn."""
    return cm.cm.prompt_renderer.render_in_flight_actions(cm.cm.in_flight_actions)


# ---------------------------------------------------------------------------
#  Ask steering tests - querying action status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_ask_task_status_after_small_talk(initialized_cm, scripted_brain):
    """
    User asks about task status after some small talk.

    Flow:
    1. User requests a CSV review (the brain starts an action)
    2. User sends small talk distractor (the action is left alone)
    3. User asks about task status
    4. Verify the ask reaches the action and its answer reaches the user
    """
    cm = initialized_cm
    query = "List the column names of every CSV file in the user's workspace."
    answer = "Two of the five CSV files are done; sales.csv has date, region, amount."
    brain = scripted_brain(query_llm=[reply(answer)])

    # Step 1: Start an action that triggers act
    result1 = await _start_action(
        cm,
        brain,
        "Go through every CSV in my workspace and list the column names.",
        query,
        "On it, going through the CSV files now.",
    )

    # Step 2: Small talk distractor (should not affect the action)
    result2 = await _chat(
        cm,
        brain,
        "Thanks! Crazy week so far haha.",
        "Ha, hang in there! The CSV review is still running.",
    )
    assert _history(cm) == ["act_started"]
    handle = _action(cm)["handle"]
    assert not handle.done() and get_handle_paused_state(handle) is False

    # Step 3: Ask about task status
    question = "How far along is the CSV review, and what have you found?"
    relay = f"Progress so far: {answer}"
    brain.next(
        turn(steer("ask", 0, question=question), wait()),
        turn(say(relay), wait()),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="How's that CSV review going? Any progress?",
        ),
    )

    # Verify: the brain called the ask_* tool
    assert_steering_called(
        cm,
        "ask_",
        "Status query should call ask_* steering tool",
        result=result3,
    )
    # The question reached the action: the actor was asked it about its request.
    (asked,) = brain.of("query_llm")
    prompt = prompt_of(asked)
    assert query in prompt and question in prompt
    # The answer is recorded on the action and relayed to the user.
    (ask_event,) = [
        a for a in _action(cm)["handle_actions"] if a["action_name"] == "ask_0"
    ]
    assert (ask_event["query"], ask_event["status"], ask_event["response"]) == (
        question,
        "completed",
        answer,
    )
    assert _sent(result3) == [relay]

    # Efficiency: Step 2 is pure small talk (acknowledge + wait)
    # Step 3 uses async ask_* which takes an extra turn (result relay)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: small talk")
    assert_efficient(result3, 5, "Step 3: ask status (async)")


@pytest.mark.asyncio
@_handle_project
async def test_ask_task_progress_mid_conversation(initialized_cm, scripted_brain):
    """
    User asks for progress update during ongoing conversation.

    Flow:
    1. User requests a knowledge query (the brain starts an action)
    2. User asks about something unrelated
    3. User asks specifically about the task progress

    The action opted into conversation context (the default), so the ask
    carries the state the brain saw, the user's latest message included.
    """
    cm = initialized_cm
    answer = "Refunds within 30 days for unopened items; store credit after that."
    brain = scripted_brain(query_llm=[reply(answer)])

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "What's our company's refund policy? I need the details.",
        "Find the company's refund policy and its details.",
        "Let me look up the refund policy.",
    )

    # Step 2: Small talk distractor
    result2 = await _chat(
        cm,
        brain,
        "Ugh, Mondays am I right?",
        "Mondays are rough! Still on the refund policy.",
    )

    handle = _action(cm)["handle"]
    contexts: list = []
    original_ask = handle.ask

    @functools.wraps(original_ask)
    async def capturing_ask(question, **kwargs):
        contexts.append(kwargs.get("_parent_chat_context"))
        return await original_ask(question, **kwargs)

    handle.ask = capturing_ask

    # Step 3: Ask about action progress
    brain.next(
        turn(steer("ask", 0, question="What have you found so far?"), wait()),
        turn(say(f"So far: {answer}"), wait()),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="What have you found so far about the refund policy?",
        ),
    )

    assert_steering_called(
        cm,
        "ask_",
        "Progress query should call ask_* steering tool",
        result=result3,
    )
    # The ask carried the state the brain saw on that turn.
    (context,) = contexts
    assert len(context) == 1 and context[0].get("_cm_state_snapshot") is True
    assert "What have you found so far about the refund policy?" in context[0]["content"]
    assert _sent(result3) == [f"So far: {answer}"]

    # Efficiency: Step 2 is pure small talk (acknowledge + wait)
    # Step 3 uses async ask_* which takes an extra turn (result relay)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: small talk")
    assert_efficient(result3, 5, "Step 3: ask progress (async)")


# ---------------------------------------------------------------------------
#  Stop steering tests - canceling actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_stop_task_after_small_talk(initialized_cm, scripted_brain):
    """
    User stops a task after some small talk.

    Flow:
    1. User requests a web search (the brain starts an action)
    2. User sends small talk
    3. User cancels the task
    4. Verify the stop reaches the action and the action leaves the in-flight set
    """
    cm = initialized_cm
    query = "Search the web for the latest news about AI regulations."
    brain = scripted_brain()

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "Search the web for the latest news about AI regulations.",
        query,
        "Searching for the latest AI regulation news.",
    )

    # Step 2: Small talk distractor
    result2 = await _chat(
        cm,
        brain,
        "Man, what a long week it's been.",
        "It sure sounds like it! The search is underway.",
    )

    # Step 3: Stop the action
    reason = "The user found what they needed."
    brain.next(
        turn(steer("stop", 0, reason=reason), say("Okay, I've cancelled it."), wait()),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Never mind, cancel that search. I found what I needed.",
        ),
    )

    assert_steering_called(
        cm,
        "stop_",
        "Cancel request should call stop_* steering tool",
        result=result3,
    )
    # The stop reached the handle, with its reason, and the action completed.
    handle = _action(cm)["handle"]
    assert handle.done()
    assert await handle.result() == f"Stopped '{query}' for reason: {reason}"
    assert 0 not in cm.cm.in_flight_actions and 0 in cm.cm.completed_actions
    (stop_event,) = [
        a for a in _action(cm)["handle_actions"] if a["action_name"] == "stop_0"
    ]
    assert stop_event["query"] == reason
    assert _sent(result3) == ["Okay, I've cancelled it."]

    # Efficiency: Step 2 is pure small talk (acknowledge + wait)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: small talk")
    assert_efficient(result3, 3, "Step 3: cancel request")


@pytest.mark.asyncio
@_handle_project
async def test_stop_task_change_of_mind(initialized_cm, scripted_brain):
    """
    User changes their mind and stops a task.

    Flow:
    1. User requests a vendor lookup
    2. User sends a small-talk distractor
    3. User decides to cancel

    When the user cancels, the brain is shown the action as executing, with
    its stop tool; once stopped, nothing is left in flight.
    """
    cm = initialized_cm
    query = "Find the Acme vendor's phone number and email in the vendors file."
    brain = scripted_brain()

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "Look up the Acme vendor's phone number and email address in the vendors file in my workspace.",
        query,
        "Looking up Acme's contact details.",
    )

    # Step 2: Small talk distractor
    result2 = await _chat(
        cm,
        brain,
        "Sorry, got distracted for a sec there.",
        "No worries! I'm still on the Acme lookup.",
    )

    # Step 3: Cancel
    turns_before = len(brain.of("brain"))
    brain.next(
        turn(
            steer("stop", 0, reason="The user will call the vendor directly."),
            say("Sure, I've dropped the lookup."),
            wait(),
        ),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="You know what, forget about that lookup. I'll just call them now.",
        ),
    )

    assert has_steering_tool_call(cm, "stop_"), (
        "Cancellation should call the stop_* steering tool. "
        f"Got: {cm.all_tool_calls}"
    )
    # What the brain saw when the user cancelled: the action, executing,
    # with the tool that stops it.
    cancel_state = state_of(brain.of("brain")[turns_before])
    assert "<action id='0'" in cancel_state and "status='executing'" in cancel_state
    assert f"<original_request>{query}</original_request>" in cancel_state
    assert "stop_action(handle_id=0, ...)" in cancel_state
    # Once stopped, the action is done and no longer in flight.
    assert _action(cm)["handle"].done()
    assert get_in_flight_action_count(cm) == 0
    assert "No actions currently executing." in _in_flight_render(cm)

    # Efficiency: Step 2 is pure small talk (acknowledge + wait)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: small talk")
    assert_efficient(result3, 3, "Step 3: cancel request")


# ---------------------------------------------------------------------------
#  Pause steering tests - temporarily halting actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_pause_task_for_meeting(initialized_cm, scripted_brain):
    """
    User pauses a task because they need to step away.

    Flow:
    1. User requests a research task
    2. User mentions something unrelated
    3. User asks to hold the task (natural language, no "pause" word)
    4. Verify the pause reaches the action, which stays in flight
    """
    cm = initialized_cm
    brain = scripted_brain()

    # Step 1: Start a research action
    result1 = await _start_action(
        cm,
        brain,
        "Research our competitors' pricing strategies and summarize.",
        "Research competitors' pricing strategies and summarize them.",
        "I'll research competitor pricing.",
    )

    # Step 2: Small talk distractor
    result2 = await _chat(
        cm,
        brain,
        "This coffee is terrible lol.",
        "Oh no! Hope the research makes up for it.",
    )

    # Step 3: Pause request (natural language)
    brain.next(
        turn(steer("pause", 0), say("Paused. Good luck in the meeting!"), wait()),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Put that on hold for now. I need to step into a meeting.",
        ),
    )

    assert_steering_called(
        cm,
        "pause_",
        "'Put on hold' should call pause_* steering tool",
        result=result3,
    )
    # The pause reached the handle; the action is paused, not finished.
    handle = _action(cm)["handle"]
    assert get_handle_paused_state(handle) is True and not handle.done()
    assert 0 in cm.cm.in_flight_actions
    assert _history(cm) == ["act_started", "pause_0"]

    # Efficiency: Step 2 is pure small talk (acknowledge + wait)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: small talk")
    assert_efficient(result3, 3, "Step 3: pause request")


@pytest.mark.asyncio
@_handle_project
async def test_pause_task_hold_on(initialized_cm, scripted_brain):
    """
    User says hold on, and the brain pauses the task.

    Flow:
    1. User requests a notes search
    2. User says hold on (natural language, no "pause" word)

    After the pause, the brain's next view shows the action paused and
    offers resume instead of pause.
    """
    cm = initialized_cm
    brain = scripted_brain()

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "Search my notes files for anything about the project deadline.",
        "Search the user's notes files for anything about the project deadline.",
        "Searching your notes for the deadline.",
    )
    assert "status='executing'" in _in_flight_render(cm)

    # Step 2: Hold on (natural language)
    brain.next(turn(steer("pause", 0), say("Sure, I've paused the search."), wait()))
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Wait, hold on a second. Let me think about what I actually need.",
        ),
    )

    assert_steering_called(
        cm,
        "pause_",
        "'Hold on' should call pause_* steering tool",
        result=result2,
    )
    assert get_handle_paused_state(_action(cm)["handle"]) is True
    pane = _in_flight_render(cm)
    assert "<action id='0'" in pane and "status='paused'" in pane
    assert "resume_action(handle_id=0, ...)" in pane
    assert "pause_action(handle_id=0, ...)" not in pane

    # Efficiency: Step 2 is explicit pause request (pause + wait)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: hold on request")


# ---------------------------------------------------------------------------
#  Resume steering tests - continuing paused actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_resume_after_pause(initialized_cm, scripted_brain):
    """
    User resumes a task after pausing it.

    Flow:
    1. User requests a task
    2. User puts it on hold (natural language, no "pause" word)
    3. User asks to continue (natural language, no "resume" word)
    """
    cm = initialized_cm
    brain = scripted_brain()

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "Find every file in my workspace I haven't touched in over a month.",
        "List every workspace file not modified in over a month.",
        "Looking for files untouched in over a month.",
    )
    handle = _action(cm)["handle"]

    # Step 2: Hold (natural language)
    brain.next(turn(steer("pause", 0), say("Paused. Take your time."), wait()))
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Wait, please put a pin in that request for now, I might be able to share a few more details, hold on a moment",
        ),
    )
    assert get_handle_paused_state(handle) is True

    # Step 3: Continue (natural language)
    brain.next(turn(steer("resume", 0), say("Resuming the file search."), wait()))
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="OK, I'm back. Never mind there were no more details of importance, please continue as you were.",
        ),
    )

    assert_steering_called(
        cm,
        "resume_",
        "'Go ahead' should call resume_* steering tool",
        result=result3,
    )
    # The resume reached the handle: running again, still in flight.
    assert get_handle_paused_state(handle) is False and not handle.done()
    assert _history(cm) == ["act_started", "pause_0", "resume_0"]

    # Efficiency: Steps 2 and 3 are explicit steering requests (pause/resume + wait)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: hold request")
    assert_efficient(result3, 3, "Step 3: resume request")


@pytest.mark.asyncio
@_handle_project
async def test_resume_continue_where_left_off(initialized_cm, scripted_brain):
    """
    User asks to continue from where they left off.

    Flow:
    1. User requests a task
    2. User puts it on hold (natural language, no "pause" word)
    3. Some small talk: the action stays paused, and the brain is shown so
    4. User asks to pick up where they left off (natural language, no "resume" word)
    """
    cm = initialized_cm
    brain = scripted_brain()

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "Search my workspace for the Henderson notes and any recent drafts.",
        "Find the Henderson notes and any recent drafts in the workspace.",
        "Searching for the Henderson notes and drafts.",
    )
    handle = _action(cm)["handle"]

    # Step 2: Hold (natural language)
    brain.next(turn(steer("pause", 0), say("Holding the search."), wait()))
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Wait, hold on a sec.",
        ),
    )

    # Step 3: Small talk
    result3 = await _chat(
        cm,
        brain,
        "Sorry, had to take another call.",
        "No problem, the Henderson search is on hold.",
    )
    assert get_handle_paused_state(handle) is True
    small_talk_state = brain.last_state()
    assert "<action id='0'" in small_talk_state and "status='paused'" in small_talk_state

    # Step 4: Pick up where left off (natural language)
    brain.next(turn(steer("resume", 0), say("Picking the search back up."), wait()))
    result4 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="OK where were we? Go ahead with that search for the Henderson notes.",
        ),
    )

    assert_steering_called(
        cm,
        "resume_",
        "'Go ahead' should call resume_* steering tool",
        result=result4,
    )
    assert get_handle_paused_state(handle) is False
    assert "status='executing'" in _in_flight_render(cm)
    assert _history(cm) == ["act_started", "pause_0", "resume_0"]

    # Efficiency: Step 3 is pure explanation (acknowledge + wait)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: hold request")
    assert_efficient(result3, 3, "Step 3: small talk")
    assert_efficient(result4, 3, "Step 4: resume request")


# ---------------------------------------------------------------------------
#  Interject steering tests - providing new information to running actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interject_additional_constraint(initialized_cm, scripted_brain):
    """
    User interjects with an additional constraint for the running task.

    Flow:
    1. User requests a file search
    2. User mentions something else
    3. User adds a constraint to the search; the interjection reaches the action
    """
    cm = initialized_cm
    brain = scripted_brain(query_llm=[reply("Narrowing to the Berlin office.")])

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "Find every file in my workspace about engineering.",
        "Find every workspace file about engineering.",
        "Searching for engineering files.",
    )

    # Step 2: Small talk distractor
    result2 = await _chat(
        cm,
        brain,
        "Phew, finally caught a breather.",
        "Glad to hear it! The search is running.",
    )

    # Step 3: Interject with constraint
    message = "Only include files about people in the Berlin office."
    brain.next(
        turn(steer("interject", 0, message=message), say("Got it, Berlin only."), wait()),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Actually, for that search, only include people in the Berlin office.",
        ),
    )

    assert_steering_called(
        cm,
        "interject_",
        "Additional constraint should call interject_* steering tool",
        result=result3,
    )
    # The interjection reached the handle and the actor's own model.
    handle = _action(cm)["handle"]
    assert handle._guidance == [message]
    (interjected,) = brain.of("query_llm")
    assert user_texts(interjected)[-2] == message
    (event,) = [
        a for a in _action(cm)["handle_actions"] if a["action_name"] == "interject_0"
    ]
    assert event["query"] == message

    # Efficiency: Step 2 is pure small talk (acknowledge + wait)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: small talk")
    assert_efficient(result3, 3, "Step 3: interject constraint")


@pytest.mark.asyncio
@_handle_project
async def test_interject_extension(initialized_cm, scripted_brain):
    """
    User interjects to extend/add to the task.

    Flow:
    1. User requests a knowledge query
    2. User expresses anticipation (neutral small talk)
    3. User extends the request with additional requirements

    The action takes the extension into its work: its result reports the
    interjection as guidance it applied.
    """
    cm = initialized_cm
    brain = scripted_brain(query_llm=[reply("Adding the attached notes.")])
    query = "Summarize what the Q3 revenue report says."

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "What's the Q3 revenue report say?",
        query,
        "Pulling up the Q3 revenue report.",
    )

    # Step 2: Anticipation (neutral, should not trigger pause/stop)
    result2 = await _chat(
        cm,
        brain,
        "Looking forward to seeing the results.",
        "Me too, I'll share them as soon as they're in.",
    )
    assert _history(cm) == ["act_started"]

    # Step 3: Extension via interjection (clearly adding to existing action)
    message = "Also include any notes or comments attached to the report."
    brain.next(
        turn(steer("interject", 0, message=message), say("Will do."), wait()),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Also include any notes or comments attached to the report.",
        ),
    )

    assert has_steering_tool_call(cm, "interject_"), (
        "Extension should call the interject_* steering tool. "
        f"Got: {cm.all_tool_calls}"
    )
    # When the action finishes, its result reports the extension as applied.
    handle = _action(cm)["handle"]
    handle.trigger_completion()
    result = await handle.result()
    assert result.startswith(f"Completed '{query}'")
    assert f"Guidance applied: '{message}'" in result

    # Efficiency: Step 2 is pure anticipation (acknowledge + wait)
    assert_efficient(result1, 5, "Step 1: initial action")
    assert_efficient(result2, 5, "Step 2: anticipation")
    assert_efficient(result3, 6, "Step 3: extension")


@pytest.mark.asyncio
@_handle_project
async def test_interject_new_priority(initialized_cm, scripted_brain):
    """
    User interjects to change priority or focus of the task.

    Flow:
    1. User requests a broad task
    2. Some conversation
    3. User narrows the focus

    The interjection carries what the action has not seen since it started:
    the conversation after its request, not the request again.
    """
    cm = initialized_cm
    brain = scripted_brain(query_llm=[reply("Focusing on pricing.")])

    # Step 1: Start a broad action
    result1 = await _start_action(
        cm,
        brain,
        "What's the Q3 revenue report say?",
        "Summarize what the Q3 revenue report says.",
        "Pulling up the Q3 revenue report.",
    )

    # Step 2: Conversation
    result2 = await _chat(
        cm,
        brain,
        "I just got out of a meeting.",
        "Welcome back! The report summary is in progress.",
    )

    handle = _action(cm)["handle"]
    contexts: list = []
    original_interject = handle.interject

    @functools.wraps(original_interject)
    async def capturing_interject(message, **kwargs):
        contexts.append(kwargs.get("_parent_chat_context_cont"))
        return await original_interject(message, **kwargs)

    handle.interject = capturing_interject

    # Step 3: Narrow focus
    brain.next(
        turn(
            steer("interject", 0, message="Focus specifically on pricing; it's urgent."),
            say("Understood, focusing on pricing."),
            wait(),
        ),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="For that research, focus specifically on pricing - that's the most urgent.",
        ),
    )

    assert has_steering_tool_call(cm, "interject_"), (
        "Focus change should call the interject_* steering tool. "
        f"Got: {cm.all_tool_calls}"
    )
    # The interjection's context is the conversation since the action started.
    (context,) = contexts
    (diff,) = context
    assert diff["role"] == "user" and diff.get("_cm_context_diff") is True
    content = diff["content"]
    assert "<new_messages>" in content
    assert "I just got out of a meeting." in content
    assert "focus specifically on pricing - that's the most urgent." in content
    assert "What's the Q3 revenue report say?" not in content
    assert handle._guidance == ["Focus specifically on pricing; it's urgent."]

    # Efficiency: Step 2 is pure statement (acknowledge + wait)
    assert_efficient(result1, 5, "Step 1: initial action")
    assert_efficient(result2, 5, "Step 2: conversation")
    assert_efficient(result3, 6, "Step 3: narrow focus")


# ---------------------------------------------------------------------------
#  Complex steering scenarios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_pause_interject_resume_sequence(initialized_cm, scripted_brain):
    """
    User pauses, adds new information, then resumes.

    This tests the full pause -> interject -> resume flow: each steer reaches
    the action in order, and the interjection does not resume it.
    """
    cm = initialized_cm
    message = "Focus specifically on agile methodology."
    brain = scripted_brain(query_llm=[reply("Noted: agile methodology.")])

    # Step 1: Start an action (web search)
    result1 = await _start_action(
        cm,
        brain,
        "Search the web for information about project management best practices.",
        "Search the web for project management best practices.",
        "Searching for project management best practices.",
    )
    handle = _action(cm)["handle"]

    # Step 2: Pause
    brain.next(turn(steer("pause", 0), say("Search paused."), wait()))
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Hold on, put that search on hold.",
        ),
    )

    # Step 3: Interject while paused
    brain.next(
        turn(
            steer("interject", 0, message=message),
            say("Noted, I'll focus on agile once it resumes."),
            wait(),
        ),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Actually, focus specifically on agile methodology.",
        ),
    )
    assert get_handle_paused_state(handle) is True
    assert handle._guidance == [message]

    # Step 4: Resume
    brain.next(turn(steer("resume", 0), say("Resuming the search."), wait()))
    result4 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="OK, go ahead with the search now.",
        ),
    )

    has_pause = has_steering_tool_call(cm, "pause_")
    has_interject = has_steering_tool_call(cm, "interject_")
    has_resume = has_steering_tool_call(cm, "resume_")
    assert has_pause and has_interject and has_resume, assertion_failed(
        expected="All three steering tools (pause_, interject_, resume_)",
        actual=f"pause_: {has_pause}, interject_: {has_interject}, resume_: {has_resume}",
        reasoning=[],
        description="Pause->interject->resume sequence should trigger steering tools",
        context_data=build_cm_context(cm=cm, result=result4),
    )
    # Each steer reached the action, in order, and it is running again.
    assert _history(cm) == ["act_started", "pause_0", "interject_0", "resume_0"]
    assert get_handle_paused_state(handle) is False and not handle.done()

    # Efficiency: All steps are explicit steering requests
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: pause request")
    assert_efficient(result3, 3, "Step 3: interject while paused")
    assert_efficient(result4, 3, "Step 4: resume request")


@pytest.mark.asyncio
@_handle_project
async def test_multiple_distractors_then_stop(initialized_cm, scripted_brain):
    """
    Multiple distractor messages before stopping an action.

    The brain is shown the in-flight action on every distractor turn, so it
    still has the action to steer when the user stops it.
    """
    cm = initialized_cm
    query = "Generate a report on all customer interactions this month."
    brain = scripted_brain()

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "Generate a report on all customer interactions this month.",
        query,
        "Generating the customer interactions report.",
    )

    # Step 2-4: Multiple distractors
    distractors = []
    for message, answer in (
        ("It's really busy today.", "Sounds hectic! The report is in progress."),
        ("Ugh, Mondays am I right?", "Mondays are the worst."),
        ("Anyway...", "Anyway, the report is still going."),
    ):
        distractors.append(await _chat(cm, brain, message, answer))
        state = brain.last_state()
        assert "<action id='0'" in state and "status='executing'" in state
        assert f"<original_request>{query}</original_request>" in state
    result2, result3, result4 = distractors
    assert _history(cm) == ["act_started"]

    # Step 5: Stop after distractors
    brain.next(
        turn(
            steer("stop", 0, reason="Someone else is already doing it."),
            say("Stopped the report."),
            wait(),
        ),
    )
    result5 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Actually, stop that report. Someone else is already doing it.",
        ),
    )

    assert_steering_called(
        cm,
        "stop_",
        "Stop after distractors should call stop_* steering tool",
        result=result5,
    )
    assert _action(cm)["handle"].done()
    assert 0 in cm.cm.completed_actions and get_in_flight_action_count(cm) == 0

    # Efficiency: Steps 2-4 are pure chit-chat (acknowledge + wait)
    assert_efficient(result1, 3, "Step 1: initial action")
    assert_efficient(result2, 3, "Step 2: distractor 1")
    assert_efficient(result3, 3, "Step 3: distractor 2")
    assert_efficient(result4, 3, "Step 4: distractor 3")
    assert_efficient(result5, 3, "Step 5: stop request")


# ---------------------------------------------------------------------------
#  In-flight ask tests - async ask status tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_ask_shows_pending_then_completed(initialized_cm, scripted_brain):
    """
    Verify the async ask flow properly tracks pending/completed states.

    The ask operation is non-blocking: when the brain calls ask_*, the tool
    returns immediately and records the ask as pending; the action's answer
    completes it, and the brain receives another turn.

    Flow:
    1. User starts a task
    2. User asks about progress (the brain calls ask_*, which is async)
    3. Verify the ask is pending while the action answers, then completed
       with the response
    """
    cm = initialized_cm
    answer = "Found two notes: the budget review moved to Thursday."
    seen_while_answering: list = []

    def answering(call):
        seen_while_answering.append(
            [
                (a["action_name"], a.get("status"))
                for a in _action(cm)["handle_actions"]
                if a["action_name"].startswith("ask_")
            ],
        )
        return reply(answer)

    brain = scripted_brain(query_llm=[answering])

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "Search my notes files for anything about the quarterly budget review.",
        "Search the user's notes files for the quarterly budget review.",
        "Searching your notes for the budget review.",
    )

    # Step 2: Ask about progress
    brain.next(
        turn(steer("ask", 0, question="What have you found so far?"), wait()),
        turn(say(answer), wait()),
    )
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="What's the status of that search?",
        ),
    )

    # Verify ask_* was called
    assert_steering_called(
        cm,
        "ask_",
        "Progress query should call ask_* steering tool",
        result=result2,
    )

    # While the action answered, its ask was pending.
    assert seen_while_answering == [[("ask_0", "pending")]]

    # The action history now holds the ask, completed with the response.
    all_actions = {**cm._cm.in_flight_actions, **cm._cm.completed_actions}
    handle_actions = all_actions[0].get("handle_actions", [])
    ask_actions = [
        a for a in handle_actions if a.get("action_name", "").startswith("ask_0")
    ]
    assert len(ask_actions) == 1
    last_ask = ask_actions[-1]
    assert last_ask.get("status") == "completed", assertion_failed(
        expected="Ask action with status='completed'",
        actual=f"Ask action with status='{last_ask.get('status')}'",
        reasoning=["Async ask should complete and update status"],
        description="Ask action should be marked completed",
        context_data={
            "handle_actions": handle_actions,
            "last_ask": last_ask,
        },
    )
    assert last_ask.get("response") == answer, assertion_failed(
        expected=f"Ask action with response={answer!r}",
        actual=f"Ask action with response='{last_ask.get('response')}'",
        reasoning=["Async ask should have a response from the action"],
        description="Ask action should have a response",
        context_data={
            "handle_actions": handle_actions,
            "last_ask": last_ask,
        },
    )

    # Efficiency: Step 1 is a straightforward action request
    assert_efficient(result1, 3, "Step 1: initial action")


@pytest.mark.asyncio
@_handle_project
async def test_ask_response_triggers_llm_followup(initialized_cm, scripted_brain):
    """
    Verify that when the async ask completes, the brain gets another turn
    to process the response and communicate it to the user.

    Flow:
    1. User starts a task
    2. User asks about progress
    3. The brain receives the ask response on a turn of its own and sends a
       chat message with the info
    """
    cm = initialized_cm
    answer = "Three conversations mention Henderson; the latest is from Tuesday."
    brain = scripted_brain(query_llm=[reply(answer)])

    # Step 1: Start an action
    result1 = await _start_action(
        cm,
        brain,
        "Look through all my past conversations for anything mentioning the Henderson account.",
        "Find past conversations that mention the Henderson account.",
        "Searching past conversations for Henderson.",
    )

    # Step 2: Ask about progress
    relay = f"Update: {answer}"
    brain.next(
        turn(steer("ask", 0, question="How far along is the Henderson search?"), wait()),
        turn(say(relay), wait()),
    )
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="How's it going with the Henderson search?",
        ),
    )

    # Verify ask_* was called
    assert_steering_called(
        cm,
        "ask_",
        "Progress query should call ask_* steering tool",
        result=result2,
    )

    # The answer opened a second brain turn within the step, which was shown
    # the response; the ask turn itself saw it pending.
    assert result2.llm_step_count == 2
    ask_turn, followup_turn = brain.of("brain")[-2:]
    assert answer not in state_of(ask_turn)
    followup = state_of(followup_turn)
    assert "type='ask_0'" in followup and "status='completed'" in followup
    assert f"<response>{answer}</response>" in followup

    # The brain sent a chat message after receiving the ask response.
    sent_events = [e for e in result2.output_events if isinstance(e, UnifyMessageSent)]
    assert [e.content for e in sent_events] == [relay], assertion_failed(
        expected="The relay sent after the ask response",
        actual=f"{len(sent_events)} UnifyMessageSent events",
        reasoning=[
            "When async ask completes, the brain gets another turn",
            "The brain sends a chat message with the status info",
        ],
        description="The brain should communicate the ask result to the user",
        context_data={
            "output_events": [type(e).__name__ for e in result2.output_events],
        },
    )

    # Efficiency: Step 1 is a straightforward action request
    assert_efficient(result1, 3, "Step 1: initial action")
