"""
File-focused ConversationManager → CodeActActor integration tests.

These validate the production path where a user provides a file path and the actor:
- reads/parses the file (PDF/CSV fixtures)
- extracts/summarizes content
- handles missing paths gracefully
- reads downloaded attachments from Attachments/

The slow brain and the actor's model are scripted (``scripted_brain``):
what a live model writes is not the subject. The actor's cells run for
real, confined to the workspace, so the PDF and CSV fixtures are copied into
the workspace first and named by their workspace path. Each test checks
that the file's content (or its absence) reached the actor as the result of
the cell that read it, and that the action completes with the actor's
answer.
"""

import shutil
from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    scripted_brain_and_actor,
    tool_results,
    wait_for_actor_completion,
    wait_until_done,
)
from tests.legacy.conversation_manager.actions.scripted_brain import (
    act,
    say,
    turn,
    user_texts,
    wait,
)
from tests.scripted_model import cell, reply
from unify.legacy.conversation_manager.events import UnifyMessageReceived

pytestmark = [pytest.mark.integration]


def _into_workspace(source: str) -> str:
    """Copy *source* into the workspace's Documents/ folder; return the
    workspace path a user would name it by."""
    from unify.workspace import get_local_root

    relative = f"Documents/{Path(source).name}"
    target = Path(get_local_root()) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return relative


def _actor_saw(model, text: str) -> bool:
    """Whether the actor's first request carries *text* in a user message."""
    first = model.of("actor")[0]
    return any(text in t for t in user_texts(first))


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_file_summarize_pdf_by_path(
    initialized_cm_codeact,
    test_files,
    scripted_brain,
):
    """PDF summarize requests route to act() with the provided file path,
    and the actor opens the file at that path."""
    cm = initialized_cm_codeact
    pdf_path = _into_workspace(test_files["test_report.pdf"])
    summary = "- The report covers the quarter's results.\n- It closes with next steps."
    model = scripted_brain_and_actor(
        scripted_brain,
        reply(calls=[cell(f"print(open({pdf_path!r}, 'rb').read(5))")]),
        reply(summary),
    )
    query = f"Summarize the PDF at {pdf_path} in 2 bullet points."
    model.next(turn(act(query), say("Reading the PDF now."), wait()))

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content=f"Please summarize the PDF at {pdf_path} in 2 bullet points.",
        ),
    )

    actor_event = get_actor_started_event(result)
    act_query = actor_event.query.lower()
    assert (
        pdf_path.lower() in act_query or "test_report.pdf" in act_query
    ), f"Expected act() query to reference the PDF path. Got: {actor_event.query}"
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]

    final = await wait_for_actor_completion(cm, handle_id, timeout=120)
    assert final == summary
    # The actor got the path in its request, and the file opened at that path.
    assert _actor_saw(model, pdf_path)
    assert "%PDF" in tool_results(model.of("actor")[1])[-1]
    await wait_until_done(handle)
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_file_read_csv_extracts_names(
    initialized_cm_codeact,
    test_files,
    scripted_brain,
):
    """Read a CSV by file path and extract simple structured facts (rows + names)."""
    cm = initialized_cm_codeact
    csv_path = _into_workspace(test_files["test_data.csv"])
    answer = "The CSV has 2 rows; the names listed are Alice and Bob."
    model = scripted_brain_and_actor(
        scripted_brain,
        reply(
            calls=[
                cell(
                    "import csv\n"
                    f"rows = list(csv.DictReader(open({csv_path!r})))\n"
                    "print(len(rows), [r['name'] for r in rows])",
                ),
            ],
        ),
        reply(answer),
    )
    model.next(
        turn(
            act(f"Read the CSV at {csv_path}; report its row count and the names listed."),
            say("Reading the CSV."),
            wait(),
        ),
    )

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                f"Read the CSV at {csv_path} and tell me how many rows it has and the names listed."
            ),
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]
    final = await wait_for_actor_completion(cm, handle_id, timeout=120)

    # The cell read the file: its rows reached the actor.
    read = tool_results(model.of("actor")[1])[-1]
    assert "2 ['Alice', 'Bob']" in read, read
    lower = final.lower()
    assert "alice" in lower and "bob" in lower
    await wait_until_done(handle)
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_file_missing_path_returns_helpful_error(
    initialized_cm_codeact,
    scripted_brain,
):
    """Missing file path is handled gracefully: the error reaches the actor as
    its cell's result, no crash, and the action completes with its answer."""
    cm = initialized_cm_codeact
    missing = "/definitely/does/not/exist.pdf"
    answer = f"The file at {missing} does not exist, so I can't summarize it."
    model = scripted_brain_and_actor(
        scripted_brain,
        reply(calls=[cell(f"print(open({missing!r}, 'rb').read(5))")]),
        reply(answer),
    )
    model.next(
        turn(
            act(f"Summarize the file at {missing} in one sentence."),
            say("Taking a look."),
            wait(),
        ),
    )

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content=f"Summarize the file at {missing} in one sentence.",
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]
    final = await wait_for_actor_completion(cm, handle_id, timeout=120)

    # The failed read reached the actor as its cell's result.
    failed_read = tool_results(model.of("actor")[1])[-1]
    assert "FileNotFoundError" in failed_read or "No such file" in failed_read, (
        failed_read
    )
    assert final == answer
    await wait_until_done(handle)
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_chat_attachment_readable_by_actor(initialized_cm_codeact, scripted_brain):
    """Actor can read a file the user attached to a chat message.

    Mirrors the production attachment flow: the chat front end copies the
    file into Attachments/ and puts its workspace path on the message, the
    brain hands that path to the actor, and the actor reads the file at that
    workspace-relative path.
    """
    cm = initialized_cm_codeact

    from pathlib import Path

    from unify.workspace import get_local_root

    attachments_dir = Path(get_local_root()) / "Attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    saved_path = "Attachments/att-notes-1_meeting_notes.txt"
    (Path(get_local_root()) / saved_path).write_bytes(
        b"Project Aurora kickoff meeting\n"
        b"Attendees: Sarah Chen, Marcus Webb, Priya Patel\n"
        b"Decision: launch date set for March 15th\n"
        b"Action item: Marcus to prepare the budget forecast by Friday\n",
    )
    answer = (
        "Sarah Chen, Marcus Webb and Priya Patel attended; the launch date is "
        "March 15th."
    )
    model = scripted_brain_and_actor(
        scripted_brain,
        reply(calls=[cell(f"print(open({saved_path!r}).read())")]),
        reply(answer),
    )
    model.next(
        turn(
            act(
                f"Read {saved_path} and report who attended the meeting and the "
                "launch date.",
            ),
            say("Reading the meeting notes."),
            wait(),
        ),
    )

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Who attended this meeting and what was the launch date?",
            attachments=[saved_path],
        ),
    )

    # The brain was shown the attachment's workspace path.
    assert f"[Attachments: {saved_path}]" in model.last_state()

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]
    final = await wait_for_actor_completion(cm, handle_id, timeout=120)

    # The actor read the attachment at its workspace path.
    assert _actor_saw(model, saved_path)
    read = tool_results(model.of("actor")[1])[-1]
    assert "Attendees: Sarah Chen" in read and "March 15th" in read, read

    lower = final.lower()
    assert (
        "sarah" in lower or "chen" in lower
    ), f"Expected actor to find attendee 'Sarah Chen' in the file. Got: {final}"
    assert (
        "march" in lower or "15" in lower
    ), f"Expected actor to find launch date 'March 15th' in the file. Got: {final}"
    await wait_until_done(handle)
