"""Symbolic: ``unify act`` driven over a piped stdin, as the benchmark runners do.

Deleting the chat command's code (443ab7a43) also removed the
``@contextlib.contextmanager`` line of ``unify.cli._channel_pump``, restored
in 92ccfbb7c. That pump runs only when stdin is not a terminal, which is how
every benchmark runner drives the CLI, so every launch there would have failed
with "'generator' object does not support the context manager protocol",
while the in-process tests of the session (which replace ``Act.start`` or
``sys.stdin``) did not show what a launch sees.

This test mirrors the runners' launch path end to end: the argv the
continual-arc-baselines adapter builds (``UnifyAgentLearner._argv``, copied in
``tests/test_cli_external_launchers.py`` and taken from there, so the two
cannot drift) starts ``unify`` in a real subprocess through its console
entry point (``unify.__main__.main``, reading ``sys.argv``), with stdin and
stdout as pipes and the real ``Act.start``. The adapter waits for the first
``response`` line, then ends the session with ``{"quit": true}`` and closes
stdin; a bare end of input ends it the same way.

Keyless and deterministic: the child installs the scripted transport of
``tests/cache_discipline_helpers.py`` before it calls the entry point, so no
request leaves the process. Its one session reply is the plain text "done"
(no tool call); the end-of-session storage review is answered "Nothing worth
storing.". No provider key reaches the child, and ``.env`` loading is off.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.test_cli_external_launchers import CONTINUAL_ARC_BASELINES_UNIFY_AGENT

pytestmark = pytest.mark.no_unify_context

REPO_ROOT = Path(__file__).resolve().parent.parent
# Bounds: the first answer includes a cold interpreter importing the actor;
# the session's end includes its storage review.
FIRST_RESPONSE_S = 180
EXIT_S = 60
FORBIDDEN = ("context manager protocol", "Traceback")

# The child: script the model, then start ``unify`` as its console script
# does. argv: ``-c <calls file> <unify argv...>``.
CHILD = r"""
import json, sys

from tests import cache_discipline_helpers as h

calls_path = sys.argv[1]
session_replies = ["done"]


def is_review(messages):
    # As tests/actor/code_act/test_provider_error_finish.py::_is_review.
    text = json.dumps(messages, default=str)
    return (
        "## Storage Review" in text
        or "You are a skill librarian" in text
        or "This is the curation step that follows" in text
    )


async def model(*, shared_session=None, client=None, **kw):
    review = is_review(kw.get("messages") or [])
    with open(calls_path, "a") as f:
        f.write(json.dumps({"review": review}) + "\n")
    if review:
        return h.completion(content="Nothing worth storing.")
    if not session_replies:
        raise AssertionError("no scripted session reply left")
    return h.completion(content=session_replies.pop(0))


with h.scripted(()):
    import unillm.clients.uni_llm as uni_llm

    uni_llm._acompletion_with_transient_retry = model
    from unify.__main__ import main

    sys.argv = ["unify", *sys.argv[2:]]
    code = main()
sys.exit(code)
"""


def _argv(home: Path) -> list[str]:
    """The adapter's argv, with this test's home in place of the fixture's."""
    argv = list(CONTINUAL_ARC_BASELINES_UNIFY_AGENT)
    argv[argv.index("--home") + 1] = str(home)
    return argv


def _child_env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        # Keyless: no provider key, and the store is the home's own.
        if not k.endswith("_API_KEY") and k != "UNIFY_STORE_PATH"
    }
    env.update(
        UNIFY_VALIDATE_LLM_PROVIDERS="false",
        PYTHON_DOTENV_DISABLED="1",
        PYTHONPATH=os.pathsep.join(
            [str(REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])],
        ),
        PYTHONUNBUFFERED="1",
    )
    return env


def _kill(proc: subprocess.Popen) -> None:
    """End the child's whole session (its sandbox worker too), then reap it."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=30)


@needs_bwrap
@pytest.mark.timeout(FIRST_RESPONSE_S + EXIT_S + 120)
@pytest.mark.parametrize("ending", ["quit", "eof"])
def test_act_over_a_piped_stdin_answers_and_exits(ending, unify_home, tmp_path):
    calls = tmp_path / "calls.jsonl"
    stderr_path = tmp_path / "stderr.log"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with open(stderr_path, "w") as stderr:
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD, str(calls), *_argv(unify_home)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            cwd=workspace,
            env=_child_env(),
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    lines: list[str] = []
    events: queue.Queue = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            events.put(line)
        events.put(None)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    def responses() -> list[dict]:
        out = []
        for line in lines:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get("type") == "response":
                out.append(item)
        return out

    exit_code = None
    try:
        # As the adapter: wait for the first answer, then end the session.
        deadline = time.monotonic() + FIRST_RESPONSE_S
        while not responses() and time.monotonic() < deadline:
            try:
                if events.get(timeout=0.5) is None:
                    break  # stdout closed: the child has ended
            except queue.Empty:
                pass
        if proc.poll() is None:
            try:
                if ending == "quit":
                    proc.stdin.write(json.dumps({"quit": True}) + "\n")
                    proc.stdin.flush()
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            exit_code = proc.wait(timeout=EXIT_S)
        except subprocess.TimeoutExpired:
            pass
    finally:
        _kill(proc)
        reader.join(10)
    stderr_text = stderr_path.read_text(errors="replace")
    model_calls = (
        [json.loads(line) for line in calls.read_text().splitlines()]
        if calls.exists()
        else []
    )
    context = (
        f"exit code {exit_code}; model calls {model_calls}; "
        f"stdout {lines[-20:]}; stderr tail {stderr_text[-4000:]}"
    )

    for marker in FORBIDDEN:
        assert not any(marker in line for line in lines), context
        assert marker not in stderr_text, context
    answered = responses()
    assert len(answered) == 1, context
    assert answered[0]["content"] == "done", context
    assert exit_code == 0, context
    # The one scripted reply answered the request; any other call was the
    # session's storage review.
    assert [c for c in model_calls if not c["review"]] == [{"review": False}], context
