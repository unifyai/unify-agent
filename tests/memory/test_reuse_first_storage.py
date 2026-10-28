"""Reuse-first storage as mechanics, never as advice; short notes served by default.

* No brief tells an agent what to prefer storing (``test_no_storage_steering`` scans for that). What the tools do is
  stated: a stored procedure called with no arguments runs on the current request, so a capture kept with its own
  values re-runs the accepted work on a new request in one call; ``memory.get`` serves a note's ``## Rule`` section.
* A captured procedure called with no arguments runs on the current request.
"""

from __future__ import annotations

from unify.memory import library_helper as lh
from unify.memory import library_prompts as lp
from unify.memory.integration import fork


def test_the_fork_is_told_what_a_no_argument_call_does_and_nothing_about_what_to_keep():
    text = " ".join(fork.CAPTURE_BULLETS.split())
    assert (
        "Called later with no arguments, a stored procedure of the `read_input` and `run(value)` form or of the "
        "`run(state)` form runs on the current request" in text
    )
    assert "one of the `run(state, env)` form is not run that way" in text
    assert "read it from `state` or the request instead" not in text and "default" not in text


def test_the_writer_is_told_what_memory_get_serves_of_a_note():
    brief = " ".join(lp.write_brief_now(helper=False).split())
    assert f"`memory.get` serves a note's front matter and its `## Rule` section, at most {lh.NOTE_RULE_CHARS}" in brief
    assert "What you store is your decision" in brief


NOTE = (
    "---\ntitle: Sorting rows\ndescription: what to check\nuses: []\n---\n"
    "## Rule\n- Read the id column as an integer.\n\n"
    "## Evidence\n" + "".join(f"- episode e{i}: sorted by int (`cell:0`)\n" for i in range(400))
)


def test_an_agent_is_served_a_notes_rule_with_the_path_to_the_rest():
    served = lh._note_rule(NOTE, "notes/files/sort-rows.md")
    assert "Read the id column as an integer." in served and "## Evidence" not in served
    assert "title: Sorting rows" in served
    assert served.rstrip().endswith(f"tools.read('{lh._ROOT.rstrip('/')}/notes/files/sort-rows.md'))")
    old = "No sections here.\n" + "x" * 5000
    bounded = lh._note_rule(old, "notes/a/b.md")
    assert len(bounded) <= lh.NOTE_RULE_CHARS + 200 and "the rest of the note" in bounded
    short = "## Rule\n- one line\n"
    assert lh._note_rule(short, "notes/a/c.md") == short


def test_a_captured_procedure_called_with_no_arguments_runs_on_the_current_request():
    cap = lh.Item(id="capture:" + "a" * 16, kind="function", summary="", applies="", evidence="unchecked",
                  status="on-trial", example="", text="", links=(), history=())  # fmt: skip
    assert cap._on_request() is cap
