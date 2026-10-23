"""Every path or call the harness prints to an agent opens for the agent it is shown to, and every refusal names a
next step. Paths are resolved by the harness: an agent never has to translate one."""

from __future__ import annotations

import re

import pytest

from unify.actor import tool_handlers as H
from unify.memory import librarian as L
from unify.memory import library_helper as lh
from unify.memory import tool_spec
from unify.memory.writer_pass import _natural_path


def _tail_path(text: str) -> str:
    return re.search(r"tools\.read\('([^']+)'\)", text).group(1)


def test_a_notes_rest_path_names_the_library_as_the_boxs_texts_do(monkeypatch):
    """The pointer names the library by the root the box's texts give it: ``/memory-library`` in the actor's and a
    teammate's box, whichever pin of the copy the package was imported from (under in_episode, a pin inside the
    mount); ``/memory`` in the writer's and the librarian's. That each opens with the agent's own ``tools.read`` is
    tested in real boxes (``test_memory_paths_one_way.test_every_printed_path_opens``)."""
    long = "## Rule\n" + "x" * 5000
    for root in (H.LIBRARY, f"{H.LIBRARY}/.pins/0-0123456789ab"):
        monkeypatch.setattr(lh, "_ROOT", root)
        assert _tail_path(lh._note_rule(long, "notes/a/b.md")) == f"{H.LIBRARY}/notes/a/b.md"
    monkeypatch.setattr(lh, "_ROOT", "/memory")
    assert _tail_path(lh._note_rule(long, "notes/a/b.md")) == "/memory/notes/a/b.md"


def test_the_calls_the_librarian_and_the_cards_print_go_through_memory():
    block = L.start_block({"fits": [{"id": "capture:" + "a" * 16, "call": "memory.get('capture:" + "a" * 16 + "')()",
                                     "why": "it fits"}]})  # fmt: skip
    call = block.splitlines()[2].split("call: ", 1)[1]
    assert L._call_problem(call) == ""
    by_hand = lh.BY_HAND.format(cid="capture:" + "a" * 16)
    shown = "print(memory.get('capture:" + "a" * 16 + "').text)"
    # the card's way to an acting capture's work is a call the agent can make: it shows the code, nothing is executed
    assert shown in by_hand and "exec(" not in by_hand and L._call_problem(shown) == ""


def test_every_refusal_of_a_path_names_a_next_step():
    for tool in ("read", "grep"):
        text = tool_spec.refusal(tool, "outside_roots", path="/x")
        assert "memory.get(" in text and "tools.take(" in text, text
    assert "/inputs/attention/index.json" in tool_spec.refusal("read", "not_found", path="/x")
    from unify.common import tool_registry as R

    assert "memory.get(" in R._WORKSPACE_LIBRARY_READ_TEXT["refusals"]["outside_roots"]


def test_the_writer_reads_a_path_in_its_natural_form(tmp_path):
    mem, inputs = tmp_path / "memory", tmp_path / "inputs"
    (mem / "notes" / "a").mkdir(parents=True)
    (mem / "notes" / "a" / "b.md").write_text("x")
    (inputs / "attention").mkdir(parents=True)
    (inputs / "attention" / "index.json").write_text("{}")
    roots = {"/memory": mem, "/inputs": inputs}
    assert _natural_path("notes/a/b.md", roots) == "/memory/notes/a/b.md"
    assert _natural_path("memory/notes/a/b.md", roots) == "/memory/notes/a/b.md"
    assert _natural_path("attention/index.json", roots) == "/inputs/attention/index.json"
    assert _natural_path("/inputs/attention/index.json", roots) == "/inputs/attention/index.json"
    assert _natural_path("../etc/passwd", roots) == "../etc/passwd"  # nothing outside a root is reached


#: What makes a refusal a next step: a call the agent can make, an agent to message, or the valid values or roots.
NEXT_STEP = re.compile(
    r"\b(?:tools|memory)\.\w+\(|@\w+|\bop='|\bone of\b|\bmust be\b|\bValid: |\bthis batch: "
    r"|\bis outside (?:/|the library)|\bon names one\b|\bonly '\w+'|'\w+' applies only|\bis a positive whole number"
)
#: Each role's refusals of reading a path, an entry, an episode or a part, and of asking the librarian, by form.
FORMS = {
    ("actor", "read"): ("outside_roots", "not_in_library"),
    ("actor", "grep"): ("outside_roots", "not_in_library"),
    ("teammate", "read"): ("outside_roots", "not_in_library"),
    ("teammate", "grep"): ("outside_roots", "not_in_library"),
    ("librarian", "read"): ("outside_library", "not_found"),
    ("librarian", "grep"): ("outside_library",),
    ("writer", "read"): ("outside_roots", "outside_box", "not_found", "outside_library", "not_in_library"),
    ("writer", "grep"): ("outside_roots", "outside_box", "outside_library", "not_in_library"),
    ("helper", "read"): ("outside_box",),
    ("writer", "pool"): ("bad_op", "bad_status", "unknown_entry", "closed_entry", "open_entry"),
    ("writer", "read_episode"): ("unknown_episode", "unknown_part", "bad_parts", "bad_offset"),
    ("actor", "ask_library"): ("no_question", "bad_schema", "bad_failed"),
}
FIELDS = dict(path="/x/none.md", episode="e9", batch="e1, e2", part="request", names="cell:9", valid="request, cell:0",
              id="ab12", status="merged", reason="r", why="w", error="e")  # fmt: skip


def test_refusals_name_a_next_step(tmp_path):
    """Every refusal of a path (the actor's, a teammate's, the librarian's, the writer's and a helper's read and grep),
    of a pool entry, an episode or a part, of a question to the librarian, and of a wrong ``tools.find``,
    ``tools.take``, ``tools.decide`` or ``tools.test`` call holds a call that works or the valid values."""
    import threading
    from types import SimpleNamespace

    from unify.common import tool_registry as R
    from unify.memory import attention, writer_tests
    from unify.memory import attention_tools as T

    missing = [(agent, tool, form) for (agent, tool), forms in FORMS.items() for form in forms
               if not NEXT_STEP.search(str(R.role_refusal(agent, tool, form, memory=True, **FIELDS)))]  # fmt: skip
    assert not missing, missing

    home = tmp_path / "home"
    led = attention.Ledger(home)
    att = T.AttentionPass(ledger=led, pass_id="p1", record_dir=home / "memory-state" / "passes" / "p1",
                          inputs=tmp_path / "inputs", home=home, handed=[])  # fmt: skip
    writer = SimpleNamespace(_attention=att, _read_logs={}, tool_events=[], box_opens=[], _tests=None, _live=None)
    ctx = SimpleNamespace(state=SimpleNamespace(writer=writer, memory_lock=threading.Lock()))
    calls = {
        "find without a query": (T.find, {"query": ""}),
        "find in no scope": (T.find, {"query": "rows", "scope": "everywhere"}),
        "find with no hits wanted": (T.find, {"query": "rows", "k": 0}),
        "take without an id": (T.take, {"entry_id": ""}),
        "decide an unknown entry": (T.decide, {"entry_ids": ["f" * 16], "decision": "nothing_to_keep", "reason": "x"}),
        "decide something else": (T.decide, {"entry_ids": ["f" * 16], "decision": "maybe", "reason": "x"}),
    }
    for name, (fn, args) in calls.items():
        reply = fn(ctx, args)
        assert reply.status == "refused" and NEXT_STEP.search(reply.text), (name, reply.text)

    from tests.memory.test_layout import _tree

    tests = writer_tests.TestContext(pass_id="p1", tests_path=tmp_path / "tests.jsonl", library=_tree(tmp_path / "lib"),
                                     episode=lambda eid: None)  # fmt: skip
    expect = {"kind": "returns", "value": 1}
    for item, on in (("dates.parse_date", "e1/request"), ("memory.text.dates:parse_date", "e9/request")):
        with pytest.raises(writer_tests.WriterTestRefused) as got:
            writer_tests.run_test(tests, item, on, expect)
        assert NEXT_STEP.search(str(got.value)), str(got.value)
