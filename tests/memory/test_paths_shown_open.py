"""Every path or call the harness prints to an agent opens for the agent it is shown to, and every refusal names a
next step. Paths are resolved by the harness: an agent never has to translate one."""

from __future__ import annotations

import re

from unify.actor import tool_handlers as H
from unify.memory import librarian as L
from unify.memory import library_helper as lh
from unify.memory import tool_spec
from unify.memory.writer_pass import _natural_path


def _tail_path(text: str) -> str:
    return re.search(r"tools\.read\('([^']+)'\)", text).group(1)


def test_a_notes_rest_path_opens_where_it_is_printed(monkeypatch):
    long = "## Rule\n" + "x" * 5000
    # the actor's box: the library mount (and, under in_episode, the current pin inside it)
    for root in ("/memory-library", "/memory-library/current"):
        monkeypatch.setattr(lh, "_ROOT", root)
        path = _tail_path(lh._note_rule(long, "notes/a/b.md"))
        assert H._in_library(path, root) == path and path == f"{root}/notes/a/b.md"
    # the writer's and the librarian's box: the library at /memory, one of their roots
    monkeypatch.setattr(lh, "_ROOT", "/memory")
    assert _tail_path(lh._note_rule(long, "notes/a/b.md")) == "/memory/notes/a/b.md"


def test_the_calls_the_librarian_and_the_cards_print_go_through_memory():
    block = L.start_block({"fits": [{"id": "capture:" + "a" * 16, "call": "memory.get('capture:" + "a" * 16 + "')()",
                                     "why": "it fits"}]})  # fmt: skip
    call = block.splitlines()[2].split("call: ", 1)[1]
    assert L._call_problem(call) == ""
    by_hand = lh.BY_HAND.format(cid="capture:" + "a" * 16)
    assert "memory.get('capture:" in by_hand and "exec(" in by_hand


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
