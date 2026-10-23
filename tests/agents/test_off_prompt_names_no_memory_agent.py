"""A run without memory (``UNIFY_MEMORY=off``) has no writer, fork or librarian: the record's common section, every
agent's roster lines, the run-record tools' texts and the ``record`` help topic name none of them. With memory on
the shipped bytes are unchanged."""

from __future__ import annotations

import re

import pytest

from unify.agents import run_record as RR
from unify.common import tool_registry as R

#: A memory agent named in a text: an address, or the role's own word.
MEMORY_NAMED = re.compile(r"@(writer|fork|librarian)\b|\bwriter\b|\blibrarian\b|\bfork\b", re.IGNORECASE)
MEMORY_ADDRESS = re.compile(r"@(writer|fork|librarian)\b")


@pytest.fixture
def no_memory(monkeypatch):
    """A run with ``UNIFY_MEMORY=off`` (the one read every text goes through)."""
    monkeypatch.setattr(R, "memory_is_on", lambda: False)


@pytest.fixture
def with_memory(monkeypatch):
    """A run with memory on, whatever the suite's environment sets."""
    monkeypatch.setattr(R, "memory_is_on", lambda: True)


def test_without_memory_the_roster_lines_name_no_memory_agent(no_memory):
    actor = RR.prompt_lines("actor-ab12", "actor")
    helper = RR.prompt_lines("helper-ab12-1", "helper", "actor-ab12")
    for text in (actor, helper):
        assert text.startswith(R.COMMON_SECTION_NO_MEMORY), text[:200]
        assert not MEMORY_NAMED.search(text), text


def test_with_memory_the_roster_lines_are_the_shipped_bytes(with_memory):
    text = RR.prompt_lines("actor-ab12", "actor")
    assert text.startswith(R.COMMON_SECTION)
    assert "Mention @writer when a served item was wrong, misleading, or something you needed was missing." in text


def test_the_no_memory_section_differs_only_in_its_role_sentences():
    full, off = R.COMMON_SECTION, R.COMMON_SECTION_NO_MEMORY
    assert off != full and off.startswith("### The agents' record\n")
    assert not MEMORY_NAMED.search(off), off
    tail = "@helpers (every running helper)."
    assert full.split(tail, 1)[1] == off.split(tail, 1)[1]
    assert R.common_section(True) == full and R.common_section(False) == off
    assert RR.COMMON_SECTION_NO_MEMORY == off


@pytest.mark.parametrize("agent", ["actor", "teammate"])
def test_without_memory_no_tool_text_addresses_a_memory_agent(no_memory, agent):
    names = [n for n in R.allow_list(agent, frozenset()) if n not in R.EMBEDDED]
    assert "send" in names
    for name in names:
        for text in (R.help_text(name, agent), R.list_text([name], agent)):
            assert not MEMORY_ADDRESS.search(text), (name, text)
    record = {topic: text for topic, (_, text) in R.help_topics().items()}["record"]
    assert record == R.COMMON_SECTION_NO_MEMORY


def test_with_memory_the_send_texts_are_unchanged(with_memory):
    assert "@writer" in R.help_text("send", "actor")
    assert R.help_text("send", "actor", memory=True) == R.help_text("send", "writer", memory=True)
