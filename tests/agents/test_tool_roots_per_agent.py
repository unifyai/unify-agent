"""Every agent's read and grep texts name the roots its handler allows: the actor and a teammate read and grep
under ``/workspace`` only (``unify.actor.tool_handlers``), the librarian inside the library, ``/memory``; the writer and
a helper keep the shared texts (``/inputs``, ``/memory``, ``/outputs``)."""

from __future__ import annotations

import re

import pytest

from unify.actor import tool_handlers as actor_tools
from unify.common import tool_registry as R


@pytest.mark.parametrize("agent", ["actor", "teammate"])
def test_workspace_agents_read_and_search_under_their_one_root(agent):
    root = actor_tools.WORKSPACE
    assert root == "/workspace"
    for name in ("read", "grep"):
        text = R.help_text(name, agent) + "\n" + R.list_text([name], agent)
        assert root in text, (name, text)
        assert not re.search(r"/inputs|/outputs|/memory|/work\b", text), (name, text)
    assert "outside_roots: <path> is outside /workspace" in R.help_text("read", agent)


def test_the_librarian_reads_and_searches_the_library():
    for name in ("read", "grep"):
        text = R.help_text(name, "librarian") + "\n" + R.list_text([name], "librarian")
        assert "/memory" in text and not re.search(r"/inputs|/outputs|/workspace", text), (name, text)


def test_the_memory_agents_keep_the_shared_texts():
    for name in ("read", "grep"):
        assert R.help_text(name, "writer") == R.help_text(name)
        assert "/inputs" in R.list_text([name], "writer")
