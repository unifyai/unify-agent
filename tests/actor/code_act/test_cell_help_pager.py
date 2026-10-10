"""Symbolic: ``help(obj)`` in a cell prints its page and never starts a pager.

``help()`` with no argument read stdin (the Continual-ARC hang of 5 October,
fixed by the cell's empty stdin). ``help(obj)`` instead pages its text:
pydoc picks a pager once per process (``pydoc.getpager``), and pipes to
``$MANPAGER``/``$PAGER`` or ``less`` only when both stdin and stdout are
terminals. Inside a cell stdin is the empty stream and stdout the capture,
neither a terminal, so pydoc's plain pager writes the page to the cell's
output. That holds even with a pager configured, which is set here to a
command that would block. In the sandboxed worker ``help`` is not a
builtin; ``pydoc.help`` behaves the same way there (descriptor 0 is
``/dev/null``).
"""

from __future__ import annotations

import time

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import parts_to_text

BOUND = 2.0
BLOCKING_PAGER = "sleep 30"


async def _run(code: str) -> tuple[str, dict, float]:
    ex = SessionExecutor(environments={}, timeout=30)
    try:
        started = time.monotonic()
        res = await ex.execute(code=code, state_mode="stateful", session_id=0)
        return parts_to_text(res["stdout"]), res, time.monotonic() - started
    finally:
        await ex.close()


@pytest.fixture
def pager_configured(monkeypatch):
    """A pager that would block, and pydoc's choice not yet made."""
    import pydoc

    monkeypatch.setenv("PAGER", BLOCKING_PAGER)
    monkeypatch.setenv("MANPAGER", BLOCKING_PAGER)
    monkeypatch.setenv("TERM", "xterm")

    def pager(text, title=""):
        # pydoc's own first-call behaviour: decide, remember, page.
        pydoc.pager = pydoc.getpager()
        pydoc.pager(text)

    monkeypatch.setattr(pydoc, "pager", pager)


@needs_bwrap
@pytest.mark.asyncio
async def test_pydoc_help_in_the_worker_prints_its_page(world, monkeypatch):
    monkeypatch.setenv("PAGER", BLOCKING_PAGER)
    monkeypatch.setenv("MANPAGER", BLOCKING_PAGER)
    out, res, elapsed = await _run("import pydoc\npydoc.help(str)")
    assert res["error"] is None, res["error"]
    # The worker starts a sandboxed process first; the page itself is quick.
    assert elapsed < 30
    assert "class str" in out
