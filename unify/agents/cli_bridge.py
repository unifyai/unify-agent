"""``unify act`` in record mode: the person or host at the CLI is one more participant."""

from __future__ import annotations

from typing import Any, Callable, Optional

from unify.agents.entry import Entry
from unify.agents.pool import Pool
from unify.agents.record import USER


class CliBridge:
    """Routes the driver's lines into the record and record entries out to it.

    A message is a ``user`` post. While the main agent works it waits for the
    next boundary; when a persistent session is parked between requests, the
    post wakes it as its next request. A cancel is recorded and then acts as
    before. Entries that mention ``@user`` go out as ``{"type": "record"}``.
    """

    def __init__(self, pool: Pool, handle: Any, emit: Callable[[dict], None]) -> None:
        self.pool = pool
        self.handle = handle
        self._emit = emit
        self._root_busy = True
        pool.record.add_listener(self._on_entry)

    def _on_entry(self, entry: Entry) -> None:
        # The main agent's reply already goes out as the response/result line.
        if entry.author == self.pool.record.root and entry.kind == "reply":
            return
        if USER in entry.mentions:
            self._emit({"type": "record", **entry.to_dict()})

    async def user_message(self, text: str) -> None:
        """A line from the driver: a ``user`` post (mentions resolved from the text)."""
        self.pool.record.append_harness(USER, text, full_text_at="the driver's input")
        if not self._root_busy:
            await self._wake_if_waiting()

    def cancel_posted(self) -> None:
        root = self.pool.record.root
        self.pool.record.append_harness(
            USER,
            f"@{root} cancel",
            kind="cancel",
            mentions=[root],
        )

    async def root_replied(self, text: str) -> None:
        """A persistent session's turn answered: record it, end its helpers."""
        await self.pool.finish_request(text, reason="the main agent replied")
        self._root_busy = False
        # A message that arrived during the turn's last model call was not shown
        # to it (the turn ended there); it is the session's next request.
        await self._wake_if_waiting()

    async def _wake_if_waiting(self) -> None:
        # A parked session takes a new request only through its request queue:
        # hand it the block, which moves the cursor, so no boundary repeats it.
        record = self.pool.record
        block = record.take_block(record.root)
        if block:
            self._root_busy = True
            await self.handle.submit(block)


def attach_bridge(handle: Any, emit: Callable[..., None]) -> Optional[CliBridge]:
    """A bridge for a record-mode handle (``emit`` takes keyword fields), else None."""
    pool = getattr(handle, "agents_pool", None)
    if pool is None:
        return None
    return CliBridge(pool, handle, lambda payload: emit(**payload))
