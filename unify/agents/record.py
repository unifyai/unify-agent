"""The run's shared record: the single writer, the in-memory view, waiting, and cursors."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from unify.agents.delivery import render_block
from unify.agents.entry import KINDS, MAX_TEXT_BYTES, Entry, mention_tokens
from unify.agents.log import RecordLog
from unify.agents.options import Options
from unify.transcripts import scrub

logger = logging.getLogger(__name__)

USER = "user"
HARNESS = "harness"
ALL = "all"
MAX_WAIT_S = 600.0


def _now() -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


# Indirections the tests replace to check the cap without sleeping.
_wait_for = asyncio.wait_for


def _deadline_passed(loop: asyncio.AbstractEventLoop, deadline: float) -> bool:
    return loop.time() >= deadline


class PostRefused(ValueError):
    """A post the record does not accept; the message says why and what to do instead."""


@dataclass
class Participant:
    name: str
    role: str  # "user" | "harness" | "agent"
    spawner: Optional[str] = None
    state: str = "running"  # running | replied | stopped | failed | lost
    task_seq: Optional[int] = None
    last_seq: Optional[int] = None
    cursor: int = 0
    returned: set[int] = field(default_factory=set)
    last_post_at: float = float("-inf")


def _wake(future: asyncio.Future) -> None:
    def _set() -> None:
        if not future.done():
            future.set_result(None)

    try:
        future.get_loop().call_soon_threadsafe(_set)
    except RuntimeError:  # its loop is closed: nobody is waiting any more
        pass


class Record:
    def __init__(
        self,
        log: Optional[RecordLog],
        *,
        options: Options,
        root: str = "root",
        user_reads: bool = False,
        clock: Callable[[], str] = _now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.log = log
        self.options = options
        self.root = root
        self.user_reads = user_reads
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._waiters: list[asyncio.Future] = []
        self._listeners: list[Callable[[Entry], None]] = []
        self.entries: list[Entry] = []
        self.participants: dict[str, Participant] = {
            USER: Participant(USER, "user"),
            HARNESS: Participant(HARNESS, "harness"),
            root: Participant(root, "agent"),
        }
        if log is not None:
            self.entries, cursors = log.load()
            for name, upto in cursors.items():
                if name in self.participants:
                    self.participants[name].cursor = upto

    @property
    def path(self) -> Optional[Path]:
        return self.log.path if self.log is not None else None

    @property
    def last_seq(self) -> int:
        return self.entries[-1].seq if self.entries else 0

    def add_agent(self, name: str, *, spawner: str) -> Participant:
        with self._lock:
            if name in self.participants:
                raise PostRefused(f"the name {name!r} is taken in this run")
            participant = Participant(
                name,
                "agent",
                spawner=spawner,
                cursor=self.last_seq,
            )
            self.participants[name] = participant
            return participant

    def live_agents(self) -> list[str]:
        return [
            n
            for n, p in self.participants.items()
            if p.role == "agent" and p.state == "running"
        ]

    def add_listener(self, fn: Callable[[Entry], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def _resolve(self, text: str, author: str) -> tuple[list[str], list[str]]:
        resolved: list[str] = []
        unresolved: list[str] = []
        for token in mention_tokens(text):
            if token == ALL:
                targets = [n for n in self.live_agents() if n != author]
            elif token in self.participants and token != HARNESS:
                targets = [token]
            else:
                unresolved.append(token)
                continue
            resolved.extend(t for t in targets if t not in resolved)
        return resolved, unresolved

    def append(
        self,
        author: str,
        text: str,
        *,
        kind: str = "post",
        mentions: Optional[list[str]] = None,
    ) -> tuple[Entry, list[str]]:
        """Accept one entry; return it and the @-names that matched nobody."""
        if kind not in KINDS:
            raise PostRefused(f"unknown kind {kind!r}; use one of {', '.join(KINDS)}")
        text = scrub(str(text))
        size = len(text.encode("utf-8"))
        if size > MAX_TEXT_BYTES:
            raise PostRefused(
                f"the entry is {size} bytes and the limit is {MAX_TEXT_BYTES}; "
                "write long content to a file and post its path",
            )
        with self._lock:
            if len(self.entries) >= self.options.max_entries:
                raise PostRefused(
                    f"the record is full ({self.options.max_entries} entries)",
                )
            if mentions is None:
                resolved, unresolved = self._resolve(text, author)
            else:
                resolved, unresolved = list(mentions), []
            entry = Entry(
                seq=self.last_seq + 1,
                ts=self._clock(),
                author=author,
                kind=kind,
                text=text,
                mentions=tuple(resolved),
            )
            if self.log is not None:
                self.log.append(entry)
            self.entries.append(entry)
            if author in self.participants:
                self.participants[author].last_seq = entry.seq
            waiters, self._waiters = self._waiters, []
            listeners = list(self._listeners)
        for future in waiters:
            _wake(future)
        for listener in listeners:
            try:
                listener(entry)
            except Exception:
                logger.exception("record listener failed on entry %s", entry.seq)
        return entry, unresolved

    def check_rate(self, name: str) -> None:
        participant = self.participants[name]
        now = self._monotonic()
        gap = self.options.min_post_interval_s - (now - participant.last_post_at)
        if gap > 0:
            raise PostRefused(f"posting too fast; retry after {gap:.1f} s")
        participant.last_post_at = now

    def read(
        self,
        *,
        since: int = 0,
        mentions: Optional[str] = None,
        author: Optional[str] = None,
        limit: int = 50,
    ) -> list[Entry]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a whole number ≥ 1")
        with self._lock:
            matched = [
                e
                for e in self.entries
                if e.seq > since
                and (mentions is None or mentions in e.mentions)
                and (author is None or e.author == author)
            ]
        return matched[-limit:]

    def is_for(self, entry: Entry, name: str) -> bool:
        if entry.author == name:
            return False
        if self.options.delivery == "all" or name in entry.mentions:
            return True
        return name == self.root and entry.author == USER

    def _unseen(self, name: str, mentions_only: bool) -> list[Entry]:
        p = self.participants[name]
        return [
            e
            for e in self.entries
            if e.seq > p.cursor
            and e.seq not in p.returned
            and e.author != name
            and (self.is_for(e, name) or not mentions_only)
        ]

    async def wait_for(
        self,
        name: str,
        *,
        timeout: float,
        mentions_only: bool = True,
    ) -> list[Entry]:
        """Entries ``name`` has not been given yet; block until one arrives or the timeout."""
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not timeout > 0
        ):
            raise ValueError(
                "timeout is required: a number of seconds above 0, for example "
                "timeout=300",
            )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + min(float(timeout), MAX_WAIT_S)
        while True:
            with self._lock:
                found = self._unseen(name, mentions_only)
                if found:
                    self.participants[name].returned.update(e.seq for e in found)
                    return found
                if _deadline_passed(loop, deadline):
                    return []
                future = loop.create_future()
                self._waiters.append(future)
            try:
                await _wait_for(
                    asyncio.shield(future),
                    min(deadline - loop.time(), MAX_WAIT_S),
                )
            except asyncio.TimeoutError:
                pass
            finally:
                with self._lock:
                    if future in self._waiters:
                        self._waiters.remove(future)

    def take_block(self, name: str) -> Optional[str]:
        """The boundary block for ``name``, or None when nothing new concerns it."""
        with self._lock:
            p = self.participants[name]
            new = [e for e in self.entries if e.seq > p.cursor and e.author != name]
            mine = [e for e in new if self.is_for(e, name)]
            if not mine:
                return None
            others = [e for e in new if not self.is_for(e, name)]
            block = render_block(
                mine,
                others,
                p.returned,
                since=p.cursor,
                cap_tokens=self.options.deliver_max_tokens,
                others_line=self.options.others_line,
            )
            upto = self.last_seq
            p.cursor = upto
            p.returned = {s for s in p.returned if s > upto}
            if self.log is not None:
                self.log.append_cursor(name, upto)
            return block
