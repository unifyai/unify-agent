"""Request text shared by most episodes, shown to the writer once.

The actor's requests open with text every request carries (in a replay the writer re-read the same opening 121
times: 17% of all it read). That text is found by code, never named: each request is split into blocks at blank
lines, and a block whose exact bytes appear in at least half of the requests seen so far (and in at least two) is
**shared**. The writer's views (``batch_map.json``'s request, ``read_episode``'s request part) show a shared block
as one marker line, ``[shared request text <id>: /inputs/shared/<id>.txt]``, and the block itself is one file
under ``/inputs/shared/``. Nothing is deleted: the raw episode files keep every byte, and the block is one read
away. Blocks shorter than :data:`MIN_CHARS` stay inline (a one-line heading is not worth a marker); that is a
floor, not a budget.

The counts live in the evidence store (lazy tables, so a store that never runs this keeps its schema), updated
once per episode as passes see it: nothing here depends on a stream or a task id.
"""

from __future__ import annotations

import hashlib
import re

MIN_CHARS = 200
# Blocks alone left 6.7-7.8% repeated read text at medium effort: a single line shared by most episodes is shared
# too, inside a block that also holds episode-specific lines, and in observations; floor, not a budget
MIN_LINE = 40
_BLOCKS = re.compile(r"\n[ \t]*\n")
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS request_blocks(block TEXT PRIMARY KEY, episodes INTEGER, text TEXT)",
    "CREATE TABLE IF NOT EXISTS request_seen(episode_id TEXT PRIMARY KEY)",
)


def blocks(text: str) -> list[str]:
    """*text* split at blank lines, each block stripped of surrounding blank space; empty blocks dropped."""
    return [b.strip("\n") for b in _BLOCKS.split(text or "") if b.strip()]


def block_id(block: str) -> str:
    return hashlib.sha256(block.encode()).hexdigest()[:12]


def units(texts) -> set[str]:
    """The countable units of *texts*: blocks of at least MIN_CHARS and lines of at least MIN_LINE."""
    out: set[str] = set()
    for text in texts:
        out |= {b for b in blocks(text or "") if len(b) >= MIN_CHARS}
        out |= {
            ln.strip()
            for ln in (text or "").splitlines()
            if len(ln.strip()) >= MIN_LINE
        }
    return out


def update_counts(ev, eid: str, request_text: str, observations=()) -> None:
    """Count *eid*'s request and observation units once (a second call for the same episode changes nothing)."""
    with ev.db:
        for stmt in _SCHEMA:
            ev.db.execute(stmt)
        if ev.db.execute(
            "SELECT 1 FROM request_seen WHERE episode_id=?",
            (eid,),
        ).fetchone():
            return
        ev.db.execute("INSERT INTO request_seen VALUES(?)", (eid,))
        for b in units([request_text, *[str(o) for o in observations]]):
            ev.db.execute(
                "INSERT INTO request_blocks VALUES(?,1,?) ON CONFLICT(block) DO UPDATE SET episodes=episodes+1",
                (block_id(b), b),
            )


def shared(ev) -> dict[str, str]:
    """The shared blocks (id -> text): in at least two of the episodes seen and in at least half of them."""
    if not ev._has_table("request_seen"):
        return {}
    seen = ev.db.execute("SELECT COUNT(*) FROM request_seen").fetchone()[0]
    rows = ev.db.execute(
        "SELECT block, text FROM request_blocks WHERE episodes >= 2 AND 2 * episodes >= ? ORDER BY block",
        (seen,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def marker(bid: str) -> str:
    """Short on purpose: batch_map.json's ``shared_blocks`` says once what a marker stands for."""
    return f"[shared:{bid}]"


def mark(text: str, shared_blocks: dict[str, str]) -> str:
    """*text* with every shared block replaced by its marker (the other text, and its blank lines, unchanged)."""
    if not shared_blocks or not text:
        return text or ""
    by_text = {t: b for b, t in shared_blocks.items()}
    parts = re.split(r"(\n[ \t]*\n)", text)
    text = "".join(
        marker(by_text[p.strip("\n")]) if p.strip("\n") in by_text else p for p in parts
    )
    lines = text.splitlines(keepends=True)
    out = []
    for ln in lines:
        body = ln.rstrip("\n")
        key = body.strip()
        if len(key) >= MIN_LINE and key in by_text:
            out.append(
                body[: len(body) - len(body.lstrip())]
                + marker(by_text[key])
                + ln[len(body) :],
            )
        else:
            out.append(ln)
    return "".join(out)
