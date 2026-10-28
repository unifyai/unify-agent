"""The block of record entries an agent is shown at its turn boundary."""

from __future__ import annotations

from typing import Iterable

from unify.agents.entry import Entry

_TAGS = {"post": "", "reply": " (reply)", "cancel": " (cancel)", "system": " (notice)"}


def _priority(entry: Entry) -> bool:
    """Instructions and control: always shown in full, never cut."""
    return entry.author == "user" or entry.kind in ("cancel", "system")


def _span(entries: list[Entry]) -> str:
    lo, hi = entries[0].seq, entries[-1].seq
    return f"#{lo}" if lo == hi else f"#{lo}–#{hi}"


def render_block(
    mine: list[Entry],
    others: list[Entry],
    returned: Iterable[int],
    *,
    since: int,
    cap_tokens: int,
    others_line: bool,
) -> str:
    returned = set(returned)
    lines: list[str] = []
    overflow: list[Entry] = []
    used = 0
    for entry in mine:
        tag = _TAGS.get(entry.kind, "")
        if entry.seq in returned and not _priority(entry):
            line = f"#{entry.seq} {entry.author}{tag} — returned to your code by record.wait"
        else:
            line = f"#{entry.seq} {entry.author}{tag}: {entry.text}"
        cost = len(line) // 4 + 1
        if not _priority(entry) and lines and used + cost > cap_tokens:
            overflow.append(entry)
            continue
        lines.append(line)
        used += cost
    shown = len(mine) - len(overflow)
    both = sorted(mine + others, key=lambda e: e.seq)
    out = [f"[record · {shown} new for you · {_span(both)}]", *lines]
    if overflow:
        out.append(
            f"(+{len(overflow)} more for you, {_span(overflow)}; "
            f"record.read(since={overflow[0].seq - 1}, mentions_me=True) shows them)",
        )
    if others and others_line:
        authors = ", ".join(sorted({e.author for e in others}))
        out.append(
            f"(+{len(others)} other entries, {_span(others)}, by {authors}; "
            f"record.read(since={since}) shows them)",
        )
    return "\n".join(out)
