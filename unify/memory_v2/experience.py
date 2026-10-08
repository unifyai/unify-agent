"""Experience tokens: how much new experience one trajectory recorded (the batched trigger's measure).

:func:`experience_tokens` counts the content an episode records once, never what was sent to a model:

* the request and every later user or observation message (``Episode.request``);
* the agent's reply texts: ``Episode.replies`` when the episode has them, else the assistant messages of
  the transcript that carry text and no tool calls, each counted once by identity (its ``id`` or
  ``seq`` when the line has one, else its role and content), so a transcript that re-sends earlier
  context (an in-context replay, a compaction) counts the same as one that does not; lines marked as
  replays or compactions are skipped;
* every cell's code (the tool-call arguments), printed output and error;
* every action's observation: a tool call's response or error, a shell command's output tail, a file's
  recorded shape (never its body), a dialogue action's observation, unless that observation is a
  request message already counted (a dialogue observation is the next user message).

Provider usage is never consulted. Tokens are counted with tiktoken's ``o200k_base`` when it imports,
else as ``ceil(utf-8 bytes / 4)``; :data:`COUNTER` names which, and every result carries it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from typing import Any, Callable

from .episodes import Episode

_REPLAY_FLAGS = (
    "in_context_replay",
    "replay",
    "compaction",
    "is_compaction",
    "replayed",
)


@lru_cache(maxsize=1)
def _encoder() -> tuple[str, Callable[[str], int]]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return "tiktoken:o200k_base", lambda s: len(
            enc.encode(s, disallowed_special=()),
        )
    except Exception:
        return "bytes/4", lambda s: math.ceil(len(s.encode("utf-8")) / 4)


def counter() -> str:
    """Which counter :func:`count_text` uses: ``tiktoken:o200k_base`` or ``bytes/4``."""
    return _encoder()[0]


def count_text(text: str) -> int:
    return _encoder()[1](text) if text else 0


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def _replayed(line: dict, msg: dict) -> bool:
    return any(bool(d.get(f)) for d in (line, msg) for f in _REPLAY_FLAGS)


def transcript_replies(transcript: list[dict]) -> list[str]:
    """The assistant reply texts of a transcript, each message once (see the module docstring)."""
    seen: set[str] = set()
    out: list[str] = []
    for line in transcript or []:
        if not isinstance(line, dict):
            continue
        msg = line.get("message") if isinstance(line.get("message"), dict) else line
        if (
            msg.get("role") != "assistant"
            or msg.get("tool_calls")
            or _replayed(line, msg)
        ):
            continue
        text = _content_text(msg.get("content"))
        if not text:
            continue
        ident = next(
            (
                f"{k}:{d[k]}"
                for d in (line, msg)
                for k in ("id", "seq")
                if d.get(k) is not None
            ),
            None,
        ) or _text(["assistant", text])
        if ident in seen:
            continue
        seen.add(ident)
        out.append(text)
    return out


def _observation(a: Any) -> Any:
    kind = getattr(a, "kind", "tool")
    r = a.response
    if kind == "shell":
        return r.get("tail") if isinstance(r, dict) else a.error
    if kind == "worktree":
        if isinstance(r, dict):
            return r.get("shape") if "shape" in r else r.get("entries")
        return a.error
    if r is None:
        return a.error
    return r


def experience_pieces(ep: Episode) -> list[str]:
    """The texts :func:`experience_tokens` counts, in a fixed order."""
    pieces: list[str] = [m for m in ep.request if isinstance(m, str)]
    requests = Counter(pieces)
    replies = list(getattr(ep, "replies", None) or []) or transcript_replies(
        ep.transcript,
    )
    pieces += replies
    for c in ep.cells:
        pieces += [c.code or "", c.output or "", c.error or ""]
    for a in ep.actions:
        obs = _observation(a)
        if (
            getattr(a, "kind", "tool") == "dialogue"
            and isinstance(obs, str)
            and requests[obs] > 0
        ):
            requests[
                obs
            ] -= 1  # the next user message, already counted as a request message
            continue
        pieces.append(_text(obs))
    return [p for p in pieces if p]


def experience_tokens(ep: Episode) -> tuple[int, str]:
    """(tokens, counter) for the content *ep* recorded once."""
    return sum(count_text(p) for p in experience_pieces(ep)), counter()
