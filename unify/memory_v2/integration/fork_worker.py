"""The episode-end fork's worker (memory v2.1 design r5 §2-§3): it continues the actor's own conversation through the
actor's proxy route and writes what the reply says is worth keeping into the episode's staging directory.

It runs by path as ``/usr/bin/python3 -I /s/fork_worker.py``, never as ``-m unify...``, inside a bubblewrap box
(:mod:`.fork`). It is standalone: standard library only, no package-relative import, so no credential lookup can
start. It holds no credential: the proxy route takes the actor's placeholder key. :mod:`.staging` imports
:func:`parse_reply` and the constants from here, so every reader of staged replies applies the same rules.

* **Input:** one JSON document read from stdin, a fresh pipe; never argv, the environment or a file. It holds the
  URL, the placeholder key, the headers, the request body and the episode id.
* **Turns:** one POST per turn. A reply with tool calls is never executed: each call is answered with
  :data:`TOOL_REPLY` and the conversation continues. The fork ends at the first reply without tool calls.
* **Output:** files parsed from fenced blocks whose first line is ``path: <relative path>``, from the reply text and
  from every tool call's arguments. Only ``candidates/<name>.py``, ``cases/<name>.json`` and ``notes.md`` are
  written; anything else is refused and listed. Then ``fork.json``.

Operational guards, which are run safety and not design: :data:`QUOTA_BYTES` and :data:`QUOTA_FILES` per episode,
:data:`TURN_GUARD` turns, and :data:`HTTP_TIMEOUT_S` per call. Each is reported in ``fork.json`` when reached.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

TOOL_REPLY = "Tools are not available in this review; write your findings as files in the format above."
QUOTA_BYTES = 4 * 1024 * 1024
QUOTA_FILES = 200
TURN_GUARD = 16
HTTP_TIMEOUT_S = 900.0
ALLOWED = (
    re.compile(r"candidates/[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}\.py"),
    re.compile(r"cases/[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}\.json"),
    re.compile(r"notes\.md"),
)
_FENCE = re.compile(
    r"^(?P<fence>`{3,}|~{3,})[^\n]*\n(?P<body>.*?)^(?P=fence)[ \t]*$",
    re.M | re.S,
)
_PATH = re.compile(r"\A[ \t]*path:[ \t]*(?P<rel>[^\n]*?)[ \t]*\n")


def blocks(text: str) -> list[tuple[str, str]]:
    """(relative path, content) for each fenced block of *text* whose first line is ``path: <rel>``."""
    out = []
    for m in _FENCE.finditer(text):
        p = _PATH.match(m.group("body"))
        if p:
            out.append((p.group("rel"), m.group("body")[p.end() :]))
    return out


def _strings(value) -> list[str]:
    if isinstance(value, str):
        try:
            inner = json.loads(value)
        except ValueError:
            return [value]
        return [value] if isinstance(inner, str) else _strings(inner)
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def reply_blocks(message: dict) -> list[tuple[str, str, str]]:
    """(path, content, source) for *message*'s text and every tool call's arguments; nothing is executed."""
    out = []
    content = message.get("content")
    texts = (
        [content]
        if isinstance(content, str)
        else [
            p.get("text", "")
            for p in (content or [])
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
    )
    for t in texts:
        out += [(rel, body, "text") for rel, body in blocks(t)]
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        name = str(fn.get("name") or "?")[:64]
        for s in _strings(fn.get("arguments")):
            out += [(rel, body, f"tool_call:{name}") for rel, body in blocks(s)]
    return out


def allowed(rel: str) -> str | None:
    """Why *rel* is refused, or None."""
    if (
        not rel
        or rel.startswith("/")
        or "\\" in rel
        or any(part in ("", ".", "..") for part in rel.split("/"))
    ):
        return "not a plain relative path"
    if not any(rx.fullmatch(rel) for rx in ALLOWED):
        return "outside candidates/*.py, cases/*.json and notes.md"
    return None


def _sorted(
    found: list[tuple[str, str, str]],
) -> tuple[dict[str, tuple[str, str]], list[dict]]:
    latest: dict[str, tuple[str, str]] = {}
    refused = []
    for rel, body, source in found:
        why = allowed(rel)
        if why:
            refused.append({"path": rel[:200], "why": why, "from": source})
        else:
            latest[rel] = (
                body,
                source,
            )  # a later block for the same path replaces an earlier one
    return latest, refused


def parse_reply(message: dict) -> tuple[dict[str, str], list[dict]]:
    """The files one assistant *message* stages, by relative path (text and every tool call's arguments; nothing
    executed), and the refused blocks ``{path, why, from}``. The operational quota applies when they are written.
    """
    latest, refused = _sorted(reply_blocks(message))
    return {rel: body for rel, (body, _source) in latest.items()}, refused


def write_files(
    staging: Path,
    found: list[tuple[str, str, str]],
) -> tuple[list, list, bool]:
    """Write the allowed blocks under *staging*, within the operational quota; (files, refused, quota reached)."""
    latest, refused = _sorted(found)
    files, total, over = [], 0, False
    for rel, (body, source) in latest.items():
        data = body.encode("utf-8", "replace")
        if len(files) >= QUOTA_FILES or total + len(data) > QUOTA_BYTES:
            refused.append(
                {"path": rel, "why": "operational quota reached", "from": source},
            )
            over = True
            continue
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.parent.is_symlink():
            refused.append({"path": rel, "why": "parent is a link", "from": source})
            continue
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        total += len(data)
        files.append({"path": rel, "from": source, "bytes": len(data)})
    return files, refused, over


def _post(url: str, headers: dict, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(
        req,
        timeout=HTTP_TIMEOUT_S,
    ) as resp:  # noqa: S310 - the proxy route only
        return json.loads(resp.read().decode("utf-8"))


def _add_usage(total: dict, usage: dict) -> None:
    def n(v):
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    pt, ct = (usage.get("prompt_tokens_details") or {}), (
        usage.get("completion_tokens_details") or {}
    )
    for key, value in (
        ("prompt_tokens", n(usage.get("prompt_tokens"))),
        ("cached_tokens", n(pt.get("cached_tokens"))),
        ("completion_tokens", n(usage.get("completion_tokens"))),
        ("reasoning_tokens", n(ct.get("reasoning_tokens"))),
    ):
        total[key] = (
            None
            if value is None or total.get(key, 0) is None
            else total.get(key, 0) + value
        )
    cost = usage.get("cost")
    try:
        usd = (
            Decimal(str(cost))
            if cost is not None and not isinstance(cost, bool)
            else None
        )
    except InvalidOperation:
        usd = None
    total["_usd"] = (
        None
        if usd is None or total.get("_usd", Decimal(0)) is None
        else total.get("_usd", Decimal(0)) + usd
    )


def run(req: dict, staging: Path, post=_post) -> dict:
    """The fork's turns and files; returns the ``fork.json`` document (also written to *staging*)."""
    doc = {
        "episode": req.get("episode"),
        "status": "ok",
        "model": (req.get("body") or {}).get("model"),
        "turns": 0,
        "usage": {},
        "usd": None,
        "files": [],
        "refused_files": [],
    }
    body = dict(req["body"])
    messages = list(body["messages"])
    found: list[tuple[str, str, str]] = []
    usage: dict = {}
    try:
        while True:
            if doc["turns"] >= TURN_GUARD:
                doc["status"] = f"error: operational turn guard ({TURN_GUARD}) reached"
                break
            headers = dict(req["headers"])
            headers["x-unify-request"] = str(doc["turns"])
            headers["x-unify-msg-count"] = str(len(messages))
            resp = post(req["url"], headers, {**body, "messages": messages})
            doc["turns"] += 1
            _add_usage(usage, resp.get("usage") or {})
            message = ((resp.get("choices") or [{}])[0].get("message")) or {}
            found += reply_blocks(message)
            calls = message.get("tool_calls") or []
            if not calls:
                break
            messages.append(
                {
                    k: message[k]
                    for k in ("role", "content", "tool_calls")
                    if k in message
                },
            )
            messages += [
                {"role": "tool", "tool_call_id": c.get("id"), "content": TOOL_REPLY}
                for c in calls
            ]
    except Exception as exc:  # noqa: BLE001 - recorded, never raised
        doc["status"] = f"error: {type(exc).__name__}"
    files, refused, over = write_files(staging, found)
    doc["files"], doc["refused_files"] = files, refused
    if over and doc["status"] == "ok":
        doc["status"] = "ok: operational quota reached"
    usd = usage.pop("_usd", None) if usage else None
    doc["usage"], doc["usd"] = usage, (str(usd) if usd is not None else None)
    (staging / "fork.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", type=Path, required=True)
    a = ap.parse_args(argv)
    req = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    run(req, a.staging)
    return 0


if __name__ == "__main__":
    sys.exit(main())
