"""The episode-end fork's worker: it continues the actor's own conversation through the key proxy's fork lane and
writes what the reply says is worth keeping into the episode's staging directory.

It runs by path as ``/usr/bin/python3 -I /s/fork_worker.py``, never as ``-m unify...``, inside a bubblewrap box
(:mod:`.fork`). It is standalone: standard library only, no package-relative import, so no credential lookup can
start. It holds no provider credential: the key proxy's fork lane takes only its per-stage fork token, which arrives
in the stdin request's headers. :mod:`.staging` imports :func:`parse_reply` and the constants from here, so every
reader of staged replies applies the same rules, and the offline replay imports :func:`parse_reply` by path.

* **Input:** one JSON document read from stdin, a fresh pipe; never argv, the environment or a file. It holds the
  fork lane's URL, the headers (with the fork token), the request body and the episode id.
* **Turns:** one POST per turn. A reply with tool calls is never executed: each call is answered with
  :data:`TOOL_REPLY` and the conversation continues. The fork ends at the first reply without tool calls.
* **Output:** the files the reply's blocks name (:func:`parse_reply`), from the reply text and from every tool
  call's arguments. Only ``summary.md``, ``candidates/<name>.py``, ``cases/<name>.json``, ``notes.md``,
  ``procedure.py`` and ``broke.md`` are written; every block found is recorded as parsed, refused or dropped
  with a reason. Written in this order: the staged files, the raw reply (``fork-reply.json``), the block ledger
  (``fork-blocks.json``, with the raw reply's sha256), then ``fork.json``. ``fork.json`` is written last, and
  its existence means the fork has settled.

Operational guards, which are run safety and not design: :data:`QUOTA_BYTES` and :data:`QUOTA_FILES` per episode,
:data:`TURN_GUARD` turns, :data:`HTTP_TIMEOUT_S` per call and :data:`WALL_GUARD_S` for the whole fork (on reaching
it the fork stages what it has and records status ``"timeout"``). Each is reported in ``fork.json`` when reached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import tempfile
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

TOOL_REPLY = "Tools are not available in this review; write your findings as fenced blocks in the format above."
QUOTA_BYTES = 4 * 1024 * 1024
QUOTA_FILES = 200
TURN_GUARD = 16
HTTP_TIMEOUT_S = 900.0
#: The fork's own wall bound; below the CLI's wait guard (``fork.WAIT_GUARD_S``), so the worker settles first.
WALL_GUARD_S = 1500.0
ALLOWED = (
    re.compile(r"summary\.md"),
    re.compile(r"candidates/[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}\.py"),
    re.compile(r"cases/[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}\.json"),
    re.compile(r"notes\.md"),
    re.compile(r"procedure\.py"),
    re.compile(r"broke\.md"),
)
FORK_JSON = "fork.json"
#: The fork's account of its episode (what was asked, the approach, what worked or failed, the outcome, with pointers
#: into the record); the writer reads it first.
SUMMARY_FILE = "summary.md"
BLOCKS_JSON = "fork-blocks.json"
REPLY_JSON = "fork-reply.json"
LOCK_FILE = "fork.lock"
#: Files in a staging dir that record the fork rather than stage findings; never exported as staged files.
RECORD_FILES = (FORK_JSON, BLOCKS_JSON, REPLY_JSON, LOCK_FILE)
FORMS = ("fence", "inner", "split", "line_before", "split_unfenced")
STATUSES = ("parsed", "refused", "dropped")
RENAMED = "renamed: not an importable module name"

# --- the block parser ---------------------------------------------------------------------------------------------

# A fence line: optional indentation, three or more backticks or tildes, an optional info string.
_FENCE = re.compile(
    r"(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n]*?)[ \t]*\Z",
)
# A line naming a file: ``path: X``, ``# path: X``, ``// path: X``, ``<!-- path: X -->``, ``**path:** X``,
# ``**path: X**``, and the path in backticks.
_MARK = re.compile(
    r"[ \t]*(?:(?P<hash>\#+)|(?P<slashes>//)|(?P<html><!--))?[ \t]*"
    r"(?:\*\*|__)?[ \t]*path[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*"
    r"`?(?P<rel>[^\s`*]+?)`?"
    r"[ \t]*(?:\*\*|__)?[ \t]*(?(html)-->)[ \t]*\Z",
    re.IGNORECASE,
)
# A fence info string naming a file: ``path: X`` or ``<lang> path: X`` (``python path: X``).
_INFO = re.compile(
    r"(?:[\w+.#-]+[ \t]+)?(?:\*\*)?path[ \t]*:(?:\*\*)?[ \t]*`?(?P<rel>[^\s`*]+?)`?[ \t]*\Z",
    re.IGNORECASE,
)
_KINDS = (
    (re.compile(r"summary\.md"), "summary"),
    (re.compile(r"candidates/[^/]+\.py"), "candidate"),
    (re.compile(r"cases/[^/]+\.json"), "case"),
    (re.compile(r"notes\.md"), "notes"),
    (re.compile(r"procedure\.py"), "procedure"),
    (re.compile(r"broke\.md"), "broke"),
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def kind(rel: str | None) -> str:
    """summary, candidate, case, notes or other, by the path's shape."""
    for rx, name in _KINDS:
        if rel is not None and rx.fullmatch(rel):
            return name
    return "other"


def _mark(line: str) -> re.Match | None:
    return _MARK.fullmatch(line)


def _open(line: str) -> re.Match | None:
    m = _FENCE.fullmatch(line)
    if m is None or (m.group("fence")[0] == "`" and "`" in m.group("info")):
        return None  # a backtick fence's info string never holds a backtick
    return m


def _closes(line: str, fence: str) -> bool:
    """A bare fence of the same character, at least as long as the opening one, closes it: a block opened with a
    longer fence keeps shorter nested fences as content."""
    s = line.strip()
    return len(s) >= len(fence) and s == s[0] * len(s) and s[0] == fence[0]


def _close_of(lines: list[str], start: int, fence: str) -> int | None:
    return next(
        (j for j in range(start + 1, len(lines)) if _closes(lines[j], fence)),
        None,
    )


def _dedent(lines: list[str], indent: str) -> list[str]:
    if not indent:
        return lines
    n = len(indent)
    return [ln[n:] if ln[:n].strip() == "" else ln.lstrip(" \t") for ln in lines]


def _trim(lines: list[str], a: int, b: int) -> tuple[int, int]:
    """Lines a..b without leading and trailing blank lines (the ledger's content rule)."""
    while a < b and not lines[a].strip():
        a += 1
    while b > a and not lines[b - 1].strip():
        b -= 1
    return a, b


def _plain(rel: str) -> bool:
    return bool(rel) and not (
        rel.startswith("/")
        or "\\" in rel
        or any(part in ("", ".", "..") for part in rel.split("/"))
    )


def importable_name(rel: str) -> tuple[str, str | None]:
    """A candidate's or case's module name must be a Python identifier for the writer to import it. A name that is
    not one is recovered under the name with each other character replaced by ``_`` (a leading digit gets a ``_``
    prefix); returns ``(path, original path or None)``. The case file of the same name gets the same new name.
    """
    if kind(rel) not in ("candidate", "case"):
        return rel, None
    folder, name = rel.split("/", 1)
    stem, dot, ext = name.rpartition(".")
    if stem.isidentifier():
        return rel, None
    fixed = re.sub(r"\W", "_", stem, flags=re.ASCII) or "_"
    if fixed[0].isdigit():
        fixed = "_" + fixed
    return f"{folder}/{fixed}{dot}{ext}", rel


def allowed(rel: str) -> str | None:
    """Why *rel* is refused, or None."""
    if not _plain(rel):
        return "not a plain relative path"
    if not any(rx.fullmatch(rel) for rx in ALLOWED):
        return "outside summary.md, candidates/*.py, cases/*.json, notes.md, procedure.py and broke.md"
    return None


def _row(form, rel, lines, a, b, *, start, unclosed=False, marker=None) -> dict:
    a, b = _trim(lines, a, b)
    body = lines[a:b]
    content = "\n".join(body)
    row = {
        "form": form,
        "path": rel,
        "lines": [a, b],
        "fence_line": start,
        "content": content + "\n" if body else "",
        "content_sha256": _sha(content),
        "unclosed": unclosed,
    }
    if marker:
        row["marker"] = marker
    return row


def _no_file(form, rel, at, stop, reason, unclosed=False) -> dict:
    return {
        "form": form,
        "path": rel,
        "lines": [at, stop],
        "fence_line": at,
        "content": None,
        "content_sha256": None,
        "unclosed": unclosed,
        "dropped": reason,
    }


def _marker_style(m: re.Match) -> str:
    if m.group("hash"):
        return "hash"
    if m.group("slashes"):
        return "slashes"
    return "html" if m.group("html") else "plain"


def _scan(text: str) -> list[dict]:
    """Every block in *text*, in order, before any path policy.

    Forms:

    * ``fence``: the fence's info string names the file (``path: X`` or ``<lang> path: X``);
    * ``inner``: the block's first non-blank line is a path line (``path: X``, ``# path: X``, ``// path: X``,
      ``<!-- path: X -->``), which is removed from the content;
    * ``line_before``: the line right before the fence is a path line;
    * ``split``: a block holding only a path line, closed, then (after blank lines only) the content's own block;
    * ``split_unfenced``: a block holding only a path line, then plain text with no further fence, to the end.

    A fence closes at the first later bare fence of the same character at least as long as it. A fence that never
    closes runs to the end of the text. An indented fence's content loses that indentation. A block that names no
    file, a path line no block takes, and a path-only block with no content are recorded as dropped.
    """
    lines = text.split("\n")
    n = len(lines)
    out: list[dict] = []
    i = 0
    used: set[int] = set()  # path lines a block has taken
    covered: set[int] = set()  # lines inside a block, fences included
    pending: tuple[str, int, int] | None = (
        None  # (path, the line its block opens on, the path block's line)
    )
    while i < n:
        m = _open(lines[i])
        if m is None:
            i += 1
            continue
        fence, indent, info = m.group("fence"), m.group("indent"), m.group("info")
        close = _close_of(lines, i, fence)
        end = n if close is None else close
        # the body in the text's own numbering: view[k] is line k, with the fence's indentation removed
        view = lines[: i + 1] + _dedent(lines[i + 1 : end], indent) + lines[end:]
        first = next((k for k in range(i + 1, end) if view[k].strip()), None)
        named = _INFO.fullmatch(info)
        inner = _mark(view[first]) if first is not None else None
        rest_blank = first is None or not any(
            view[k].strip() for k in range(first + 1, end)
        )
        nxt = n if close is None else close + 1
        covered.update(range(i, nxt))
        rel, form, a, marker, path_only = None, None, i + 1, None, False
        if named:
            rel, form = named.group("rel"), "fence"
            if inner and inner.group("rel") == rel:
                a = first + 1
            path_only = first is None or (
                inner is not None and inner.group("rel") == rel and rest_blank
            )
        elif inner:
            rel, form, a, marker = (
                inner.group("rel"),
                "inner",
                first + 1,
                _marker_style(inner),
            )
            path_only = rest_blank
        if pending is not None and pending[1] == i:
            path, _line, at = pending
            pending = None
            if rel is None:
                rel, form, a = path, "split", i + 1
            elif rel != path:
                out.append(
                    _no_file(
                        "split",
                        path,
                        at,
                        at,
                        "path_block_then_block_naming_another_path",
                    ),
                )
        if (
            rel is None
            and i > 0
            and (i - 1) not in used
            and (before := _mark(lines[i - 1]))
        ):
            rel, form, a = before.group("rel"), "line_before", i + 1
            used.add(i - 1)
        if rel is not None and path_only and form in ("fence", "inner"):
            # a block holding only its path: the content is the next block, or the plain text to the end
            k = nxt
            while k < n and not lines[k].strip():
                k += 1
            if k < n and _open(lines[k]):
                pending = (rel, k, i)
                i = k
                continue
            if k < n and not any(_open(lines[x]) for x in range(k, n)):
                out.append(_row("split_unfenced", rel, lines, k, n, start=i))
                covered.update(range(k, n))
                break
            reason = (
                "path_block_without_content"
                if k >= n
                else "path_block_then_text_then_block"
            )
            out.append(_no_file("split", rel, i, nxt, reason, unclosed=close is None))
            i = nxt
            continue
        if form is None:
            row = _row(None, None, view, i + 1, end, start=i, unclosed=close is None)
            row["dropped"] = "no_path"
        else:
            row = _row(
                form,
                rel,
                view,
                a,
                end,
                start=i,
                unclosed=close is None,
                marker=marker,
            )
        out.append(row)
        i = nxt
    # a path line that no block took names nothing: recorded, so the ledger shows it
    for k, line in enumerate(lines):
        if k not in used and k not in covered and (mk := _mark(line)):
            out.append(
                _no_file(
                    "line_before",
                    mk.group("rel"),
                    k,
                    k,
                    "path_line_without_block",
                ),
            )
    out.sort(key=lambda r: r["fence_line"])
    return out


def _decide(rows: list[dict], source: str, offset: int = 0) -> list[dict]:
    """Apply the path policy to scanned rows: parsed, refused (path not allowed) or dropped (no file), each with a
    reason (a parsed block's reason is None, or says why it was renamed)."""
    blocks = []
    for row in rows:
        rel = row["path"]
        b = {
            "index": offset + len(blocks),
            "form": row["form"],
            "path": rel[:200] if isinstance(rel, str) else None,
            "kind": kind(rel),
            "status": "parsed",
            "reason": None,
            "content_sha256": row["content_sha256"],
            "source": source,
            "lines": row["lines"],
            "content": row["content"],
        }
        for key in ("marker", "unclosed"):
            if row.get(key):
                b[key] = row[key]
        if row.get("dropped"):
            b["status"], b["reason"] = "dropped", row["dropped"]
        else:
            if _plain(rel):
                fixed, original = importable_name(rel)
                if original is not None:
                    b.update(
                        path=fixed,
                        renamed_from=original,
                        kind=kind(fixed),
                        reason=RENAMED,
                    )
            why = allowed(b["path"])
            if why:
                code = "path_not_plain" if not _plain(rel) else "path_not_allowed"
                b.update(status="refused", reason=code, refusal=why)
        if b["status"] != "parsed":
            b["content"] = None
        blocks.append(b)
    return blocks


def _supersede(blocks: list[dict]) -> None:
    """A later parsed block for the same path replaces an earlier one, which is recorded as dropped."""
    last: dict[str, dict] = {}
    for b in blocks:
        if b["status"] != "parsed":
            continue
        prev = last.get(b["path"])
        if prev is not None:
            prev.update(
                status="dropped",
                reason=f"replaced_by_block_{b['index']}",
                content=None,
            )
        last[b["path"]] = b


def counts(blocks: list[dict]) -> dict:
    c = {"blocks": len(blocks), **{s: 0 for s in STATUSES}}
    for b in blocks:
        c[b["status"]] += 1
    return c


def ledger(blocks: list[dict]) -> dict:
    """The hop record of *blocks*: every block without its content, and the counts (blocks found = parsed + refused
    + dropped). A reply without blocks says so in ``note``. :func:`write_record` adds the episode and the reply's
    digest to make ``fork-blocks.json``."""
    doc = {
        "blocks": [{k: v for k, v in b.items() if k != "content"} for b in blocks],
        "counts": counts(blocks),
    }
    if not blocks:
        doc["note"] = "no blocks"
    return doc


def write_record(
    staging: Path | str,
    episode_id: str | None,
    reply_text: str,
    hop_record: dict,
    turns: list | None = None,
) -> dict:
    """Write the raw reply (``fork-reply.json``) and the block ledger (``fork-blocks.json``) inside the episode's
    staging dir *staging*, which lives under the state dir, outside Git.

    ``fork-reply.json`` holds ``episode``, ``reply_text`` (every assistant turn's text joined by ``"\\n"``, in
    order), ``turns`` (each turn's content; *turns* may be the turns' message dicts, whose tool calls are then kept
    in ``tool_calls``, one list per turn) and ``reply_sha256`` (the sha256 of ``reply_text``). ``fork-blocks.json``
    is *hop_record* with ``episode`` and ``reply_sha256``. Returns the ``fork-blocks.json`` document.
    """
    staging = Path(staging)
    digest = _sha(reply_text)
    raw_doc: dict = {
        "episode": episode_id,
        "reply_text": reply_text,
        "turns": (
            [reply_text]
            if turns is None
            else [message_text(t) if isinstance(t, dict) else t for t in turns]
        ),
        "reply_sha256": digest,
    }
    if turns is not None and any(
        isinstance(t, dict) and t.get("tool_calls") for t in turns
    ):
        raw_doc["tool_calls"] = [
            (t.get("tool_calls") or []) if isinstance(t, dict) else [] for t in turns
        ]
    raw = (json.dumps(raw_doc, indent=1, sort_keys=True, default=str) + "\n").encode(
        "utf-8",
        "surrogatepass",
    )
    fd = os.open(
        staging / REPLY_JSON,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
        0o644,
    )
    with os.fdopen(fd, "wb") as fh:
        fh.write(raw)
    doc = {"episode": episode_id, "reply_sha256": digest, **hop_record}
    atomic_json(staging / BLOCKS_JSON, doc)
    return doc


def parse_reply(text: str) -> tuple[list[dict], dict]:
    """``(blocks, record)`` for one reply *text* (a reply's message contents joined by ``"\\n"``).

    *blocks* lists every block found, in order, each a dict: ``index``, ``form`` (:data:`FORMS`, or None for a block
    naming no file), ``path``, ``kind`` (candidate, case, notes, other), ``status`` (parsed, refused or dropped),
    ``reason`` (None when parsed as named), ``content_sha256`` (sha256 of the content's lines joined by ``"\\n"``,
    without the path line, the fence lines and leading or trailing blank lines), ``lines`` (that content's line
    range ``[a, b)`` in *text* split on newlines), ``content`` (the file's text with a final newline; None unless
    parsed), ``source``, and where they apply ``renamed_from``, ``marker``, ``unclosed`` and ``refusal``. *record*
    is the hop record (:func:`ledger`): ``blocks`` without their content, ``counts``, and ``note`` when there are no
    blocks; :func:`write_record` adds the episode and the reply's digest.
    """
    text = text if isinstance(text, str) else ""
    blocks = _decide(_scan(text), "text")
    _supersede(blocks)
    return blocks, ledger(blocks)


# --- the reply's messages -----------------------------------------------------------------------------------------


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


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(
        p.get("text", "")
        for p in (content or [])
        if isinstance(p, dict) and isinstance(p.get("text"), str)
    )


def message_blocks(message: dict, offset: int = 0) -> list[dict]:
    """Every block in *message*'s text and in every tool call's arguments (nothing is executed), with ``source``
    ``text`` or ``tool_call:<name>``; indexes continue from *offset*. :func:`settle` applies replacement across
    texts."""
    out: list[dict] = []
    text = message_text(message)
    if text:
        out += _decide(_scan(text), "text", offset)
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        name = str(fn.get("name") or "?")[:64]
        for s in _strings(fn.get("arguments")):
            out += _decide(_scan(s), f"tool_call:{name}", offset + len(out))
    return out


def reply_text(turns: list[dict]) -> str:
    """The fork's reply text: every assistant turn's text joined by ``"\\n"``, in order."""
    return "\n".join(message_text(t) for t in turns)


def parse_turns(turns: list[dict]) -> tuple[list[dict], dict, str]:
    """``(blocks, hop record, reply text)`` for the fork's assistant *turns*, live and replayed alike: the joined
    reply text parsed once (:func:`parse_reply`), then the blocks in every tool call's arguments (source
    ``tool_call:<name>``; nothing is executed), with a later block for the same path replacing an earlier one.
    """
    text = reply_text(turns)
    blocks = _decide(_scan(text), "text") if text else []
    for turn in turns:
        for call in turn.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            name = str(fn.get("name") or "?")[:64]
            for arg in _strings(fn.get("arguments")):
                blocks += _decide(_scan(arg), f"tool_call:{name}", len(blocks))
    settle(blocks)
    return blocks, ledger(blocks), text


def settle(blocks: list[dict]) -> list[dict]:
    """Index *blocks* gathered from several texts in order and apply replacement across them (a later block wins)."""
    for k, b in enumerate(blocks):
        b["index"] = k
    _supersede(blocks)
    return blocks


def write_blocks(staging: Path, blocks: list[dict]) -> tuple[list, list, bool]:
    """Write the parsed *blocks* under *staging* within the operational quota; a block over the quota or under a
    linked folder becomes refused. Returns ``(files, refused, quota reached)`` as ``fork.json`` lists them.
    """
    files, refused, total, over = [], [], 0, False
    for b in blocks:
        if b["status"] == "refused":
            refused.append(
                {
                    "path": b["path"],
                    "why": b.get("refusal") or b["reason"],
                    "from": b["source"],
                },
            )
        if b["status"] != "parsed":
            continue
        data = b["content"].encode("utf-8", "replace")
        if len(files) >= QUOTA_FILES or total + len(data) > QUOTA_BYTES:
            b.update(
                status="refused",
                reason="quota",
                refusal="operational quota reached",
                content=None,
            )
            refused.append(
                {
                    "path": b["path"],
                    "why": "operational quota reached",
                    "from": b["source"],
                },
            )
            over = True
            continue
        dest = staging / b["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.parent.is_symlink():
            b.update(
                status="refused",
                reason="parent_is_link",
                refusal="parent is a link",
                content=None,
            )
            refused.append(
                {"path": b["path"], "why": "parent is a link", "from": b["source"]},
            )
            continue
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        total += len(data)
        files.append({"path": b["path"], "from": b["source"], "bytes": len(data)})
    return files, refused, over


def write_files(
    staging: Path,
    found: list[tuple[str, str, str]],
) -> tuple[list, list, bool]:
    """Write ``(path, content, source)`` triples under *staging* within the operational quota (writer-side analysts
    stage through this); a later triple for the same path replaces an earlier one. ``(files, refused, quota
    reached)``."""
    latest: dict[str, tuple[str, str]] = {}
    refused = []
    for rel, body, source in found:
        why = allowed(rel)
        if why:
            refused.append({"path": rel[:200], "why": why, "from": source})
        else:
            latest[rel] = (body, source)
    blocks = [
        {
            "path": rel,
            "status": "parsed",
            "reason": None,
            "source": source,
            "content": body,
        }
        for rel, (body, source) in latest.items()
    ]
    files, more, over = write_blocks(staging, blocks)
    return files, refused + more, over


def atomic_json(path: Path, doc: dict) -> None:
    """Write *doc* to *path* through a temporary file and a rename, so a reader never sees half of it."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --- the turns ----------------------------------------------------------------------------------------------------


class WallTimeout(Exception):
    """The fork's wall bound was reached."""


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


def _stop_alarm() -> None:
    try:
        signal.setitimer(signal.ITIMER_REAL, 0)
    except (ValueError, OSError):  # not the main thread: no timer was set here
        pass


def run(
    req: dict,
    staging: Path,
    post=_post,
    *,
    wall_s: float = WALL_GUARD_S,
    clock=time.monotonic,
) -> dict:
    """The fork's turns and files; returns the ``fork.json`` document (also written to *staging*, last). It records
    ``seconds``, the fork's own wall time, and ``summary_bytes``, the size of the ``summary.md`` it staged (None when
    it staged none).

    *wall_s* bounds the fork: past it no further turn starts, and :func:`main` also raises :class:`WallTimeout` from
    a timer inside a turn. Either way the fork stages what it has and records status ``"timeout"``.
    """
    doc = {
        "episode": req.get("episode"),
        "status": "ok",
        "model": (req.get("body") or {}).get("model"),
        "turns": 0,
        "usage": {},
        "usd": None,
        "files": [],
        "refused_files": [],
        "wall_guard_s": wall_s,
    }
    body = dict(req["body"])
    messages = list(body["messages"])
    replies: list[dict] = []
    usage: dict = {}
    started = clock()
    try:
        while True:
            if doc["turns"] >= TURN_GUARD:
                doc["status"] = f"error: operational turn guard ({TURN_GUARD}) reached"
                break
            if clock() - started >= wall_s:
                raise WallTimeout()
            headers = dict(req["headers"])
            headers["x-unify-request"] = str(doc["turns"])
            headers["x-unify-msg-count"] = str(len(messages))
            resp = post(req["url"], headers, {**body, "messages": messages})
            doc["turns"] += 1
            _add_usage(usage, resp.get("usage") or {})
            message = ((resp.get("choices") or [{}])[0].get("message")) or {}
            kept = {
                k: message[k] for k in ("role", "content", "tool_calls") if k in message
            }
            replies.append(kept)
            calls = message.get("tool_calls") or []
            if not calls:
                break
            messages.append(kept)
            messages += [
                {"role": "tool", "tool_call_id": c.get("id"), "content": TOOL_REPLY}
                for c in calls
            ]
    except WallTimeout:
        doc["status"] = "timeout"
        doc["detail"] = f"the fork's wall bound ({wall_s:g} s) was reached"
    except Exception as exc:  # noqa: BLE001 - recorded, never raised
        doc["status"] = f"error: {type(exc).__name__}"
    _stop_alarm()
    blocks, _hop, text = parse_turns(replies)
    files, refused, over = write_blocks(staging, blocks)
    doc["files"], doc["refused_files"] = files, refused
    # the captured attempt's files (the harness captures procedure.py at once once the fork settles)
    written = {f.get("path") for f in files if isinstance(f, dict)}
    doc["procedure"] = "procedure.py" in written
    doc["broke"] = "broke.md" in written
    if over and doc["status"] == "ok":
        doc["status"] = "ok: operational quota reached"
    usd = usage.pop("_usd", None) if usage else None
    doc["usage"], doc["usd"] = usage, (str(usd) if usd is not None else None)
    doc["summary_bytes"] = next(
        (f["bytes"] for f in files if f["path"] == SUMMARY_FILE),
        None,
    )
    doc["seconds"] = round(max(0.0, clock() - started), 3)
    record = write_record(staging, req.get("episode"), text, ledger(blocks), replies)
    doc["reply_sha256"], doc["blocks"] = record["reply_sha256"], counts(blocks)
    atomic_json(staging / FORK_JSON, doc)
    return doc


def _alarm(_signum, _frame):
    raise WallTimeout()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", type=Path, required=True)
    a = ap.parse_args(argv)
    req = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, WALL_GUARD_S)
    run(req, a.staging)
    return 0


if __name__ == "__main__":
    sys.exit(main())
