"""Episode records (spec §5): one append-only commit per request in the episodes repo."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .blobs import BlobStore
from .gitio import GitError, Repo
from .redact import Redactor


@dataclass
class Cell:
    index: int
    code: str
    output: str
    error: str | None = None
    language: str = "python"  # the cell's language: python | bash | ...


ACTION_KINDS = frozenset({"tool", "shell", "worktree", "dialogue"})
_NOT_NAME = re.compile(r"[^a-z0-9_]+")


def env_channel(kind: str, channel: str) -> str | None:
    """The memory channel (``env/<name>``) whose items may cover an action of *kind* on *channel*.

    A channel key without a ``:`` names the memory channel itself (every tool namespace, and any other
    kind recorded with a bare key). A kind-qualified key ``<kind>:<key>`` (``shell:uv``,
    ``worktree:workspace``, ``dialogue:user``) maps to ``<kind>_<key>``, lower-cased, with every run of
    characters other than ``a-z``, ``0-9`` and ``_`` replaced by one ``_`` (``shell:python3.12`` is
    ``shell_python3_12``). A qualifier naming another kind maps to nothing (None). Deterministic and
    structural: the key is how the harness saw the surface, never a task.
    """
    if not isinstance(channel, str) or not channel:
        return None
    if kind == "tool" or ":" not in channel:
        return channel
    prefix, key = channel.split(":", 1)
    if prefix != kind:
        return None
    name = _NOT_NAME.sub("_", key.lower()).strip("_")
    return f"{kind}_{name}" if name else None


@dataclass
class Action:
    cell: int
    channel: str
    method: str
    args: list
    kwargs: dict
    response: Any = None
    status: str = "unrecorded"  # ok | error | unrecorded
    effect: str = "unknown"  # read | write | unknown
    error: str | None = None
    # Which of the four action kinds this is (spec §B): a typed tool call, a
    # shell command, a worktree file change, or a dialogue turn. Rows recorded
    # before kinds existed load as "tool".
    kind: str = "tool"

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(
                f"action kind must be one of {sorted(ACTION_KINDS)}, got {self.kind!r}",
            )


@dataclass
class CostRow:
    purpose: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    usd: str  # decimal string, or "unknown"


@dataclass
class Episode:
    episode_id: str
    started_at: str
    ended_at: str
    build: str
    model: str
    effort: str
    regime: str
    memory_main: str
    worktree_before: str | None
    worktree_after: str | None
    request: list[str]
    transcript: list[dict]
    cells: list[Cell]
    actions: list[Action]
    memory_diff: str = ""
    # only from integration.adapters.worktree.WorkTreeRecorder.diff (the records' redactor): text
    # hunks, never a binary's bytes (R28)
    worktree_diff: str = ""
    costs: list[CostRow] = field(default_factory=list)
    fingerprints: dict = field(default_factory=dict)
    # Assistant replies in order. ``request`` holds every user message in order: the first is
    # the request, the rest are observations.
    replies: list[str] = field(default_factory=list)
    # How the request used the memory library (``analysis.use.request_use``), written as
    # ``memory_use.json``; None (no file) when nothing computed it.
    memory_use: dict | None = None


def episode_dir(ep: Episode) -> str:
    return f"{ep.started_at[0:4]}/{ep.started_at[5:7]}/{ep.episode_id}"


def _jsonl(rows: list[Any]) -> str:
    return "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows)


#: A reference to a value stored as a blob ({"__capped__": {excerpt, blob, bytes}}, BlobStore.cap_text).
_REF = "__capped__"
#: Record format 2 (UNIFY_MEMORY_V21): a recorded value that is itself a dict with a _REF or _LITERAL key is
#: written as {_LITERAL: value}, so it is never read as a reference (RUNTIME's T7 checklist, point 2).
_LITERAL = "__literal__"
#: The record format a v2.1 writer stamps in meta.json; a record without the key is format 1.
RECORD_FORMAT = 2


class EpisodeRecordError(ValueError):
    """A recorded episode that cannot be loaded as recorded: a blob it references is missing, or its bytes do
    not hash to its id. Loading fails closed rather than return a value with a hole in it.
    """


def _blob_value(blobs: BlobStore, blob: Any, episode_id: str) -> Any:
    sha = str(blob)
    if not blobs.has(sha):
        raise EpisodeRecordError(f"episode {episode_id}: blob {sha} missing")
    data = blobs.get(sha)
    if hashlib.sha256(data).hexdigest() != sha:
        raise EpisodeRecordError(f"episode {episode_id}: blob {sha} corrupt")
    return json.loads(data.decode())


class EpisodeWriter:
    def __init__(
        self,
        repo: Repo,
        blobs: BlobStore,
        redactor: Redactor,
        response_cap: int = 16384,
        *,
        v21: bool = False,
    ) -> None:
        self.repo, self.blobs, self.redactor, self.cap = (
            repo,
            blobs,
            redactor,
            response_cap,
        )
        # UNIFY_MEMORY_V21: record format 2 (every large value as a blob; values like a reference escaped)
        self.v21 = v21

    def _capped(self, value: Any) -> Any:
        text = json.dumps(value, sort_keys=True, default=str)
        out = self.blobs.cap_text(text, self.cap)
        return value if "text" in out else {"__capped__": out}

    def _stored(self, value: Any) -> Any:
        """Format 2: *value* (already redacted) inline, escaped when it looks like a reference, or a reference
        to its canonical JSON stored whole as a blob when that is over the inline cap.
        """
        out = self.blobs.cap_text(
            json.dumps(value, sort_keys=True, default=str),
            self.cap,
        )
        if "text" not in out:
            return {_REF: out}
        if isinstance(value, dict) and (_REF in value or _LITERAL in value):
            return {_LITERAL: value}
        return value

    def write(self, ep: Episode) -> str:
        r = self.redactor
        meta = {
            k: getattr(ep, k)
            for k in (
                "episode_id",
                "started_at",
                "ended_at",
                "build",
                "model",
                "effort",
                "regime",
                "memory_main",
                "worktree_before",
                "worktree_after",
                "fingerprints",
            )
        }
        actions = []
        for a in ep.actions:
            row = r.obj(asdict(a))
            if self.v21:  # format 2: args, kwargs and the response, each stored whole
                row["args"] = [self._stored(x) for x in row["args"]]
                row["kwargs"] = {k: self._stored(v) for k, v in row["kwargs"].items()}
                row["response"] = self._stored(row["response"])
            else:
                row["response"] = self._capped(row["response"])
            actions.append(row)
        request = r.obj(ep.request)
        if self.v21:
            meta["record_format"] = RECORD_FORMAT
            request = [self._stored(x) for x in request]
        files = {
            "meta.json": json.dumps(r.obj(meta), sort_keys=True, indent=1) + "\n",
            "request.json": json.dumps(request, indent=1) + "\n",
            "replies.json": json.dumps(r.obj(ep.replies), indent=1) + "\n",
            "transcript.jsonl": _jsonl(r.obj(ep.transcript)),
            "cells.jsonl": _jsonl([r.obj(asdict(c)) for c in ep.cells]),
            "actions.jsonl": _jsonl(actions),
            "memory.diff": r.text(ep.memory_diff),
            "worktree.diff": r.text(ep.worktree_diff),
            "cost.jsonl": _jsonl([asdict(c) for c in ep.costs]),
        }
        if ep.memory_use is not None:
            files["memory_use.json"] = (
                json.dumps(r.obj(ep.memory_use), sort_keys=True, indent=1) + "\n"
            )
        rel = episode_dir(ep)
        base = self.repo.head()
        with self.repo.temp_checkout() as wt:
            d = wt / rel
            if d.exists():
                raise FileExistsError(f"episode {ep.episode_id} already recorded")
            d.mkdir(parents=True)
            for name, text in files.items():
                (d / name).write_text(text)
            sha = self.repo.commit_all(
                wt,
                f"episode {ep.episode_id}",
                {"Episode": ep.episode_id},
            )
        self.repo.fast_forward("main", sha, expected_old=base)
        return sha


def _uncap(value: Any, blobs: BlobStore, episode_id: str = "") -> Any:
    """Format 1: an action response stored as a blob, resolved (a missing or corrupt blob fails closed)."""
    if isinstance(value, dict) and _REF in value:
        return _blob_value(blobs, value[_REF]["blob"], episode_id)
    return value


def _decode2(value: Any, blobs: BlobStore, episode_id: str) -> Any:
    """Format 2, strictly: a reference only when _REF is the dict's only key; an escaped literal unwrapped."""
    if isinstance(value, dict) and len(value) == 1:
        if _REF in value and isinstance(value[_REF], dict) and "blob" in value[_REF]:
            return _blob_value(blobs, value[_REF]["blob"], episode_id)
        if _LITERAL in value:
            return value[_LITERAL]
    return value


def decode_action_row(row: dict, blobs: BlobStore, episode_id: str) -> dict:
    """One format-2 ``actions.jsonl`` row with its args, kwargs and response resolved (for readers of raw
    records, such as the offline use analysis; a missing or corrupt blob raises EpisodeRecordError).
    """
    row = dict(row)
    row["args"] = [_decode2(x, blobs, episode_id) for x in row.get("args") or []]
    row["kwargs"] = {
        k: _decode2(v, blobs, episode_id) for k, v in (row.get("kwargs") or {}).items()
    }
    row["response"] = _decode2(row.get("response"), blobs, episode_id)
    return row


def load_episode(repo: Repo, rev: str, rel: str, blobs: BlobStore) -> Episode:
    def read(name: str) -> str:
        return repo.show(rev, f"{rel}/{name}").decode()

    meta = json.loads(read("meta.json"))
    fmt = meta.pop("record_format", 1)
    eid = str(meta.get("episode_id", ""))
    if isinstance(fmt, bool) or fmt not in (
        1,
        RECORD_FORMAT,
    ):  # fail closed: never read another format as 1
        raise EpisodeRecordError(f"episode {eid}: unknown record_format {fmt!r}")

    def lines(name: str) -> list[Any]:
        return [json.loads(ln) for ln in read(name).splitlines() if ln.strip()]

    try:  # absent from episodes recorded before replies were kept
        replies = json.loads(read("replies.json"))
    except GitError:
        replies = []
    try:  # absent from episodes recorded without use telemetry; unreadable is the same as absent
        memory_use = json.loads(read("memory_use.json"))
    except (GitError, ValueError, RecursionError):
        memory_use = None
    actions = []
    request = json.loads(read("request.json"))
    if fmt == RECORD_FORMAT:
        request = [_decode2(x, blobs, eid) for x in request]
    for row in lines("actions.jsonl"):
        if fmt == RECORD_FORMAT:
            row = decode_action_row(row, blobs, eid)
        else:
            row["response"] = _uncap(row["response"], blobs, eid)
        actions.append(Action(**row))
    return Episode(
        **meta,
        request=request,
        transcript=lines("transcript.jsonl"),
        cells=[Cell(**c) for c in lines("cells.jsonl")],
        actions=actions,
        memory_diff=read("memory.diff"),
        worktree_diff=read("worktree.diff"),
        costs=[CostRow(**c) for c in lines("cost.jsonl")],
        replies=replies,
        memory_use=memory_use if isinstance(memory_use, dict) else None,
    )
