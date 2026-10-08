"""Episode records (spec §5): one append-only commit per request in the episodes repo."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .blobs import BlobStore
from .gitio import Repo
from .redact import Redactor


@dataclass
class Cell:
    index: int
    code: str
    output: str
    error: str | None = None


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
    worktree_diff: str = ""
    costs: list[CostRow] = field(default_factory=list)
    fingerprints: dict = field(default_factory=dict)


def episode_dir(ep: Episode) -> str:
    return f"{ep.started_at[0:4]}/{ep.started_at[5:7]}/{ep.episode_id}"


def _jsonl(rows: list[Any]) -> str:
    return "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows)


class EpisodeWriter:
    def __init__(
        self,
        repo: Repo,
        blobs: BlobStore,
        redactor: Redactor,
        response_cap: int = 16384,
    ) -> None:
        self.repo, self.blobs, self.redactor, self.cap = (
            repo,
            blobs,
            redactor,
            response_cap,
        )

    def _capped(self, value: Any) -> Any:
        text = json.dumps(value, sort_keys=True, default=str)
        out = self.blobs.cap_text(text, self.cap)
        return value if "text" in out else {"__capped__": out}

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
            row["response"] = self._capped(row["response"])
            actions.append(row)
        files = {
            "meta.json": json.dumps(r.obj(meta), sort_keys=True, indent=1) + "\n",
            "request.json": json.dumps(r.obj(ep.request), indent=1) + "\n",
            "transcript.jsonl": _jsonl(r.obj(ep.transcript)),
            "cells.jsonl": _jsonl([r.obj(asdict(c)) for c in ep.cells]),
            "actions.jsonl": _jsonl(actions),
            "memory.diff": r.text(ep.memory_diff),
            "worktree.diff": r.text(ep.worktree_diff),
            "cost.jsonl": _jsonl([asdict(c) for c in ep.costs]),
        }
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


def _uncap(value: Any, blobs: BlobStore) -> Any:
    if isinstance(value, dict) and "__capped__" in value:
        return json.loads(blobs.get(value["__capped__"]["blob"]).decode())
    return value


def load_episode(repo: Repo, rev: str, rel: str, blobs: BlobStore) -> Episode:
    def read(name: str) -> str:
        return repo.show(rev, f"{rel}/{name}").decode()

    meta = json.loads(read("meta.json"))

    def lines(name: str) -> list[Any]:
        return [json.loads(ln) for ln in read(name).splitlines() if ln.strip()]

    actions = []
    for row in lines("actions.jsonl"):
        row["response"] = _uncap(row["response"], blobs)
        actions.append(Action(**row))
    return Episode(
        **meta,
        request=json.loads(read("request.json")),
        transcript=lines("transcript.jsonl"),
        cells=[Cell(**c) for c in lines("cells.jsonl")],
        actions=actions,
        memory_diff=read("memory.diff"),
        worktree_diff=read("worktree.diff"),
        costs=[CostRow(**c) for c in lines("cost.jsonl")],
    )
