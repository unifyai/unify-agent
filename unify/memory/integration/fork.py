"""The episode-end fork.

After the reply has been returned and the episode recorded, the actor's own conversation is continued once more by a
separate process: its last request's messages and tools unchanged, so the provider serves the prefix from cache,
plus one user message after that prefix (:func:`fork_message`): the fixed instruction (:func:`fork_instruction`) and
the recorded episode's part list (:func:`part_list`), so the pointers in the fork's ``summary.md`` name parts the
writer can read. The process writes what the reply stages into ``<state>/memory-staging/<episode>/`` and nothing
else.

* **The boundary:**
  - :mod:`.fork_worker` runs by path with the system python (``/usr/bin/python3 -I /s/fork_worker.py``). It is
    standard library only.
  - The request reaches it on stdin, a fresh pipe, never argv, the environment or a file.
  - Its environment is PATH and PYTHONDONTWRITEBYTECODE, nothing else.
  - It calls the fork lane (``/fork/v1``): its base URL and token come from the descriptor the launcher hands this
    controller (``switch.settle_fork_route_env``), and the token reaches the worker only inside its stdin request. Without the lane there is no fork (``refused: no fork lane``): never the actor's route, never a
    provider directly, never a provider key.
* **Mounts:** a bubblewrap box with a read-only root (``/usr``) and a private ``/tmp``. Home, the state dir and the
  library are not mounted. The worker file is read-only at ``/s``. The only writable bind is the episode's fresh
  staging dir.
  - It keeps only the network of the process that starts it (``--share-net``), which reaches the lane's loopback
    relay.
* It starts after the episode is recorded, from bytes serialised once. Nothing flows back to the reply or the
  transcript. Its relay (:mod:`unify.agents.fork_relay`) answers its calls to the run record.

**Settling and admission.** ``fork.json`` is written last and means the fork has settled, whatever its status. An
episode enters a pass only once it has settled (:class:`Admission`); an unsettled one waits. Settlement is always reached:

* the request reserves the staging dir before the episode is recorded (:func:`reserve`) and holds an exclusive lock
  on its ``fork.lock``; the lock passes to the worker process when it starts, so the lock is held exactly while the
  request or the worker is alive;
* the worker's own wall bound (:data:`.fork_worker.WALL_GUARD_S`) ends it with status ``"timeout"``;
* a worker that dies without writing ``fork.json`` (killed, out of memory) releases the lock with it; the first
  admission check that finds the lock free and no ``fork.json`` records status ``"lost"`` (:func:`settled`);
* a lock still held past :data:`STALE_S` is recorded as ``"timeout"`` and admitted, so no episode waits forever.

This keys on a per-episode signal only, never on the stream.

**Lifetime.** The fork runs in its own session, and :func:`finish` holds the CLI after the reply until the fork ends,
so a host whose sandbox ends every process of the controller with it never cuts a fork short, and the lane outlives
it. That wait is bounded by :data:`WAIT_GUARD_S`, an operational guard reported in ``fork.json``, not a design limit.

**Spend.** The calls carry ``x-unify-call-kind: actor_fork``, ``x-unify-session: fork.<episode>`` (what separates
their spend), ``x-unify-parent: <episode>`` and ``x-unify-episode`` (and a segment fork's ``x-unify-segment``). There
is no token cap; the lane's cap bounds the money.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..library_prompts import CANDIDATE_CONTRACT_IN_CODE, SUMMARY_FORMAT
from .fork_worker import (
    FORK_JSON,
    LOCK_FILE,
    QUOTA_BYTES,
    QUOTA_FILES,
    TURN_GUARD,
    atomic_json,
)

FORMAT_EXAMPLE = (
    "```path: summary.md\n"
    "## Asked\n"
    "Read a table of numbers from a text file and report its size.\n"
    "## Approach\n"
    '- Parsed the first line as the size (`cell:0` "int(lines[0])").\n'
    '- Skipped the header line, then parsed the first number (`cell:1` "for line in lines:").\n'
    "## Worked\n"
    '- Skipping lines until the first number (`cell:1` "size = 12").\n'
    "## Failed\n"
    '- Reading the size from the first line, which was a header (`cell:0` "ValueError: invalid literal for int()").\n'
    "## Outcome\n"
    'Reported the size 12; the record does not show whether it was right: not shown (`action:2` "size: 12").\n'
    "```\n"
    "\n"
    "```path: candidates/double.py\n"
    "class MemoryInputError(ValueError):\n"
    '    """An input of a shape this module was not built for."""\n'
    "\n"
    "\n"
    "def double(x):\n"
    '    """Return twice the number x."""\n'
    "    if isinstance(x, bool) or not isinstance(x, (int, float)):\n"
    '        raise MemoryInputError(f"x is a number, not {type(x).__name__}")\n'
    "    return 2 * x\n"
    "```\n"
    "\n"
    "```path: cases/double.json\n"
    '[{"inputs": [21], "expected": 42}, {"inputs": [-3], "expected": -6}]\n'
    "```\n"
    "\n"
    "```path: notes.md\n"
    "- Reading the size from the first line failed when the file had a header line; skip lines until the first "
    'number. From the error "ValueError: invalid literal for int()".\n'
    "```"
)

#: How the fork names the parts of its own conversation in its summary's pointers: it sees a conversation, while the
#: writer reads the recorded episode's parts. The part list that follows the instruction (:func:`part_list`) gives the
#: recorded names themselves.
NUMBERING = (
    "Here, the request is your first user message. `observation:<n>` is the n-th message from the user or the "
    "environment after it, counting from 0. `cell:<n>` is the n-th code cell you ran, counting from 0. `action:<n>` "
    "is the n-th tool call the harness ran for you, counting from 0."
)
#: The heading of the part list that follows the instruction when the recorded episode could be read.
PARTS_HEAD = (
    "The record of this episode has these parts, in this order, each with its first line or what it was, so you "
    "can match it to this conversation. Use exactly these names, in backticks, in the summary's pointers:"
)
#: The characters of a part's first line the part list shows.
PART_GIST_CHARS = 80
#: What the fork writes for the harness to capture (:mod:`..captures`): the procedure that did the work, in the form the
#: work needed (``value``, ``state`` or ``state_env``: :mod:`..capture_trace`), returning a result only when one was
#: handed back; and what broke.
CAPTURE_BULLETS = (
    "- `procedure.py`: the procedure that did this work, from code that ran here, as functions with docstrings, in "
    "the form that fits what the work needed:\n"
    "  - `run(state, env)` when the work acted through tools. `env` is this episode's tool namespace: "
    "`env.<channel>.<method>(...)` makes the call the record lists as an action of that channel and method (the part "
    "list below shows each action as `<kind> <channel>.<method>`); make every tool call through `env`. `state` is "
    "the episode so far: `state.request` (the first message), `state.messages` (every message from the user or the "
    "environment, in order), `state.last` (the latest one) and `state.data` (the JSON objects and arrays those "
    "messages contain, in order).\n"
    "  - `run(state)` when the work needed more than the first message and no tool.\n"
    "  - `read_input(request)` and `run(value)` when the whole job was in the first message: `read_input` takes the "
    "request (`request` is its text, `request.data` the list of the JSON objects and arrays it contains) and returns "
    "the value `run` works on.\n"
    "  `run` returns the result you handed back when you handed one back, and None when the outcome of the work is "
    "what it did (its tool calls, the files it changed). Where your code used a value you typed into a cell, read it "
    "from `state` or the request instead. If the work has conditions on its input or its result, write them as "
    "`applies(x)` and `holds(x, result)`, where `x` is what `run` takes first, each returning True or False. Write it "
    "whether or not the work was accepted; say which in its docstring, quoting the line of the record that shows it, "
    'or write "not shown".\n'
    "- `broke.md`: each thing that broke in this attempt and why, one line each, with the part name and a short quote "
    'from the record. Write "none" if nothing broke.\n'
)
#: The sentence of the instructions that says which calls run: the record's comms calls only (through the fork's relay,
#: :mod:`unify.agents.fork_relay`); any other tool call gets the reply saying tools are not available
#: (``fork_worker.TOOL_REPLY``).
RECORD_CALLS_ONLY = (
    "Call no tool except the record's: tools.send, tools.messages, tools.wait and tools.who, each a statement of its "
    "own in a cell (the record's section below says how they work); nothing else runs here."
)
#: The fork's instruction: what to write, its limits, and the one block format with an example. The per-reply time is
#: not stated, since nothing the fork writes can act on it.
INSTRUCTION = (
    "Your work on this request is finished and your reply has been sent; nothing you write now changes it. "
    "Now write down what would help with similar requests later, as files: write each file as a fenced block in "
    "your reply, in the format below. " + RECORD_CALLS_ONLY + "\n\n"
    "What to write:\n"
    "- `summary.md`: a short account of this episode for a reader who has not seen it, in the format below.\n"
    "- `candidates/<name>.py`: for code you wrote here that ran and worked and could be reused, a function "
    "with a docstring. <name> uses only letters, digits and underscores. Only code that actually ran here; never "
    "say code worked if it did not. " + CANDIDATE_CONTRACT_IN_CODE + "\n"
    "- `cases/<name>.json`, with the same <name>: the inputs that function had in this conversation and the "
    'outputs it should give, as a JSON list of objects with "inputs" (the list of arguments) and "expected" '
    "(the return value). Each value must appear in this conversation as it is (a tool's reply, a file's content "
    "or a cell's printed output), so it can be copied from the record; leave out a case with a value that does "
    "not.\n"
    "- `notes.md`: each thing that failed and why, and any other lesson, in general terms so it applies beyond this "
    "request, each quoting the short line or error message from this conversation it comes from. If nothing here "
    "is worth keeping, write `notes.md` saying so.\n"
    + CAPTURE_BULLETS
    + "\n"
    + SUMMARY_FORMAT
    + "\n\n"
    + NUMBERING
    + "\n\n"
    f"Limits: at most {QUOTA_FILES} files and {QUOTA_BYTES // 1024**2} MiB in all are kept. A reply that calls a "
    f"tool gets another turn, up to {TURN_GUARD} turns in all.\n\n"
    "Format: each file is one fenced block. Its opening line is three backticks followed by `path: ` and the "
    "file's relative path; then the file's full content; then a line of three backticks. If a file itself contains "
    "a line of three backticks, open and close its block with four. Example:\n\n"
    + FORMAT_EXAMPLE
)
#: A segment fork's instruction (``UNIFY_MEMORY=in_episode``, at a checkpoint inside the episode): the same summary and
#: capture contract, in a short form, for the cells since the last checkpoint; the work goes on after it.
SEGMENT_FORMAT_EXAMPLE = (
    "```path: summary.md\n"
    "## Asked\n"
    "Read a table of numbers from a text file and report its size.\n"
    "## Approach\n"
    '- Parsed the first line as the size (`cell:0` "int(lines[0])").\n'
    "## Worked\n"
    "- none so far\n"
    "## Failed\n"
    '- Reading the size from the first line, which was a header (`cell:0` "ValueError: invalid literal for int()").\n'
    "## Outcome\n"
    "Not finished: the work goes on (`cell:0` \"int(lines[0])\").\n"
    "```\n"
    "\n"
    "```path: broke.md\n"
    '- `cell:0`: the first line is a header, not the size ("ValueError: invalid literal for int()").\n'
    "```"
)
SEGMENT_INSTRUCTION = (
    "This is a pause in your work on this request, which goes on after it; nothing you write now changes it. Write "
    "down what would help with similar work later, for the part of the work since the last pause: the cells "
    "{cells}. Write each file as a fenced block in your reply, in the format below. " + RECORD_CALLS_ONLY + "\n\n"
    "What to write:\n"
    "- `summary.md`: a short account of this part of the episode for a reader who has not seen it, in the format "
    "below; its Outcome says the work is not finished, and what the record shows so far.\n"
    + CAPTURE_BULLETS
    + "\n"
    + SUMMARY_FORMAT
    + "\n\n"
    + NUMBERING
    + "\n\n"
    f"Limits: at most {QUOTA_FILES} files and {QUOTA_BYTES // 1024**2} MiB in all are kept. A reply that calls a "
    f"tool gets another turn, up to {TURN_GUARD} turns in all.\n\n"
    "Format: each file is one fenced block. Its opening line is three backticks followed by `path: ` and the "
    "file's relative path; then the file's full content; then a line of three backticks. If a file itself contains "
    "a line of three backticks, open and close its block with four. Example:\n\n"
    + SEGMENT_FORMAT_EXAMPLE
)
WORKER_FILE = Path(__file__).resolve().parent / "fork_worker.py"
BOX_WORKER = "/s/fork_worker.py"
STAGING_IN_BOX = "/staging"
#: Operational guard on the CLI's wait for the fork (run safety, reported; not a design limit).
WAIT_GUARD_S = 1800.0
WRITE_GUARD_S = 120.0
#: A staging dir with no lock file this long after it was made was never started: recorded as lost.
UNSTARTED_S = 120.0
#: A lock still held this long after the reservation: the fork is recorded as timed out and admitted.
STALE_S = WAIT_GUARD_S + 300.0
#: The file that keeps which episodes earlier passes rolled over (under the state dir).
ROLLED_FILE = "pass-admission.json"


def fork_instruction() -> str:
    """The fixed instruction appended to the actor's conversation for the fork (:data:`INSTRUCTION`)."""
    return INSTRUCTION


def segment_instruction(first_cell: int | None, last_cell: int | None) -> str:
    """A segment fork's instruction (:data:`SEGMENT_INSTRUCTION`) for the cells from *first_cell* to *last_cell*."""
    if first_cell is None or last_cell is None:
        cells = "since the last pause (no cell ran in it)"
    elif first_cell == last_cell:
        cells = f"`cell:{first_cell}`"
    else:
        cells = f"`cell:{first_cell}` to `cell:{last_cell}`"
    return SEGMENT_INSTRUCTION.replace("{cells}", cells)


def _first_line(text: object) -> str:
    line = next((ln.strip() for ln in str(text or "").splitlines() if ln.strip()), "")
    return line[:PART_GIST_CHARS] + (" ..." if len(line) > PART_GIST_CHARS else "")


def part_list(ep: Any) -> str:
    """The recorded episode's part list as the fork reads it: one line per part, in the order the writer's table of
    contents lists them (:func:`..writer_tools.part_names`), each with its first line (a cell's first line of code
    and whether it raised; an action's kind, call and status), so the fork can match each part to its own
    conversation and its pointers use the names the writer reads."""
    from ..writer_tools import part_names

    obs = list(ep.request[1:])
    cells = {f"cell:{c.index}": c for c in ep.cells}
    lines = [PARTS_HEAD]
    for name in part_names(ep):
        if name == "request":
            gist = _first_line(ep.request[0] if ep.request else "")
        elif name.startswith("observation:"):
            gist = _first_line(obs[int(name[12:])])
        elif name.startswith("cell:"):
            c = cells[name]
            raised = c.error is not None or "Traceback (most recent call last)" in (c.output or "")
            gist = _first_line(c.code) + ("  (raised an error)" if raised else "")
        elif name.startswith("action:"):
            a = ep.actions[int(name[7:])]
            gist = f"{a.kind} {a.channel}.{a.method}: {a.status}"
        else:
            gist = "the work-tree diff"
        lines.append(f"- `{name}`: {gist}" if gist else f"- `{name}`")
    return "\n".join(lines)


def fork_message(parts: str | None = None, instruction: str | None = None) -> str:
    """The user message the fork's request ends with: :func:`fork_instruction` (or *instruction*: a segment fork's,
    :func:`segment_instruction`), then the episode's part list when there is one. Both follow the cached prefix."""
    text = fork_instruction() if instruction is None else instruction
    return f"{text}\n\n{parts}" if parts else text


def staging_dir(paths: Any, eid: str) -> Path:
    return Path(paths.state_dir) / "memory-staging" / eid


def route() -> tuple[tuple[str, str] | None, str | None]:
    """The fork lane's route ``(chat completions URL, token)``, or why the fork has none: only the key proxy's own
    fork lane, read once from its descriptor (``switch.settle_fork_route_env``); never the actor's route, never a
    provider. With the fork on and no lane, the episode records ``refused: no fork lane``.
    """
    from . import switch

    switch.settle_fork_route_env(os.environ, _settings())
    return switch.fork_lane_route()


def _settings() -> Any:
    from unify.settings import SETTINGS

    return SETTINGS


def _model(client: Any) -> str | None:
    endpoint = str(
        getattr(client, "endpoint", "") or getattr(client, "model", "") or "",
    )
    return endpoint.rsplit("@", 1)[0] or None


def fork_tool_choice(choice: Any) -> Any:
    """The fork's tool choice. The actor's tool schemas stay in the request for the cached prefix, but a choice that
    forces a tool call would never let the fork answer in text, so anything other than ``auto`` or ``none`` becomes
    ``auto``."""
    if choice is None or choice in ("auto", "none"):
        return choice
    return "auto"


def request_doc(
    handle: Any,
    eid: str,
    url: str,
    key: str,
    parts: str | None = None,
    instruction: str | None = None,
) -> tuple[dict | None, str | None]:
    """The fork's request, from the session's last request as sent, or why the session cannot be forked; *instruction*
    replaces :func:`fork_instruction` (a segment fork's, :func:`segment_instruction`)."""
    from unify.actor.code_act_actor import _session_fork_source

    inner = getattr(handle, "_inner", None) or handle
    from .switch import in_episode

    # UNIFY_MEMORY=in_episode: a compaction was a pause (its segment fork covered the cells before it), so the forks
    # after it continue the compacted conversation; off, a compacted session is not forked, as built
    kw = {"allow_compressed": True} if in_episode(_settings()) else {}
    source, why = _session_fork_source(inner, getattr(handle, "_actor", None), **kw)
    if source is None:
        return None, why
    return request_from_source(eid, source, url, key, parts=parts, instruction=instruction)


def as_sent(message: Any) -> Any:
    """*message* in the form the actor's own request carried it: unillm strips keys that start with ``_`` (the
    harness's annotations, e.g. which reply ended a turn) from a message and its content blocks, and LiteLLM leaves out
    a top-level key whose value is ``None``. The fork posts its body itself, so it applies both, and its prefix is
    the actor's, byte for byte, rather than the history as the harness holds it."""
    if not isinstance(message, dict):
        return message
    out = {k: v for k, v in message.items() if not (isinstance(k, str) and k.startswith("_")) and v is not None}
    content = out.get("content")
    if isinstance(content, list):
        out["content"] = [
            {k: v for k, v in block.items() if not (isinstance(k, str) and k.startswith("_"))}
            if isinstance(block, dict)
            else block
            for block in content
        ]
    return out


def request_from_source(
    episode_id: str,
    source: dict,
    base_url: str,
    token: str,
    *,
    parts: str | None = None,
    instruction: str | None = None,
) -> tuple[dict | None, str | None]:
    """``(request document, None)`` for the fork from a conversation *source* (``client``, ``sent_messages``,
    ``tools`` and optionally ``tool_choice``, as the actor's session reader gives them), or ``(None, why)`` when it
    cannot be forked: the messages as the actor sent them (:func:`as_sent`) and the tool schemas unchanged, then
    :func:`fork_message` (the instruction, or *instruction* in its place, and *parts*, the episode's part list from
    :func:`part_list`, when given).
    *base_url* is the fork lane's URL (``/chat/completions`` is added when it is not already there); *token* goes only
    into the headers, which reach the worker on its stdin."""
    eid = episode_id
    url = (
        base_url
        if base_url.rstrip("/").endswith("/chat/completions")
        else base_url.rstrip("/") + "/chat/completions"
    )
    key = token
    client = source.get("client")
    model = _model(client)
    if not model:
        return None, "the session's client names no model"
    body: dict = {
        "model": model,
        "messages": [
            *(as_sent(m) for m in source["sent_messages"]),
            {"role": "user", "content": fork_message(parts, instruction)},
        ],
        "tools": source["tools"],
    }
    choice = fork_tool_choice(source.get("tool_choice"))
    if choice is not None:
        body["tool_choice"] = choice
    effort = getattr(client, "reasoning_effort", None)
    if isinstance(effort, str) and effort:
        # as the actor's own transport sends it: LiteLLM's OpenRouter route passes reasoning_effort at the top level
        body["reasoning_effort"] = "xhigh" if effort == "max" else effort
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "x-unify-call-kind": "actor_fork",
        "x-unify-session": f"fork.{eid}",
        "x-unify-parent": eid,
    }
    return {"episode": eid, "url": url, "headers": headers, "body": body}, None


def _record(staging: Path, eid: str, status: str, detail: str | None = None) -> None:
    doc = {
        "episode": eid,
        "status": status,
        "model": None,
        "turns": 0,
        "usage": {},
        "usd": None,
        "files": [],
        "refused_files": [],
    }
    if detail:
        doc["detail"] = detail
    atomic_json(staging / FORK_JSON, doc)


# --- the reservation and its lock ----------------------------------------------------------------------------------


class Reservation:
    """An episode's fresh staging dir and the exclusive lock on its ``fork.lock``, held by this process until the
    worker inherits it (:func:`start`) or the reservation is closed."""

    def __init__(self, staging: Path, eid: str, fd: int | None):
        self.staging, self.eid, self.fd = staging, eid, fd

    def close(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def abandon(self, status: str, detail: str | None = None) -> None:
        """Settle the episode without a fork (``fork.json`` with *status*), then release the lock."""
        try:
            if not (self.staging / FORK_JSON).exists():
                _record(self.staging, self.eid, status, detail)
        except OSError:
            pass
        self.close()


def reserve(
    paths: Any,
    eid: str,
    progress: Callable[[str], None] = lambda _m: None,
) -> Reservation | None:
    """Make the episode's fresh staging dir and lock it, before the episode is recorded, so that an admission check
    never sees a recorded episode without its dir. None when the dir exists.
    """
    staging = staging_dir(paths, eid)
    try:
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(exist_ok=False)
    except OSError as exc:
        progress(f"memory: fork not started: staging dir: {type(exc).__name__}")
        return None
    fd = None
    try:
        fd = os.open(
            staging / LOCK_FILE,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        _record(staging, eid, f"error: lock: {type(exc).__name__}")
        return None
    return Reservation(staging, eid, fd)


def _age_s(path: Path) -> float:
    return time.time() - os.lstat(path).st_mtime


def settled(root: Path | str, eid: str) -> bool:
    """Whether *eid*'s fork has settled under *root* (``<state>/memory-staging``): first :func:`resolve` (which records
    ``"lost"`` or ``"timeout"`` for a fork that can no longer settle itself), then the pure rule
    :func:`..trigger.admit_settled`."""
    from ..trigger import admit_settled

    resolve(root, eid)
    return bool(admit_settled(root, [eid])[0])


def resolve(root: Path | str, eid: str) -> None:
    """Record ``"lost"`` when *eid*'s fork process is gone without its record (its lock is free) or was never
    started, and ``"timeout"`` when its lock is still held past :data:`STALE_S`; otherwise leave it alone.
    """
    from ..staging import STATUS_FILES

    try:
        _resolve(Path(root), eid, STATUS_FILES)
    except OSError:
        pass  # unreadable state: nothing recorded; the pure rule decides on the files as they are


def _resolve(root: Path, eid: str, status_files: Any) -> bool:
    d = root / eid
    try:
        st = os.lstat(d)
    except FileNotFoundError:
        return True
    except OSError:
        return True
    if not stat.S_ISDIR(st.st_mode):
        return True  # a link or a file is never read as staging
    if any(os.path.lexists(d / name) for name in status_files):
        return True
    try:
        fd = os.open(d / LOCK_FILE, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        if _age_s(d) < UNSTARTED_S:
            return False
        _settle_record(d, eid, "lost", "the fork was never started: no lock file")
        return True
    except OSError as exc:
        _settle_record(
            d,
            eid,
            "lost",
            f"the fork's lock cannot be opened: {type(exc).__name__}",
        )
        return True
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                raise
            # held: the request or its fork worker is alive
            if _age_s(d / LOCK_FILE) < STALE_S:
                return False
            _settle_record(
                d,
                eid,
                "timeout",
                f"the fork was still running {STALE_S:g} s after it was reserved; admitted without it",
            )
            return True
        if not os.path.lexists(d / FORK_JSON):
            _settle_record(
                d,
                eid,
                "lost",
                "the fork process ended without writing its record",
            )
        return True
    finally:
        os.close(fd)


def _settle_record(d: Path, eid: str, status: str, detail: str) -> None:
    try:
        _record(d, eid, status, detail)
    except OSError:
        pass


class Admission:
    """With the fork on: which pending episodes may enter a pass batch (:func:`settled`), and the episodes earlier
    passes rolled over, kept in ``<state>/pass-admission.json`` so they enter a later batch.
    """

    def __init__(self, state_dir: Path | str):
        self.state_dir = Path(state_dir)
        self.root = self.state_dir / "memory-staging"
        self.file = self.state_dir / ROLLED_FILE

    def settled(self, eid: str) -> bool:
        return settled(self.root, eid)

    def rolled(self) -> list[str]:
        try:
            doc = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        rolled = doc.get("rolled") if isinstance(doc, dict) else None
        return [e for e in rolled or [] if isinstance(e, str)]

    def set_rolled(self, eids: list[str]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.file, {"rolled": list(dict.fromkeys(eids))})


def admission(paths: Any) -> Admission:
    """The admission rule: an episode enters a pass once its fork has settled."""
    return Admission(paths.state_dir)


# --- the box and the process ---------------------------------------------------------------------------------------


def box_argv(staging: Path, *, die_with_parent: bool) -> list[str]:
    """bubblewrap around the system python: read-only /usr, private /tmp, the worker file read-only, the staging
    dir the only writable bind, the starting process's network only."""
    bwrap, prlimit = shutil.which("bwrap"), shutil.which("prlimit")
    if not bwrap or not prlimit:
        raise RuntimeError(
            "bubblewrap and prlimit are required: the fork never runs outside a box",
        )
    args = [bwrap, "--unshare-all", "--share-net", "--new-session"]
    if die_with_parent:
        args.append("--die-with-parent")
    args += ["--ro-bind", "/usr", "/usr"]
    for link, target in (
        ("/lib", "usr/lib"),
        ("/lib64", "usr/lib64"),
        ("/bin", "usr/bin"),
        ("/sbin", "usr/sbin"),
    ):
        if os.path.islink(link):
            args += ["--symlink", target, link]
    for path in ("/etc/hosts", "/etc/ssl", "/etc/ca-certificates"):
        if os.path.exists(path):
            args += ["--ro-bind", path, path]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(WORKER_FILE), BOX_WORKER,
        "--bind", str(staging), STAGING_IN_BOX,
        "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        prlimit, "--nproc=64:64", "--nofile=256:256", "--as=2147483648:2147483648", "--fsize=8388608:8388608",
        "--", "/usr/bin/python3", "-I", BOX_WORKER, "--staging", STAGING_IN_BOX,
    ]  # fmt: skip
    return args


class Fork:
    """A started fork: its process, the thread writing its request, and its staging dir."""

    def __init__(self, proc: Any, writer: threading.Thread, staging: Path, eid: str):
        self.proc, self.writer, self.staging, self.eid = proc, writer, staging, eid


def launch(
    reservation: Reservation,
    argv: list[str],
    data: bytes,
    log_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    env: dict | None = None,
    cwd: Path | str | None = None,
) -> Fork:
    """Start *argv* in its own session with *data* on its stdin and the reservation's lock passed to it; this process
    then closes its copy, so the lock is held exactly while the process (or a child that kept the descriptor) lives.
    *env*: the process's environment (default ``PATH`` alone, the box's; the run record's relay gets
    :func:`relay_env`, never this process's whole environment).
    """
    log = open(log_path, "ab")
    try:
        os.set_inheritable(reservation.fd, True)
        try:
            proc = popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
                pass_fds=(reservation.fd,),
                env=env if env is not None else {"PATH": "/usr/bin:/bin"},
                **({"cwd": str(cwd)} if cwd is not None else {}),
            )
        finally:
            reservation.close()
    finally:
        log.close()

    def feed() -> None:
        try:
            proc.stdin.write(data)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    writer = threading.Thread(
        target=feed,
        name=f"fork-request-{reservation.eid}",
        daemon=True,
    )
    writer.start()
    return Fork(proc, writer, reservation.staging, reservation.eid)


def start(
    paths: Any,
    handle: Any,
    eid: str,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    progress: Callable[[str], None] = lambda _m: None,
    reservation: Reservation | None = None,
    parts: str | None = None,
    instruction: str | None = None,
) -> Fork | None:
    """Start the episode's fork, or record why not; never raises. Without a *reservation* (made before the episode was
    recorded), one is made here. *parts* is the recorded episode's part list (:func:`part_list`), which follows the
    instruction; without it the fork names parts by :data:`NUMBERING` alone. *instruction* replaces the episode
    fork's (a segment fork's, :func:`segment_instruction`; ``UNIFY_MEMORY=in_episode``)."""
    if reservation is not None and reservation.eid != eid:
        reservation.abandon("error: reserved for another episode")
        reservation = None
    if reservation is None:
        reservation = reserve(paths, eid, progress)
        if reservation is None:
            return None
    staging = reservation.staging
    try:
        r, why = route()
        doc = None
        # the fork's place on the run record, named before its message is written
        on_record = run_record_name(paths, eid)
        from unify.agents.run_record import prompt_lines

        lines = prompt_lines(on_record[1], "fork")
        parts = f"{parts}\n\n{lines}" if parts else lines
        if r is not None:
            doc, why = request_doc(handle, eid, *r, parts=parts, instruction=instruction)
        if doc is None:
            _leave(on_record)
            reservation.abandon(f"refused: {why}")
            progress(f"memory: fork refused: {why}")
            return None
        # metering: the fork's calls carry its episode, and a segment fork its segment (ids only), for the lane's
        # journal
        episode, segment = split_segment(eid)
        doc["headers"]["x-unify-episode"] = episode
        if segment is not None:
            doc["headers"]["x-unify-segment"] = str(segment)
        box = box_argv(staging, die_with_parent=True)
        log = Path(paths.state_dir) / "fork-worker.log"
        # the relay starts the box with --comms and answers its comms calls from the run record
        spec = {
            "home": str(paths.home),
            "name": on_record[1],
            "argv": [*box, "--comms"],
            "doc": doc,
            "lock_fd": reservation.fd,
            "log": str(log),
        }
        data = json.dumps(spec, default=str).encode("utf-8")
        try:
            # least privilege: by path through relay_boot (never the unify package's __init__), in a folder of its own
            # with an empty HOME, and the minimal environment
            folder = relay_folder(paths, on_record[1])
            fork = launch(
                reservation,
                relay_argv(),
                data,
                log,
                popen=popen,
                env=relay_env(folder / RELAY_HOME),
                cwd=folder,
            )
        except BaseException:
            _leave(on_record)
            raise
        progress("memory: fork started")
        return fork
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the request is already answered and recorded
        reservation.abandon(f"error: {type(exc).__name__}")
        progress(f"memory: fork failed to start: {type(exc).__name__}")
        return None


#: The relay's start file: run by path with ``-I`` (no ``PYTHONPATH``, no current folder on the path), it never runs
#: the ``unify`` package's ``__init__`` and refuses every module that reads provider settings or keys.
RELAY_BOOT = Path(__file__).resolve().parents[2] / "agents" / "relay_boot.py"
#: The only names the relay's environment takes from this process's (least privilege): the locale and the time zone.
#: ``PATH`` is fixed, ``HOME`` is the relay's own empty folder; no route base URL, token or key name ever reaches it,
#: nor ``PYTHONPATH``. The fork's request, its auth header included, goes on the relay's stdin.
RELAY_ENV_NAMES = ("LANG", "LC_ALL", "LC_CTYPE", "TZ")
#: The relay's HOME, an empty folder inside its own folder (removed when it ends).
RELAY_HOME = "relay-home"


def relay_argv() -> list[str]:
    return [sys.executable, "-I", str(RELAY_BOOT)]


def relay_folder(paths: Any, name: str) -> Path:
    """The relay's own folder (its current folder: no ``.env`` there), under the state directory."""
    folder = Path(paths.state_dir) / "fork-relays" / name
    (folder / RELAY_HOME).mkdir(parents=True, exist_ok=True)
    return folder


def relay_env(home: Path | str) -> dict[str, str]:
    """The relay's environment: a fixed ``PATH``, *home* as ``HOME``, and :data:`RELAY_ENV_NAMES` as this process
    has them; nothing else."""
    env = {k: os.environ[k] for k in RELAY_ENV_NAMES if os.environ.get(k)}
    env.update(PATH="/usr/bin:/bin", HOME=str(home))
    return env


def run_record_name(paths: Any, eid: str) -> tuple[Any, str]:
    """The fork put on the run record's roster for its relay, as ``fork-<hex of the episode>`` with its episode, and its
    name. The relay takes the place over when it starts (``attach``)."""
    from unify.agents.record import PostRefused
    from unify.agents.run_record import join_run_record

    record = join_run_record(paths.home)
    episode, segment = split_segment(eid)
    for _ in range(20):
        name = record.unique_name("fork", episode, "" if segment is None else f"-s{segment}")
        try:
            record.join(name, "fork", episode=episode)
            return record, name
        except PostRefused:
            continue
    raise PostRefused("no free name for the fork on the run record")


#: A segment's staging name (in-episode memory): ``<episode>.s<n>``.
_SEGMENT = re.compile(r"(.+)\.s([0-9]+)")


def split_segment(eid: str) -> tuple[str, int | None]:
    """``(episode, segment)`` of a staging name: a segment fork's ``<episode>.s<n>``, else ``(eid, None)``."""
    m = _SEGMENT.fullmatch(eid or "")
    return (m.group(1), int(m.group(2))) if m else (eid, None)


def _leave(on_record: tuple[Any, str] | None) -> None:
    """The fork never started: its place on the record ends."""
    if on_record is not None:
        try:
            on_record[0].leave(on_record[1])
        except Exception:  # noqa: BLE001 - leaving never changes how the episode settles
            pass


def finish(
    fork: Fork | None,
    progress: Callable[[str], None] = lambda _m: None,
) -> None:
    """After the reply: hand over the request and wait for the fork (bounded by :data:`WAIT_GUARD_S`, operational). A
    fork that leaves no ``fork.json`` gets one: ``"timeout"`` when the wait guard stopped it, ``"lost"`` when it ended
    without one.
    """
    if fork is None:
        return
    fork.writer.join(WRITE_GUARD_S)
    try:
        fork.proc.wait(timeout=WAIT_GUARD_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(fork.proc.pid, 9)
        except OSError:
            pass
        fork.proc.wait()
        if not (fork.staging / FORK_JSON).exists():
            _record(
                fork.staging,
                fork.eid,
                "timeout",
                f"the operational wait guard ({WAIT_GUARD_S:g} s) stopped the fork",
            )
    if not (fork.staging / FORK_JSON).exists():
        _record(
            fork.staging,
            fork.eid,
            "lost",
            f"the fork process exited with code {fork.proc.returncode} without writing its record",
        )
    progress("memory: fork ended")
