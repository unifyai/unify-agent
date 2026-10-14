"""The episode journal: the request's transcript, kept durably turn by turn until its episode is recorded.

The episode record is written once, when the request ends (``RequestRun.finish``). So that a process that dies in the
middle of a request keeps the turns that ran, the turn that stopped it included, the run journals its transcript as
it is written: every line the episode's transcript session writes (:mod:`unify.transcripts`: the session's start,
the system prompt, each model reply, each tool result and its later version, each compaction) is handed over by the
transcript's tap (``transcripts.tap``) as soon as it is on disk, and appended to
``<state dir>/episode-journal/<episode id>.jsonl`` (:attr:`.paths.Paths.journal`) as one JSON line, through the
run's redactor (the episode writer's redaction as far as it is known before the episode ends: the environment's and
the process's registered credentials, and key-shaped strings), then flushed and fsynced. A line is the transcript's
line whole: nothing is cut or capped (a 20 MB reply is one 20 MB line here, as it is one line of the episode's
``transcript.jsonl``). The first line is the run's header (:data:`HEADER`): what the episode's meta needs when no
request finishes it.

The journal is in the harness-only state directory: no box mounts it and the actor never reads it.

When the episode is recorded the journal is removed. A request that ends unfinished (``RequestRun.abort``: a timeout
or an error) keeps it, with one last line saying why it stopped (:data:`STOPPED`: ``"timeout"``, or ``"error"`` and the
exception's class name, never its text). A journal still there when the next request opens (under the request lock,
so no other request writes it) is one whose request stopped or whose process died mid-request: :func:`recover`
records it as an incomplete episode, from the journal's lines alone, with ``incomplete`` (why it stopped, else
:data:`UNKNOWN_REASON`, and when it was recovered) in its meta, and removes it. A journal whose header cannot be read
is renamed with :data:`UNRECORDED`, kept and never retried.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: The ``type`` of the journal's first line, and of the line an unfinished request ends it with.
HEADER = "episode_journal"
STOPPED = "episode_journal_stopped"
#: Why a request stopped, as the journal may say it: these reasons, and an exception's class name.
STOP_REASONS = ("timeout", "error")
_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,99}\Z")
#: Why a recovered episode ended, as far as its journal tells.
UNKNOWN_REASON = "unknown"
#: The suffix a journal is renamed with when it cannot be recorded (kept as it is, never retried).
UNRECORDED = ".unrecorded"
#: The header's fields: the episode's meta that no request finishes, and how the episode is written.
_FIELDS = ("episode_id", "started_at", "build", "model", "effort", "memory_main", "worktree_before", "request")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class EpisodeJournal:
    """One request's journal (module docstring). *redactor* makes the run's redactor (called for each line, so a
    credential the process learns later is known from then on). :meth:`append` takes a transcript line as written;
    it never raises, and once the journal is removed or closed it does nothing."""

    def __init__(self, path: Path, redactor: Callable[[], Any]) -> None:
        self.path = Path(path)
        self.redactor = redactor
        self.lines = 0
        self._fd: int | None = None
        self._lock = threading.Lock()
        self._failed = False
        # a write that failed part way left a line without its end: the next line starts on a line of its own
        self._torn = False

    @classmethod
    def open(cls, directory: Path, header: dict, redactor: Callable[[], Any]) -> "EpisodeJournal":
        """A new journal for the episode *header* names, with its header written and on disk. Raises OSError
        when it cannot be created."""
        directory = Path(directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        journal = cls(directory / f"{header['episode_id']}.jsonl", redactor)
        fd = os.open(journal.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC, 0o600)
        journal._fd = fd
        row = {"type": HEADER, **{k: header.get(k) for k in _FIELDS}, "library_mode": bool(header.get("library_mode"))}
        if header.get("profile"):  # UNIFY_PROFILE, only when set: a default run's header is as before
            row["profile"] = str(header["profile"])
        try:
            journal._write(json.dumps(redactor().obj(row), ensure_ascii=False, default=str))
        except BaseException:
            journal.remove()
            raise
        _fsync_dir(directory)
        return journal

    def _write(self, text: str) -> None:
        data = ("\n" if self._torn else "") + text + "\n"
        view = memoryview(data.encode("utf-8", "surrogatepass"))
        self._torn = True
        while view:
            n = os.write(self._fd, view)
            view = view[n:]
        self._torn = False
        os.fsync(self._fd)

    def append(self, line: str) -> None:
        """Append one transcript line (JSON text, as the transcript wrote it) through the redactor, durably."""
        with self._lock:
            if self._fd is None:
                return
            try:
                row = json.loads(line)
                text = json.dumps(self.redactor().obj(row), ensure_ascii=False, default=str)
                self._write(text)
                self.lines += 1
            except Exception as exc:  # noqa: BLE001 - the journal never stops a turn
                if not self._failed:
                    self._failed = True
                    logger.warning("memory: a turn was not journaled (%s)", type(exc).__name__)

    def stopped(self, why: dict) -> None:
        """End the journal with why the request stopped (:data:`STOP_REASONS`, and an error's class name only), then
        stop journaling; the file stays for the next start."""
        row = {"type": STOPPED, **_stop_fields(why)}
        with self._lock:
            if self._fd is not None:
                try:
                    self._write(json.dumps(row))
                except OSError as exc:
                    logger.warning("memory: the journal's stop was not written (%s)", type(exc).__name__)
        self.close()

    def close(self) -> None:
        """Stop journaling; the file stays."""
        with self._lock:
            fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def remove(self) -> None:
        """Stop journaling and remove the file (its episode is recorded, or the request records nothing)."""
        self.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        _fsync_dir(self.path.parent)


def _stop_fields(why: Any) -> dict:
    """*why* as the journal keeps it: a known reason, and for an error the exception's class name; anything else is
    :data:`UNKNOWN_REASON`."""
    why = why if isinstance(why, dict) else {}
    reason = why.get("reason") if why.get("reason") in STOP_REASONS else UNKNOWN_REASON
    out = {"reason": reason}
    if reason == "error" and isinstance(why.get("error"), str) and _CLASS_NAME.match(why["error"]):
        out["error"] = why["error"]
    return out


def read(path: Path) -> tuple[dict | None, list[dict]]:
    """A journal's header (None when its first line is not one) and its lines; a line that cannot be read (one cut
    off when the process died) is skipped."""
    rows = []
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                row = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    if not rows or rows[0].get("type") != HEADER or not isinstance(rows[0].get("episode_id"), str):
        return None, rows
    return rows[0], rows[1:]


def _last_ts(lines: list[dict]) -> str | None:
    for row in reversed(lines):
        if isinstance(row.get("ts"), str):
            return row["ts"]
    return None


def recover(
    paths: Any,
    stores: Any,
    *,
    library_index: Callable[[Any, Any], None] | None = None,
) -> list[str]:
    """Record every journal left in ``paths.journal`` as an incomplete episode (module docstring), oldest name
    first, and remove it; the episode ids recorded. Called with the request lock held, before this request's own
    journal exists. A journal whose episode is already in the episodes repo (its process died after recording it)
    is removed; one that cannot be recorded for another reason stays for the next request. *library_index* is
    called with each recorded episode and its redactor (the similar index in library mode). Never raises."""
    directory = Path(paths.journal)
    try:
        found = sorted(p for p in directory.glob("*.jsonl") if p.is_file() and not p.is_symlink())
    except OSError:
        return []
    done: list[str] = []
    for path in found:
        try:
            eid = _record(path, stores, library_index)
        except Exception as exc:  # noqa: BLE001 - kept for the next request
            from ..redact import redact_error

            why = redact_error(f"{type(exc).__name__}: {exc}")[:300]
            logger.warning("memory: the journal %s was not recorded (%s)", path.name, why)
            continue
        if eid is not None:
            done.append(eid)
    return done


def _record(path: Path, stores: Any, library_index: Callable[[Any, Any], None] | None) -> str | None:
    from ..episodes import EpisodeWriter
    from . import trajectory

    header, lines = read(path)
    stops = [r for r in lines if r.get("type") == STOPPED]
    lines = [r for r in lines if r.get("type") != STOPPED]
    if header is None:
        path.rename(path.with_name(path.name + UNRECORDED))
        logger.warning("memory: the journal %s has no header; kept as %s", path.name, path.name + UNRECORDED)
        return None
    recovered_at = _now()
    run = SimpleNamespace(
        episode_id=header["episode_id"],
        started_at=str(header.get("started_at") or recovered_at),
        build=str(header.get("build") or "unknown"),
        model=str(header.get("model") or ""),
        effort=str(header.get("effort") or ""),
        pin=str(header.get("memory_main") or ""),
        request=str(header.get("request") or ""),
        profile=str(header.get("profile") or ""),
        observer=None,
        costs=None,
    )
    ep, redactor = trajectory.assemble(
        run,
        lines,
        "",
        _last_ts(lines) or run.started_at,
        worktree_before=header.get("worktree_before") if isinstance(header.get("worktree_before"), str) else None,
    )
    ep.incomplete = {**_stop_fields(stops[-1] if stops else None), "recovered_at": recovered_at}
    writer = EpisodeWriter(
        stores.episodes,
        stores.blobs,
        redactor,
        **({"library_mode": True} if header.get("library_mode") else {}),
    )
    try:
        sha = writer.write(ep)
    except FileExistsError:  # recorded before the process died: nothing is lost by removing the journal
        path.unlink()
        return None
    stores.evidence.index_episode(ep, sha)
    if library_index is not None:
        try:
            library_index(ep, redactor)
        except Exception as exc:  # noqa: BLE001 - the index never stops a recovery
            logger.warning("memory: the similar index missed the recovered episode (%s)", type(exc).__name__)
    path.unlink()
    _fsync_dir(path.parent)
    logger.warning("memory: recorded the incomplete episode %s from its journal (%d lines)", ep.episode_id, len(lines))
    return ep.episode_id


__all__ = ["EpisodeJournal", "HEADER", "STOPPED", "STOP_REASONS", "UNKNOWN_REASON", "UNRECORDED", "read", "recover"]
