"""The record's file: one JSON entry per line, appended, fsynced, never rewritten."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from unify.agents.entry import Entry
from unify.transcripts import scrub


class RecordLog:
    """``<run>.jsonl`` holds the entries; ``<run>.cursors.jsonl`` what each agent was shown."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.cursor_path = self.path.with_name(f"{self.path.stem}.cursors.jsonl")
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: Entry) -> None:
        self._append_line(self.path, json.dumps(entry.to_dict(), ensure_ascii=False))

    def append_cursor(self, agent: str, upto: int) -> None:
        self._append_line(
            self.cursor_path,
            json.dumps({"agent": agent, "upto": upto}),
        )

    def _append_line(self, path: Path, line: str) -> None:
        data = (scrub(line) + "\n").encode("utf-8")
        with self._lock:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                while data:
                    data = data[os.write(fd, data) :]
                os.fsync(fd)
            finally:
                os.close(fd)

    def load(self) -> tuple[list[Entry], dict[str, int]]:
        """Every whole entry, and each agent's highest cursor. A line cut by a crash is skipped."""
        entries: list[Entry] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(Entry.from_dict(json.loads(line)))
                except (ValueError, KeyError, TypeError):
                    continue
        cursors: dict[str, int] = {}
        if self.cursor_path.exists():
            for line in self.cursor_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    agent, upto = str(item["agent"]), int(item["upto"])
                except (ValueError, KeyError, TypeError):
                    continue
                cursors[agent] = max(cursors.get(agent, 0), upto)
        return entries, cursors
