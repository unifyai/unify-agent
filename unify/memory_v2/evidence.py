"""SQLite evidence store (spec §3, D5): a derived index over the episodes and memory repos."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from .episodes import Episode
from .experience import experience_tokens

if TYPE_CHECKING:
    from .signals import Signal

# Input-shape descriptors kept per function in a commit's snapshot (the catalogue's ``input_shapes``).
MAX_INPUT_SHAPES = 16

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(seq INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT UNIQUE, commit_sha TEXT,
  started_at TEXT, regime TEXT, memory_main TEXT, request TEXT);
CREATE TABLE IF NOT EXISTS env_touch(episode_id TEXT, channel TEXT, n_calls INTEGER, PRIMARY KEY(episode_id, channel));
CREATE TABLE IF NOT EXISTS signals(signal_id TEXT PRIMARY KEY, episode_id TEXT, source TEXT, label TEXT, ts TEXT,
  refers_to TEXT, regime TEXT, revealed INTEGER, reveal_p TEXT);
CREATE TABLE IF NOT EXISTS item_evidence(item TEXT, episode_id TEXT, role TEXT, PRIMARY KEY(item, episode_id, role));
CREATE TABLE IF NOT EXISTS covers(item TEXT, episode_id TEXT, action_index INTEGER,
  PRIMARY KEY(item, episode_id, action_index));
CREATE TABLE IF NOT EXISTS passes(pass_id TEXT PRIMARY KEY, kind TEXT, channel TEXT, parent TEXT, candidate TEXT,
  passed INTEGER, reasons TEXT, usd TEXT, patch_blob TEXT);
CREATE TABLE IF NOT EXISTS cursors(channel TEXT PRIMARY KEY, seq INTEGER);
CREATE TABLE IF NOT EXISTS experience(episode_id TEXT PRIMARY KEY, tokens INTEGER, counter TEXT);
"""
# The input-shape snapshots of ``UNIFY_MEMORY_V2_SURFACING=catalogue`` (:mod:`.shape_rows`), created by the
# first write, so a store that never freezes a snapshot (``index``, the default) keeps the v2 schema.
_SHAPE_SCHEMA = """
CREATE TABLE IF NOT EXISTS shape_commits(commit_sha TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS commit_shapes(commit_sha TEXT, item TEXT, body TEXT, shapes TEXT, backfilled INTEGER,
  PRIMARY KEY(commit_sha, item));
"""


class EvidenceStore:
    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(str(path))
        self.db.executescript(_SCHEMA)

    def index_episode(self, ep: Episode, commit_sha: str) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO episodes(episode_id, commit_sha, started_at, regime, memory_main, request) VALUES(?,?,?,?,?,?)",
                (
                    ep.episode_id,
                    commit_sha,
                    ep.started_at,
                    ep.regime,
                    ep.memory_main,
                    "\n".join(ep.request),
                ),
            )
            counts: dict[str, int] = {}
            for a in ep.actions:
                counts[a.channel] = counts.get(a.channel, 0) + 1
            self.db.executemany(
                "INSERT INTO env_touch VALUES(?,?,?)",
                [(ep.episode_id, ch, n) for ch, n in counts.items()],
            )
            tokens, how = experience_tokens(ep)
            self.db.execute(
                "INSERT INTO experience VALUES(?,?,?)",
                (ep.episode_id, tokens, how),
            )
            return int(cur.lastrowid)

    def seq_of(self, eid: str) -> int:
        row = self.db.execute(
            "SELECT seq FROM episodes WHERE episode_id=?",
            (eid,),
        ).fetchone()
        if row is None:
            raise KeyError(eid)
        return int(row[0])

    def episode_ref(self, eid: str) -> tuple[str, str]:
        """``(commit_sha, started_at)`` of an indexed episode; ``KeyError`` when absent."""
        row = self.db.execute(
            "SELECT commit_sha, started_at FROM episodes WHERE episode_id=?",
            (eid,),
        ).fetchone()
        if row is None:
            raise KeyError(eid)
        return str(row[0]), str(row[1])

    def regime_of(self, eid: str) -> str:
        row = self.db.execute(
            "SELECT regime FROM episodes WHERE episode_id=?",
            (eid,),
        ).fetchone()
        if row is None:
            raise KeyError(eid)
        return str(row[0])

    def episode_exists(self, eid: str) -> bool:
        return (
            self.db.execute(
                "SELECT 1 FROM episodes WHERE episode_id=?",
                (eid,),
            ).fetchone()
            is not None
        )

    def episode_ids_since(self, channel: str, after_seq: int) -> list[str]:
        rows = self.db.execute(
            "SELECT e.episode_id FROM episodes e JOIN env_touch t ON t.episode_id=e.episode_id "
            "WHERE t.channel=? AND e.seq>? ORDER BY e.seq",
            (channel, after_seq),
        ).fetchall()
        return [r[0] for r in rows]

    def experience_of(self, eid: str) -> tuple[int, str]:
        """(experience tokens, counter) recorded for *eid* (:mod:`.experience`); KeyError if none."""
        row = self.db.execute(
            "SELECT tokens, counter FROM experience WHERE episode_id=?",
            (eid,),
        ).fetchone()
        if row is None:
            raise KeyError(eid)
        return int(row[0]), str(row[1])

    def experience_since(self, after_seq: int) -> list[tuple[str, int]]:
        """(episode id, experience tokens) of every episode after *after_seq*, in order (0 if unknown)."""
        rows = self.db.execute(
            "SELECT e.episode_id, COALESCE(x.tokens, 0) FROM episodes e "
            "LEFT JOIN experience x ON x.episode_id=e.episode_id WHERE e.seq>? ORDER BY e.seq",
            (after_seq,),
        ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    def channels_of(self, eid: str) -> list[str]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT channel FROM env_touch WHERE episode_id=? ORDER BY channel",
                (eid,),
            )
        ]

    def add_signal(self, sig: "Signal") -> None:
        """Raw index write, used by ``signals.post_signal`` and by ``rebuild``.

        It performs no regime-mask check; all harness code must post signals
        through ``signals.post_signal``.
        """
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO signals VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    sig.signal_id,
                    sig.episode_id,
                    sig.source,
                    sig.label,
                    sig.ts,
                    sig.refers_to,
                    sig.regime,
                    int(sig.revealed),
                    sig.reveal_p,
                ),
            )

    def signals_for(self, eid: str) -> list["Signal"]:
        from .signals import Signal

        rows = self.db.execute(
            "SELECT * FROM signals WHERE episode_id=? ORDER BY ts, signal_id",
            (eid,),
        ).fetchall()
        return [
            Signal(r[0], r[1], r[2], r[3], r[4], r[5], r[6], bool(r[7]), r[8])
            for r in rows
        ]

    def add_item_evidence(self, item: str, eid: str, role: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO item_evidence VALUES(?,?,?)",
                (item, eid, role),
            )

    def item_episodes(self, item: str) -> list[str]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT DISTINCT episode_id FROM item_evidence WHERE item=? ORDER BY episode_id",
                (item,),
            )
        ]

    def items_citing(self, eid: str) -> list[str]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT DISTINCT item FROM item_evidence WHERE episode_id=? ORDER BY item",
                (eid,),
            )
        ]

    def add_cover(self, item: str, eid: str, action_index: int) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO covers VALUES(?,?,?)",
                (item, eid, action_index),
            )

    def covered(self) -> set[tuple[str, int]]:
        return {
            (r[0], int(r[1]))
            for r in self.db.execute("SELECT episode_id, action_index FROM covers")
        }

    def write_commit_shapes(self, commit: str, rows: dict[str, dict]) -> bool:
        """Freeze the input-shape snapshot of memory commit *commit* (:mod:`.shape_rows`): item ->
        ``{"body": digest, "shapes": [...], "backfilled": bool}``. A commit's snapshot is written once and
        never changed, so every export of that commit renders the same catalogue; False if it existed.
        """
        with self.db:
            for ddl in _SHAPE_SCHEMA.split(";"):
                if ddl.strip():
                    self.db.execute(ddl)
            cur = self.db.execute(
                "INSERT OR IGNORE INTO shape_commits VALUES(?)",
                (commit,),
            )
            if cur.rowcount == 0:
                return False
            self.db.executemany(
                "INSERT INTO commit_shapes VALUES(?,?,?,?,?)",
                [
                    (
                        commit,
                        item,
                        row["body"],
                        json.dumps(row["shapes"], sort_keys=True, ensure_ascii=False),
                        int(bool(row.get("backfilled"))),
                    )
                    for item, row in sorted(rows.items())
                ],
            )
        return True

    def commit_shapes(self, commit: str) -> dict[str, dict] | None:
        """The frozen snapshot of *commit*, or None when it has none."""
        if not self._has_shape_tables():
            return None
        if (
            self.db.execute(
                "SELECT 1 FROM shape_commits WHERE commit_sha=?",
                (commit,),
            ).fetchone()
            is None
        ):
            return None
        return {
            r[0]: {"body": r[1], "shapes": json.loads(r[2]), "backfilled": bool(r[3])}
            for r in self.db.execute(
                "SELECT item, body, shapes, backfilled FROM commit_shapes WHERE commit_sha=? "
                "ORDER BY item",
                (commit,),
            )
        }

    def _has_shape_tables(self) -> bool:
        return (
            self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='shape_commits'",
            ).fetchone()
            is not None
        )

    def covers_of(self, item: str) -> list[tuple[str, int]]:
        """The validated covers recorded for *item* by every landed merge, sorted."""
        return [
            (r[0], int(r[1]))
            for r in self.db.execute(
                "SELECT episode_id, action_index FROM covers WHERE item=? "
                "ORDER BY episode_id, action_index",
                (item,),
            )
        ]

    def record_pass(self, row: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO passes VALUES(?,?,?,?,?,?,?,?,?)",
                tuple(
                    row.get(k)
                    for k in (
                        "pass_id",
                        "kind",
                        "channel",
                        "parent",
                        "candidate",
                        "passed",
                        "reasons",
                        "usd",
                        "patch_blob",
                    )
                ),
            )

    def add_pass_notes(self, pass_id: str, notes: list[str]) -> None:
        """Append *notes* to a recorded pass's reasons (after the gate's); KeyError if it is not recorded."""
        if not notes:
            return
        with self.db:
            row = self.db.execute(
                "SELECT reasons FROM passes WHERE pass_id=?",
                (pass_id,),
            ).fetchone()
            if row is None:
                raise KeyError(pass_id)
            reasons = json.loads(row[0]) if row[0] else []
            self.db.execute(
                "UPDATE passes SET reasons=? WHERE pass_id=?",
                (json.dumps(list(reasons) + list(notes)), pass_id),
            )

    def pass_exists(self, pass_id: str) -> bool:
        return (
            self.db.execute(
                "SELECT 1 FROM passes WHERE pass_id=?",
                (pass_id,),
            ).fetchone()
            is not None
        )

    def cursor(self, channel: str) -> int:
        row = self.db.execute(
            "SELECT seq FROM cursors WHERE channel=?",
            (channel,),
        ).fetchone()
        return int(row[0]) if row else 0

    def set_cursor(self, channel: str, seq: int) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO cursors VALUES(?,?)",
                (channel, seq),
            )
