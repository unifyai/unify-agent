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
CREATE TABLE IF NOT EXISTS item_use(item TEXT, episode_id TEXT, PRIMARY KEY(item, episode_id));
"""
# The input-shape snapshots of ``UNIFY_MEMORY_V2_SURFACING=catalogue`` (:mod:`.shape_rows`), created by the
# first write, so a store that never freezes a snapshot (``index``, the default) keeps the v2 schema.
_SHAPE_SCHEMA = """
CREATE TABLE IF NOT EXISTS shape_commits(commit_sha TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS commit_shapes(commit_sha TEXT, item TEXT, body TEXT, shapes TEXT, backfilled INTEGER,
  PRIMARY KEY(commit_sha, item));
"""

# The item_use counts, in column order. Each is added to a store that lacks it (a store opened before
# the column existed), as ``INTEGER NOT NULL DEFAULT 0``, or ``DEFAULT 1`` for :data:`_UNKNOWN_BY_DEFAULT`.
_USE_COUNTS = (
    "imported",
    "called",
    "refused",
    "errored",
    "refused_accepted",
    "referenced",
    "guarded",
    "unknown_calls",
    "shown",
    "channel_shown",
    "modified",
    "refused_modified",
    "errored_modified",
    "exposure_record",
    "exposure_legacy_text",
    "exposure_unknown",
    "outcome_unknown",
    "prompt_unconfirmed",
)
# Added with ``DEFAULT 1``: a row indexed before the column existed reads as unknown (its outcomes and
# where its shown lists came from), never as known. Every insert sets every column, so the default only
# ever fills the rows already there when an old store is migrated.
_UNKNOWN_BY_DEFAULT = frozenset({"outcome_unknown", "exposure_unknown"})
# Where a record's shown lists came from (``analysis.use``'s ``exposure_source``); a record without one
# (from before the field) counts as ``unknown``.
_EXPOSURE_SOURCES = ("record", "legacy_text", "unknown")
# The item_use columns :meth:`EvidenceStore.request_flags` counts requests by (a request counts when any
# of its rows has it). All but ``outcome_unknown`` hold one value per request; ``outcome_unknown`` is per
# item (the items a cell with an unknown outcome could reach).
_REQUEST_FLAGS = (
    "exposure_record",
    "exposure_legacy_text",
    "exposure_unknown",
    "outcome_unknown",
    "prompt_unconfirmed",
)
# Per-item sums over the requests whose outcome for the item was known (``outcome_unknown = 0``).
_KNOWN_SUMS = (
    ("refused_known", "SUM(CASE WHEN outcome_unknown=0 THEN refused ELSE 0 END)"),
    ("errored_known", "SUM(CASE WHEN outcome_unknown=0 THEN errored ELSE 0 END)"),
    (
        "refused_accepted_known",
        "SUM(CASE WHEN outcome_unknown=0 THEN refused_accepted ELSE 0 END)",
    ),
    (
        "requests_refusing_known",
        "SUM(CASE WHEN outcome_unknown=0 AND refused>0 THEN 1 ELSE 0 END)",
    ),
    (
        "requests_erroring_known",
        "SUM(CASE WHEN outcome_unknown=0 AND errored>0 THEN 1 ELSE 0 END)",
    ),
)
_IN_CHUNK = 500


def _migrate_item_use(db: sqlite3.Connection) -> None:
    """Add the item_use columns a store lacks. Rows already there read ``outcome_unknown = 1`` and
    ``exposure_unknown = 1`` (:data:`_UNKNOWN_BY_DEFAULT`): what they did not record is unknown.
    """
    have = {r[1] for r in db.execute("PRAGMA table_info(item_use)")}
    with db:
        for col in _USE_COUNTS:
            if col not in have:
                default = 1 if col in _UNKNOWN_BY_DEFAULT else 0
                db.execute(
                    f"ALTER TABLE item_use ADD COLUMN {col} INTEGER NOT NULL DEFAULT {default}",
                )


def _exposure_source(use: dict) -> str:
    """The record's ``exposure_source``; a record from before the field (``version`` below 4) whose
    legacy reading found a section is ``legacy_text``, any other record without one ``unknown``.
    """
    source = use.get("exposure_source")
    if source in _EXPOSURE_SOURCES:
        return source
    version, section = use.get("version"), use.get("memory_section_shown")
    if (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version < 4
        and isinstance(section, dict)
        and section.get("shown") is True
    ):
        return "legacy_text"
    return "unknown"


def _unknown_items(use: dict, items: set[str]) -> set[str]:
    """The items whose outcome the record leaves unknown: its ``items_outcome_unknown`` (every item when
    that list was cut short), or, for a record without the list (an older or failed record), every item
    unless ``outcomes_known`` is true."""
    listed = use.get("items_outcome_unknown")
    if isinstance(listed, list):
        count = use.get("items_outcome_unknown_count")
        if (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > len(listed)
        ):
            return set(items)
        return {v for v in listed if isinstance(v, str)}
    return set() if use.get("outcomes_known") is True else set(items)


def _use_rows(eid: str, use: dict) -> list[tuple]:
    """One ``item_use`` row per item at the episode's pin (an exposure, zeros included) and per item the
    record counts, in :data:`_USE_COUNTS` order. ``unknown_calls`` is the dynamic calls on the item's
    channel plus on ``env`` itself; ``shown`` whether the item's own line was in the prompt's memory
    section, ``channel_shown`` whether its channel was; ``modified`` whether the request edited its
    channel's files (its refusals and errors are then in the ``_modified`` columns only);
    ``exposure_<source>`` is 1 in the column of the record's ``exposure_source`` (the harness's record of
    what the prompt showed, the legacy reading of the prompt's text, or unknown); ``prompt_unconfirmed``
    is 1 when the shown lists came from the record but no recorded system prompt ends with its text;
    ``outcome_unknown`` is 1 for the items a cell with an unknown outcome could reach in this request
    (``items_outcome_unknown``; for a record without that list, every item unless ``outcomes_known``),
    and that row's refusal and error columns are then lower bounds.
    """

    def listed(key: str) -> set[str]:
        value = use.get(key)
        return (
            {v for v in value if isinstance(v, str)}
            if isinstance(value, list)
            else set()
        )

    rows = use.get("items") if isinstance(use.get("items"), dict) else {}
    pinned = listed("items_at_pin")
    shown, channels, changed = (
        listed("shown_items"),
        listed("shown_channels"),
        listed("modified_channels"),
    )
    unknown = (
        use.get("unknown_calls") if isinstance(use.get("unknown_calls"), dict) else {}
    )
    source = _exposure_source(use)
    exposure = tuple(int(source == s) for s in _EXPOSURE_SOURCES)
    section = use.get("memory_section_shown")
    unconfirmed = int(
        source == "record"
        and isinstance(section, dict)
        and section.get("prompt_confirmed") is False,
    )

    def n(value: object) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    every = pinned | {k for k in rows if isinstance(k, str)}
    unknown_items = _unknown_items(use, every)
    out = []
    for item in sorted(every):
        r = rows.get(item) if isinstance(rows.get(item), dict) else {}
        channel = item.split(":", 1)[0].removeprefix("env/")
        out.append(
            (
                item,
                eid,
                n(r.get("imported")),
                n(r.get("called")),
                n(r.get("refused")),
                n(r.get("errored")),
                n(r.get("refused_then_accepted")),
                n(r.get("referenced")),
                n(r.get("guarded")),
                n(unknown.get(channel)) + n(unknown.get("*")),
                int(item in shown),
                int(channel in channels),
                int(channel in changed),
                n(r.get("refused_modified")),
                n(r.get("errored_modified")),
                *exposure,
                int(item in unknown_items),
                unconfirmed,
            ),
        )
    return out


class EvidenceStore:
    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(str(path))
        self.db.executescript(_SCHEMA)
        _migrate_item_use(self.db)

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
            use = getattr(ep, "memory_use", None)
            if isinstance(use, dict):
                cols = ", ".join(("item", "episode_id", *_USE_COUNTS))
                marks = ",".join("?" * (2 + len(_USE_COUNTS)))
                self.db.executemany(
                    f"INSERT OR REPLACE INTO item_use({cols}) VALUES({marks})",
                    _use_rows(ep.episode_id, use),
                )
            return int(cur.lastrowid)

    # -- item use (memory v2.1 telemetry) --------------------------------------------------------------

    def item_use(
        self,
        eids: list[str] | None = None,
        item: str | None = None,
    ) -> dict[str, dict]:
        """Per item (only *item* when given), the summed ``item_use`` counts over *eids* (every indexed
        episode when None).

        Each entry: ``requests`` (episodes whose pin held the item), ``used_requests`` (of those, the ones
        with a call site), the sums of :data:`_USE_COUNTS` (``shown`` and ``channel_shown`` are then
        the requests whose memory section showed the item's line and its channel; ``outcome_unknown`` the
        requests in which a cell with an unknown outcome could reach the item), and the sums of
        :data:`_KNOWN_SUMS` over the requests whose outcome for the item was known. Items are in id order.
        """
        cols = ", ".join(
            [f"SUM({c})" for c in _USE_COUNTS] + [expr for _, expr in _KNOWN_SUMS],
        )
        known_keys = [k for k, _ in _KNOWN_SUMS]
        base = (
            f"SELECT item, COUNT(*), SUM(CASE WHEN called > 0 THEN 1 ELSE 0 END), {cols} "
            "FROM item_use"
        )
        out: dict[str, dict] = {}
        if eids is None:
            chunks: list[list[str] | None] = [None]
        else:
            uniq = sorted(set(eids))
            chunks = [uniq[i : i + _IN_CHUNK] for i in range(0, len(uniq), _IN_CHUNK)]
        for chunk in chunks:
            where: list[str] = []
            params: list[str] = []
            if item is not None:
                where.append("item=?")
                params.append(item)
            if chunk is not None:
                where.append(f"episode_id IN ({','.join('?' * len(chunk))})")
                params.extend(chunk)
            clause = f" WHERE {' AND '.join(where)}" if where else ""
            rows = self.db.execute(f"{base}{clause} GROUP BY item", params).fetchall()
            for r in rows:
                cur = out.setdefault(
                    r[0],
                    {
                        "requests": 0,
                        "used_requests": 0,
                        **dict.fromkeys(_USE_COUNTS, 0),
                        **dict.fromkeys(known_keys, 0),
                    },
                )
                cur["requests"] += int(r[1] or 0)
                cur["used_requests"] += int(r[2] or 0)
                for c, v in zip([*_USE_COUNTS, *known_keys], r[3:]):
                    cur[c] += int(v or 0)
        return dict(sorted(out.items()))

    def request_flags(self, eids: list[str] | None = None) -> dict[str, int]:
        """Per column of :data:`_REQUEST_FLAGS`, how many requests among *eids* (every indexed one when
        None) have it set on any of their rows (``outcome_unknown``: some item's outcome was unknown); a
        request counts when its pin held at least one item."""
        out = dict.fromkeys(_REQUEST_FLAGS, 0)
        cols = ", ".join(f"MAX({c})" for c in _REQUEST_FLAGS)
        if eids is None:
            chunks: list[list[str] | None] = [None]
        else:
            uniq = sorted(set(eids))
            chunks = [uniq[i : i + _IN_CHUNK] for i in range(0, len(uniq), _IN_CHUNK)]
        for chunk in chunks:
            clause, params = "", []
            if chunk is not None:
                clause = f" WHERE episode_id IN ({','.join('?' * len(chunk))})"
                params = list(chunk)
            rows = self.db.execute(
                f"SELECT episode_id, {cols} FROM item_use{clause} GROUP BY episode_id",
                params,
            ).fetchall()
            for r in rows:
                for c, v in zip(_REQUEST_FLAGS, r[1:]):
                    out[c] += int(bool(v))
        return out

    def last_call_seq(self, item: str) -> int | None:
        """The ``seq`` of the latest indexed episode with a call site of *item*, or None."""
        row = self.db.execute(
            "SELECT MAX(e.seq) FROM item_use u JOIN episodes e ON e.episode_id=u.episode_id "
            "WHERE u.item=? AND u.called > 0",
            (item,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def latest_seq(self) -> int:
        """The ``seq`` of the latest indexed episode (0 when none)."""
        row = self.db.execute("SELECT MAX(seq) FROM episodes").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

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
