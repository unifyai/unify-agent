"""The system suite's driver: one memory home, CLI sessions in process, and readers of the real records.

Every reader here opens a real file the harness wrote (the capture store, the evidence store, the attention ledger,
``spend.jsonl``, ``pass-results.jsonl``, the episode record in ``episodes.git``, the library copy a request mounts).
None of them computes what the harness should have written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from tests.memory.conformance import fake_lanes as FL
from tests.memory.conformance import harness as H
from tests.system import sysworld as SW

MEMORY_MODES = ("on", "in_episode")
ALL_MODES = ("off", "on", "in_episode")


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    except OSError:
        return []


@dataclass
class System:
    world: dict
    monkeypatch: Any
    mode: str
    lanes: FL.Lanes
    server: Any
    port: int
    writer: SW.WriterLane = field(default_factory=SW.WriterLane)
    sessions: list = field(default_factory=list)
    models: list = field(default_factory=list)

    # ── set-up and tear-down ─────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def open(cls, world: dict, monkeypatch: Any, mode: str) -> "System":
        import unify.common.embeddings as embeddings
        from tests.memory.conformance.test_surfaces import _fake_embed

        lanes = FL.Lanes()
        server, port = FL.serve(lanes)
        env = H.redesign_env(port)
        env["UNIFY_MEMORY"] = mode
        H.apply_settings(monkeypatch, env)
        H.settle_lanes(monkeypatch, port)
        monkeypatch.setattr(embeddings, "embed", _fake_embed)
        s = cls(world, monkeypatch, mode, lanes, server, port)
        lanes.script("fork", *([SW.fork_script] * 400))
        lanes.script("writer", *([lambda body: s.writer(body)] * 400))
        return s

    def close(self) -> None:
        try:
            if self.mode != "off" and (self.home / "memory-state").exists():
                report = H.drain(self.home)
                self.drain_report = report
                H.check_terminated(report, self.home)
        finally:
            self.server.shutdown()

    # ── places ───────────────────────────────────────────────────────────────────────────────────────────────────

    @property
    def home(self) -> Path:
        return Path(self.world["state"])

    @property
    def workspace(self) -> Path:
        return Path(self.world["workspace"])

    @property
    def paths(self):
        from unify.memory.integration.paths import Paths

        return Paths.under(self.home)

    def stores(self):
        from unify.memory.integration.consolidate import open_stores

        return open_stores(self.paths)

    def copy(self) -> Path:
        """The library copy the last request mounted (under ``in_episode``, its ``current`` pin)."""
        c = self.paths.checkout
        return c / "current" if (c / "current").exists() else c

    # ── running ──────────────────────────────────────────────────────────────────────────────────────────────────

    async def request(self, text: str, model: Any, host: Any = None, *, timeout_s: float = 600):
        self.models.append(model)
        s = await H.cli_session(self.world, self.monkeypatch, model, text, host or H.replies(), timeout_s=timeout_s)
        self.sessions.append(s)
        assert s.code == 0, s.lines[-6:]
        return s

    async def accepted_words(self, words: list[str], op: str = "reverse", *, librarian: Any = None):
        """An accepted episode of the words world (a verdict the actor sees, then a message); settled after."""
        from tests.scripted_model import Always, ScriptedModel, cell, reply

        kinds = {"actor": [reply(calls=[cell(SW.words_cell(words, op=op))]),
                           reply(json.dumps(SW.words_answer(words, op=op))), reply("Good.")]}
        kinds["librarian"] = librarian or Always(lambda c: SW.librarian_answer([], nearest=[]))
        r = await self.request(SW.words_request(words, op=op), ScriptedModel(**kinds), SW.accepted_host())
        self.settle()
        return r

    async def accepted_tally(self, *, librarian: Any = None):
        """An accepted episode of the tally world (an acting procedure); settled after."""
        from tests.scripted_model import Always, ScriptedModel, cell, reply

        SW.seed_tally(self.workspace)
        model = ScriptedModel(actor=[reply(calls=[cell(SW.TALLY_CELL)]), reply("The tally is updated."),
                                     reply("Good.")],
                              librarian=librarian or Always(lambda c: SW.librarian_answer([], nearest=[])))
        r = await self.request(SW.TALLY_REQUEST, model, SW.accepted_host())
        self.settle()
        return r

    def settle(self) -> None:
        if self.mode != "off":
            H.wait_until_settled(self.home)

    # ── records ──────────────────────────────────────────────────────────────────────────────────────────────────

    def captures(self) -> dict[str, dict]:
        from unify.memory import captures as C

        return C.load(C.store_dir(self.paths.state_dir))

    def capture_of(self, episode_id: str) -> str:
        """The capture id whose record names *episode_id* among its episodes (fails when there is none)."""
        found = [cid for cid, d in self.captures().items()
                 if episode_id in ((d.get("record") or {}).get("episodes") or [])]
        assert found, f"no capture records episode {episode_id}: {sorted(self.captures())}"
        return found[0]

    def entries(self) -> list:
        return list(H.ledger(self.home).entries().values())

    def ledger_rows(self, op: str) -> list[dict]:
        return [r for r in _jsonl(self.paths.state_dir / "memory-attention" / "ledger.jsonl") if r.get("op") == op]

    def pass_results(self) -> list[dict]:
        return _jsonl(self.paths.state_dir / "pass-results.jsonl")

    def spend_rows(self) -> list[dict]:
        from unify.memory import writer_service as ws

        return _jsonl(ws.service_dir(self.home) / ws.SPEND_FILE)

    def item_use(self, item: str) -> dict:
        return self.stores().evidence.item_use(item=item).get(item) or {}

    def episode_file(self, episode_id: str, name: str) -> str:
        repo = self.stores().episodes
        names = repo.run("ls-tree", "-r", "--name-only", "main").split()
        hits = [n for n in names if f"/{episode_id}/" in f"/{n}" and n.endswith("/" + name)]
        assert hits, f"episode {episode_id} has no {name}"
        return repo.show("main", hits[0]).decode()

    def library_main_has(self, rel: str) -> bool:
        repo = self.stores().memory
        return repo.run("ls-tree", "--name-only", "main", "--", rel).strip() == rel

    # ── seeding (a fixture's own writes, named as such: used only where the harness writes nothing to test against) ──

    def seed_served_row(self, episode_id: str, item: str) -> None:
        """Write an ``item_use`` row saying *item* was only served (shown) in *episode_id*, never called: what a build
        that records served items writes. Used to prove a test would see served items counted as used."""
        db = self.stores().evidence.db
        db.execute("INSERT OR IGNORE INTO item_use(item, episode_id) VALUES (?, ?)", (item, episode_id))
        db.execute("UPDATE item_use SET shown = 1 WHERE item = ? AND episode_id = ?", (item, episode_id))
        db.commit()

    def seed_library(self, files: dict[str, str], message: str = "seed") -> None:
        """Commit *files* onto the library's ``main`` (a seeded library: content the harness would refuse to land)."""
        repo = self.stores().memory
        base = repo.head()
        with repo.temp_checkout() as wt:
            for rel, text in files.items():
                (wt / rel).parent.mkdir(parents=True, exist_ok=True)
                (wt / rel).write_text(text)
            sha = repo.commit_all(wt, message, {"Seed": "system-test"})
        repo.fast_forward("main", sha, expected_old=base)

    def item_evidence(self) -> dict:
        """The served evidence of the last request's copy, by item id (``.memory/item_evidence.json``'s ``items``)."""
        p = self.copy() / ".memory" / "item_evidence.json"
        assert p.is_file(), f"the copy serves no evidence file: {p}"
        return json.loads(p.read_text()).get("items") or {}

    @staticmethod
    def usd(value: Any) -> Decimal:
        return Decimal(str(value))
