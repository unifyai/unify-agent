"""The writer pass's driver (``consolidate``): an attention pass's records, the episode lookup, the gate's reports and
the pass's events.

The writer is a fake turn factory (:class:`FakeWriter`): no network and no key. The fake ends at once (``finish``)
without a manifest, so no sandboxed cell runs; the tests marked ``needs_bwrap`` run the real gate.
"""

import asyncio
import json
import shutil
from decimal import Decimal
from types import SimpleNamespace

import pytest

from unify.memory.library_checks import LibraryConfig

from unify.memory.admission import cover_problem
from unify.memory.episodes import Action, CostRow, EpisodeWriter
from unify.memory.integration import consolidate
from unify.memory.integration.consolidate import EpisodeLookup, open_stores
from unify.memory.integration.paths import Paths
from unify.memory.writer_pass import TRANSCRIPT_REF
from unify.memory.redact import Redactor
from tests.memory.test_episodes import _ep

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

CSV = b"vendor_id,invoice_no,amount,due_date,status\nV-17,INV-0042,1250.50,2026-10-14,open\n"


def _settings(model="openai/gpt-6-sol", cap="20.00"):
    """Memory on, the writer's model and memory's cap for the run."""
    return SimpleNamespace(UNIFY_MEMORY="on", UNIFY_MEMORY_WRITER_MODEL=model, UNIFY_MEMORY_CAP_USD=cap)


def _finish():
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "f",
                "type": "function",
                "function": {"name": "finish", "arguments": '{"summary": "nothing"}'},
            },
        ],
    }


class FakeWriter:
    """A turn factory: each turn ends the pass at once (no manifest), or follows ``script``."""

    def __init__(self, usd="0.01", script=None):
        self.usd, self.script = usd, script
        self.made: list[tuple[str, str]] = []
        self.calls = 0
        self.seen: list[list[dict]] = []

    def __call__(self, model, effort, **kw):
        self.made.append((model, effort))

        async def turn(messages, tools):
            self.calls += 1
            self.seen.append([dict(m) for m in messages])
            if self.script is not None:
                step = self.script[min(self.calls, len(self.script)) - 1]
                if isinstance(step, BaseException):
                    raise step
                if step == "sleep":
                    await asyncio.sleep(30)
                return step
            return _finish(), self.usd

        return turn


def _stores(tmp_path):
    return open_stores(Paths.under(tmp_path / "home"))


def _record(stores, eid, actions=None, request="Pay my Venmo friends back", minute=0):
    ep = _ep(
        episode_id=eid,
        started_at=f"2026-10-08T01:{minute:02d}:00Z",
        request=[request],
        actions=(
            actions
            if actions is not None
            else [
                Action(0, "venmo", "me", [], {}, {"user_id": "u-1"}, "ok", "read"),
            ]
        ),
        costs=[CostRow("actor", "openai/gpt-6-luna", 10, 2, "0.000123")],
    )
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    stores.evidence.index_episode(ep, sha)
    return sha, ep


# --- the writer's transcript as a note ------------------------------------------------------------------


def _attention_pass(stores, fake, *, entry_episodes=("e1",)):
    """One attention pass (``consolidate.run_attention_pass``) over a message entry naming *entry_episodes*, with
    *fake* as the writer's turns and the writer lane's journal unread; returns its report."""
    from unify.memory import attention
    from unify.memory import writer_service as ws
    from unify.memory.integration.async_pass import Supervisor
    from unify.memory.reconcile import unavailable

    home = stores.paths.home
    ledger = attention.Ledger(home)
    ledger.create("message", attention.Refs(episodes=tuple(entry_episodes), record_seq=1), 1)
    pid = "ab" * 8
    handed = ledger.hand(pid, 4)
    folder = ws.service_dir(home) / ws.TURNS_DIR / pid
    folder.mkdir(parents=True, exist_ok=True)

    async def recon(self, **kw):
        return unavailable(Decimal("13.125"))

    mp = pytest.MonkeyPatch()
    mp.setattr(Supervisor, "reconcile", recon)
    try:
        return asyncio.run(
            consolidate.run_attention_pass(
                home, pid, handed, resume=False, folder=folder, journal_start=None, effort="low",
                settings=_settings(), turn_factory=lambda: fake("m", "low"),
            ),
        )  # fmt: skip
    finally:
        mp.undo()


def test_writer_transcript_is_kept_redacted_on_the_episodes_head(tmp_path, monkeypatch):
    """An attention pass keeps its writer's transcript as a note on the episodes repo's head, redacted: a secret the
    writer saw never reaches the record."""
    secret = "planted-fake-token-6c1d2e"  # pragma: allowlist secret
    monkeypatch.setenv("UNIFY_TEST_FAKE_TOKEN", secret)
    msg = {"role": "assistant", "content": f"the token is {secret}"}
    fake = FakeWriter(script=[(msg, "0.0000001"), (_finish(), "0.0000001")])
    stores = _stores(tmp_path)
    _record(stores, "e1")
    report = _attention_pass(stores, fake)
    assert report.pass_id == "ab" * 8
    assert secret in json.dumps(fake.seen)  # the writer itself saw it; the record must not
    head, *rows = [json.loads(n) for n in stores.episodes.notes(stores.episodes.head(), ref=TRANSCRIPT_REF)]
    # the header names the brief and the tool JSON by hash; then the messages
    assert head["pass_id"] == "ab" * 8 and head["header"]["brief_sha256"]
    assert rows and {r["pass_id"] for r in rows} == {"ab" * 8}
    assert any(r["message"].get("content") == "the token is <secret:UNIFY_TEST_FAKE_TOKEN>" for r in rows)
    assert secret not in stores.episodes.run("log", "-p", "--all")
    assert secret not in json.dumps(rows) and secret not in json.dumps(head)


# --- reason codes: structural, never read from reason text ----------------------------------------------


def test_reason_codes_are_known_codes_deduplicated_and_capped():
    from unify.memory.writer_pass import PassOutcome

    def oc(passed, codes):
        return PassOutcome("p", passed, None, "0", 1, ["free text G1"], "", 0, codes)

    assert consolidate.reason_codes(oc(True, ["ok"]), None) == ["ok"]
    assert consolidate.reason_codes(oc(False, ["G2", "G2", "x", "G5"]), None) == [
        "G2",
        "G5",
    ]
    assert consolidate.reason_codes(oc(False, []), None) == ["writer_error"]
    assert consolidate.reason_codes(None, "deadline") == ["deadline"]
    assert consolidate.reason_codes(None, None) == ["writer_error"]
    many = [f"G{n}" for n in range(1, 7)] + [
        "no_manifest",
        "over_quota",
        "deadline",
        "pass_cap",
        "writer_error",
    ]
    assert len(consolidate.reason_codes(oc(False, many), None)) == 10


def test_the_gate_names_the_checks_that_refused_structurally(tmp_path):
    from unify.memory.gate import Gate

    stores = _stores(tmp_path)
    head = stores.memory.head()
    res = Gate(stores.memory, stores.evidence, stores.blobs, library_mode=LibraryConfig()).check(
        head,
        head,
        {"items": "not a list"},
    )
    assert not res.passed and res.refused == ["G1"] and res.manifest_invalid
    assert not res.checks[
        "G4"
    ]  # unevaluated checks are False in checks, but not refused


# --- (f) the episode lookup ------------------------------------------------------------------------------


def test_lookup_returns_recorded_actions_of_any_kind(tmp_path):
    stores = _stores(tmp_path)
    tool = Action(0, "venmo", "me", [], {}, {"user_id": "u-1"}, "ok", "read")
    talk = Action(
        -1,
        "dialogue:user",
        "reply",
        ["do"],
        {},
        "ok then",
        "ok",
        kind="dialogue",
    )
    _record(stores, "e1", actions=[tool, talk])
    look = EpisodeLookup(stores)
    assert look.action("e1", 0) == tool
    assert look.action("e1", 1) == talk
    for eid, idx in [
        ("e1", 2),
        ("e1", -1),
        ("e1", True),
        ("e1", "0"),
        ("nope", 0),
        ("../e1", 0),
        (None, 0),
    ]:
        assert look.action(eid, idx) is None
    look.action("e1", 0).response[
        "user_id"
    ] = "changed"  # a copy: the cache is untouched
    assert look.action("e1", 0) == tool


def test_a_worktree_cover_is_admitted_by_the_g2_rule_through_the_lookup(tmp_path):
    stores = _stores(tmp_path)
    blob = stores.blobs.put(CSV)
    wt = Action(
        0,
        "worktree:workspace",
        "read",
        ["finance/ap/invoices-2026-10.csv"],
        {},
        {
            "blob_before": blob,
            "blob_after": blob,
            "size": len(CSV),
            "shape": {"format": "csv"},
        },
        "ok",
        "read",
        kind="worktree",
    )
    _record(
        stores,
        "o1",
        actions=[wt, Action(0, "venmo", "me", [], {}, {"u": 1}, "ok")],
    )
    look = EpisodeLookup(stores)
    has = stores.blobs.has
    assert cover_problem(look.action("o1", 0), "worktree_workspace", has) is None
    assert cover_problem(look.action("o1", 1), "worktree_workspace", has) is not None
    assert (
        cover_problem(look.action("o1", 9), "worktree_workspace", has)
        == "not a recorded action"
    )


@needs_bwrap
def test_the_gate_admits_a_worktree_cover_through_the_lookup(tmp_path):
    """Remote (bwrap): the real gate's G2, held-out values included, with the lookup over a written episode."""
    from tests.memory.test_kinds_gate import (
        INVENTORY,
        INVOICES,
        WT_FILES,
        WT_MAN,
        _commit,
        _wt,
    )
    from unify.memory.gate import Gate

    stores = _stores(tmp_path)
    inv, tsv = stores.blobs.put(INVOICES), stores.blobs.put(INVENTORY)
    _record(
        stores,
        "o1",
        actions=[
            _wt("finance/ap/invoices-2026-10.csv", inv),
            _wt("ops/inventory.csv", tsv),
        ],
    )
    gate = Gate(
        stores.memory,
        stores.evidence,
        stores.blobs,
        action_lookup=EpisodeLookup(stores).action,library_mode=LibraryConfig()
    )
    parent = stores.memory.head()
    cand = _commit(stores.memory, WT_FILES)
    res = gate.check(parent, cand, WT_MAN)
    assert res.checks["G2"], res.reasons
    blind = Gate(
        stores.memory,
        stores.evidence,
        stores.blobs,library_mode=LibraryConfig()
    )  # no lookup: fails closed
    assert not blind.check(parent, cand, WT_MAN).checks["G2"]


def test_open_stores_is_idempotent(tmp_path):
    a = _stores(tmp_path)
    head = (a.memory.head(), a.episodes.head())
    b = _stores(tmp_path)
    assert (b.memory.head(), b.episodes.head()) == head


def test_every_setting_the_driver_reads_is_a_build_setting():
    """The driver reads its switches by name; a renamed setting must never be read under its old name."""
    import inspect
    import re

    from unify.memory.integration import consolidate
    from unify.settings import ProductionSettings

    read = set(
        re.findall(r'"(UNIFY_MEMORY_[A-Z0-9_]+)"', inspect.getsource(consolidate)),
    )
    assert read, "the driver reads no v2 setting by name"
    assert read <= set(ProductionSettings.model_fields), sorted(
        read - set(ProductionSettings.model_fields),
    )


# --- the library a pass landed ---------------------------------------------------------------------------------


def test_the_landed_event_counts_the_library_at_the_landed_commit(tmp_path):
    """A pass that lands N items reports N at its landed commit, not the library before it landed."""
    stores = _stores(tmp_path)
    before = stores.memory.head()
    files = {
        "memory/acct/__init__.py": '"""Accounts."""\n',
        "memory/acct/users.py": (
            'def user_id(r):\n    """The id."""\n    return r\n\n\n'
            'def user_name(r):\n    """The name."""\n    return r\n'
        ),
        "notes/acct/ids.md": "---\ntitle: Ids\ndescription: How ids look.\nuses: []\n---\nIds are strings.\n",
    }
    with stores.memory.temp_checkout("main") as wt:
        for rel, text in files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        landed = stores.memory.commit_all(wt, "pass", {"Pass": "p1"})
    assert consolidate.library_items_at(stores, landed) == 3  # two functions and a note
    assert consolidate.library_items_at(stores, before) == 0
    assert consolidate.library_items_at(stores, None) is None


class WindowedWriter(FakeWriter):
    """A fake writer whose turns carry a known context window, as unillm_turn's carry the model's."""

    def __call__(self, model, effort, **kw):
        turn = super().__call__(model, effort, **kw)
        turn.max_input_tokens = 1_000
        return turn


def test_the_cost_recording_turn_keeps_the_turns_context_window():
    from unify.memory.integration.cost import recording_turn

    async def turn(messages, tools):
        return {"role": "assistant", "content": "x"}, "0.01"

    turn.max_input_tokens = 4_096
    rows: list[CostRow] = []
    wrapped = recording_turn(turn, rows, "m")
    assert wrapped.max_input_tokens == 4_096
    assert (
        asyncio.run(wrapped([], [])) == ({"role": "assistant", "content": "x"}, "0.01")
        and len(rows) == 1
    )


# --- a running pass reports its progress ---------------------------------------------------------------------------


def test_the_reporting_gate_reports_a_check_and_a_merge_and_is_the_gate_otherwise():
    reports: list[tuple[str, dict]] = []

    class Gate:
        _blob = "the gate's own"

        def check(self, *a, **k):
            return SimpleNamespace(passed=False, refused=["G2"])

        def merge(self, *a, **k):
            return SimpleNamespace(passed=True, refused=[], merged="abc")

    gate = consolidate._ReportingGate(Gate(library_mode=LibraryConfig()), lambda phase, **f: reports.append((phase, f)))
    assert gate.check("p", "c", {}).refused == ["G2"] and gate.merge("p", "c", {}, "id", "k", None, "0").merged == "abc"
    assert gate._blob == "the gate's own"
    assert reports == [
        ("gate", {"step": "check", "passed": False, "refused": ["G2"]}),
        ("gate", {"step": "merge", "passed": True, "refused": [], "landed": True}),
    ]


def test_a_failed_model_call_is_reported_as_not_returned():
    reports: list[tuple[str, dict]] = []

    async def failing(messages, tools):
        raise RuntimeError("down")

    turn = consolidate._reporting_turn(failing, lambda phase, **f: reports.append((phase, f)))
    with pytest.raises(RuntimeError):
        asyncio.run(turn([], []))
    assert reports == [("turn", {"turn": 1, "returned": False})]
