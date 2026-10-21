"""The consolidation driver: checker pass/fail, size trigger, the writer, gate, events.

The writer is a fake turn factory monkeypatched over ``consolidate.unillm_turn``: no network and no key. The
fake ends at once (``finish``) without a manifest, so no sandboxed cell runs; the tests marked
``needs_bwrap`` run the real gate.
"""

import asyncio
import json
import re
import shutil
from decimal import Decimal
from types import SimpleNamespace

import pytest

from unify.memory.admission import cover_problem
from unify.memory.episodes import Action, CostRow, EpisodeWriter
from unify.memory.experience import experience_tokens
from unify.memory.integration import consolidate
from unify.memory.integration.consolidate import EpisodeLookup, events_path, open_stores, writer_settings
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import State
from unify.memory.writer_pass import TRANSCRIPT_REF
from unify.memory.redact import Redactor
from tests.memory.test_episodes import _ep

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

A_TOK = "0.00000073"
MONEY = re.compile(r"^[0-9]+(\.[0-9]+)?\Z")
CSV = b"vendor_id,invoice_no,amount,due_date,status\nV-17,INV-0042,1250.50,2026-10-14,open\n"


def _settings(e=1, guard="", model="openai/gpt-6-sol", usage="", scale="", calls=""):
    return SimpleNamespace(
        UNIFY_MEMORY="on",
        UNIFY_MEMORY_E=e,
        UNIFY_MEMORY_WRITER_MODEL=model,
        UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS=A_TOK,
        UNIFY_MEMORY_WRITER_RUN_GUARD_USD=guard,
        UNIFY_MEMORY_WRITER_USAGE=usage,
        UNIFY_MEMORY_WRITER_EFFORT_SCALE=scale,
        UNIFY_MEMORY_WRITER_MAX_CALLS=calls,
    )


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


def _run(stores, eid, sha, state=None, *, effort="low", emit=None, **kw):
    # low: scale 1 and 40 calls (the per-effort defaults), so a pass's cap is E x a_tok
    state = state if state is not None else State(stores.paths.state)
    return asyncio.run(
        run_due_passes(
            stores,
            eid,
            sha,
            state,
            effort=effort,
            settings=_settings(**kw),
            emit=emit,
        ),
    )


def _events(stores, progress=False):
    """The events file's rows; the running passes' progress rows (``turn``, ``gate``) only with *progress*."""
    p = events_path(stores.paths)
    rows = [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []
    return rows if progress else [r for r in rows if r.get("phase") not in consolidate.PROGRESS_PHASES]


# --- (a) the checker -------------------------------------------------------------------------------------


# --- the size trigger ------------------------------------------------------------------------------------


def test_one_token_short_of_e_runs_nothing(tmp_path, monkeypatch):
    fake = FakeWriter()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, ep = _record(stores, "e1")
    assert _run(stores, "e1", sha, e=experience_tokens(ep)[0] + 1) == []
    assert fake.calls == 0


# --- (b) a pass without a manifest -----------------------------------------------------------------------


# --- (c) no channel filter -------------------------------------------------------------------------------


def test_kind_qualified_channels_take_part_in_the_pass(tmp_path, monkeypatch):
    fake = FakeWriter()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    blob = stores.blobs.put(CSV)
    acts = [
        Action(
            0,
            "worktree:workspace",
            "read",
            ["ap/invoices.csv"],
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
        ),
        Action(
            0,
            "shell:git",
            "run",
            ["git status"],
            {},
            {"exit_code": 0, "tail": "clean"},
            "ok",
            kind="shell",
        ),
    ]
    sha, _ = _record(stores, "o1", actions=acts)
    assert stores.evidence.channels_of("o1") == ["shell:git", "worktree:workspace"]
    (out,) = _run(stores, "o1", sha)
    assert fake.calls == 1
    user = fake.seen[0][1]["content"]
    assert '"channel": null' in user and '"o1"' in user


# --- (d) and the run guard -------------------------------------------------------------------------------


# --- events ----------------------------------------------------------------------------------------------


def test_cap_is_e_times_the_allowance_as_a_plain_decimal():
    cfg = writer_settings(_settings(e=150000))
    assert format(cfg.cap_usd, "f") == "0.10950000"
    assert cfg.run_guard_usd is None and cfg.model == "openai/gpt-6-sol"
    defaults = writer_settings(SimpleNamespace())
    assert (defaults.experience_budget, format(defaults.usd_per_token, "f")) == (
        150000,
        A_TOK,
    )
    with pytest.raises(ValueError):
        writer_settings(
            SimpleNamespace(UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS="7.3E-7"),
        )
    with pytest.raises(ValueError):
        writer_settings(SimpleNamespace(UNIFY_MEMORY_WRITER_RUN_GUARD_USD="1e1"))
    with pytest.raises(ValueError):
        writer_settings(SimpleNamespace(UNIFY_MEMORY_E="0"))
    for zero in ("0", "0.0", "0.00000000"):
        with pytest.raises(ValueError):
            writer_settings(
                SimpleNamespace(UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS=zero),
            )
    assert writer_settings(_settings(guard="2.50")).run_guard_usd == Decimal("2.50")
    assert defaults.show_usage is False and cfg.show_usage is False
    assert writer_settings(_settings(usage="on")).show_usage is True
    assert writer_settings(_settings(usage="off")).show_usage is False
    with pytest.raises(ValueError):
        writer_settings(SimpleNamespace(UNIFY_MEMORY_WRITER_USAGE="yes"))


# --- per-effort pass limits (UNIFY_MEMORY_WRITER_EFFORT_SCALE, UNIFY_MEMORY_WRITER_MAX_CALLS) ------------------


def test_per_effort_limits_default_to_one_rule_for_every_bed():
    cfg = writer_settings(SimpleNamespace())
    assert cfg.effort_scale == {
        "low": Decimal("1"),
        "medium": Decimal("2"),
        "high": Decimal("5"),
    }
    assert cfg.max_calls == {"low": 40, "medium": 80, "high": 80}
    assert (
        writer_settings(_settings()).effort_scale == cfg.effort_scale
    )  # empty: the defaults
    assert format(writer_settings(_settings(e=150000)).cap_usd * 5, "f") == "0.54750000"
    custom = writer_settings(
        _settings(scale="high:4, Low:1.5,medium:3", calls="low:10,medium:20,high:30"),
    )
    assert custom.effort_scale == {
        "low": Decimal("1.5"),
        "medium": Decimal("3"),
        "high": Decimal("4"),
    }
    assert custom.max_calls == {"low": 10, "medium": 20, "high": 30}


@pytest.mark.parametrize(
    "scale",
    [
        "low:1,medium:2",  # an effort missing
        "low:1,medium:2,high:5,xhigh:9",  # an unknown effort
        "low:1,low:1,medium:2,high:5",  # an effort twice
        "low:0,medium:2,high:5",  # not positive
        "low:1,medium:-2,high:5",
        "low:1,medium:2,high:5e0",  # an exponent
        "low:1,medium:2,high:.5",
        "low=1,medium=2,high=5",
        "low:1;medium:2;high:5",
        "low:1,medium:2,high:",
        "5",
    ],
)
def test_a_bad_effort_scale_map_is_refused(scale):
    with pytest.raises(ValueError, match="UNIFY_MEMORY_WRITER_EFFORT_SCALE"):
        writer_settings(_settings(scale=scale))


@pytest.mark.parametrize(
    "calls",
    [
        "low:40,medium:80",
        "low:40,medium:80,high:80,max:9",
        "low:0,medium:80,high:80",
        "low:40,medium:80,high:1.5",
        "low:40,medium:80,high:-1",
        "low:40,medium:80,high:1e2",
        "low:40,medium:80,high:many",
        "80",
    ],
)
def test_a_bad_max_calls_map_is_refused(calls):
    with pytest.raises(ValueError, match="UNIFY_MEMORY_WRITER_MAX_CALLS"):
        writer_settings(_settings(calls=calls))


def _spy_configs(monkeypatch) -> list:
    seen: list = []
    real = consolidate.WriterPass

    def spy(*args, **kwargs):
        seen.append(args[5])  # the PassConfig
        return real(*args, **kwargs)

    monkeypatch.setattr(consolidate, "WriterPass", spy)
    return seen


def _ledger_rows(stores):
    return [
        json.loads(ln)
        for ln in consolidate.ledger_path(stores.paths).read_text().splitlines()
    ]


def test_the_effort_maps_reach_the_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeWriter())
    configs = _spy_configs(monkeypatch)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    got: list[dict] = []
    _run(
        stores,
        "e1",
        sha,
        effort="High",
        emit=got.append,
        scale="low:1,medium:2,high:3.5",
        calls="low:40,medium:80,high:60",
    )
    (cfg,) = configs
    assert cfg.max_calls == 60 and cfg.max_usd == Decimal(A_TOK) * Decimal("3.5")
    assert (got[0]["max_calls"], got[0]["effort_scale"]) == (60, "3.5")


def test_the_run_guard_compares_against_the_scaled_cap(tmp_path, monkeypatch):
    fake = FakeWriter()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    guard = format(Decimal(A_TOK) * 3, "f")  # above one allowance, below five
    assert _run(stores, "e1", sha, effort="high", guard=guard) == []
    assert fake.calls == 0 and _events(stores)[0]["reason_codes"] == ["run_guard"]
    (out,) = _run(stores, "e1", sha, effort="low", guard=guard)
    assert fake.calls == 1


@pytest.mark.parametrize("usage, shown", [("", False), ("off", False), ("on", True)])
def test_the_writer_usage_switch_reaches_the_passes_first_message(
    tmp_path,
    monkeypatch,
    usage,
    shown,
):
    from unify.memory.usage import USAGE_HEADING

    fake = FakeWriter()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    _run(stores, "e1", sha, usage=usage)
    first = fake.seen[0][1]["content"]
    assert (USAGE_HEADING in first) is shown


def test_emit_failures_never_stop_the_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeWriter())
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")

    def broken(row):
        raise OSError("stdout closed")

    (out,) = _run(stores, "e1", sha, emit=broken)
    assert out.pass_id == "e1.p0" and len(_events(stores)) == 2
    assert "stdout closed" in stores.paths.errors.read_text()


# --- (e) the writer's cost rows as notes -----------------------------------------------------------------


def test_writer_transcript_is_kept_redacted_on_the_episode_commit(
    tmp_path,
    monkeypatch,
):
    secret = "planted-fake-token-6c1d2e"  # pragma: allowlist secret
    monkeypatch.setenv("UNIFY_TEST_FAKE_TOKEN", secret)
    msg = {"role": "assistant", "content": f"the token is {secret}"}
    fake = FakeWriter(script=[(msg, "0.0000001"), (_finish(), "0.0000001")])
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = _run(stores, "e1", sha)
    assert secret in json.dumps(
        fake.seen,
    )  # the writer itself saw it; the record must not
    head, *rows = [
        json.loads(n) for n in stores.episodes.notes(sha, ref=TRANSCRIPT_REF)
    ]
    # the header names the brief and the tool JSON by hash; then the messages
    assert head["pass_id"] == "e1.p0" and head["header"]["brief_sha256"]
    assert {r["pass_id"] for r in rows} == {"e1.p0"}
    assert [r["message"]["role"] for r in rows] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert (
        rows[2]["message"]["content"] == "the token is <secret:UNIFY_TEST_FAKE_TOKEN>"
    )
    assert secret not in stores.episodes.run("log", "-p", "--all")
    assert secret not in json.dumps(rows) and secret not in json.dumps(head)


# --- drift and suspect -----------------------------------------------------------------------------------


def test_drift_rides_with_the_pass_and_is_cleared_once_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeWriter())
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    state = State(stores.paths.state, drift={"venmo"}, suspect={"venmo", "slack"})
    _run(stores, "e1", sha, state)
    assert state.drift == set()
    assert state.suspect == {"venmo", "slack"}  # a failed pass clears no suspect flag


# --- checker text stays out of every record and writer input ---------------------------------------------


SENTINEL = "SENTINEL-7f3a"


# --- reason codes: structural, never read from reason text ----------------------------------------------


@pytest.mark.parametrize(
    "script, codes",
    [
        (
            [
                ({"role": "assistant", "content": "x"}, "0.0000001"),
                RuntimeError("down"),
            ],
            ["writer_error", "no_manifest"],
        ),
        (
            [({"role": "assistant", "content": "x"}, "0.00000073")],
            ["pass_cap", "no_manifest"],
        ),
        (["sleep"], ["deadline", "no_manifest"]),
    ],
    ids=["writer-error", "usd-cap", "deadline"],
)
def test_end_event_reason_codes_come_from_the_outcome(
    tmp_path,
    monkeypatch,
    script,
    codes,
):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeWriter(script=script))
    monkeypatch.setattr(consolidate, "DEADLINE_S", 0.3)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = _run(stores, "e1", sha)
    assert out.codes == codes
    assert _events(stores)[-1]["reason_codes"] == codes


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
    res = Gate(stores.memory, stores.evidence, stores.blobs).check(
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
        action_lookup=EpisodeLookup(stores).action,
    )
    parent = stores.memory.head()
    cand = _commit(stores.memory, WT_FILES)
    res = gate.check(parent, cand, WT_MAN)
    assert res.checks["G2"], res.reasons
    blind = Gate(
        stores.memory,
        stores.evidence,
        stores.blobs,
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


# --- library mode: the switches reach the gate and the writer's brief ---------------------------------------


# --- library mode (UNIFY_MEMORY_LIBRARY) ----------------------------------------------------------------------


def _library_settings(raw):
    s = _settings()
    s.UNIFY_MEMORY_LIBRARY = raw
    s.UNIFY_MEMORY_LIBRARY_E = (
        1  # library mode's own E (100k by default): one small episode makes a pass due
    )
    return s


def test_writer_settings_reads_the_library_switch():
    assert writer_settings(_settings()).library_mode is False
    assert writer_settings(_library_settings("on")).library_mode is True
    assert writer_settings(_library_settings("off")).library_mode is False
    with pytest.raises(ValueError):
        writer_settings(_library_settings("maybe"))


def test_library_has_its_own_usd_per_token_and_the_off_path_ignores_it():
    """v2's rate by default; the replay can raise library mode's without a code change."""
    assert writer_settings(_library_settings("on")).usd_per_token == Decimal(
        "0.00000073",
    )
    raised = _library_settings("on")
    raised.UNIFY_MEMORY_LIBRARY_WRITER_USD_PER_TOKEN = "0.000002"
    assert writer_settings(raised).usd_per_token == Decimal("0.000002")
    off = _library_settings("off")
    off.UNIFY_MEMORY_LIBRARY_WRITER_USD_PER_TOKEN = "0.000002"
    assert writer_settings(off).usd_per_token == Decimal("0.00000073")
    for bad in ("0", "-1", "1e-6", "abc"):
        raised.UNIFY_MEMORY_LIBRARY_WRITER_USD_PER_TOKEN = bad
        with pytest.raises(ValueError):
            writer_settings(raised)


def test_library_off_leaves_the_end_event_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeWriter())
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = _run(stores, "e1", sha)
    assert out.coverage is None
    end = [e for e in _events(stores) if e.get("phase") == "end"][-1]
    assert not {"coverage", "reads", "exported_bytes", "chain", "provenance"} & set(end)


# --- library mode: CURATE after WRITE, triggered by library state ----------------------------------------------

from tests.memory.test_gate import _merged
from tests.memory.test_layout import LIB
from tests.memory.test_overlap import SPLIT, SPLIT_PATH

#: an earlier pass left two functions doing one job
DUPLICATE = {
    "memory/text/__init__.py": '"""Text helpers."""\n',
    "memory/text/parse.py": LIB["memory/text/parse.py"],
    SPLIT_PATH: SPLIT,
}


def _settings_library(on=True, **kw):
    s = _settings(**kw)
    s.UNIFY_MEMORY_LIBRARY = "on" if on else "off"
    s.UNIFY_MEMORY_LIBRARY_E = 1  # library mode's own E (100k by default)
    return s


def _run_library(stores, eid, sha, *, on=True, **kw):
    return asyncio.run(
        run_due_passes(
            stores,
            eid,
            sha,
            State(stores.paths.state),
            effort="low",
            settings=_settings_library(on, **kw),
            emit=None,
        ),
    )


def _no_records(monkeypatch):
    # the item records are not under test here: no retired item, every item on trial
    monkeypatch.setattr(consolidate, "_lifecycle", lambda stores, sha: (None, {}, {}))


def _kinds(stores):
    return [
        r[0]
        for r in stores.evidence.db.execute("SELECT kind FROM passes ORDER BY pass_id")
    ]


def test_no_curate_on_a_library_with_nothing_to_curate(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeWriter())
    _no_records(monkeypatch)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    assert [o.pass_id for o in _run_library(stores, "e1", sha)] == ["e1.p0"]


def test_curate_never_runs_with_library_off(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeWriter())
    monkeypatch.setattr(
        consolidate,
        "_lifecycle",
        lambda *a: pytest.fail("v2 never reads CURATE's state"),
    )
    stores = _stores(tmp_path)
    _merged(stores.memory, DUPLICATE)
    sha, _ = _record(stores, "e1")
    assert [o.pass_id for o in _run_library(stores, "e1", sha, on=False)] == ["e1.p0"]
    found = stores.evidence.db.execute(
        "SELECT 1 FROM sqlite_master WHERE name IN ('curate_seen', 'curations')",
    ).fetchone()
    assert found is None


def test_curate_waits_for_the_run_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(
        consolidate,
        "unillm_turn",
        FakeWriter(),
    )  # 0.01 USD a call: the WRITE pass commits 0.01
    _no_records(monkeypatch)
    stores = _stores(tmp_path)
    _merged(stores.memory, DUPLICATE)
    sha, _ = _record(stores, "e1")
    assert [o.pass_id for o in _run_library(stores, "e1", sha, guard="0.01")] == [
        "e1.p0",
    ]
    held = [e for e in _events(stores) if e.get("phase") == "end"][
        -1
    ]  # library mode publishes served after it
    assert held["pass_id"] == "e1.p1" and held["reason_codes"] == ["run_guard"]
    assert (
        stores.evidence.curate_seen() == set()
    )  # never shown: it fires again once the guard allows


def test_no_curate_after_a_write_whose_call_may_still_be_in_flight(
    tmp_path,
    monkeypatch,
):
    fake = FakeWriter(script=[RuntimeError("upstream failed")])
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    _no_records(monkeypatch)
    stores = _stores(tmp_path)
    _merged(stores.memory, DUPLICATE)
    sha, _ = _record(stores, "e1")
    _run_library(stores, "e1", sha)
    assert fake.calls == 1 and _kinds(stores) == ["batched"]


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

    gate = consolidate._ReportingGate(Gate(), lambda phase, **f: reports.append((phase, f)))
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
