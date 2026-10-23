"""Capturing every completed attempt at once, into a store of its own beside the library.

When the episode-end fork settles, the harness (never an agent) captures the attempt: it checks the fork's
``procedure.py`` for safety and writes ``procedure.py``, ``signature.json`` and ``record.json`` under
``<store>/<capture id>/``, where the capture id is the first 16 hex digits of the sha256 of ``procedure.py``. Rev 2
(the trace contract): the procedure is never run; its record says so, its signature describes what was recorded,
and its outcome comes from the episode's signals only. The
store is not the library: only the writer edits the library, and the writer reads captures as pool entries.

The applicability signature is computed by the harness and holds no request text. Rev 2.1: an identifier is never
stripped or masked, so the goal (the summary's Asked line) keeps the ids it states, as written. An attempt with a contrary signal is captured as negative knowledge
(``outcome: "contrary"``).

Every name this file assumes for code that does not exist yet is in the block of constants below, so the code's
real names can be put in one place.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import world  # noqa: F401
from unify.memory.episodes import Action, Episode
from unify.memory.signals import Signal
from tests.memory import prompt_guards as guards

# ---------------------------------------------------------------------------------------------------------------
# Assumed names (the spec leaves these open; adapt them here)
# ---------------------------------------------------------------------------------------------------------------
#: The capture step's module (its step is ``capture_trace_step(staging_dir, episode, store, *, signals)``).
CAPTURES_MODULE = "unify.memory.captures"
#: The store's directory name under the state home, and the ``Paths`` property that names it.
STORE_DIR_NAME = "memory-captures"
PATHS_STORE_ATTR = "captures"
#: The files of one capture, and the store's event log.
PROCEDURE_FILE = "procedure.py"
SIGNATURE_FILE = "signature.json"
RECORD_FILE = "record.json"
EVENTS_FILE = "captures-events.jsonl"
#: Fork files: the procedure and what broke, and their block kinds in the fork's ledger.
FORK_PROCEDURE = "procedure.py"
FORK_BROKE = "broke.md"
FORK_PROCEDURE_KIND = "procedure"
FORK_BROKE_KIND = "broke"
#: Signature fields.
SIGNATURE_FIELDS = ("input", "output", "relations", "constraints", "goal")
#: The older reproduction outcome and failed outcome (a record the pool still reads), the trace contract's negative
#: outcome, and the renamed starting level and status.
FAILED = "failed"
CONTRARY = "contrary"
LEVELS = ("unchecked", "reproduces-its-case", "confirmed-by-outcomes")
START_STATUS = "on-trial"
#: Pool kinds the writer reads captures through, and the pool's reader of the store.
POOL_CAPTURE_KIND = "capture"
POOL_FAILURE_KIND = "failure"
#: The writer's tool that lands a capture, and its brief text (quoted from the spec).
LAND_TOOL = "land_capture"
LAND_CALL_TEXT = "land_capture(capture_id, package, name)"
BRIEF_CAPTURE_TEXT = "The harness runs no capture"
BRIEF_FAILURE_TEXT = (
    "A failed capture is knowledge too: a note of what broke, or a test marked as a known failure."
)


def _capture(staging: Path, ep: Episode, store: Path, signals: list[Signal]):
    """The capture step as the harness runs it once a fork settles (the trace contract: nothing is run)."""
    mod = importlib.import_module(CAPTURES_MODULE)
    return mod.capture_trace_step(staging, ep, store, signals=signals)


def _from_captures(store: Path):
    """The store as the pool reads it at a pass's start (``Pool.enter_captures``): each capture of
    ``captures.load(store)``, in id order, as ``pool.from_capture``'s entry."""
    from unify.memory import captures, pool

    return [pool.from_capture(cid, doc) for cid, doc in sorted(captures.load(store).items())]


# ---------------------------------------------------------------------------------------------------------------

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not sys.platform.startswith("linux"),
    reason="bubblewrap required: model-written code runs only confined",
)

#: Identifiers the request prints; none may reach a signature.
IDS = ("ALPHA-7731", "cobalt-otter", "3f9a2c1e")
OTHER_IDS = ("BRAVO-0042", "amber-lynx", "7d1e0b44")


def _request(ids=IDS, numbers="5 3 9 1") -> str:
    return (
        f"Ticket {ids[0]} (alias: {ids[1]}), ref {ids[2]}\n"
        "Sort the numbers below in ascending order and reply with the list.\n"
        f"{numbers}\n"
    )


def _summary(ids=IDS) -> str:
    return (
        "## Asked\n"
        f"Sort the numbers of ticket {ids[0]} (alias: {ids[1]}, ref {ids[2]}) in ascending order.\n\n"
        "## Approach\nParsed the last line of numbers and sorted them.\n\n"
        "## Worked\nsorted() on the parsed list.\n\n"
        "## Failed\nnone\n\n"
        "## Outcome\nThe list was handed back.\n"
    )


PROCEDURE = '''"""Sort the numbers of a request. Accepted: not shown."""
import os
import re


def read_input(text):
    """The numbers on the last line that holds only digits and spaces."""
    rows = [ln for ln in text.splitlines() if ln.strip() and re.fullmatch(r"[0-9 ]+", ln.strip())]
    return [int(x) for x in rows[-1].split()]


def run(value):
    """The numbers in ascending order."""
    return sorted(value)


def applies(value):
    """A list of whole numbers."""
    return isinstance(value, list) and all(isinstance(x, int) for x in value)


def holds(value, result):
    """The result is the input in ascending order."""
    return result == sorted(value)
'''

WRONG_PROCEDURE = '''"""Sort descending (the request asked for ascending)."""


def read_input(text):
    """The numbers on the last line."""
    return [int(x) for x in text.strip().splitlines()[-1].split()]


def run(value):
    """The numbers in descending order."""
    return sorted(value, reverse=True)
'''

#: A planted, readable, key-shaped test string (never a real credential).
PLANTED_KEY = "sk-" + "planted-test-value-" * 2  # pragma: allowlist secret
#: The planted environment value a confined run must never see.
PLANTED_ENV_NAME = "CAPTURE_TEST_PLANTED_VALUE"
PLANTED_ENV_VALUE = "planted-readable-value-for-capture-tests"  # pragma: allowlist secret


def _confinement_probe(sentinel: Path) -> str:
    """A procedure that tries to write a host file and reads a planted environment value; it reproduces only when
    it runs with an empty environment, and the host file appears only if it ran on the host."""
    return f'''"""Sort, while probing the box."""
import os


def read_input(text):
    """The numbers on the last line."""
    return [int(x) for x in text.strip().splitlines()[-1].split()]


def run(value):
    """The numbers in ascending order."""
    try:
        with open({str(sentinel)!r}, "w") as f:
            f.write("ran on the host")
    except OSError:
        pass
    if os.environ.get({PLANTED_ENV_NAME!r}):
        return []
    return sorted(value)
'''


def _cid(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _episode(eid: str, request: str, handed_back) -> Episode:
    return Episode(
        episode_id=eid,
        started_at="2026-10-13T01:00:00Z",
        ended_at="2026-10-13T01:01:00Z",
        build="0" * 9,
        model="test/model",
        effort="low",
        regime="dense",
        memory_main="0" * 40,
        worktree_before=None,
        worktree_after=None,
        request=[request],
        transcript=[{"role": "user", "content": request}],
        cells=[],
        actions=[
            Action(
                0,
                "user",
                "act",
                [handed_back],
                {},
                "received",
                "ok",
                "write",
                kind="dialogue",
            ),
        ],
        replies=[json.dumps(handed_back)],
    )


def _signal(eid: str, label: str) -> Signal:
    return Signal(
        signal_id=f"{eid}-checker",
        episode_id=eid,
        source="checker",
        label=label,
        ts="2026-10-13T01:01:00Z",
        regime="dense",
        visible_to_actor=True,
    )


def _stage(
    home: Path,
    eid: str,
    procedure: str | None,
    *,
    summary: str | None = None,
    broke: str = "none\n",
) -> Path:
    """A settled fork's staging directory, as the fork worker leaves it."""
    d = home / "memory-staging" / eid
    d.mkdir(parents=True)
    (d / "summary.md").write_text(summary if summary is not None else _summary())
    if procedure is not None:
        (d / FORK_PROCEDURE).write_text(procedure)
    (d / FORK_BROKE).write_text(broke)
    (d / "fork.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "episode": eid,
                "procedure": procedure is not None,
                "broke": True,
            },
        ),
    )
    return d


def _store(home: Path) -> Path:
    return home / STORE_DIR_NAME


def _record(store: Path, cid: str) -> dict:
    return json.loads((store / cid / RECORD_FILE).read_text())


def _served(home: Path, store: Path, cid: str) -> dict:
    """The capture's row of the harness's evidence (``serving.item_evidence``) from an evidence store with no use of
    it: what its uses give its status and level, which its record never holds."""
    from unify.memory import captures, serving
    from unify.memory.integration.consolidate import open_stores
    from unify.memory.integration.paths import Paths

    stores = open_stores(Paths.under(home))
    return serving.item_evidence(stores, captures=captures.load(store), home=None)["capture:" + cid]


def _signature_bytes(store: Path, cid: str) -> str:
    return (store / cid / SIGNATURE_FILE).read_text()


def _events(store: Path) -> list[dict]:
    p = store / EVENTS_FILE
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _held(relations) -> set[str]:
    """The relation names that held, from a list of names or a mapping of name to bool."""
    if isinstance(relations, dict):
        return {k for k, v in relations.items() if v}
    return set(relations or [])


def _tree_bytes(root: Path, skip: Path) -> dict[str, bytes]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and skip not in p.parents:
            out[p.relative_to(root).as_posix()] = p.read_bytes()
    return out


def _captured(tmp_path: Path, procedure: str = PROCEDURE, *, ids=IDS, label="pass"):
    home = tmp_path / "home"
    eid = "ep-0001"
    staging = _stage(home, eid, procedure, summary=_summary(ids))
    store = _store(home)
    _capture(staging, _episode(eid, _request(ids), [1, 3, 5, 9]), store, [_signal(eid, label)])
    return store, _cid(procedure)


# --- the fork writes the procedure and what broke ------------------------------------------------------------


def test_the_fork_may_write_a_procedure_and_what_broke():
    from unify.memory.integration import fork_worker as fw

    assert any(rx.fullmatch(FORK_PROCEDURE) for rx in fw.ALLOWED)
    assert any(rx.fullmatch(FORK_BROKE) for rx in fw.ALLOWED)
    assert fw.kind(FORK_PROCEDURE) == FORK_PROCEDURE_KIND
    assert fw.kind(FORK_BROKE) == FORK_BROKE_KIND
    # only those exact names: no directories, no other suffixes
    assert not any(rx.fullmatch("x/procedure.py") for rx in fw.ALLOWED)
    assert not any(rx.fullmatch("procedure.sh") for rx in fw.ALLOWED)


# --- a completed attempt is captured with its procedure and signature ----------------------------------------


@needs_bwrap
def test_a_completed_attempt_is_captured_with_its_procedure_signature_and_record(
    tmp_path,
):
    store, cid = _captured(tmp_path)
    d = store / cid
    assert (d / PROCEDURE_FILE).read_text() == PROCEDURE
    sig = json.loads((d / SIGNATURE_FILE).read_text())
    assert set(SIGNATURE_FIELDS) <= set(sig)
    assert sig["input"] and sig["output"]
    # rev 2: the signature describes what was recorded: the value handed back (a list of whole numbers), and no
    # relation that only a run's read value could show
    assert sig["output"]["tree"] == ["int"]
    assert "scalar_output" not in _held(sig["relations"])
    constraints = json.dumps(sig["constraints"])
    assert "applies" in constraints and "holds" in constraints
    assert sig["constraints"]["applies"]["held"] is None and sig["constraints"]["holds"]["held"] is None
    assert "ascending" in sig["goal"]  # the goal is the summary's Asked section

    rec = _record(store, cid)
    assert rec["episodes"] == ["ep-0001"]
    # rev 2: never run to classify it: no reproduction, no determinism, no conditions judged
    assert rec["own_check"] == {"ran": False, "why": "a capture is never run to classify it"}
    assert "reproduces" not in rec
    assert rec["deterministic"] is None
    assert rec["applies"] is None and rec["holds"] is None
    assert rec["value"]["present"] is True
    assert rec["outcome"] == "positive"
    # its uses, and the status and level they give it, are the evidence store's, never its record's
    from unify.memory.captures import USE_FIELDS

    assert not set(USE_FIELDS) & set(rec)
    served = _served(tmp_path / "stores", store, cid)
    assert (served["status"], served["level"]) == (START_STATUS, LEVELS[0])
    assert served["support"] == {"episodes": 1, "accepted": 1, "confirmed_uses": 0}

    rows = _events(store)
    assert len(rows) == 1
    row = rows[0]
    assert row["capture_id"] == cid and row["episode"] == "ep-0001"
    assert row["safety"] == "ok" and row["run"] == "not run" and "reproduces" not in row
    assert row["published_at"]


@needs_bwrap
def test_the_signature_carries_no_request_text_and_keeps_its_goals_identifiers(tmp_path):
    store, cid = _captured(tmp_path)
    text = _signature_bytes(store, cid)
    assert "5 3 9 1" not in text
    assert "Sort the numbers below" not in text
    goal = json.loads(text)["goal"]
    for ident in IDS:
        assert ident in goal, ident
    assert "<id>" not in text


@needs_bwrap
def test_the_signature_is_the_same_when_only_printed_identifiers_change_but_its_goal(tmp_path):
    """The structure is the same; the goal states each episode's own identifiers (rev 2.1: never masked)."""
    a, cid = _captured(tmp_path / "a", ids=IDS)
    b, cid_b = _captured(tmp_path / "b", ids=OTHER_IDS)
    assert cid == cid_b
    sig_a, sig_b = json.loads(_signature_bytes(a, cid)), json.loads(_signature_bytes(b, cid))
    assert {k: v for k, v in sig_a.items() if k != "goal"} == {k: v for k, v in sig_b.items() if k != "goal"}
    for ident in OTHER_IDS:
        assert ident in sig_b["goal"] and ident not in sig_a["goal"]


@needs_bwrap
def test_the_procedure_never_runs(tmp_path, monkeypatch, request):
    """Rev 2: the capture step runs no model-written code at all, on the host or in a box."""
    # outside /tmp on purpose: the box's /tmp is private, so a probe there would prove nothing
    root = Path(__file__).resolve().parents[2] / "logs" / "capture_store_tests"
    digest = hashlib.md5(request.node.nodeid.encode()).hexdigest()[:12]
    probe_dir = root / f"{digest}-{os.getpid()}"
    shutil.rmtree(probe_dir, ignore_errors=True)
    probe_dir.mkdir(parents=True)
    sentinel = probe_dir / "escaped.txt"
    monkeypatch.setenv(PLANTED_ENV_NAME, PLANTED_ENV_VALUE)
    try:
        source = _confinement_probe(sentinel)
        store, cid = _captured(tmp_path, source)
        assert not sentinel.exists(), "the procedure ran on the host"
        rec = _record(store, cid)
        assert rec["own_check"]["ran"] is False and [c["run"] for c in rec["cases"]] == ["not run"]
        assert [r["run"] for r in _events(store)] == ["not run"]
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


@needs_bwrap
def test_a_key_shaped_string_is_refused_before_anything_runs(tmp_path):
    source = PROCEDURE + f"\nTOKEN = {PLANTED_KEY!r}\n"
    store, cid = _captured(tmp_path, source)
    assert not (store / cid).exists()
    rows = _events(store)
    assert len(rows) == 1 and rows[0]["safety"] != "ok"
    for p in store.rglob("*"):
        if p.is_file():
            assert PLANTED_KEY not in p.read_text(errors="replace"), p


# --- a failure is captured as negative knowledge -------------------------------------------------------------


@needs_bwrap
def test_an_attempt_with_a_contrary_signal_is_captured_as_negative_knowledge(tmp_path):
    from unify.memory.captures import negative

    store, cid = _captured(tmp_path, label="fail")
    rec = _record(store, cid)
    assert rec["outcome"] == CONTRARY and negative(rec)
    assert "status" not in rec and "level" not in rec
    served = _served(tmp_path / "stores", store, cid)
    assert (served["status"], served["level"]) == (START_STATUS, LEVELS[0])
    assert served["support"] == {"episodes": 1, "accepted": 0, "confirmed_uses": 0}


# --- the store sits outside every root a box can write or read -----------------------------------------------


def test_the_store_is_harness_only_state():
    from unify.memory.integration.paths import Paths

    paths = Paths.under(Path("/srv/unify-home"))
    store = getattr(paths, PATHS_STORE_ATTR)
    assert store.name == STORE_DIR_NAME
    assert store in paths.harness_only()
    for p in (paths.checkout, paths.memory):
        assert p != store and p not in store.parents and store not in p.parents


def _within(p: Path, root: Path) -> bool:
    p, root = Path(os.path.realpath(p)), Path(os.path.realpath(root))
    return p == root or root in p.parents


def test_no_cell_can_read_or_write_the_store(world):  # noqa: F811
    from unify import sandbox, transcripts
    from unify.memory.integration.paths import Paths

    paths = Paths.under(world["state"])
    store = getattr(paths, PATHS_STORE_ATTR)
    store.mkdir(parents=True)
    (store / "probe.json").write_text("{}")
    policy = sandbox.build_policy(fresh=True)
    assert policy.readable_violation(store) is not None
    assert policy.readable_violation(store / "probe.json") is not None
    for root in (
        policy.workspace,
        *policy.readonly_state,
        *policy.root_visible,
        transcripts.transcripts_dir(),
        paths.checkout,  # the one path mounted read-write into the worker
    ):
        assert not _within(store, root), (store, root)


# --- only the writer edits the library; the writer reads captures ------------------------------------------


@needs_bwrap
def test_the_capture_step_changes_nothing_outside_its_store(tmp_path):
    home = tmp_path / "home"
    (home / "memory").mkdir(parents=True)
    (home / "memory" / "HEAD").write_text("ref: refs/heads/main\n")
    (home / "memory-checkout" / "memory").mkdir(parents=True)
    (home / "memory-checkout" / "memory" / "__init__.py").write_text("")
    staging = _stage(home, "ep-0001", PROCEDURE)
    store = _store(home)
    before = _tree_bytes(home, store)
    _capture(
        staging,
        _episode("ep-0001", _request(), [1, 3, 5, 9]),
        store,
        [_signal("ep-0001", "pass")],
    )
    assert _tree_bytes(home, store) == before
    assert (store / _cid(PROCEDURE)).is_dir()


def test_only_the_writer_has_the_tool_that_lands_a_capture():
    from unify.common import tool_registry as tr

    spec = tr.registry()[LAND_TOOL]
    assert tuple(spec.roles) == ("writer",)
    flags = frozenset(
        s.requires for s in tr.registry().values() if s.requires is not None
    )
    for agent in tr.AGENTS:
        names = tr.allow_list(agent, flags)
        assert (LAND_TOOL in names) == (agent == "writer"), agent


def _write_record(store: Path, source: str, record: dict, broke: str = "none\n") -> str:
    cid = _cid(source)
    d = store / cid
    d.mkdir(parents=True)
    (d / PROCEDURE_FILE).write_text(source)
    (d / FORK_BROKE).write_text(broke)
    (d / SIGNATURE_FILE).write_text(
        json.dumps(
            {
                "input": {"kind": "value", "tree": ["int"], "lengths": {"": "few"}},
                "output": {"kind": "value", "tree": ["int"], "lengths": {"": "few"}},
                "relations": ["same_shape", "same_outer_length"],
                "constraints": {},
                "goal": "Sort the numbers in ascending order.",
            },
        ),
    )
    base = {
        "episodes": ["ep-0001"],
        "deterministic": True,
        "applies": None,
        "holds": None,
        "outcome": "none",
        "level": LEVELS[0],
        "status": START_STATUS,
        "track": {},
    }
    base.update(record)
    (d / RECORD_FILE).write_text(json.dumps(base))
    return cid


def test_the_writer_reads_each_capture_as_a_pool_entry(tmp_path):
    from unify.memory import pool

    assert POOL_CAPTURE_KIND in pool.KINDS and POOL_FAILURE_KIND in pool.KINDS
    store = tmp_path / STORE_DIR_NAME
    ok = _write_record(store, PROCEDURE, {})
    bad = _write_record(
        store,
        WRONG_PROCEDURE,
        {"outcome": FAILED, "episodes": ["ep-0002"]},
        broke="- run: sorted descending; the record shows the ascending list was asked for\n",
    )
    entries = _from_captures(store)
    by_kind: dict[str, list] = {}
    for e in entries:
        by_kind.setdefault(e.kind, []).append(e)
    (cap,) = by_kind[POOL_CAPTURE_KIND]
    (fail,) = by_kind[POOL_FAILURE_KIND]
    # its data is what the attempt's record and signature say of it; what an earlier build's record kept (its
    # reproduction, level, status and same_as) stays in its record.json, never in the entry's data
    assert cap.data == {"id": ok, "capture_id": ok, "goal": "Sort the numbers in ascending order.", "outcome": "none"}
    assert PROCEDURE in cap.files.values()
    assert cap.provenance["episodes"] == ["ep-0001"]
    # a failed capture is negative knowledge: listed as a failure, with what broke, never as a capture
    assert fail.data["id"] == bad and fail.data["outcome"] == FAILED
    assert any("sorted descending" in v for v in fail.files.values())
    assert bad not in [e.data.get("id") for e in by_kind[POOL_CAPTURE_KIND]]
    # the same store gives the same entries
    assert [e.id for e in _from_captures(store)] == [e.id for e in entries]


def test_an_empty_or_missing_store_gives_no_entries(tmp_path):
    assert _from_captures(tmp_path / "absent") == []
    (tmp_path / "empty").mkdir()
    assert _from_captures(tmp_path / "empty") == []


def test_the_writer_brief_says_captures_exist_and_how_to_land_one():
    from unify.memory import library_prompts as lp

    brief = lp.write_brief_now()
    assert LAND_CALL_TEXT in brief
    assert BRIEF_CAPTURE_TEXT in brief and "has already run" not in brief
    assert BRIEF_FAILURE_TEXT in brief
    assert guards.benchmark_words(brief) == []
    assert guards.example_checks(brief) == []
