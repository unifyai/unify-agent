"""With UNIFY_MEMORY_V2=on and every v2.1 switch at its default, the build is the v2 screen build (9deefbfd1).

The switches: ``UNIFY_MEMORY_V2_SURFACING`` (``index``), ``UNIFY_MEMORY_V2_DOCSTRINGS`` (``off``) and
``UNIFY_MEMORY_V2_SOFT_BUDGET`` (``off``). Byte for byte at their defaults: the actor's memory section (the
v2 index and export line), the export tree (the commit's files and nothing generated), Sol's brief, tools
and first message (the current index), G4's refusal over the index budget, and the evidence schema.

The goldens (``golden/v2_screen_9deefbfd1.json``) are read out of 9deefbfd1's source by
``golden/extract_v2_screen.py`` (ast, standard library only); :func:`test_the_goldens_are_read_from_the_v2_screen_commit`
re-reads them from the commit whenever the clone holds it. The expected texts are then built from the goldens
with the v2 code paths the build keeps (``build_index``, ``_manifest_rules``), so a change to either side shows.

The gate's other outcomes at the defaults are those of ``test_gate.py``, whose tests are 9deefbfd1's and run on
the default gate (only the G4 test gained a soft-budget variant).
"""

import asyncio
import importlib.util
import inspect
import json
import random
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from unify.memory_v2 import sol_pass
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.index import IndexOverBudget, build_index
from unify.memory_v2.integration import hooks, switch
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.checkout import export_checkout
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.integration.test_request import _abort, _begin, mv2  # noqa: F401
from tests.memory_v2.test_gate import (
    FILES,
    MAN,
    _candidate,
    _lookup,
    world,
)  # noqa: F401

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN = json.loads((GOLDEN_DIR / "v2_screen_9deefbfd1.json").read_text())
REPO = Path(__file__).resolve().parents[2]
needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required (the gate runs pytest confined)",
)


def _extractor():
    spec = importlib.util.spec_from_file_location(
        "extract_v2_screen",
        GOLDEN_DIR / "extract_v2_screen.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_goldens_are_read_from_the_v2_screen_commit():
    have = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"{GOLDEN['commit']}^{{commit}}"],
        capture_output=True,
    )
    if have.returncode != 0:
        pytest.skip(
            "this clone does not hold 9deefbfd1; the goldens stand as committed",
        )
    assert _extractor().extract(REPO, GOLDEN["commit"]) == GOLDEN


# --- the switches ------------------------------------------------------------------------------------------


def test_the_defaults_are_the_v2_behaviour(monkeypatch):
    from unify.settings import ProductionSettings

    for name in (switch.SURFACING, switch.DOCSTRINGS, switch.SOFT_BUDGET):
        monkeypatch.delenv(name, raising=False)
    s = ProductionSettings()
    opts = switch.surfacing_options(s)
    assert opts == switch.SurfacingOptions()
    assert opts.gate_kwargs() == {
        "surfacing": "index",
        "docstring_standard": False,
        "soft_budget": False,
    }
    assert (
        switch.surfacing_options(object()) == opts
    )  # a missing setting is its default


def test_the_gate_defaults_are_the_switch_defaults(tmp_path):
    mem = Repo.init_bare(tmp_path / "m.git")
    gate = Gate(mem, EvidenceStore(tmp_path / "e.sqlite"), BlobStore(tmp_path / "b"))
    assert (gate.surfacing, gate.docstring_standard, gate.soft_budget) == (
        "index",
        False,
        False,
    )
    assert gate.budget == GOLDEN["gate_budget_tokens_default"]


@pytest.mark.parametrize(
    "name,parse,values",
    [
        (switch.SURFACING, switch.parse_surfacing, ("index", "catalogue")),
        (switch.DOCSTRINGS, switch.parse_docstrings, ("off", "on")),
        (switch.SOFT_BUDGET, switch.parse_soft_budget, ("off", "on")),
    ],
)
def test_each_switch_takes_only_its_values(monkeypatch, name, parse, values):
    from unify.settings import ProductionSettings

    for value in values:
        for spelt in (value, value.upper(), f" {value} "):
            monkeypatch.setenv(name, spelt)
            assert getattr(ProductionSettings(), name) == value
            assert parse(spelt) == value
    monkeypatch.setenv(name, "")
    assert getattr(ProductionSettings(), name) == values[0]  # empty is the default
    rng = random.Random(20261008)
    alphabet = "abcdefghijklmnopqrstuvwxyz01_- "
    bad = {"true", "1", "yes", "none", "catalog", "indexes", "onn", "of"}
    bad |= {
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 9)))
        for _ in range(40)
    }
    for raw in sorted(bad):
        if raw.strip().lower() in values or not raw.strip():
            continue
        with pytest.raises(ValueError, match=name):
            parse(raw)
        monkeypatch.setenv(name, raw)
        with pytest.raises(ValueError):
            ProductionSettings()


# --- Sol ---------------------------------------------------------------------------------------------------


# Declared changes to v2 behaviour that are not v2.1 switches, merged into the frozen build (memory-v2-int1):
# D26 (memory-v2-sol-hygiene) gives Sol's brief one paragraph and a longer gate summary, and its first message
# the pass's functions with their cover counts; use telemetry (memory-v2.1-tele) records each request's use of
# the library in an ``item_use`` table; the later merges' brief sentences are in ``INSERTS``. Everything else at
# the switch defaults is 9deefbfd1's, byte for byte.
D26_PARAGRAPH = (
    "Tend the library too. On this pass's channels, read the existing functions and tests (the request lists each\n"
    "function's recorded covers; /inputs/library_covers.json holds them as [episode_id, action_index] lists) and, where\n"
    "it makes the library smaller or clearer: merge near-duplicates into one function (keep an old name that code outside\n"
    "the channel may import as a thin alias calling the merged one); delete a function the episodes show is wrong or\n"
    "unused, listing every recorded input it covered in a remaining function's covers; repair a function that refused\n"
    "an input the environment accepted. Test first here as well: every old test keeps passing against the result, or\n"
    'is retired in "deleted_tests" (only a test file of deleted functions) with the reason in the summary. A pass that\n'
    "adds nothing and shrinks the library needs no red test for an edited function an old passing test exercises.\n\n"
)
D26_AFTER = "the library must earn its place.\n\n"
D26_SUMMARY = (
    "test suite, an index budget, that the library only grows when it covers new recorded calls, and safety. Its rules\n"
    "follow; a pass that breaks one is refused whole.\n",
    "test suite, an index budget, that the library only grows when it covers new recorded calls or shrinks, that what\n"
    "deleted functions covered stays covered, and safety. Its rules follow; a pass that breaks one is refused\n"
    "whole.\n",
)
D26_FIRST_MESSAGE_TAIL = (
    "\n\nFunctions on this pass's channels:\n(none yet)"  # no episode: no channel
)
TELEMETRY_TABLES = {"item_use"}
# Merged into memory-v2-int2: the override rule (memory-v2-override-rule) adds one sentence to the brief's
# covers paragraph. Each entry is (anchor, text inserted after it).
INSERTS = [
    (
        "nonzero exit) that justifies a value check (never covers made only of rejections).\n",
        "A function that replaces a value it computed from its input under a condition encodes a policy; it needs covers\n"
        "from at least two episodes.\n",
    ),
]


def _v2_template() -> str:
    """The golden v2 brief template with the declared changes applied (each must apply exactly once)."""
    text = GOLDEN["sol_prompt_template"]
    assert text.count(D26_AFTER) == 1 and text.count(D26_SUMMARY[0]) == 1
    text = text.replace(D26_AFTER, D26_AFTER + D26_PARAGRAPH).replace(*D26_SUMMARY)
    for anchor, inserted in INSERTS:
        assert text.count(anchor) == 1, anchor
        text = text.replace(anchor, anchor + inserted)
    return text


def _v2_brief() -> str:
    return (
        _v2_template()
        .replace("{entries}", str(sol_pass.QUOTA_ENTRIES))
        .replace("{file_mib}", str(sol_pass.QUOTA_FILE_BYTES // 1024**2))
        .replace("{total_mib}", str(sol_pass.QUOTA_TOTAL_BYTES // 1024**2))
        .replace("{checks}", str(sol_pass.MAX_CHECKS))
        .replace("{semantic_types}", sol_pass._manifest.describe_semantic_types())
        .replace("{input_kinds}", sol_pass._manifest.describe_input_kinds())
        + "\n"
        + sol_pass._manifest_rules()
        + "\n"
    )


def _v2_tools() -> list[dict]:
    text = json.dumps(GOLDEN["sol_tools"]).replace(
        "{MAX_CHECKS}",
        str(sol_pass.MAX_CHECKS),
    )
    return json.loads(text)


def test_sols_brief_and_tools_at_the_defaults_are_v2s():
    assert sol_pass.SOL_SYSTEM == _v2_brief()
    assert sol_pass.sol_system() == _v2_brief()
    assert sol_pass._TOOLS == _v2_tools()
    assert sol_pass.sol_tools() == _v2_tools()


def _never(eid):
    raise AssertionError(f"episode {eid} was not requested")


class _FirstTurn:
    """Records the first call's messages and tools, then ends the pass (no box is started)."""

    def __init__(self):
        self.messages = None
        self.tools = None

    async def __call__(self, messages, tools):
        self.messages, self.tools = [dict(m) for m in messages], tools
        raise RuntimeError("recorded")


def test_sols_first_message_at_the_defaults_is_the_v2_index(
    tmp_path,
    world,
):  # noqa: F811
    mem, ev, _ = world
    mem.fast_forward("main", _candidate(mem, FILES), expected_old=mem.head())
    gate = Gate(mem, ev, BlobStore(tmp_path / "b2"), action_lookup=_lookup)
    turn = _FirstTurn()
    sol = sol_pass.SolPass(
        mem,
        gate,
        ev,
        load=_never,
        model_turn=turn,
        config=sol_pass.PassConfig(),
    )
    req = PassRequest("incremental", "venmo", [], False)
    asyncio.run(sol.run(req, "p-golden"))
    export_checkout(mem.git_dir, mem.head(), tmp_path / "co")
    index = build_index(tmp_path / "co")
    want = (
        GOLDEN["sol_first_message"]
        .replace("{pass_id}", "p-golden")
        .replace("{json.dumps(req.__dict__)}", json.dumps(req.__dict__))
        .replace("{index}", index)
        + D26_FIRST_MESSAGE_TAIL
    )
    assert turn.messages[0] == {"role": "system", "content": _v2_brief()}
    assert turn.messages[1] == {"role": "user", "content": want}
    assert turn.tools == _v2_tools()
    assert "README" not in turn.messages[1]["content"]


# --- the actor's prompt and the export ---------------------------------------------------------------------


def test_the_prompt_section_and_export_at_the_defaults_are_v2s(mv2):  # noqa: F811
    run = _begin(mv2, "Say hi to ada.")
    try:
        checkout = mv2.paths.checkout
        assert run.surfacing == switch.SurfacingOptions()
        index = build_index(
            checkout,
            budget_tokens=GOLDEN["index_budget_tokens"],
            suspect=set(run.state.suspect),
        )
        want = index + "\n" + GOLDEN["export_line"].replace("{checkout}", str(checkout))
        assert run.index == want
        assert hooks.system_prompt("S").endswith("\n\n" + want)
        committed = sorted(
            Repo(mv2.paths.memory).run("ls-tree", "-r", "--name-only", run.pin).split(),
        )
        exported = sorted(
            str(p.relative_to(checkout)) for p in checkout.rglob("*") if p.is_file()
        )
        assert exported == committed
        assert run.generated == {}
    finally:
        _abort(mv2, run)
    assert request_mod.current() is None


# --- the gate ----------------------------------------------------------------------------------------------


def _tables(db: sqlite3.Connection) -> set[str]:
    return {
        r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


@needs_bwrap
def test_g4_at_the_defaults_refuses_an_index_over_budget_as_v2(
    tmp_path,
    world,
):  # noqa: F811
    mem, ev, _ = world
    gate = Gate(
        mem,
        ev,
        BlobStore(tmp_path / "b2"),
        action_lookup=_lookup,
        budget_tokens=10,
    )
    cand = _candidate(mem, FILES)
    res = gate.check(mem.head(), cand, MAN)
    export_checkout(mem.git_dir, cand, tmp_path / "co")
    with pytest.raises(IndexOverBudget) as exc:
        build_index(tmp_path / "co", budget_tokens=10)
    assert not res.passed and not res.checks["G4"]
    assert [r for r in res.reasons if r.startswith("G4")] == [f"G4: {exc.value}"]
    assert not any(r.startswith("note: G4") for r in res.reasons)
    # the kept v2 branch is the v2 code, statement for statement
    v2 = [ln.strip() for ln in GOLDEN["gate_g4"].splitlines()[1:] if ln.strip()]
    now = [ln.strip() for ln in inspect.getsource(Gate._g4).splitlines()]
    at = 0
    for line in v2:
        at = now.index(line, at) + 1


@needs_bwrap
def test_a_merge_at_the_defaults_keeps_the_v2_evidence_schema(
    tmp_path,
    world,
):  # noqa: F811
    mem, ev, gate = world
    res = gate.merge(
        mem.head(),
        _candidate(mem, FILES),
        MAN,
        "p1",
        "incremental",
        "venmo",
        "0.01",
    )
    assert res.passed, res.reasons
    base = sqlite3.connect(":memory:")
    base.executescript(GOLDEN["evidence_schema"])
    assert _tables(ev.db) == _tables(base) | TELEMETRY_TABLES
    assert ev.commit_shapes(mem.head()) is None
    assert not any(r.startswith(("note: G4", "G3: the examples")) for r in res.reasons)
