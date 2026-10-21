"""QA-level test checks in the consolidation gate (library-mode stage 5, under the gate's check G3).

G3 asks that each new or changed environment function has a test that is red on the parent and green on the
candidate. It does not judge whether the test is any good: one fixture read many times, a stand-in environment
that ignores its arguments, an assertion that would hold whatever the function did. These checks judge test
quality, each behind its own switch (all off by default; :class:`QAConfig`). With every switch off, ``on`` is
False, the gate makes exactly the calls it made before, and nothing here runs.

**Seeded random fixtures** (:attr:`QAConfig.fixtures`: ``on`` or ``strict``). Inputs are drawn, never
chosen by the consolidator. For each new or changed function, :func:`draw` takes up to :data:`SAMPLE_K`
recorded inputs of the function's family (the family G2's held-out check uses: a tool call's
``channel.method``, a work-tree path family, a dialogue ``channel.method``) from the store-wide pool G2 read
(:meth:`.gate.Gate._pool`), leaving out the function's covers, rejections, truncated recordings and inputs its
declared form cannot take. Distinct inputs (by content) are ranked by SHA-256 of the seed and their content;
the seed is SHA-256 of the candidate commit (:func:`seed_of`), so a rerun draws the same sample and the
consolidator cannot know it before the harness commits. Then:

* the **probe** (:data:`PROBE`, one confined process per function and tree) calls the function on every drawn
  input in its declared form (``env``: a :class:`~.replay.RecordedEnv` over the input's episode's recorded
  calls on its channels, the drawn call first). On an input the environment accepted, the function must
  return, or raise its ``MemoryInputError`` when the input's fields differ from every cover's (a precondition).
  Raising anything else, or refusing an input shaped like its covers, refuses the pass. A replay miss (the
  function made a call that episode did not record) is inconclusive and noted. Arguments beyond the first
  come only from the recordings (the input's own recorded keyword arguments, else a cover's); a required one
  no recording names leaves the row unbound (inconclusive), and when the function returns on none of its
  covers under this convention the drawn inputs are not judged at all;
* the **sample run**: each of the function's new or changed test files that imports ``memlab.inputs`` runs
  again with ``/qa/samples.json`` mounted, so a test parametrised over :func:`.inputs.inputs` also runs its own
  assertions on every drawn input, and must stay green. :mod:`.inputs` records which drawn input each test case
  read; a drawn input no test of the function read is a note, and with ``strict`` a refusal.

**Mutation testing** (default 0.5).
Up to :data:`MAX_MUTANTS` mutants per function (:mod:`.mutation`, chosen by the seed) each run the function's
new or changed tests, confined, at most :data:`MUTANT_S` each. A mutant is killed when a test is not green. A
surviving mutant whose probe outputs equal the original's on every informative row (the function returned,
refused or raised) of the covers, the drawn inputs and the covers' structural negatives
(:func:`.mutation.negatives`: a field dropped, retyped, emptied or added) is likely equivalent and leaves the
count; the negatives keep a mutant that drops a shape guard from looking equivalent. A kill share below the
threshold refuses the pass, naming operator kinds and lines. When survivors remain but the probe got no
informative row (every input missed the replay or was unbound), the check is "not judged", which refuses the
pass. First the tests run on the module re-printed unchanged (the control): if that is not
green, nothing is judged.

**Library mode** (a library function takes any arguments; its examples are the calls its tests make). The
covers of the draw are the recorded actions behind the function's whole-source copied input fixtures, and each
drawn input takes the form its fixture's kind gives (:data:`SOURCE_FORMS`: a tool response or a dialogue
observation as ``observation``, a work-tree file as ``path``); a sliced fixture, a message or a cell's output
has no family to draw from, which is noted. No probe runs: the drawn inputs are appended to a copy of the
recorded-inputs file and the function's listed tests run with its calls recorded (:mod:`.example_calls`). A
call on an appended line (a call key the run without them did not make) that raises refuses the pass, as one
that refuses an input whose arguments have the field names of an accepted call's. A surviving mutant is likely
equivalent when every call of its tests returns, refuses or raises as on the original, with an equal result
digest; with no recorded call to compare, mutation is not judged, which refuses the pass.

**Determinism**. Every gate pytest run gets ``PYTHONHASHSEED=0``,
``TZ=UTC`` and the plugin :mod:`.pin` (``-p _memory_pin``: a stepping clock from a fixed epoch, ``random`` seeded
before each test). Each new or changed test file green in G3 runs once more under the other pin variant
(another epoch, clock step, random seed, hash seed and time zone); different outcomes refuse it, so a test can
neither be flaky nor depend on a pinned value. A second run that times out is noted, not judged.

**The test kit** (:mod:`.testkit`). Every gate pytest run mounts the kit (``memlab``, the pin plugin, the
blobs the tests name) read-only at ``/inputs`` when a switch is on **or** the library's tests use it, so a
stored library never depends on a switch setting; the same kit is in the writer's box and the actor's export. New
or changed library code that uses it, and tests importing what it does not provide or naming ``/inputs``,
are refused (``[qa:kit]``).

**Replay fidelity**. A new or changed test that passes an environment-taking
function a stand-in of its own instead of the kit's replay (``memlab.replay.env_from`` or ``RecordedEnv``, the
harness's exact-call replay, :mod:`.replay`), while the function has recorded calls, is refused
(:func:`.qa_static.stand_ins`, structural).

**Fixture size**. A new or changed file under ``env/<channel>/tests/``, the
test kit or a support file larger than :data:`FIXTURE_MAX_BYTES` is refused: recorded payloads are referenced
by blob id (:func:`.inputs.blob`). The cause of a 1.1 MB fixture (an offline sweep, 300k) was a brief
that asked for every recorded observation pair to be copied into a data file under the 1 MiB quota; with this
switch the export also writes every response over :data:`RESPONSE_BLOB_BYTES` as a blob (``response_blobs`` in
each exported episode), the gate mounts every blob id a test-side file names at ``/inputs/blobs``, and the
brief says to reference rather than copy. Truncated recordings carry the recorder's marker (and the export
lists them in ``truncated``); a test asserting on text at the cut is refused (:func:`.qa_static.asserts_on_cut`).

Every stage-5 run is confined (:func:`.sandbox_run.run_confined`, bubblewrap with a cleared environment) with
the tree read-only at ``/memory``, the library test kit read-only at ``/inputs`` (:mod:`.testkit`: ``memlab``
with no git and no evidence store, the pin plugin, the referenced blobs), the drawn inputs read-only at
``/qa``, and only a fresh output directory writable. No credential is in any argument, environment entry or
file. All dynamic checks share one time budget (:data:`BUDGET_S`): an exhausted budget stops the checks and
refuses the pass as unjudged (``[qa:budget]``), never a pass and never an exception. Reasons and notes carry
item ids, test paths, counts, operator kinds, line numbers, byte sizes and builtin exception names, never a
recorded value. All reasons are filed under G3 with a ``[qa:<check>]`` tag.

Rejected alternatives: rewriting the consolidator's fixtures to the drawn inputs (expected values would no
longer match, so only property tests survive); a pytest plugin wrapping library functions to watch their
arguments (it changes the code under test); ``sitecustomize`` for the pins (shadowed by any in the venv);
mutating by text substitution (cannot be kept syntactically valid). Known limits: an input's "shape" is its
field-name set, so a refusal of a same-named but differently typed value counts as a false refusal; the stand-in
check resolves only literal data flow (an environment built behind a helper's parameter or an attribute is
``unknown``, never refused); the truncation check sees only ``assert`` statements of the test source (an
expected value kept in a data file is not seen); module code under test shares the probe's process (the
consolidator is assumed careless, not malicious).
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import hashlib
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from . import qa_static, testkit
from .episodes import Action
from .held_out import (
    _blob_sha,
    _form,
    _key,
    _read_results,
)
from .manifest import INPUT_KINDS, TESTKIT, TESTS_DIR, item_path
from .sandbox_run import SandboxResult, run_confined

SAMPLE_K = 8  # drawn inputs per function
MAX_MUTANTS = 8  # mutants per function
RUN_S = 120.0  # one stage-5 pytest run or probe
MUTANT_S = 60.0  # one mutant's test run
BUDGET_S = 900.0  # every dynamic stage-5 check of one gate run together
PROBE_CASE_S = 2  # one call in the probe
FIXTURE_MAX_BYTES = 64 * 1024
MIN_KILL = Decimal("0.5")
# library mode (from the test-quality study): the mutation kill share is gated per changed function at
# MUTATION_MIN (0.7 after the first replay). Guard coverage, the kill rate of dropped-guard (``drop_raise``) mutants,
# is measured and recorded only; GUARD_MIN_KILL is its planned gate. MAX_GUARD_MUTANTS bounds the
# extra dropped-guard mutants run to measure it.
MUTATION_MIN = Decimal("0.6")
MAX_GUARD_MUTANTS = 8
# Under the fixture-size switch, an exported response at least 1/RESPONSE_BLOB_SHARE of the fixture bound also
# becomes a blob (1024 bytes at the default 64 KiB bound): a test file then holds at least 64 inline copies of
# responses under the threshold, and the brief's "handful" (8 or so) takes at most 1/8 of the bound. Dialogue
# observations are capped at 4000 characters by the recorder, so a capped Crafter screen (about 4 KB, which the
# old fixed 4096-byte threshold just missed) is always a blob; a blob reference costs 66 bytes of JSON, so the
# 1.1 MB Crafter fixture, if it held ~275 screens of ~4 KB (inferred from its size), would take ~18 KB as
# references.
RESPONSE_BLOB_SHARE = 64
RESPONSE_BLOB_BYTES = FIXTURE_MAX_BYTES // RESPONSE_BLOB_SHARE
MAX_INPUT_FILE_BYTES = testkit.MAX_INPUT_FILE_BYTES
PIN_MODULE = testkit.PIN_MODULE
PINNED_ENV = {"PYTHONHASHSEED": "0", "TZ": "UTC"}
# TZ per pin variant: UTC, and UTC+05:45 as a POSIX string (no tz database needed in the box)
PIN_TZ = ("UTC", "<+0545>-05:45")
PIN_VARIANT_ENV = "MEMORY_PIN"  # :data:`.pin.VARIANT_ENV`
_NAMED = 5


# --- configuration -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QAConfig:
    """Which stage-5 checks run, and their bounds. The default runs none of them."""

    fixtures: str = ""  # "", "on" or "strict"
    mutation: bool = False
    min_kill: Decimal = MIN_KILL
    determinism: bool = False
    replay: bool = False
    fixture_size: bool = False
    sample_k: int = SAMPLE_K
    max_mutants: int = MAX_MUTANTS
    run_s: float = RUN_S
    mutant_s: float = MUTANT_S
    budget_s: float = BUDGET_S
    fixture_max_bytes: int = FIXTURE_MAX_BYTES
    library_mode: bool = (
        False  # library mode: plain recorded-inputs files, recorded measures, guard coverage
    )
    guard_mutants: int = MAX_GUARD_MUTANTS
    clock: Callable[[], float] = field(default=time.monotonic, compare=False)
    runner: Callable[..., SandboxResult] = field(default=run_confined, compare=False)

    @property
    def on(self) -> bool:
        return bool(
            self.fixtures
            or self.mutation
            or self.determinism
            or self.replay
            or self.fixture_size,
        )

    @property
    def draws(self) -> bool:
        """Whether inputs are drawn and probed (the fixture check, or the mutants' equivalence)."""
        return bool(self.fixtures or self.mutation)

    @property
    def strict(self) -> bool:
        """``fixtures="strict"``: what the checks could not judge refuses the pass."""
        return self.fixtures == "strict"

    @property
    def response_blob_bytes(self) -> int:
        """The size from which the writer's export writes a response as a blob (relative to the fixture bound)."""
        return max(1, self.fixture_max_bytes // RESPONSE_BLOB_SHARE)


def seed_of(candidate: str) -> bytes:
    """The stage-5 seed: SHA-256 of the candidate commit (reproducible, unknown before the commit)."""
    return hashlib.sha256(f"memory-qa\0{candidate}".encode()).digest()


def pytest_env(cfg: QAConfig, variant: int = 0) -> dict[str, str]:
    """The environment of every gate pytest run while the kit is mounted (a switch on, or the tests use it).

    Under determinism the pins of *variant* (:mod:`.pin`): 0 for every run but the determinism rerun, which
    uses 1 (another epoch, clock step, random seed, hash seed and time zone).
    """
    env = {
        "PYTHONPATH": "/memory:/inputs",
        "PYTEST_ADDOPTS": "-c /dev/null --import-mode=importlib",
    }
    if cfg.determinism:
        env["PYTEST_ADDOPTS"] += f" -p {PIN_MODULE}"
        env.update(PINNED_ENV)
        env["PYTHONHASHSEED"] = str(variant)
        env["TZ"] = PIN_TZ[variant]
        env[PIN_VARIANT_ENV] = str(variant)
    return env


def library_config(base: QAConfig) -> QAConfig:
    """The stage-5 checks the library mode gates: drawn inputs ``strict``, mutation at MUTATION_MIN."""
    return dataclasses.replace(
        base,
        fixtures="strict",
        mutation=True,
        min_kill=MUTATION_MIN,
        library_mode=True,
    )


@dataclass
class QAEnv:
    """The read-only directory mounted at ``/inputs`` in the gate's runs and their environment."""

    inputs: Path
    env: dict[str, str]


def stage_inputs(
    dest: Path,
    cfg: QAConfig,
    has_blob: Callable[[str], bool],
    read_blob: Callable[[str], bytes],
    blob_size: Callable[[str], int],
    refs: list[str],
) -> tuple[QAEnv, list[str]]:
    """The library test kit (:mod:`.testkit`: ``memlab``, the pin plugin, the referenced blobs) under *dest*."""
    notes = testkit.stage(
        dest,
        refs,
        has_blob=has_blob,
        read_blob=read_blob,
        blob_size=blob_size,
    )
    return QAEnv(dest, pytest_env(cfg)), notes


# --- drawn inputs --------------------------------------------------------------------------------------------


@dataclass
class Row:
    """One input the probe (and the sample run) feeds a function."""

    role: str  # "cover", "sample" or "negative" (a structurally broken cover; mutant equivalence only)
    action: Action
    context: list[Action]
    covered_shape: bool
    form: str
    file: tuple[str, bytes] | None = None
    text: str | None = None
    value: Any = (
        None  # the input itself, when it is not built from the action (a negative)
    )
    has_value: bool = False


def _content(a: Action, form: str | None) -> str:
    """What makes two recorded inputs the same input (as G2's constancy counts observations)."""
    kind = getattr(a, "kind", "tool")
    if kind == "worktree":
        return "blob:" + str(_blob_sha(a))
    if kind == "tool" and _form(a, form) == "env":
        return "call:" + _key([a.method, a.kwargs, a.response])
    return "response:" + _key(a.response)


# --- the probe -----------------------------------------------------------------------------------------------

_OUTCOMES = frozenset(
    {
        "handled",
        "refused",
        "miss",
        "recorded_error",
        "error",
        "timeout",
        "unbuilt",
        "unbound",
    },
)


def builtin_error(name: object) -> str | None:
    """*name* if it names a builtin exception class (so no text the function controls reaches a reason)."""
    if isinstance(name, str) and name.isidentifier() and len(name) <= 64:
        cls = getattr(builtins, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            return name
    return None


def probe(
    item: str,
    tree: Path,
    qa_env: QAEnv,
    samples: Path,
    *,
    python: Path,
    work: Path,
    timeout_s: float,
    runner: Callable[..., SandboxResult] = run_confined,
) -> dict[int, dict] | None:
    """The probe's result row per drawn-input id, or None when the function could not be loaded."""
    out = work / "out"
    out.mkdir(parents=True)
    runner(
        [str(python), "-s", "/qa/probe.py", item, str(PROBE_CASE_S)],
        ro={tree: "/memory", qa_env.inputs: "/inputs", samples: "/qa"},
        rw={out: "/out"},
        cwd="/memory",
        timeout_s=timeout_s,
        env=dict(PINNED_ENV),
    )
    rows = _read_results(out / "results.jsonl")
    if any("fatal" in r for r in rows):
        return None
    return {
        r["id"]: r
        for r in rows
        if isinstance(r.get("id"), int) and r.get("outcome") in _OUTCOMES
    }


def _signature(r: dict) -> tuple:
    return (
        r.get("outcome"),
        r.get("digest") if r.get("outcome") == "handled" else None,
        builtin_error(r.get("raised")),
    )


# --- library mode: judged from the tests' recorded calls ----------------------------------------------------

# --- the checks ----------------------------------------------------------------------------------------------


def _names(items: list[str]) -> str:
    more = f" and {len(items) - _NAMED} more" if len(items) > _NAMED else ""
    return ", ".join(items[:_NAMED]) + more


#: Stage-5 findings about how much an item's evidence shows (its drawn recorded inputs, its mutation score); in
#: library mode they set the item's evidence level instead of refusing it. The others (the test kit, determinism,
#: replay fidelity, truncation, fixture size, the budget) still refuse.
EVIDENCE_TAGS = frozenset({"fixtures", "mutation"})


class QAChecks:
    """The stage-5 checks of one gate run (*run* is the gate's :class:`~.gate._Run`)."""

    def __init__(self, gate: Any, run: Any) -> None:
        self.gate, self.run, self.cfg = gate, run, gate.qa
        # library mode: the pass's seed when the gate was given one (every round and the merge draw alike)
        self.seed = getattr(run, "seed", None) or seed_of(run.candidate)
        self.started: float | None = None

    # -- shared -------------------------------------------------------------------------------------------
    def _fail(
        self,
        tag: str,
        reason: str,
        item: str | list[str] | None = None,
    ) -> None:
        """A G3 refusal; *item* names the manifest item(s) it belongs to (None: the whole pass), for the
        gate's per-item admission (:meth:`.gate.Gate.merge`). The drawn recorded inputs and the mutation score
        measure how much an item's evidence shows: in library mode they set its evidence level instead
        (:meth:`.gate._Run.cap`)."""
        if tag in EVIDENCE_TAGS and item is not None:
            found = getattr(self.run, "cap", None)
            if callable(found):
                found("G3", f"[qa:{tag}] {reason}", item)
                return
        self.run.fail("G3", f"[qa:{tag}] {reason}", item)

    def _owners(self, path: str) -> list[str]:
        """The manifest items listing test file *path* ([]: none, so a refusal of it is pass-wide)."""
        return [it.item for it in self.run.man.items if path in it.tests]

    def _tests(self, it: Any) -> list[str]:
        changed = set(self.run.changed)
        return sorted(t for t in it.tests if t in changed and t in self.run.c_files)

    def _form(self, it: Any) -> str | None:
        if self.cfg.library_mode:
            return None  # a library function takes any arguments; a drawn input's form is its fixture's
        declared = self.gate._doc_inputs(self.run).get(it.item, "")
        return it.input or (declared if declared in INPUT_KINDS else None)

    def _covers(
        self,
        item: str,
        lookup: Callable[[str, int], Action | None],
    ) -> list[tuple[str, int, Action]]:
        out = []
        for i, eid, idx in sorted(self.run.covers):
            if i == item:
                a = lookup(eid, idx)
                if a is not None:
                    out.append((eid, idx, a))
        return out

    def _kit(self) -> bytes | None:
        p = self.run.c_tree / TESTKIT
        return p.read_bytes() if TESTKIT in self.run.c_files else None

    # -- before G3: the /inputs mount ---------------------------------------------------------------------
    def prepare(self) -> bool:
        """Mount the library test kit (:mod:`.testkit`) for every pytest run of this check when a switch is
        on or a test-side file of the parent or the candidate uses it; whether it is mounted.

        A library whose tests never use the kit, checked with every switch off, gets exactly the screen
        build's calls (nothing is mounted). One that does is checked the same way under every switch setting.
        """
        run = self.run
        sources, refs = self._scan()
        blobs = self.gate.blobs
        if not (self.cfg.on or testkit.uses_kit(sources, blobs.has, refs)):
            return False
        run.qa_env, notes = stage_inputs(
            run.tmp / "qa-inputs",
            self.cfg,
            blobs.has,
            blobs.get,
            blobs.size,
            refs,
        )
        for n in notes:
            run.note(n)
        return True

    def _scan(self) -> tuple[list[bytes], list[str]]:
        """The test-side files of the parent and the candidate (bounded), and the blob ids they name."""
        run = self.run
        sources: list[bytes] = []
        for tree, files in ((run.p_tree, run.p_files), (run.c_tree, run.c_files)):
            if files:
                sources += testkit.read_sources(tree, files).values()
        return sources, qa_static.blob_refs(sources)

    def uses_kit(self) -> bool:
        """Whether the parent's or the candidate's tests use the test kit (:func:`.testkit.uses_kit`)."""
        sources, refs = self._scan()
        return testkit.uses_kit(sources, self.gate.blobs.has, refs)

    # -- the kit stays a test-side tool ---------------------------------------------------------------------
    def kit(self) -> None:
        """Refuse new or changed library code that uses the kit, test imports the kit does not provide, test
        imports of a module named only at run time, and tests that name the ``/inputs`` path (each would work
        in one place a library's tests run, not all).

        Run wherever the kit is mounted (a switch on, or the library's tests use it); static, no test runs.
        """
        run = self.run
        for p in sorted(set(run.changed) & set(run.c_files)):
            if not p.endswith(".py"):
                continue
            source = (run.c_tree / p).read_bytes()
            if not testkit.is_test_side(p):
                for line in testkit.library_uses(source):
                    self._fail(
                        "kit",
                        f"{p} line {line} uses the test kit (memlab, the pin plugin or /inputs) outside a "
                        "test; the working model imports the library without it",
                    )
                continue
            for line, what in testkit.unresolved(source):
                what = what if len(what) <= 80 else what[:77] + "..."
                self._fail(
                    "kit",
                    f"{p} line {line} imports {what}, which the test kit (version "
                    f"{testkit.KIT_VERSION}) does not provide",
                    self._owners(p),
                )
            for line in testkit.dynamic_unnamed(source):
                self._fail(
                    "kit",
                    f"{p} line {line} imports a module whose name is not a constant string "
                    "(import_module, importorskip or __import__); name it with a constant, so the test "
                    "kit is staged wherever the library's tests run",
                    self._owners(p),
                )
            for line in testkit.names_inputs_path(source):
                self._fail(
                    "kit",
                    f"{p} line {line} names a path under /inputs; read recorded blobs through "
                    "memlab.inputs.blob(<id>), which works wherever the library's tests run",
                    self._owners(p),
                )

    # -- static: fixture size, replay fidelity, cuts ------------------------------------------------------
    def static(self, lookup: Callable[[str, int], Action | None]) -> None:
        run, cfg = self.run, self.cfg
        if cfg.fixture_size:
            support = set(run.man.support)
            for p in sorted(set(run.changed) & set(run.c_files)):
                if not (TESTS_DIR.match(p) or p == TESTKIT or p in support):
                    continue
                size = (run.c_tree / p).stat().st_size
                if size > cfg.fixture_max_bytes:
                    self._fail(
                        "fixture-size",
                        f"{p} has {size} bytes, over the {cfg.fixture_max_bytes}-byte bound for test "
                        "files: reference recorded payloads by blob id (memlab.inputs.blob) instead of "
                        "copying them",
                        self._owners(p),
                    )
        items = [it for it in run.man.items if it.kind == "env_function"]
        if cfg.replay:
            env_items: dict[str, str] = {}
            for it in items:
                covers = self._covers(it.item, lookup)
                tool_calls = any(
                    getattr(a, "kind", "tool") == "tool" and a.status == "ok"
                    for _, _, a in covers
                )
                form = self._form(it) or ("env" if tool_calls else None)
                first = _first_param(run.c_tree, it.item)
                if form == "env" and tool_calls and first is not None:
                    env_items[it.item] = first
            if env_items:
                kit = self._kit()
                for t in sorted({t for it in items for t in self._tests(it)}):
                    for line, item in qa_static.stand_ins(
                        (run.c_tree / t).read_bytes(),
                        kit,
                        env_items,
                    ):
                        self._fail(
                            "replay",
                            f"{t} line {line} passes {item} an environment the tests make (a class, "
                            "function or literal of their own); test it through memlab.replay.env_from "
                            "over its recorded calls",
                            item,
                        )
        if cfg.fixture_size:
            for it in items:
                actions = [a for _, _, a in self._covers(it.item, lookup)]
                if not any(qa_static.truncated(a) for a in actions):
                    continue
                for t in self._tests(it):
                    for line in qa_static.asserts_on_cut(
                        (run.c_tree / t).read_bytes(),
                        actions,
                    ):
                        self._fail(
                            "truncation",
                            f"{t} line {line} asserts on text at the recorder's cut of a truncated "
                            "recording; assert on what the environment returned, not on the cut",
                            it.item,
                        )


def _first_param(tree: Path, item: str) -> str | None:
    """The name of *item*'s first parameter in the candidate tree, or None."""
    path = tree / item_path(item)
    try:
        module = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError, RecursionError):
        return None
    name = item.split(":", 1)[1]
    fn = None
    for node in module.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            fn = node
    if fn is None:
        return None
    params = fn.args.posonlyargs + fn.args.args
    return params[0].arg if params else None


# --- the consolidator's brief --------------------------------------------------------------------------------

# Sentences of the base brief that the stage-5 switches change: (switch test, old text, new text). A test keeps
# every old text present in :data:`.writer_pass.WRITER_SYSTEM`, so the brief cannot drift from these rewrites.
REWRITES: tuple[tuple[str, str, str], ...] = (
    (
        "on",
        'env={"PYTHONPATH": "/memory", "PYTHONDONTWRITEBYTECODE": "1"})',
        'env={"PYTHONPATH": "/memory:/inputs", "PYTHONDONTWRITEBYTECODE": "1"})',
    ),
    # the harness provides the replay (memlab.replay.env_from) wherever the kit is mounted, so the writer no longer
    # writes the root unify_memory_testkit.py to fake the environment (one already in a library stays accepted)
    (
        "on",
        "use the recorded responses; /memory/unify_memory_testkit.py, the only helper\nmodule allowed at the root, "
        "may build the fake environment",
        "serve the recorded calls with memlab.replay.env_from(<recorded actions>), the harness's exact-call\n"
        "replay; never a fake environment of your own",
    ),
    ("on", '"support":["unify_memory_testkit.py"]', '"support":[]'),
    (
        "fixture_size",
        "or every\nrecorded (action, next observation) pair in its scope",
        "or a handful of\nrecorded (action, next observation) pairs in its scope",
    ),
    (
        "on",
        "The gate mounts only /memory, so copy the fixtures a test\nneeds into data files under "
        "env/<channel>/tests/ (never memlab)",
        "The gate mounts /memory, and memlab and the recorded blobs your tests name at /inputs; copy the small "
        "fixtures a test\nneeds into data files under env/<channel>/tests/",
    ),
)


def brief(cfg: QAConfig) -> str:
    """The paragraph the writer's brief gains for the stage-5 checks that are on ("" when none is)."""
    if not cfg.on:
        return ""
    lines = [
        "The gate also checks test quality. Run tests with PYTHONPATH=/memory:/inputs (memlab importable), as "
        "the gate then does; tests may import memlab.replay and memlab.inputs, which the gate provides (the "
        "same test kit wherever the library's tests run). Read a recorded blob with memlab.inputs.blob(<id>), "
        "never by its /inputs path. Library code outside tests never imports memlab: the working model "
        "imports the library without it. env = memlab.replay.env_from(<recorded actions>) answers only the "
        "identical recorded call, repeated calls in recorded order (a call with no recording raises "
        "ReplayMiss, a recorded failure re-raises RecordedError), and env.issued() lists the calls it served: "
        "a test asserts env.issued() == memlab.replay.calls(<the function's own recorded actions>) and not "
        'env.misses. Filtering on effect="read" or "write" works only where the recordings carry an effect; '
        "it raises UnknownEffect where one is unknown, as for every dialogue action.",
    ]
    if cfg.fixtures:
        strict = (
            " Every drawn input must be read by the function's own tests."
            if cfg.fixtures == "strict"
            else ""
        )
        lines.append(
            "- Drawn inputs: after you finish, the gate draws a seeded random sample of recorded inputs of each "
            "new or changed function's family (beyond its covers) and calls the function on each in its declared "
            "form. It must return, or raise MemoryInputError only for inputs whose fields differ from its covers'; "
            "any other exception, or refusing an input shaped like its covers, refuses the pass. Parametrise "
            'tests over memlab.inputs.inputs("env/<channel>:<function>", fixtures) (fixtures from '
            "memlab.inputs.from_action(<exported action>, form=...) or from_blob(<blob id>, <file name>, "
            "form=...)); the gate appends its drawn inputs, x.value() gives the input in the function's form and "
            "x.response / x.kwargs the recorded call. Assert what holds for every recorded input (the result "
            "agrees with x.response), not copied values." + strict,
        )
    if cfg.mutation:
        lines.append(
            "- Mutants: the gate makes small changes to each new or changed function (a comparison flipped, a "
            "condition negated, a raise dropped, a constant off by one, and/or swapped, a return replaced by "
            f"None); its tests must fail on at least {cfg.min_kill} of the changes that alter its outputs on "
            "recorded inputs or on copies of its covers with a field dropped, retyped, emptied or added. Test "
            "the values it returns and that it refuses inputs of another shape.",
        )
    if cfg.determinism:
        lines.append(
            "- Determinism: tests run twice with the clock, the hash seed and random pinned to different "
            "values on each run; a test whose outcome differs between the runs is refused (never assert the "
            "current time, a random draw or a hash order).",
        )
    if cfg.replay:
        lines.append(
            "- Replay: test a function that takes the environment through memlab.replay.env_from(<recorded "
            "actions>) (exact-call replay); a class, lambda, function or literal of your own standing in for "
            "the environment is refused.",
        )
    if cfg.fixture_size:
        lines.append(
            f"- Fixture size: a test or data file over {cfg.fixture_max_bytes} bytes is refused. Reference "
            "recorded payloads by id instead of copying them: memlab.inputs.blob(<id>) reads a recorded file "
            "(blob_before/blob_after) or a recorded response (the episode's response_blobs[i], for responses of "
            f"at least {cfg.response_blob_bytes} bytes). Do not copy "
            "every observation; test a handful and let the gate draw more. A truncated recording is marked "
            "(the episode's truncated[i]); never assert on text at the cut.",
        )
    return "\n".join(lines)


def system(base: str, cfg: QAConfig) -> str:
    """The writer's system prompt with the stage-5 rewrites and paragraph (*base* unchanged when no switch is on)."""
    if not cfg.on:
        return base
    text = base
    for when, old, new in REWRITES:
        if (cfg.on if when == "on" else bool(getattr(cfg, when))) and old in text:
            text = text.replace(old, new)
    return text + "\n" + brief(cfg) + "\n"
