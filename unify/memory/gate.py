"""The deterministic gate that decides every merge into memory ``main``.

Nothing a consolidation pass writes reaches the working model unless :meth:`Gate.merge` passes it. Both
revisions are resolved to commit SHAs once, and the trees inspected are exactly the committed blobs
(``ls-tree`` + ``cat-file``; no ``git archive``, so no ``.gitattributes`` can hide a file).

Threat model
------------------------------------
The gate guards against a careless or mistaken consolidator, not a malicious one; malice is contained by
the runtime confinement of every run (:mod:`.sandbox_run`). Its rules are structural: they pin what a pass
may touch and keep every old test passing against the new library. Documented limits:

* **Test adequacy** is not judged: a weak test that is red on the parent and green on the candidate passes
  (unless a stage-5 switch of :mod:`.qa` is on: drawn inputs, mutants, determinism, replay, fixture size).
* **Effect: lines** are checked for presence, and for ``write`` only where the recording shows it: a
  function taking the environment that covers a tool call recorded with effect ``write`` must declare
  ``Effect: write``. A call the environment did not declare an effect for (``unknown``) proves nothing,
  and a ``read`` or ``unknown`` declaration on a function covering no recorded write is not checked.
  These docstring checks apply to environment functions only. A library function (``memory/<package>``)
  declares no effect: its effect is derived from what it reproduces and recorded in its item record
  (:func:`.library_checks.derive_effect`), so nothing is asked of the writer.
* **Covers** prove that a real recorded observation of the right kind on the item's channel exists (a
  successful tool call, a shell output tail, a file blob, a dialogue observation; :mod:`.admission`), not
  that the item agrees with it: the item's own tests over recorded fixtures check that.
* **pass_id** uniqueness is checked before the gate runs, not atomically with ``record_pass``; two
  concurrent merges with one id race (passes run one at a time).

The checks:

* **G1 provenance and scope.** The manifest is well formed and its ids strict (:mod:`.manifest`); the
  candidate descends from the parent; every file has mode 100644 and sits where the layout admits it;
  every changed file is declared by an item, a listed test, a ``support`` helper, a ``deleted``/
  ``unlisted`` item, a ``skeleton`` channel (its ``__init__.py`` and its ``NOTES.md``) or a
  ``deleted_tests`` entry; every added, changed or removed item (by id and body) is in ``items``,
  ``deleted`` or ``unlisted``; deleted items are gone; unlisted ones are present, unlisted and otherwise
  unchanged; a channel module's code outside its public functions (docstring, imports, helpers,
  statements) changes only under ``skeleton``, which then lists every public function of that module in
  ``items``; a notes file's preamble also changes only under ``skeleton``; a retired test file imports
  only deleted items; every item names source episodes that exist; every new or changed environment
  function declares its ``input`` form, and every declared form equals the function's docstring
  ``Input:`` line. No commit writes a path the harness reserves for its generated catalogue
  (``README.md``, ``memory.py``, ``.memory/``; :func:`.catalogue.reserved`), in every mode. With
  ``docstring_standard`` (library mode; ``UNIFY_MEMORY_DOCSTRINGS=on``), every new or changed environment
  function's docstring meets the lean standard (:mod:`.docstrings`: summary, ``Args:`` naming each
  parameter and the first one's input form, ``Returns:``, ``Raises:`` naming ``MemoryInputError``, an
  ``Example:`` with a ``>>>`` example), and each ``raise MemoryInputError(...)`` in it carries a message of
  at least :data:`.docstrings.MIN_REFUSAL_CHARS` characters; each missing part is its own reason.
* **G2 evidence.** An ``env_function`` covers recorded actions on its own channel, each a real recorded
  observation of its kind (:func:`.admission.cover_problem`): a tool call with status ``ok`` and a
  response, a shell command with an output tail, a file read or write with a blob in the blob store, or a
  dialogue action with an observation; or a recorded environment rejection (status ``error`` with its
  error), never alone. Scope is shape, not observed values: run confined on each covered
  input, passed in its declared ``input`` form (the docstring's ``Input:`` line for an unchanged function
  without one; else the kind's convention), with one value at a time replaced by an unseen value of its
  type (a field with one value across at least 3 recorded observations of its family and field-name set,
  from at least 2 episodes of the whole evidence store, is identity or format and is kept, with a note)
  (:mod:`.held_out`); a declared form a covered input cannot be given in fails; the item
  must not raise its ``MemoryInputError`` before any environment call unless one of its covers is a
  recorded rejection of the same family that varied that field (each such allowance is noted); the reason
  names the field only. A field declared with a semantic type (``field_types``) is checked two-sided:
  unseen in-domain values must be accepted and an out-of-domain value refused. Other exceptions, unused keywords, timeouts and refused baselines are notes.
  Shell covers are skipped with a note. A ``workflow`` is promotable under the job-level rule over its recorded and source episodes.
  An added or changed environment function that replaces a value it computed from its input under a condition
  (:mod:`.analysis.overrides`; an expense case paid an over-cap claim 0 instead of the cap) encodes a
  decision rule its recordings cannot mark wrong: its validated covers must come from at least
  :data:`RULE_EPISODES` episodes, else it is refused with its first such line (a function the analysis cannot
  bound is noted, not refused).
* **Evidence levels (library mode).** A finding about how much an item's evidence shows does not refuse it: it
  sets the item's evidence level (:mod:`.item_states`) through :meth:`_Run.cap`, and the item lands at
  ``unchecked`` with the findings as its ``level_reasons``; an item with none lands at ``reproduces-its-case``.
  Those findings are G2's grounding when some call that returns still uses a value from a recorded episode, the
  episodes rule, the check with unseen values (special-cased, refused or unjudged), the literal and rule checks,
  a typed cover that does not validate, the docstring's ``Example:`` line, the example-shape check, G3's exact
  and oracle checks, and stage 5's drawn recorded inputs and mutation score. Every other check refuses as
  before: layout, provenance (every fixture re-derives from the recording, parsed fixtures by their stored
  reader), the manifest, safety (G6), tests that are not plain or not green, regressions, the behaviour check,
  an edited function without a red test, deleted or weakened tests without a reason, an example call that
  raises, an incomplete call record, and a function no call of which returns on a value from a recorded
  episode in any form (it is not landed).
* **G3 red→green and regression.** Every added or edited function has a new or changed listed test that passes on
  the candidate and is red on the parent: it fails on the parent tree with the changed tests, their fixtures and
  the ``support`` helpers copied in (a run there that times out counts as red once the candidate's run is green).
  Other new or changed tests need only pass on the candidate; each is noted as a keeping test (it passes on the
  parent, repairs a test the parent had red, or cannot be compared there). A pass that changes only tests or
  only notes is judged by the checks that apply to what it changes. A listed function whose body is unchanged
  (listed to change its tests) skips the checks that judge its own code (its source episodes, its stored typed
  covers); the checks that judge its tests still apply.

  **Behaviour check: any change in what a stored function returns on recorded inputs needs a
  failing-then-passing test.** In every pass, for every channel it changes (any file under
  ``env/<channel>/`` added, edited or deleted: a public function, a private helper, a constant, a
  decorator, an alias or other module-level code), every public function of the parent's module (a
  ``def``, or a name bound by assignment or import that has recorded covers) whose name the candidate's
  module still binds (a ``def``, an assignment such as ``A = B`` or an import) runs confined on the
  parent's and the candidate's version through G2's runner and replay. It runs on its recorded covers
  (the evidence store's and this manifest's; beyond :data:`.held_out.MAX_OUTPUT_COVERS` a hash-seeded
  sample) and on the recorded actions of its channel in the pass's own episodes (at most
  :data:`MAX_EPISODE_INPUTS`, chosen by hash), each in the parent's declared input form. Both must
  return the same value (canonical JSON), refuse alike, raise the same exception class and issue the
  same environment calls. A function whose results differ on any of them, or cannot be compared (not
  JSON, timed out, not run in time), or whose input form changed, must be listed in ``items`` with a
  test of this pass that is red on the parent's library and green on the candidate's; otherwise the
  pass is refused, naming the function and the number of differing inputs (ids, never values). A cover
  that cannot be read or given is skipped with a note; an action that cannot be given in the declared
  form is not compared; a deleted name is G5's. New functions have no parent version and are exempt
  (G2 and the red run hold them). This replaces task-10's rule that a helper or constant change needs no
  test: an unchanged body whose results change is a change. The check is bounded per pass at
  :data:`BEHAVIOUR_BUDGET_S` seconds and :data:`BEHAVIOUR_MAX_CASES` compared inputs; exhausting either
  refuses the pass ("behaviour check not completed"), never merges it unchecked.

  A *clean-up pass* (it adds no item and the library shrinks: the channel modules' function
  definitions or the AST nodes of those definitions fall, and neither they nor the nodes of other module
  code grow; comments and docstrings never count) may exempt an edited function from the red run only
  when the change keeps its behaviour: a parent test file that passed on the parent, calls it (an import
  alone does not count) and still passes in both runs below; and the behaviour check above shows it
  doing the same on every recorded cover (none unread or ungiven, at most the cap; a recorded rejection
  that gives no input aside) and on the pass's recorded episode actions. Without it the function needs a
  red test as in any pass (a repair changes behaviour, so it always does); new tests of a clean-up pass
  need only be green. Then, per channel, against the parent's suite (the baseline):

  - a readable baseline protects every test it passed (except in retired files). Each must still pass
    twice: in the candidate's own suite and in an isolated regression run of the parent's own tests and
    test kit against the candidate's channel modules and notes (nothing else from the candidate). The
    candidate's suite must be readable and fail only tests the baseline already failed (no new failures);
  - an unreadable or timed-out baseline protects nothing, and its regression run is skipped (a ``note:``
    says so). If the pass touches the channel (any path under ``env/<channel>/``), the candidate's suite
    must be readable and free of failures: that is the repair path;
  - failures already on the parent that the candidate keeps, and an unreadable suite in a channel the pass
    does not touch, are reported under ``note: pre-existing`` and do not fail the gate.

  ``note:`` lines follow the failure reasons in :attr:`GateResult.reasons`. Every run is confined,
  read-only at ``/memory`` with ``PYTHONPATH=/memory``, ``--import-mode=importlib`` and ``-c /dev/null`` (no
  configuration discovery). A report :class:`PytestOutcome` marks invalid never counts as green, and
  counts as red only for pytest's collection-error status (2) with the erroring module in the report, or
  for a timed-out red run as above. Each run is bounded at 300 s. With ``docstring_standard``, every
  ``>>>`` example of each new or changed environment function runs as a doctest in one more confined
  pytest run of the same kind, on the candidate tree (the module's names in scope, ``/memory`` the
  working directory, read-only); a failing or unrun example refuses that function, and so does an example
  whose channel fixtures (the files its docstring names) include none shaped like an input the function
  was admitted on (its validated covers' shapes: the same shape, or a structural match of keyed shapes; a
  function without recorded shapes is not checked).
* **G4 size budget.** By default (``UNIFY_MEMORY_SOFT_BUDGET=off``, as in v2) :func:`.index.build_index`
  of the candidate fits ``budget_tokens``, or the candidate is refused. With ``soft_budget`` the library's
  surface is measured (:func:`.catalogue.catalogue_tokens`, the generated README and the channel lines,
  under ``surfacing="catalogue"``; the index otherwise); over ``budget_tokens`` a ``note: G4 hygiene due``
  records that a hygiene pass is due and growth is never refused; only a surface that cannot be built
  fails.
* **G5 description length.** If ``env/*/__init__.py`` grew, a cover G2 validated is new to the evidence
  and is a successful observation (a recorded rejection cover, status ``error``, is not new coverage), or
  the library shrinks by the clean-up measure above. Every recorded input a deleted item covered (except
  recorded rejections) is covered by a remaining item: one of this pass's validated covers, or a recorded
  cover of a kept item whose channel the pass leaves unchanged or that G3's behaviour check showed doing
  the same. A function changed under a red test holds only the covers this manifest lists for it.
* **G6 safety.** No links, executables, submodules, or git, pytest or interpreter configuration files; no
  bytecode or native code, ``__pycache__`` path, ``sitecustomize``/``usercustomize`` under any suffix or root
  directory named like a standard-library or pytest module; and no other root entry than ``env/``,
  ``workflows/`` and the test kit (G1) (:func:`.manifest.unsafe_path`, all refused before anything is
  extracted or run); no key-shaped string in a changed file or the manifest; every public function of a
  changed module declares ``Effect:`` and is defined once; every module parses. **Structural Effect:** an
  environment function in ``items`` that covers a tool call whose recorded effect is ``write``
  (:attr:`.episodes.Action.effect`, the effect the environment declared for the method; never inferred from
  names) declares ``Effect: write``, unless its declared input is a recorded value (``path``, ``text``,
  ``bytes``, ``observation``: it parses what a call returned and makes no call). A library function's effect
  is derived, not declared (:func:`.library_checks.derive_effect`).

Per-item admission (:meth:`Gate.merge`): a failure that belongs to manifest items refuses those items
(and what depends on them, :mod:`.reduction`) instead of the pass, when no failure is pass-wide; the
reduced candidate passes the whole gate again before it lands. The item-scoped checks are G1's per-item
checks (and the docstring standard's), G2 (covers, held-out values, the rule check, workflow status),
G3's per-item checks (an item's own tests, its examples and their fixtures, a clean-up edit no old test
holds or that the behaviour check does not show doing the same, and new failures only in its own test
files), the behaviour check's refusal of a function whose channel the pass changes only through one
environment function in ``items`` (it belongs to that item; with a ``skeleton`` change, a deleted or
unlisted item, or more than one such function in the channel it is pass-wide, as are the check's bounds),
the stage-5 checks of one function or of tests only items list (drawn inputs, mutants, replay, truncation,
determinism), and G6's ``Effect:`` checks.

:meth:`Gate.preview` runs the cheap, read-only part (the manifest, G1, G2's covers, G4 to G6) on an
uncommitted tree, for the consolidator's ``check`` tool; it never decides or records a merge.

Under ``surfacing="catalogue"`` (``UNIFY_MEMORY_SURFACING``) a landed merge also freezes the candidate
commit's input-shape snapshot (:mod:`.shape_rows`): per
environment function, the parent's shapes for the same body plus the descriptors of the covers this merge
validated (each covered file's blob, each covered observation), for the export's catalogue
(:mod:`.catalogue`). Computing it never fails the gate. The examples run, like G3's runs, is a drift check
under the threat model above, not a security boundary: module code the runner imports can patch ``doctest``.

Stage-5 test checks (library mode, :mod:`.qa`) extend G3 when a :class:`.qa.QAConfig` switch is on: seeded
random draws of recorded inputs, mutation testing, pinned determinism, replay fidelity and fixture size. Their
reasons are G3 reasons tagged ``[qa:<check>]``; the static ones (fixture size, replay fidelity, cuts of
truncated recordings) also run in :meth:`Gate.preview`. Every gate test run also mounts the library test kit
(:mod:`.testkit`: ``memlab``, the pin plugin, the blobs the tests name) read-only at ``/inputs``
(``PYTHONPATH=/memory:/inputs``) when a switch is on **or** a test-side file of the parent or the candidate
uses the kit, so a stored library is checked the same way under every switch setting; library code that uses
the kit is then refused (``[qa:kit]``). With every switch off (the default) and a library whose tests never
use the kit, the gate is exactly as described above. The dynamic stage-5 checks run last, after G4 to G6, so
they are spent only on a candidate the rest of the gate accepts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import docstrings, layout
from .admission import cover_problem, is_rejection
from .blobs import BlobStore
from .catalogue import body_digest, catalogue_tokens, reserved
from .episodes import Action, env_channel
from .evidence import EvidenceStore
from .gitio import GitError, Repo
from .held_out import (
    MAX_ACTIONS_PER_EPISODE,
    MAX_OUTPUT_COVERS,
    MAX_POOL_ACTIONS,
    MAX_POOL_EPISODES,
    OUTPUTS_BUDGET_S,
    Case,
    _form,
    _unfit_reason,
    output_cases,
    plan,
    pool_actions,
    run_outputs,
    run_plan,
    seen_actions,
)
from . import reduction
from .library_checks import LibraryConfig
from .manifest import (
    MODULE_PATH,
    NOTES_PATH,
    Manifest,
    ManifestError,
    forbidden,
    INPUT_KINDS,
    item_path,
    layout_allowed,
    parse_manifest,
    unsafe_path,
)
from .index import IndexOverBudget, build_index, estimate_tokens
from .memory_helper import compare, file_shape, same_shape, value_shape
from .shape_rows import descriptors, shapes_at, snapshot_rows
from .memory_repo import ItemsReport, items
from .item_states import UNCHECKED as _LOWEST_LEVEL
from .qa import QAChecks, QAConfig, QAEnv
from .redact import KEY_SHAPED
from .sandbox_run import PYTHON, PytestOutcome, run_pytest
from .signals import job_item_status
from .snapshot import (
    calls_item,
    code_size,
    duplicate_public_defs,
    env_references,
    library_references,
    item_bodies,
    listing,
    materialise,
    module_skeleton,
    notes_preamble,
    public_bindings,
    tree_listing,
    without_listed,
)

CHECKS = ("G1", "G2", "G3", "G4", "G5", "G6")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_COLLECTION_ERROR = (
    2  # pytest's "interrupted" status, which a collection error produces
)
_TIMEOUT_S = 300.0
# No configuration discovery and no sys.path insertion of test directories in any confined run.
_PYTEST_ENV = {
    "PYTHONPATH": "/memory",
    "PYTEST_ADDOPTS": "-c /dev/null --import-mode=importlib",
}
_REDACTED = "<redacted:key-shaped>"
# G3's two per-function refusals (an edited function with no new or changed test, or with none red on the parent):
# what would satisfy the rule, and what a test that already passes on the parent shows
RED_TEST_HINT = (
    "A function you edit needs at least one listed test that fails on the parent's version and passes on yours. A "
    "test that already passes on the parent is accepted, but it records existing behaviour and cannot show this "
    "change."
)


# The generated doctest runner of G3's examples run (a path the layout admits for no candidate file).
EXAMPLES_TEST = "_memory_examples/test_examples.py"
# Input-shape descriptors computed per item at a merge (the evidence store keeps at most its own cap).
MAX_SHAPE_COVERS = 32

# An environment function that encodes a rule (G2) needs validated covers from at least this many episodes.
RULE_EPISODES = 2

# A check (:meth:`Gate.preview`) examines at most this many covers, naming at most this many episodes.
PREVIEW_MAX_COVERS = 500
PREVIEW_MAX_EPISODES = 32

# G3's behaviour check, per pass: the seconds its confined runs may take and the recorded inputs it may
# compare (each run on both versions); either exhausted refuses the pass. Each changed channel's functions are
# also run on at most this many recorded actions of the channel in the pass's own episodes (chosen by hash).
BEHAVIOUR_BUDGET_S = 600.0
BEHAVIOUR_MAX_CASES = 2000
MAX_EPISODE_INPUTS = 64


@dataclass
class _Reduced:
    """A refused candidate reduced to its admissible items (:meth:`Gate._reduce`), committed on the parent."""

    sha: str
    manifest: dict
    refused: dict[
        str,
        list[str],
    ]  # item -> value-free codes (checks, or ``dependency``)
    reasons: list[str]  # why each dependent item goes with a refused one


@dataclass
class ParentSnapshot:
    """A parent revision taken once for repeated previews: its sha, its files and their extracted tree."""

    sha: str
    files: dict[str, tuple[str, str]]
    tree: Path


@dataclass
class GateResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    # The checks that refused, in order (structured; ``checks`` also marks unevaluated checks False).
    refused: list[str] = field(default_factory=list)
    # The manifest could not be parsed (refused under G1).
    manifest_invalid: bool = False
    # The commit ``main`` moved to: the candidate, or its reduction to the admitted items (None: nothing landed).
    merged: str | None = None
    # Per-item admission: the manifest items that landed, and each refused item's value-free reason codes
    # (the checks that refused it, ``dependency`` for an item refused with one it depends on, ``pass`` for
    # an item refused only because the whole pass was).
    items_merged: list[str] = field(default_factory=list)
    items_refused: dict[str, list[str]] = field(default_factory=dict)
    # library mode (library_checks): the item verification records and whether CURATE is due (G4)
    verification: dict[str, dict] = field(default_factory=dict)
    curate_due: bool = False
    # library-mode CURATE: the aliases, retirements and dropped aliases of the checked candidate; a landed
    # merge records them (EvidenceStore.record_curations)
    curations: list[dict] = field(default_factory=list)
    # Library mode: each failing run's whole output (sandbox_run.PytestOutcome.full_output: bounded
    # only by the sandbox's marked capture), key-shaped strings redacted, by run label. Never stored in the pass
    # row, whose reasons keep v2's 300-character tail.
    outputs: dict[str, str] = field(default_factory=dict)
    # library layout: one ``{"item", "adopts"}`` row per landed item, the proposals (staged files, helper output,
    # pool entries) its manifest entry says it came from; empty when nothing landed
    provenance: list[dict] = field(default_factory=list)
    # library mode: the fixture provenance entries G1 verified (path -> (sha256, entry)), persisted on merge
    fixtures_verified: dict[str, tuple[str, dict]] = field(default_factory=dict)
    # each refused item's own reason lines, as they stand in ``reasons`` (a failure attributed to items); the
    # writer pass's repair rounds compare them round to round (:func:`.repair.refusals`)
    item_reasons: dict[str, list[str]] = field(default_factory=dict)
    # library mode: each manifest function's evidence level (:mod:`.item_states`; the gate gives ``unchecked`` or
    # ``reproduces-its-case``) and the findings that set it (``level_reasons``: evidence findings, never refusals)
    levels: dict[str, str] = field(default_factory=dict)
    level_reasons: dict[str, list[str]] = field(default_factory=dict)
    # library mode: each parsed fixture G1 re-derived in this check: path -> {"parser" (the readers it names),
    # "parser_sha256" (their digests as the candidate has them), "rederived" (it re-derived), "parse_seconds"}
    parsed: dict[str, dict] = field(default_factory=dict)


def _failed(reason: str, check: str | None = None) -> GateResult:
    return GateResult(
        False,
        {c: False for c in CHECKS},
        [KEY_SHAPED.sub(_REDACTED, reason)],
        [check] if check else [],
    )


# --- outcome rules -----------------------------------------------------------------------------------------


def _red(o: PytestOutcome) -> bool:
    """At least one failure in a trustworthy report, or a collection error naming the module."""
    if o.timed_out or not o.failed:
        return False
    return o.valid or o.returncode == _COLLECTION_ERROR


def _green(o: PytestOutcome) -> bool:
    return (
        o.valid
        and not o.timed_out
        and o.returncode == 0
        and not o.failed
        and bool(o.passed)
    )


def _shown(o: PytestOutcome) -> str:
    """The output a reason quotes: v2's last 300 characters, or in library mode the head and the whole output's blob."""
    blob = getattr(o, "output_blob", None)
    if blob is None:
        return o.output[-300:]
    return o.output[:300] + f" [full output: blob {blob}]"


def _describe(o: PytestOutcome) -> str:
    return (
        f"rc={o.returncode} valid={o.valid} timed_out={o.timed_out} "
        f"failed={sorted(o.failed)[:5]} passed={len(o.passed)}"
    )


def _exercised(source: bytes, *bodies: dict) -> set[str] | None:
    """The stored functions a test file imports, by id, by module or package, or all of them (None: unparsable).

    v2: a channel or ``env`` import stands for every environment function of *bodies* in its scope. Library mode:
    ``from memory.<package>.<module> import <name>`` names one function; an import of a module, of a package
    (``from memory.<package> import <name>`` names the module or a function of that package by that name) or of
    ``memory`` itself stands for every function of *bodies* in its scope.
    """
    refs = env_references(source)
    lib = library_references(source)
    if refs is None or lib is None:
        return None
    names, channels, everything = refs
    functions = {i for b in bodies for i, v in b.items() if v[0] == "env_function"}
    out = set(names) | {
        i
        for i in functions
        if everything or i.split(":", 1)[0].split("/", 1)[1] in channels
    }
    lib_names, scopes, lib_all = lib
    stored = {i for b in bodies for i, v in b.items() if v[0] == "function"}
    for i in stored:
        module, name = i.split(":", 1)
        package = module.rsplit(".", 1)[0]
        if (
            lib_all
            or i in lib_names
            or module in scopes
            or package in scopes
            or f"{package}.{name}" in lib_names
        ):
            out.add(i)
    return out


def _unchanged(run: Any, it: Any) -> bool:
    """A listed function whose body (and kind) is the parent's: the pass lists it only to change its tests."""
    return (
        it.kind in ("env_function", "function")
        and it.item in run.p_bodies
        and it.item in run.c_bodies
        and run.p_bodies[it.item][:2] == run.c_bodies[it.item][:2]
    )


def _is_fixture(path: str) -> bool:
    """A fixture path: a file under a ``tests/data/`` directory (library mode ``memory/<package>/tests/data/``)."""
    return "/tests/data/" in f"/{path}"


def _library_size(files: dict, tree: Path) -> tuple[int, int, int] | None:
    """The channel modules' code (:func:`.snapshot.code_size`, summed): function definitions, AST nodes of
    function definitions, AST nodes of other module code; comments and docstrings never count. None when a
    module does not parse.
    """
    total = [0, 0, 0]
    for p in files:
        if MODULE_PATH.match(p):
            size = code_size((tree / p).read_bytes())
            if size is None:
                return None
            total = [a + b for a, b in zip(total, size)]
    return total[0], total[1], total[2]


def _owners(man: Manifest, test: str) -> list[str]:
    """The manifest items that list test file *test* ([]: none, so a failure in it is pass-wide)."""
    return [it.item for it in man.items if test in it.tests]


# The input forms that hand a function a recorded value, not the environment: such a function parses what a
# call returned and makes no call itself, so the effect of the calls it covers is not its own.
_VALUE_FORMS = frozenset({"path", "text", "bytes", "observation"})


def _covers_a_write(covers: list[tuple[str, int, Action]], form: str | None) -> bool:
    """Whether a function makes a recorded write call: structural, from the recording, never from names.

    A tool call's effect is what the recording holds (:attr:`.episodes.Action.effect`, the effect the
    environment declared for the method; ``unknown`` when undeclared, which proves nothing). No other
    kind counts: a file, shell or dialogue cover is an observation the function reads. A function whose
    declared input is a recorded value (:data:`_VALUE_FORMS`) parses a response and makes no call.
    """
    if form in _VALUE_FORMS:
        return False
    return any(
        getattr(a, "kind", "tool") == "tool" and getattr(a, "effect", "") == "write"
        for _, _, a in covers
    )


def _test_file(test_id: str, tests_dir: str, library_layout: bool = False) -> str:
    """The test file of a suite's test id, as a path from the tree's root (ids may be relative to *tests_dir*).

    *library_layout* (library mode): a ``memory/`` id is already from the root; under v2 only ``env/`` is.
    """
    path = test_id.split("::", 1)[0]
    roots = ("env/", "memory/") if library_layout else ("env/",)
    return path if path.startswith(roots) else f"{tests_dir}/{path}"


def _unfit_forms(
    covers: list[tuple[str, int, Action]],
    input_kind: str | None,
) -> list[str]:
    """The reasons :func:`.held_out.plan` gives as ``unfit`` (run_plan's first failures), without planning.

    The same covers in the same order (status ``ok``, not a rejection, not shell, each once) through the
    same :func:`.held_out._unfit_reason`, deduplicated, so :meth:`Gate.preview` names exactly what G2 will.
    """
    out: list[str] = []
    done: set[tuple[str, int]] = set()
    for eid, idx, a in covers:
        if (
            getattr(a, "kind", "tool") == "shell"
            or a.status != "ok"
            or is_rejection(a)
            or (eid, idx) in done
        ):
            continue
        done.add((eid, idx))
        why = _unfit_reason(a, input_kind)
        if why is not None and why not in out:
            out.append(why)
    return out


_EXAMPLES_RUNNER = """\
import doctest
import functools
import importlib
import io


def _run(module, name):
    mod = importlib.import_module(module)
    fn = getattr(mod, name)
    calls = []

    @functools.wraps(fn)
    def counted(*args, **kwargs):
        calls.append(1)
        return fn(*args, **kwargs)

    setattr(mod, name, counted)  # an example that imports the function gets the counted one too
    globs = dict(vars(mod))
    tests = doctest.DocTestFinder(recurse=False).find(fn, name, module=False, globs=globs)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    out = io.StringIO()
    results = [runner.run(t, out=out.write) for t in tests]
    assert sum(r.failed for r in results) == 0, out.getvalue()[-2000:]
    assert sum(r.attempted for r in results) >= 1, f"no example of {module}.{name} ran"
    assert calls, f"no example called {module}.{name}"
"""


#: At most this many elements of a collection fixture are shaped (:func:`_fixture_elements`).
FIXTURE_ELEMENTS = 16


def _fixture_elements(data: bytes) -> list:
    """The first :data:`FIXTURE_ELEMENTS` elements of a fixture holding a collection of recorded inputs: the
    items of a JSON list, the values of a JSON object, or the lines of a JSON-lines file; ``[]`` for anything
    else. One level down, only a collection is opened as well (``{"rows": [...]}``, ``{"by_id": {"a": {...}}}``):
    a member that is a list, or an object whose values are all objects or lists. A record (an object with any
    other value) is one input and is never opened, so its field values are never elements. At most
    ``2 * FIXTURE_ELEMENTS`` elements in all. Bounded and value-free."""
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError:
        return []
    try:
        top = json.loads(text)
    except (ValueError, RecursionError):
        out = []
        for line in text.splitlines():
            if len(out) >= FIXTURE_ELEMENTS:
                break
            if line.strip():
                try:
                    out.append(json.loads(line))
                except (ValueError, RecursionError):
                    return []
        return out

    def members(v):
        if isinstance(v, list):
            return v[:FIXTURE_ELEMENTS]
        if isinstance(v, dict):
            return list(v.values())[:FIXTURE_ELEMENTS]
        return []

    def collection(v) -> bool:
        return isinstance(v, list) or (
            isinstance(v, dict)
            and bool(v)
            and all(isinstance(x, (dict, list)) for x in v.values())
        )

    out = list(members(top))
    for v in members(top):
        if not collection(v):
            continue  # a record is one input: its field values are not elements
        if len(out) >= 2 * FIXTURE_ELEMENTS:
            break
        out += members(v)[: 2 * FIXTURE_ELEMENTS - len(out)]
    return out


def fixture_fits(mine: dict, recorded: dict) -> bool:
    """Whether a fixture's descriptor fits a recorded input's: the same exact shape, or a structural match
    of keyed shapes (see :meth:`Gate._fixture_shape`)."""
    return same_shape(mine, recorded) or compare(mine, recorded) is not None


def _examples_source(items: list[str]) -> str:
    """A pytest module running each item's docstring examples as a doctest, one test per item, in order.

    Each test fails unless no example fails, at least one example ran (a skipped one does not count) and
    the function itself was called (it is wrapped by a counter in its module and the examples' globals).
    """
    out = [_EXAMPLES_RUNNER]
    for n, item in enumerate(items):
        channel, name = item.split(":", 1)
        module = channel.replace("/", ".")
        out.append(f"\n\ndef test_example_{n}():\n    _run({module!r}, {name!r})\n")
    return "".join(out)


@dataclass
class _Probe:
    """One function's behaviour check: the inputs to run on both versions and what they showed."""

    item: str
    strict: bool  # an edited function claiming the clean-up exemption: every recorded cover, none unread
    cases: list[Case] = field(default_factory=list)
    # recorded covers among the cases
    covers: set[tuple[str, int]] = field(default_factory=set)
    why: str | None = None  # not shown to do the same before anything runs
    # recorded covers that cannot be read or given (not strict: noted, not compared)
    skipped: int = 0
    differ: list[tuple[str, int]] = field(default_factory=list)
    unjudged: list[tuple[str, int]] = field(default_factory=list)

    def verdict(self) -> str | None:
        """None when it does the same on every compared input; else why not (cover ids, never a value)."""
        if self.why is not None:
            return self.why
        parts = []
        for ids, what in (
            (self.differ, "its results differ from the parent's on {n} {of}"),
            (
                self.unjudged,
                "its results on {n} {of} cannot be compared (not run in time, timed out or not JSON)",
            ),
        ):
            for of, chosen in (
                ("recorded covers", [c for c in ids if c in self.covers]),
                (
                    "recorded actions of this pass's episodes",
                    [c for c in ids if c not in self.covers],
                ),
            ):
                if chosen:
                    parts.append(
                        what.format(n=len(chosen), of=of)
                        + f": {[list(c) for c in sorted(chosen)[:5]]}",
                    )
        return "; ".join(parts) or None


class _Budget:
    """G3's behaviour-check budget for one pass: wall-clock seconds left."""

    def __init__(self, seconds: float) -> None:
        self.end = time.monotonic() + seconds

    def left(self) -> float:
        return self.end - time.monotonic()


def _sample(covers: set[tuple[str, int]], seed: str, n: int) -> set[tuple[str, int]]:
    """At most *n* of *covers*, chosen by a hash seeded with *seed* (deterministic, independent of order)."""
    if len(covers) <= n:
        return set(covers)

    def rank(c: tuple[str, int]) -> str:
        return hashlib.sha256(f"{seed}\0{c[0]}\0{c[1]}".encode()).hexdigest()

    return set(sorted(covers, key=rank)[:n])


class _WithSources:
    """The evidence store, with an item's manifest source episodes counted as its episodes.

    A new job-level item has no recorded item evidence yet (that is written on merge); its source
    episodes are where its supporting or contrary signals live.
    """

    def __init__(self, evidence: EvidenceStore, item: str, sources: list[str]) -> None:
        self._ev, self._item, self._sources = evidence, item, sources

    def item_episodes(self, item: str) -> list[str]:
        eids = set(self._ev.item_episodes(item))
        if item == self._item:
            eids |= set(self._sources)
        return sorted(eids)

    def signals_for(self, eid: str):
        return self._ev.signals_for(eid)


@dataclass
class _Run:
    """One check's state: the resolved revisions, both trees and what the checks found."""

    res: GateResult
    man: Manifest
    manifest_raw: object
    parent: str
    candidate: str
    tmp: Path
    p_files: dict[str, tuple[str, str]] = field(
        default_factory=dict,
    )  # path -> (mode, blob)
    c_files: dict[str, tuple[str, str]] = field(default_factory=dict)
    changed: list[str] = field(default_factory=list)
    p_tree: Path = Path()
    c_tree: Path = Path()
    p_bodies: dict[str, tuple[str, str, bool]] = field(default_factory=dict)
    c_bodies: dict[str, tuple[str, str, bool]] = field(default_factory=dict)
    c_report: ItemsReport | None = None
    report_error: str | None = None
    covers: set[tuple[str, str, int]] = field(
        default_factory=set,
    )  # validated (item, episode, action)
    rejections: set[tuple[str, int]] = field(
        default_factory=set,
    )  # validated covers that are recorded rejections, never new coverage for G5
    notes: list[str] = field(default_factory=list)  # informational; never fail the gate
    # covered channels -> (the recorded observations pool of :mod:`.held_out`, whether the cap stopped it)
    pools: dict[tuple[str, ...], tuple[list[tuple[str, Action]], bool]] = field(
        default_factory=dict,
    )
    # item -> (body digest, input-shape descriptors of its validated covers), recorded on a landed merge
    shapes: dict[str, tuple[str, list[dict]]] = field(default_factory=dict)
    # stage 5 (:mod:`.qa`): the /inputs mount of the gate's runs (None: every switch off and the library's
    # tests do not use the test kit), and G3's first candidate run of each new or changed test file
    qa_env: QAEnv | None = None
    qa_first: dict[str, PytestOutcome] = field(default_factory=dict)
    # environment functions that cover a recorded write call (structural Effect; G6)
    writes: set[str] = field(default_factory=set)
    # per-item admission: item -> the checks that refused it, and whether any failure is pass-wide
    item_fail: dict[str, list[str]] = field(default_factory=dict)
    pass_wide: bool = False
    # functions of the changed channels that G3's behaviour check showed doing what their parent versions did
    # on every compared input. None: not judged (a preview runs no G3).
    same: set[str] | None = None
    # the parent's declared input forms by function, read once (False: its library cannot be read)
    p_inputs: dict[str, str] | bool | None = None
    # changed channel -> recorded actions of the channel in the pass's own episodes (G3's behaviour check)
    episode_inputs: dict[str, list[tuple[str, int, Action]]] = field(
        default_factory=dict,
    )
    # library mode: per item, what the checks measured (library_checks, qa); whether the index is over its view (G4)
    verification: dict[str, dict] = field(default_factory=dict)
    curate_due: bool = False
    # library-mode CURATE: what LibraryChecks.curate found the candidate does to aliases and retirements
    curations: list[dict] = field(default_factory=list)
    # run label -> the whole recorded output of a failing run (GateResult.outputs; library mode)
    outputs: dict[str, str] = field(default_factory=dict)
    # the stage-5 seed (library mode: one per pass, so every round's check and the final merge draw the same
    # recorded inputs and mutants); None: :func:`.qa.seed_of` the candidate, as in v2
    seed: bytes | None = None
    # library mode: the import graph of both versions (G3's behaviour scope; None in v2)
    graph: dict[str, set[str]] | None = None
    # library mode: the fixture provenance entries G1 verified in this pass (path -> (sha256, entry)), persisted on
    # merge
    fixtures_verified: dict[str, tuple[str, dict]] = field(default_factory=dict)
    # library mode: each function's input fixtures (example_perturb.InputFixture), for stage 5's drawn inputs
    input_fixtures: dict[str, list] = field(default_factory=dict)
    # library mode: the call records of the parent's suites on the parent's library (baseline), on the
    # candidate's library (regression run), and of the candidate's suites (suite path -> record; example_calls)
    calls_base: dict[str, Any] = field(default_factory=dict)
    calls_reg: dict[str, Any] = field(default_factory=dict)
    calls_cand: dict[str, Any] = field(default_factory=dict)
    # library mode: evidence findings set an item's evidence level instead of refusing it (:meth:`cap`)
    levels_on: bool = False
    # item -> the evidence findings that set its level (in order, each once)
    level_reasons: dict[str, list[str]] = field(default_factory=dict)
    # library mode: per function listed only to change its tests, the episodes of the stored typed ``episode`` covers
    # its manifest entry repeats; they were checked when it landed and are not run again
    typed_kept: dict[str, list[str]] = field(default_factory=dict)

    def cap(
        self,
        check: str,
        reason: str,
        item: str | list[str] | None = None,
    ) -> None:
        """Record an evidence finding of *check* about *item*: it sets the item's evidence level to
        :data:`.item_states.UNCHECKED` and refuses nothing (library mode).

        Outside library mode, or for a finding no item owns, it is a failure (:meth:`fail`), as before levels.
        """
        owners = [item] if isinstance(item, str) else list(item or [])
        if not self.levels_on or not owners:
            self.fail(check, reason, item)
            return
        line = KEY_SHAPED.sub(_REDACTED, f"{check}: {reason}")
        for owner in owners:
            found = self.level_reasons.setdefault(owner, [])
            if line not in found:
                found.append(line)
                # the writer sees it in the gate's reply as a note, never as a refusal
                self.note(f"evidence level: {owner} is {_LOWEST_LEVEL}: {line}")

    def fail(
        self,
        check: str,
        reason: str,
        item: str | list[str] | None = None,
    ) -> None:
        """Record a failure of *check*; *item* names the manifest item(s) it belongs to (None: pass-wide).

        Only a failure attributed to items can be resolved by refusing those items alone (per-item
        admission, :meth:`Gate.merge`); an unattributed failure refuses the whole pass.
        """
        # reasons are stored in the evidence store; test output in them is model-controlled
        self.res.checks[check] = False
        self.res.passed = False
        if check not in self.res.refused:
            self.res.refused.append(check)
        line = KEY_SHAPED.sub(_REDACTED, f"{check}: {reason}")
        self.res.reasons.append(line)
        owners = [item] if isinstance(item, str) else list(item or [])
        if not owners:
            self.pass_wide = True
        for owner in owners:
            codes = self.item_fail.setdefault(owner, [])
            if check not in codes:
                codes.append(check)
            self.res.item_reasons.setdefault(owner, []).append(line)

    def note(self, reason: str) -> None:
        self.notes.append(KEY_SHAPED.sub(_REDACTED, f"note: {reason}"))

    def output(self, label: str, o: PytestOutcome) -> None:
        """Keep a failing run's whole recorded output under *label* for the full gate result (library mode):
        :attr:`.sandbox_run.PytestOutcome.full_output`, which only the sandbox's own marked capture bound cuts.

        Never part of a reason: the reasons, and so the pass row, keep v2's 300-character tail.
        """
        text = KEY_SHAPED.sub(_REDACTED, o.full_output or o.output)
        blob = getattr(o, "output_blob", None)
        # library mode: the whole output, past the capture bound, is a blob
        self.outputs[label] = (
            text if blob is None else f"{text}\n[whole output: blob {blob}]"
        )

    def stop(self, failures: list[tuple[str, str]]) -> None:
        for check, reason in failures:
            self.fail(check, reason)
        hit = {c for c, _ in failures}
        for c in CHECKS:
            if c not in hit:
                self.res.checks[c] = False
                self.res.reasons.append(f"{c}: not evaluated")


class Gate:
    def __init__(
        self,
        memory: Repo,
        evidence: EvidenceStore,
        blobs: BlobStore,
        *,
        python: Path = PYTHON,
        budget_tokens: int = 4000,
        reader_promotes: bool = False,
        action_lookup: Callable[[str, int], Action | None] | None = None,
        pytest_runner: Callable[..., PytestOutcome] = run_pytest,
        docstring_standard: bool = False,
        surfacing: str = "index",
        soft_budget: bool = False,
        qa: QAConfig | None = None,
        library_mode: LibraryConfig | None = None,
    ) -> None:
        """The library-mode switches (:mod:`.integration.switch`; each default is the v2 behaviour):
        *docstring_standard* turns on the lean docstring standard (G1) and its examples run (G3) for new or
        changed environment functions; *surfacing* ``"catalogue"`` records and freezes input shapes per
        commit for the export's catalogue; *soft_budget* makes *budget_tokens* G4's soft budget (a note,
        never a refusal) instead of the index's hard cap. *qa* is the stage-5 test checks' configuration
        (:class:`.qa.QAConfig`; None runs none of them). The writer's brief follows the gate's switches.
        *library_mode* (``UNIFY_MEMORY_LIBRARY``, :class:`.library_checks.LibraryConfig`; None is v2): with ``layout``, covers have
        no channel rule (G2), and G3's behaviour check and per-item reduction are scoped by the import graph.
        With ``checks``, the library gate checks (:mod:`.library_checks`) and stage 5 run in library mode.
        """
        if surfacing not in ("index", "catalogue"):
            raise ValueError(
                f"surfacing must be 'index' or 'catalogue', not {surfacing!r}",
            )
        self.mem, self.ev, self.blobs = memory, evidence, blobs
        self.docstring_standard = bool(docstring_standard)
        self.surfacing = surfacing
        self.soft_budget = bool(soft_budget)
        self.python, self.budget, self.reader_promotes = (
            python,
            budget_tokens,
            reader_promotes,
        )
        # Fail closed: without an episodes reader no covered call can be confirmed.
        self.lookup = action_lookup or (lambda eid, i: None)
        self.pytest = pytest_runner
        # stage-5 test checks (library mode); the default runs none of them
        self.qa = qa if qa is not None else QAConfig()
        self.library_mode = library_mode
        if library_mode is not None and library_mode.checks:
            if not library_mode.layout:
                raise ValueError(
                    "the library gate checks need the library layout (LibraryConfig.layout)",
                )
            from .qa import library_config

            self.qa = library_config(self.qa)

    @property
    def _library_layout(self) -> bool:
        """The library layout rules are on."""
        library_mode = getattr(self, "library_mode", None)
        return library_mode is not None and bool(library_mode.layout)

    # -- public API ---------------------------------------------------------------------------------------
    def check(
        self,
        parent: str,
        candidate: str,
        manifest: dict,
        *,
        seed: bytes | None = None,
    ) -> GateResult:
        """The whole gate on *candidate*, without merging; no per-item reduction is tried.

        ``items_refused`` names the items its failures belong to, as :meth:`merge` would refuse them. *seed*
        (library mode): the pass's stage-5 seed, so a repair round's check and the final merge draw the same
        sample; None draws by the candidate, as in v2.
        """
        res, _, notes, _, _ = self._check(parent, candidate, manifest, seed=seed)
        res.reasons.extend(notes)
        return res

    def parent_snapshot(self, parent: str, dest: Path) -> ParentSnapshot | None:
        """Resolve *parent* and extract its committed tree into *dest* once, for repeated :meth:`preview` calls.

        None when *parent* does not resolve. The snapshot is read only by previews; its owner removes *dest*.
        """
        sha = self._resolve(parent)
        if sha is None:
            return None
        files, _ = listing(self.mem, sha)
        return ParentSnapshot(sha, files, materialise(self.mem, files, Path(dest)))

    def preview(
        self,
        parent: ParentSnapshot | str,
        tree: Path,
        manifest: object,
    ) -> list[str]:
        """The cheap, read-only checks of :meth:`check` on an uncommitted *tree*: failure reasons, [] if none.

        For a consolidator's ``check`` tool before it finishes. *tree* holds the files the pass would commit
        (:func:`.snapshot.tree_listing` lists it without git; empty directories are no entries). *parent* is
        a :meth:`parent_snapshot` (then the preview runs no git at all) or a revision to snapshot now.
        Runs the manifest parse, the early layout and safety refusals, G1, G2's cover checks (each cover's
        kind, record and channel, and the structural check that the covers can give the declared ``input``
        form, without the held-out perturbation; at most :data:`PREVIEW_MAX_COVERS` covers over
        :data:`PREVIEW_MAX_EPISODES` episodes, each looked up once), G4, G5 and G6. Never G3 (no test runs),
        G2's held-out runs or a workflow's signals (nothing about outcomes), and never writes:
        no commit, git object, evidence or pass row. A clean preview does not mean the gate will pass.
        """
        res = GateResult(True, {c: True for c in CHECKS})
        tmp = Path(tempfile.mkdtemp(prefix="memory-preview-"))
        run = _Run(res, Manifest(), manifest, "", "", tmp)
        run.levels_on = self._library_checks()
        try:
            try:
                run.man = self._parse_manifest(manifest)
            except ManifestError as exc:
                run.fail("G1", f"malformed manifest: {exc}")
                return list(res.reasons)
            base = (
                parent
                if isinstance(parent, ParentSnapshot)
                else self.parent_snapshot(parent, tmp / "parent")
            )
            if base is None:
                run.fail("G1", "unresolvable parent revision")
                return list(res.reasons)
            run.parent, run.p_files, run.p_tree = base.sha, base.files, base.tree
            run.c_files, refused = tree_listing(
                Path(tree),
                "sha256" if len(base.sha) == 64 else "sha1",
            )
            early = self._early(run, refused, self._library_layout)
            if early:
                for check, reason in early:
                    run.fail(check, reason)
                return list(res.reasons)
            run.c_tree = tmp / "cand"
            run.c_tree.mkdir()
            # exactly the listed files: no empty directory, nothing skipped
            for rel in run.c_files:
                (run.c_tree / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    Path(tree) / rel,
                    run.c_tree / rel,
                    follow_symlinks=False,
                )
            self._read_trees(run)
            self._g1(run)
            covers = [
                c
                for it in run.man.items
                if it.kind in ("env_function", "function")
                for c in it.covers
            ]
            episodes = {eid for eid, _ in covers}
            bounded = (
                len(covers) <= PREVIEW_MAX_COVERS
                and len(episodes) <= PREVIEW_MAX_EPISODES
            )
            memo: dict[tuple[str, int], Action | None] = {}

            def lookup(eid: str, idx: int) -> Action | None:
                if (eid, idx) not in memo:
                    memo[(eid, idx)] = self.lookup(eid, idx)
                return memo[(eid, idx)]

            if not bounded:
                run.fail(
                    "G2",
                    f"the manifest names {len(covers)} covers over {len(episodes)} episodes; a check "
                    f"examines at most {PREVIEW_MAX_COVERS} covers over {PREVIEW_MAX_EPISODES} episodes",
                )
            else:
                self._g2(run, preview=True, lookup=lookup)
                self._g5(run)  # needs G2's validated covers
            self._g4(run)
            self._g6(run)
            v = self._library_mode(run)
            if v is not None:
                v.static()
            qa = QAChecks(self, run)
            if self.qa.on or qa.uses_kit():
                qa.kit()  # library code never uses the test kit (static)
            if self.qa.on and bounded:
                # stage 5's static checks: fixture size, replay fidelity, cuts (no test runs)
                qa.static(lookup)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return list(res.reasons)

    def merge(
        self,
        parent: str,
        candidate: str,
        manifest: dict,
        pass_id: str,
        kind: str,
        channel: str | None,
        usd: str,
        *,
        seed: bytes | None = None,
    ) -> GateResult:
        """Check, then fast-forward ``main``; evidence is recorded only for a landed candidate.

        *seed*: as for :meth:`check` (library mode); the reduced candidate is checked with the same seed.

        Per-item admission (stage 7): when the candidate is refused and every failure belongs to manifest
        items (an item's G1 checks, its G2 evidence and held-out runs, its own tests' red→green, examples
        and new failures in its own test files under G3, a behaviour change G3's behaviour check attributes
        to it (:meth:`_behaviour_owner`), its G6 ``Effect:`` checks), those items are refused with
        everything that depends on them (:mod:`.reduction`). The candidate reduced to the other items is
        committed as a child of *parent* and the whole gate runs on it once more; it lands if it passes,
        and otherwise the pass is refused whole. A pass-wide failure (layout, unsafe or forbidden files,
        a malformed manifest, undeclared changes, the suites' lost tests and regression runs, the
        behaviour check's bounds and the changes it cannot attribute to one item, G4, G5,
        key-shaped strings, unparsable modules) refuses the whole pass at once. One round; no search over
        subsets. :attr:`GateResult.items_merged` and :attr:`GateResult.items_refused` (value-free codes)
        are recorded in the pass row.
        """
        if self.ev.pass_exists(pass_id):
            # never overwrite an earlier attempt's record
            return _failed(f"pass {pass_id} is already recorded")
        p_sha, c_sha = self._resolve(parent), self._resolve(candidate)
        if p_sha is None or c_sha is None:
            res = _failed(
                f"G1: unresolvable revision {parent!r} or {candidate!r}"[:300],
                "G1",
            )
            self._record(
                parent,
                candidate,
                pass_id,
                kind,
                channel,
                usd,
                res,
                patch_ok=False,
            )
            return res
        landed, landed_manifest = c_sha, manifest
        if seed is None and self._library_checks():
            from .qa import seed_of

            seed = seed_of(
                c_sha,
            )  # the reduced re-check draws the candidate's sample
        try:
            res, covers, notes, shapes, reduced = self._check(
                p_sha,
                c_sha,
                manifest,
                reduce_as=pass_id,
                seed=seed,
            )
            if reduced is not None:
                again, covers2, notes2, shapes2, _ = self._check(
                    p_sha,
                    reduced.sha,
                    reduced.manifest,
                    seed=seed,
                )
                if again.passed:
                    again.reasons = [
                        f"item refused: {r}" for r in res.reasons + reduced.reasons
                    ]
                    again.refused = []
                    again.outputs = {
                        **{f"item refused: {k}": v for k, v in res.outputs.items()},
                        **again.outputs,
                    }
                    res, covers, notes, shapes = again, covers2, notes2, shapes2
                    landed, landed_manifest = reduced.sha, reduced.manifest
                else:
                    res.reasons.append(
                        "per-item admission: the candidate without the refused items "
                        f"({reduced.sha[:12]}) was refused too",
                    )
                    res.reasons += [f"reduced: {r}" for r in again.reasons]
                    res.outputs.update(
                        {f"reduced: {k}": v for k, v in again.outputs.items()},
                    )
                    res.refused += [c for c in again.refused if c not in res.refused]
                res.items_refused = reduced.refused
        except BaseException as exc:
            self._record(
                p_sha,
                c_sha,
                pass_id,
                kind,
                channel,
                usd,
                _failed(f"gate error: {exc!r}"[:500]),
            )
            raise
        if res.passed:
            try:
                if isinstance(self.library_mode, LibraryConfig):
                    # library mode: the one writer; re-targets over status commits only
                    from .memory_writer import land

                    landed = land(self.mem, landed, p_sha)
                else:
                    self.mem.fast_forward("main", landed, expected_old=p_sha)
            except (
                GitError
            ) as exc:  # StaleParent is a GitError: refused, and _record keeps the whole patch
                res.passed = False
                res.reasons.append(f"merge: {exc}")
        res.reasons.extend(notes)  # after every failure reason, including the merge's
        if res.passed:
            res.merged = landed
            landed_items = self._parse_manifest(landed_manifest).items
            res.items_merged = [it.item for it in landed_items]
            if self.library_mode is not None and self.library_mode.layout:
                # a renamed item names its old id, so adoption joins across the rename (the item records' uid)
                was = {
                    r["target"]: r["item"]
                    for r in res.curations
                    if r.get("action") == "rename"
                }
                res.provenance = [
                    {
                        "item": it.item,
                        "adopts": list(it.adopts),
                        "how": dict(it.adopt_how),
                    }
                    | ({"renamed_from": was[it.item]} if it.item in was else {})
                    for it in landed_items
                ]
            for it in landed_items:
                for eid in it.source_episodes:
                    self.ev.add_item_evidence(it.item, eid, "source")
            for item, eid, idx in sorted(covers):
                self.ev.add_cover(item, eid, idx)
            if self._library_checks():
                # typed covers persist beside covers, for the behaviour check and deletion (G5)
                from .procedures import cover_raw

                for it in landed_items:
                    for c in it.typed_covers:
                        self.ev.add_typed_cover(
                            it.item,
                            c.episode,
                            json.dumps(cover_raw(c), sort_keys=True),
                        )
            if self._library_checks():
                # fixture provenance this pass verified, for later passes (fixtures.ledger_for)
                for path, (sha, entry) in sorted(res.fixtures_verified.items()):
                    self.ev.add_fixture_provenance(path, sha, entry, landed)
            if self.surfacing == "catalogue" or self._library_layout:
                # library mode: shapes per library id, frozen per commit for the actor's find (library_export)
                self.ev.write_commit_shapes(landed, shapes)
        else:
            try:
                ids = [it.item for it in self._parse_manifest(manifest).items]
            except ManifestError:
                ids = []
            res.items_merged = []
            res.items_refused = {i: res.items_refused.get(i) or ["pass"] for i in ids}
        self._record(p_sha, c_sha, pass_id, kind, channel, usd, res)
        if self._library_checks() and res.verification:
            # the item verification records, for the item records; measured, never a refusal
            self.ev.add_pass_notes(
                pass_id,
                [
                    "library_mode-verification "
                    + json.dumps(res.verification, sort_keys=True, default=str),
                ],
            )
        if self.library_mode is not None and res.passed and res.curations:
            # CURATE's aliases, retirements and dropped aliases, for the item records (alias_of,
            # deprecated), and any pass's renames, which carry an item's uid to its new id
            self.ev.record_curations(pass_id, res.merged, res.curations)
        return res

    # -- helpers ------------------------------------------------------------------------------------------
    def _library_checks(self) -> bool:
        return self.library_mode is not None and self.library_mode.checks

    def _parse_manifest(self, manifest: object) -> Manifest:
        """The manifest under the gate's layout: the library mode's (:func:`.manifest.parse_manifest` ``library_mode``) or v2's."""
        return parse_manifest(
            manifest,
            library_mode=self.library_mode is not None and self.library_mode.layout,
        )

    def _library_mode(self, run: _Run):
        """The library-mode checks of *run* (:mod:`.library_checks`), or None with them off."""
        if not self._library_checks():
            return None
        from .library_checks import LibraryChecks

        return LibraryChecks(self, run)

    def _resolve(self, rev: str) -> str | None:
        if not isinstance(rev, str) or not rev or rev.startswith("-") or "\n" in rev:
            return None
        try:
            sha = self.mem.run(
                "rev-parse",
                "--verify",
                "--quiet",
                f"{rev}^{{commit}}",
            ).strip()
        except GitError:
            return None
        return sha if _SHA.match(sha) else None

    def _pytest(
        self,
        tree: Path,
        target: str,
        extra_env: dict[str, str] | None = None,
        qa_env: QAEnv | None = None,
        calls: list[str] | None = None,
    ) -> PytestOutcome:
        # library mode: the whole output is spooled and kept as a blob; v2 passes nothing new
        spool = (
            Path(tempfile.mkdtemp(prefix="memlib-out-"))
            if self._library_checks()
            else None
        )
        more: dict = {"spool_dir": spool} if spool is not None else {}
        if calls is not None:
            # library mode: record the calls test code makes of these functions (example_calls)
            more["calls"] = list(calls)
        try:
            if qa_env is not None:
                # a stage-5 switch is on or the library's tests use the test kit: memlab and the referenced
                # blobs at /inputs, and a skip for a failed import fails (the kit must never be silently missing)
                o = self.pytest(
                    target,
                    python=self.python,
                    ro={tree: "/memory", qa_env.inputs: "/inputs"},
                    rw={},
                    cwd="/memory",
                    timeout_s=_TIMEOUT_S,
                    env={**qa_env.env, **(extra_env or {})},
                    import_skips_fail=True,
                    **more,
                )
            else:
                o = self.pytest(
                    target,
                    python=self.python,
                    ro={tree: "/memory"},
                    rw={},
                    cwd="/memory",
                    timeout_s=_TIMEOUT_S,
                    env={**_PYTEST_ENV, **(extra_env or {})},
                    **more,
                )
            if spool is not None:
                from .library_checks import keep_output

                keep_output(o, spool, self.blobs)
            return o
        finally:
            if spool is not None:
                shutil.rmtree(spool, ignore_errors=True)

    def _record(
        self,
        parent: str,
        candidate: str,
        pass_id: str,
        kind: str,
        channel: str | None,
        usd: str,
        res: GateResult,
        patch_ok: bool = True,
    ) -> None:
        patch = None
        # a refused pass keeps its whole patch; a reduced one keeps what it refused
        base = parent if not res.passed else res.merged
        if patch_ok and base is not None and base != candidate:
            try:
                patch = self.blobs.put(self.mem.diff(base, candidate).encode())
            except GitError:
                patch = None
        self.ev.record_pass(
            {
                "pass_id": pass_id,
                "kind": kind,
                "channel": channel,
                "parent": parent,
                "candidate": candidate,
                "passed": int(res.passed),
                "reasons": json.dumps(res.reasons),
                "usd": usd,
                "patch_blob": patch,
                "merged": res.merged,
                "items_merged": json.dumps(res.items_merged),
                "items_refused": json.dumps(res.items_refused, sort_keys=True),
            },
        )

    # -- the checks ---------------------------------------------------------------------------------------
    def _check(
        self,
        parent: str,
        candidate: str,
        manifest: dict,
        reduce_as: str | None = None,
        seed: bytes | None = None,
    ) -> tuple[
        GateResult,
        set[tuple[str, str, int]],
        list[str],
        dict[str, dict],
        _Reduced | None,
    ]:
        """The result (failure reasons only), the validated covers if it passed, the ``note:`` lines, the
        candidate's shape snapshot to freeze if it passed (:mod:`.shape_rows`), and with *reduce_as* (the
        pass id) the committed reduction of a candidate refused item by item.
        """
        res = GateResult(True, {c: True for c in CHECKS})
        tmp = Path(tempfile.mkdtemp(prefix="memory-gate-"))
        run = _Run(res, Manifest(), manifest, parent, candidate, tmp)
        run.seed = seed
        run.levels_on = self._library_checks()
        reduced = None
        try:
            if self._prepare(run):
                v = self._library_mode(run)
                self._g1(run)
                if v is not None:
                    v.g1()
                self._g2(run)
                if v is not None:
                    v.g2()
                # stage 5: the test kit when a switch is on or the library's tests use it (else nothing)
                qa = QAChecks(self, run)
                if qa.prepare():
                    qa.kit()
                    if self.qa.on:
                        qa.static(self.lookup)
                self._g3(run)
                if v is not None:
                    v.g3()
                    self._learned(run)
                    v.g4()
                    v.curate()
                else:
                    self._g4(run)
                self._g5(run)
                if v is not None:
                    v.g5()
                self._g6(run)
                if self.qa.on:
                    # last: only a candidate the rest of the gate accepts, or, under per-item admission,
                    # one refused item by item only, so that a stage-5 refusal joins the reduction
                    qa.dynamic(item_scoped=reduce_as is not None)
                if reduce_as is not None and not res.passed:
                    reduced = self._reduce(run, reduce_as)
            snapshot = (
                self._snapshot(run)
                if res.passed
                and (self.surfacing == "catalogue" or self._library_layout)
                else {}
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        res.items_refused = {i: list(c) for i, c in run.item_fail.items()}
        if run.levels_on:
            self._levels(run)
        res.verification = {i: dict(r) for i, r in sorted(run.verification.items())}
        res.curate_due = run.curate_due
        res.curations = [dict(r) for r in run.curations]
        res.outputs = dict(run.outputs)
        res.fixtures_verified = dict(run.fixtures_verified)
        passed = res.passed
        return res, (run.covers if passed else set()), run.notes, snapshot, reduced

    @staticmethod
    def _levels(run: _Run) -> None:
        """Library mode: each manifest function's evidence level, from the findings :meth:`_Run.cap` recorded.

        A function with a finding is ``unchecked``; one without is ``reproduces-its-case``. Both go in the gate
        result and in the function's verification record (``level``, ``level_reasons``), from which the item
        records take them; the gate never gives ``confirmed-by-outcomes``.
        """
        from .item_states import REPRODUCES_ITS_CASE, UNCHECKED

        res = run.res
        for it in run.man.items:
            if it.kind != "function" or it.item not in run.c_bodies:
                continue
            found = list(run.level_reasons.get(it.item, []))
            level = UNCHECKED if found else REPRODUCES_ITS_CASE
            res.levels[it.item] = level
            res.level_reasons[it.item] = found
            rec = run.verification.setdefault(it.item, {})
            rec["level"] = level
            rec["level_reasons"] = found

    def _learned(self, run: _Run) -> None:
        """``UNIFY_MEMORY_LEARNED_SKILLS=on`` (library mode), under G3: each learned module the candidate adds or changes
        (its module or its fixture file, :mod:`.learned_store`) must carry fixture cases that its skill, from its
        initial state, reproduces after the fit (examples and queries), checked in a confined run. A failure belongs
        to the module's items in the manifest (else the pass)."""
        from unify.settings import SETTINGS

        from .integration.switch import learned_skills

        if not learned_skills(SETTINGS):
            return
        from . import learned_store

        mods: dict[str, bytes] = {}
        for p in run.c_files:
            parts = p.split("/")
            if len(parts) == 3 and parts[0] == "memory" and p.endswith(".py"):
                try:
                    mods[p] = (run.c_tree / p).read_bytes()
                except OSError:
                    continue
        changed = set(run.changed)
        ids = {it.item for it in run.man.items}
        declared = learned_store.learned_modules(mods)
        for p, data in sorted(mods.items()):
            module = p[:-3].replace("/", ".")
            if p in changed and b"LEARNED_FAMILY" in data and module not in declared:
                owners = sorted(i for i in ids if i.startswith(module + ":"))
                run.fail(
                    "G3",
                    f"learned skill {module}: LEARNED_FAMILY, LEARNED_OPTIONS and LEARNED_INITIAL must be literals "
                    "naming a known family with options it accepts",
                    owners or None,
                )
        for module, decl in declared.items():
            if (
                learned_store.module_path(module) not in changed
                and learned_store.fixture_path(module) not in changed
            ):
                continue
            ok, reason = learned_store.check_landing(run.c_tree, module, decl)
            if ok:
                run.note(
                    f"learned skill {module}: every fixture case is reproduced after the fit",
                )
                continue
            owners = sorted(i for i in ids if i.startswith(module + ":"))
            run.fail("G3", f"learned skill {module}: {reason}", owners or None)

    def _reduce(self, run: _Run, pass_id: str) -> _Reduced | None:
        """The candidate without the items its failures belong to, committed on the parent; None if none.

        Only when no failure is pass-wide and each names a manifest item (:meth:`_Run.fail`).
        """
        ids = {it.item for it in run.man.items}
        if run.pass_wide or not run.item_fail or not set(run.item_fail) <= ids:
            return None
        dest = run.tmp / "reduced"
        try:
            refused, reasons = reduction.with_dependents(
                run.man,
                run.item_fail,
                run.c_tree,
                library_mode=self._library_layout,
            )
            manifest = reduction.build(
                run.man,
                run.manifest_raw,
                refused,
                set(run.p_files),
                {i for i, b in run.p_bodies.items() if b[0] == "env_function"},
                run.p_tree,
                run.c_tree,
                dest,
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - no reduction: the pass stays refused whole
            run.note(
                f"per-item admission: the candidate could not be reduced ({exc!r})"[
                    :300
                ],
            )
            return None
        if (
            manifest is None
        ):  # nothing would be admitted, or a refused item cannot be separated
            return None
        sources = sorted(
            {
                e
                for it in run.man.items
                if it.item not in refused
                for e in it.source_episodes
            },
        )
        try:
            with self.mem.temp_checkout(run.parent) as wt:
                for entry in wt.iterdir():
                    if entry.name == ".git":
                        continue
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                shutil.copytree(dest, wt, dirs_exist_ok=True)
                sha = self.mem.commit_all(
                    wt,
                    f"consolidation pass {pass_id}: the admitted items",
                    {
                        "Pass": pass_id,
                        "Reduced-From": run.candidate,
                        "Episode": sources,
                        "Evidence": sources,
                    },
                )
        except (GitError, OSError) as exc:
            run.note(
                f"per-item admission: the reduced candidate was not committed ({exc})"[
                    :300
                ],
            )
            return None
        if sha == run.parent:
            return None
        return _Reduced(sha, manifest, refused, reasons)

    def _prepare(self, run: _Run) -> bool:
        """Parse, resolve and extract; refuse what must never be extracted or run. False stops the check."""
        try:
            run.man = self._parse_manifest(run.manifest_raw)
        except ManifestError as exc:
            run.res.manifest_invalid = True
            run.stop([("G1", f"malformed manifest: {exc}")])
            return False
        p_sha, c_sha = self._resolve(run.parent), self._resolve(run.candidate)
        if p_sha is None or c_sha is None:
            run.stop([("G1", "unresolvable parent or candidate revision")])
            return False
        run.parent, run.candidate = p_sha, c_sha
        try:
            self.mem.run("merge-base", "--is-ancestor", p_sha, c_sha)
        except GitError:
            run.stop([("G1", f"{c_sha[:12]} does not descend from {p_sha[:12]}")])
            return False
        run.p_files, _ = listing(self.mem, p_sha)
        run.c_files, refused = listing(self.mem, c_sha)
        early = self._early(run, refused, self._library_layout)
        if early:
            run.stop(early)
            return False
        run.p_tree = materialise(self.mem, run.p_files, run.tmp / "parent")
        run.c_tree = materialise(self.mem, run.c_files, run.tmp / "cand")
        self._read_trees(run)
        return True

    @staticmethod
    def _early(
        run: _Run,
        refused: list[str],
        library_layout: bool = False,
    ) -> list[tuple[str, str]]:
        """What must never be extracted or run (refused entries, forbidden files, layout); sets ``changed``.

        *library_layout* (library mode): the layout is the library's (:func:`.layout.classify`): a path it has no place for, or a
        file the harness generates (never committed: the writer's box drops unchanged copies), is refused, and
        only the G6 classes of :func:`.manifest.unsafe_path` apply (its root rule is v2's).
        """
        early = [("G6", f"the candidate holds {r}") for r in refused[:10]]
        early += [
            ("G6", f"forbidden file {p}") for p in sorted(run.c_files) if forbidden(p)
        ]
        if library_layout:
            from .layout import misplaced

            early += [
                ("G1", f"file {p}: {why}")
                for p in sorted(run.c_files)
                if (why := misplaced(p)) is not None
            ]
            unsafe = [
                u
                for p in sorted(run.c_files)
                if not forbidden(p) and (u := unsafe_path(p)) and u[0] == "G6"
            ]
            early += unsafe[:10]
            if len(unsafe) > 10:
                early.append(("G6", f"and {len(unsafe) - 10} more such paths"))
            run.changed = sorted(
                p
                for p in set(run.p_files) | set(run.c_files)
                if run.p_files.get(p) != run.c_files.get(p)
            )
            return early
        early += [
            ("G1", f"file {p} is outside the layout (no submodules or root modules)")
            for p in sorted(run.c_files)
            if not layout_allowed(p)
        ]
        # Compiled code, start-up hooks and root entries the layout above admits; one reason each.
        unsafe = [
            u
            for p in sorted(run.c_files)
            if not forbidden(p) and layout_allowed(p) and (u := unsafe_path(p))
        ]
        early += unsafe[:10]
        if len(unsafe) > 10:
            early.append(("G6", f"and {len(unsafe) - 10} more such paths"))
        run.changed = sorted(
            p
            for p in set(run.p_files) | set(run.c_files)
            if run.p_files.get(p) != run.c_files.get(p)
        )
        early += [
            (
                "G1",
                f"file {p} is reserved: the harness generates README.md, memory.py and .memory/ in "
                "every export, and no root entry may shadow `import memory` or `import env`",
            )
            for p in run.changed
            if p in run.c_files and reserved(p)
        ]
        return early

    def _read_trees(self, run: _Run) -> None:
        if self._library_layout:
            # library mode: items are the layout's function ids and note paths (:mod:`.layout`)
            from .layout import classify, function_bodies

            def bodies(tree: Path, files: dict) -> dict[str, tuple[str, str, bool]]:
                out = {
                    i: ("function", b, True) for i, b in function_bodies(tree).items()
                }
                for p in sorted(files):
                    if classify(p) == "note":
                        out[p] = (
                            "note",
                            (tree / p).read_bytes().decode("utf-8", "replace"),
                            True,
                        )
                return out

            run.p_bodies, run.c_bodies = bodies(run.p_tree, run.p_files), bodies(
                run.c_tree,
                run.c_files,
            )
        else:
            run.p_bodies, run.c_bodies = item_bodies(run.p_tree), item_bodies(
                run.c_tree,
            )
        try:
            run.c_report = items(run.c_tree)
        except ValueError as exc:  # an undecodable notes file, say
            run.report_error = str(exc)

    def _g1(self, run: _Run) -> None:
        man, pb, cb = run.man, run.p_bodies, run.c_bodies
        declared = set(man.support) | set(man.deleted_tests)
        declared |= {
            ch + f for ch in man.skeleton for f in ("/__init__.py", "/NOTES.md")
        }
        raw = run.manifest_raw if isinstance(run.manifest_raw, dict) else {}
        if self._library_checks() and isinstance(raw.get("fixtures"), dict):
            # the fixtures the harness recorded: the only files under tests/data/ a pass may add (library_checks)
            declared |= {p for p in raw["fixtures"] if isinstance(p, str)}
        for it in man.items:
            declared.add(it.path)
            declared.update(it.tests)
        listed_ids = {it.item for it in man.items}
        for eid in man.deleted:
            if eid not in pb:
                run.fail("G1", f"deleted item {eid} is not in the parent")
            elif eid in cb:
                run.fail("G1", f"deleted item {eid} is still in the candidate")
            else:
                declared.add(self._item_path(eid))
        for eid in man.unlisted:
            if self._unlisted_ok(run, eid):
                declared.add(self._item_path(eid))
        for t in man.deleted_tests:
            self._retired_ok(run, t)
        self._pins(run)
        if self._library_layout:
            # library mode: a package's __init__.py (its one-paragraph docstring) goes with what the pass declares there
            declared |= {
                f"memory/{p.split('/')[1]}/__init__.py"
                for p in list(declared)
                if p.startswith("memory/") and p.count("/") >= 2
            }
            # a folder's README.md is the writer's description of the folder, kept by any pass (library_checks)
            from .layout import classify as _classify

            declared |= {p for p in run.changed if _classify(p) == "readme"}
        for p in run.changed:
            if p not in declared:
                run.fail("G1", f"undeclared change {p}")
        accounted = listed_ids | set(man.deleted) | set(man.unlisted)
        for iid in sorted(set(pb) | set(cb)):
            if pb.get(iid) != cb.get(iid) and iid not in accounted:
                what = (
                    "added"
                    if iid not in pb
                    else "removed" if iid not in cb else "changed"
                )
                run.fail("G1", f"undeclared item {iid} ({what})")
        self._inputs(run)
        if self.docstring_standard:
            self._standard(run)
        for it in man.items:
            if cb.get(it.item, ("",))[0] != it.kind:
                run.fail(
                    "G1",
                    f"{it.item} ({it.kind}) is not in the candidate",
                    it.item,
                )
            # a function listed only to change its tests keeps the provenance it landed with (its own code is
            # unchanged); every other item names its source episodes
            if not it.source_episodes and not _unchanged(run, it):
                run.fail("G1", f"{it.item} names no source episode", it.item)
            for eid in it.source_episodes:
                if not self.ev.episode_exists(eid):
                    run.fail("G1", f"{it.item} cites unknown episode {eid}", it.item)

    @staticmethod
    def _doc_inputs(run: _Run) -> dict[str, str]:
        """Each candidate environment function's docstring ``Input:`` form ("" without one)."""
        if run.c_report is None:
            return {}
        return {
            i.item_id: i.input for i in run.c_report.items if i.kind == "env_function"
        }

    def _inputs(self, run: _Run) -> None:
        """A new or changed environment function declares its input; a declared input equals its one
        ``Input:`` line."""
        doc = self._doc_inputs(run)
        lines = {
            i.item_id: i.input_lines
            for i in (run.c_report.items if run.c_report is not None else [])
        }
        for it in run.man.items:
            if it.kind != "env_function" or it.item not in doc:
                continue  # an absent item (or an unreadable module) is refused elsewhere
            if lines.get(it.item, 0) > 1:
                run.fail("G1", f"{it.item} has more than one Input: line", it.item)
                continue
            if it.input is None:
                changed = (
                    run.p_bodies.get(it.item, ("", ""))[:2]
                    != run.c_bodies.get(it.item, ("", ""))[:2]
                )
                if changed:
                    run.fail(
                        "G1",
                        f"{it.item} declares no input (one of {', '.join(INPUT_KINDS)})",
                        it.item,
                    )
            elif doc[it.item] != it.input:
                run.fail(
                    "G1",
                    f"{it.item} declares input {it.input} but its docstring's Input: line says "
                    f"{doc[it.item] or 'missing'}",
                    it.item,
                )

    @staticmethod
    def _edited(run: _Run) -> list[str]:
        """The manifest's environment functions whose body (or kind) differs from the parent's, present now."""
        return [
            it.item
            for it in run.man.items
            if it.kind in ("env_function", "function")
            and it.item in run.c_bodies
            and run.p_bodies.get(it.item, ("", ""))[:2]
            != run.c_bodies.get(it.item, ("", ""))[:2]
        ]

    @staticmethod
    def _function_node(
        run: _Run,
        item: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """The candidate's definition of environment function *item* (None: unreadable, reported by G6)."""
        try:
            module = ast.parse((run.c_tree / item_path(item)).read_bytes())
        except (SyntaxError, ValueError, OSError):
            return None
        name = item.split(":", 1)[1]
        for node in module.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ):
                return node
        return None

    def _standard(self, run: _Run) -> None:
        """The lean docstring standard and self-explaining refusals (:mod:`.docstrings`), per new or
        changed environment function; each missing part is its own reason."""
        declared = {it.item: it.input for it in run.man.items}
        doc_inputs = self._doc_inputs(run)
        for item in self._edited(run):
            node = self._function_node(run, item)
            if node is None:
                continue
            doc = ast.get_docstring(node) or ""
            form = declared.get(item) or doc_inputs.get(item) or None
            name, channel = (
                item.split(":", 1)[1],
                item.split("/", 1)[1].split(":", 1)[0],
            )
            for problem in docstrings.problems(
                doc,
                docstrings.params(node),
                form,
                name=name,
                channel=channel,
            ):
                run.fail("G1", f"{item} docstring {problem}", item)
            for path in docstrings.fixture_paths(docstrings.parse(doc), channel):
                if path not in run.c_files:
                    run.fail(
                        "G1",
                        f"{item} docstring Example: reads {path}, which the commit does not hold",
                        item,
                    )
            for problem in docstrings.refusal_problems(node):
                run.fail("G1", f"{item} {problem}", item)

    def _unlisted_ok(self, run: _Run, eid: str) -> bool:
        """An unlisted item is present on both sides, unlisted now, and otherwise byte-for-byte the same."""
        pb, cb = run.p_bodies, run.c_bodies
        if eid not in pb or eid not in cb:
            run.fail(
                "G1",
                f"unlisted item {eid} is not in both the parent and the candidate",
            )
        elif cb[eid][2]:
            run.fail("G1", f"unlisted item {eid} is still listed")
        elif cb[eid][0] == "workflow":
            before = without_listed((run.p_tree / eid).read_bytes())
            if before != without_listed((run.c_tree / eid).read_bytes()):
                run.fail(
                    "G1",
                    f"unlisted workflow {eid} changed beyond its listed: line",
                )
            else:
                return True
        elif cb[eid][1] != pb[eid][1]:
            run.fail("G1", f"unlisted item {eid} changed beyond its listing")
        else:
            return True
        return False

    def _item_path(self, item: str) -> str:
        if self._library_layout:
            from . import layout

            return layout.item_path(item)
        return item_path(item)

    def _retired_ok(self, run: _Run, test: str) -> None:
        """A retired test file is removed and imports only items this manifest deletes.

        Library mode: removed only; whether it may go is the ``tests_changed`` reason rule (:mod:`.library_checks`,
        and :meth:`_suites`).
        """
        if test not in run.p_files or test in run.c_files:
            run.fail("G1", f"deleted test {test} is not removed by the candidate")
            return
        if self._library_layout:
            return
        named = _exercised((run.p_tree / test).read_bytes(), run.p_bodies)
        if named is None:
            run.fail("G1", f"deleted test {test} does not parse")
            return
        if not named:
            run.fail("G1", f"deleted test {test} imports no library item")
        elif not named <= set(run.man.deleted):
            others = sorted(named - set(run.man.deleted))[:5]
            run.fail(
                "G1",
                f"deleted test {test} also exercises {others}, which are not deleted",
            )

    def _pins(self, run: _Run) -> None:
        """Module skeletons and notes preambles change only when declared."""
        listed_ids = {it.item for it in run.man.items}
        everything = set(run.p_files) | set(run.c_files)

        def raw(files: dict, tree: Path, path: str) -> bytes | None:
            return (tree / path).read_bytes() if path in files else None

        for path in sorted(p for p in everything if MODULE_PATH.match(p)):
            before = module_skeleton(raw(run.p_files, run.p_tree, path))
            after = module_skeleton(raw(run.c_files, run.c_tree, path))
            if before == after:
                continue
            channel = path.rsplit("/", 1)[0]
            if channel not in run.man.skeleton:
                run.fail(
                    "G1",
                    f"module-level code of {channel} changed (docstring, imports, helpers or "
                    "statements) without a skeleton declaration",
                )
                continue
            missing = sorted(
                i
                for i, b in run.c_bodies.items()
                if b[0] == "env_function"
                and i.startswith(channel + ":")
                and i not in listed_ids
            )
            if missing:
                run.fail(
                    "G1",
                    f"the skeleton change of {channel} must list every public function; "
                    f"missing {missing[:5]}",
                )
        for path in sorted(p for p in everything if NOTES_PATH.match(p)):
            texts = [
                (b.decode("utf-8", "replace") if b is not None else None)
                for b in (
                    raw(run.p_files, run.p_tree, path),
                    raw(run.c_files, run.c_tree, path),
                )
            ]
            channel = path.rsplit("/", 1)[0]
            if (
                notes_preamble(texts[0]) != notes_preamble(texts[1])
                and channel not in run.man.skeleton
            ):
                run.fail(
                    "G1",
                    f"the preamble of {path} changed without a skeleton declaration of {channel}",
                )

    def _g2(
        self,
        run: _Run,
        *,
        preview: bool = False,
        lookup: Callable[[str, int], Action | None] | None = None,
    ) -> None:
        """*preview* (:meth:`preview`) checks the covers and their declared input form only: no held-out
        runs, no workflow signals.

        *lookup* replaces the gate's action lookup (the preview's memoised one).
        """
        lookup = lookup or self.lookup
        seen: list[Action] | None = (
            None  # the recorded actions of every episode the manifest names
        )
        library = self._library_checks()
        for n, it in enumerate(run.man.items):
            if library and it.kind == "function":
                # library mode: a function's examples are the calls its listed tests make, judged after G3's runs
                # (example_checks); its typed covers are validated by library_checks
                continue
            if it.kind in ("env_function", "function"):
                if not it.covers and not getattr(it, "typed_covers", None):
                    run.fail("G2", f"{it.item} covers no recorded action", it.item)
                valid: list[tuple[str, int, Action]] = []
                for eid, idx in it.covers:
                    action = lookup(eid, idx)
                    problem = cover_problem(
                        action,
                        it.channel,
                        self.blobs.has,
                        channel_rule=not self._library_layout,
                    )
                    if problem is not None:
                        run.fail(
                            "G2",
                            f"{it.item} covers ({eid},{idx}), {problem}",
                            it.item,
                        )
                    else:
                        run.covers.add((it.item, eid, idx))
                        valid.append((eid, idx, action))
                        if (
                            is_rejection(action)
                            and getattr(action, "kind", "tool") != "shell"
                        ):
                            run.rejections.add((eid, idx))
                declared = self._doc_inputs(run).get(it.item, "")
                form = it.input or (declared if declared in INPUT_KINDS else None)
                if _covers_a_write(valid, form):
                    run.writes.add(it.item)
                if valid:
                    self._rule(run, it.item, valid)
                if valid and all(
                    is_rejection(a) and getattr(a, "kind", "tool") != "shell"
                    for _, _, a in valid
                ):
                    run.fail(
                        "G2",
                        f"{it.item} covers only recorded rejections",
                        it.item,
                    )
                elif valid and preview:
                    # the structural input-form check only: no held-out perturbation
                    for why in _unfit_forms(valid, form)[:5]:
                        run.fail("G2", f"{it.item} {why}", it.item)
                elif valid:
                    if seen is None:
                        seen = seen_actions(self._named_episodes(run), self.lookup)
                    if (
                        self.surfacing == "catalogue"
                        or self.docstring_standard
                        or self._library_layout
                    ):
                        # the catalogue's shapes and the examples' fixture check
                        self._record_shapes(
                            run,
                            it.item,
                            valid,
                            it.input or (declared if declared in INPUT_KINDS else None),
                        )
                    self._held_out(
                        run,
                        it.item,
                        valid,
                        seen,
                        run.tmp / f"held-out-{n}",
                        it.field_types,
                        form,
                        self._pool(run, valid),
                    )
            elif it.kind == "workflow" and not preview:
                status = job_item_status(
                    it.item,
                    _WithSources(self.ev, it.item, it.source_episodes),
                    reader_promotes=self.reader_promotes,
                )
                if status != "promotable":
                    run.fail("G2", f"{it.item} is {status}", it.item)

    def _record_shapes(
        self,
        run: _Run,
        item: str,
        covers: list[tuple[str, int, Action]],
        input_kind: str | None,
    ) -> None:
        """The input-shape descriptors of *item*'s validated covers (:func:`.shape_rows.descriptor`), kept
        for the candidate's shape snapshot on a landed merge; never fails."""
        body = run.c_bodies.get(item)
        if body is None:
            return
        found = descriptors(
            [a for _, _, a in covers[:MAX_SHAPE_COVERS]],
            input_kind,
            self.blobs,
        )
        if found:
            run.shapes[item] = (body_digest(body[1]), found)

    def _snapshot(self, run: _Run) -> dict[str, dict]:
        """The candidate's shape snapshot: the parent's rows (backfilled where it has none, not frozen) for
        unchanged bodies plus this merge's shapes (:func:`.shape_rows.snapshot_rows`); never fails.
        """
        try:
            prev = shapes_at(
                self.mem,
                self.ev,
                run.parent,
                run.p_tree,
                lookup=self.lookup,
                blobs=self.blobs,
                library_mode=self._library_layout,
            )
            return snapshot_rows(
                run.c_tree,
                prev,
                run.shapes,
                library_mode=self._library_layout,
            )
        except (
            Exception
        ):  # noqa: BLE001 - shapes are a catalogue aid, never a gate reason
            return snapshot_rows(
                run.c_tree,
                {},
                run.shapes,
                library_mode=self._library_layout,
            )

    @staticmethod
    def _rule(run: _Run, item: str, valid: list[tuple[str, int, Action]]) -> None:
        """An added or changed function that replaces a computed value under a condition needs covers from
        :data:`RULE_EPISODES` episodes (module docstring, G2)."""
        if run.p_bodies.get(item, ("", ""))[:2] == run.c_bodies.get(item, ("", ""))[:2]:
            return
        found = [
            i for i in (run.c_report.items if run.c_report else []) if i.item_id == item
        ]
        if not found:
            return  # an absent item (or an unreadable module) is refused elsewhere
        info = found[0]
        if info.rule_unchecked:
            run.note(f"G2 rule check: {item} is too large to analyse; not checked")
        elif info.rule_line and len({eid for eid, _, _ in valid}) < RULE_EPISODES:
            run.fail(
                "G2",
                f"{item} replaces a value it computed from its input under a condition (line "
                f"{info.rule_line} of {info.path}); a function that encodes such a rule needs covers from at "
                f"least {RULE_EPISODES} episodes",
                item,
            )

    @staticmethod
    def _named_episodes(run: _Run) -> list[str]:
        eids = {e for it in run.man.items for e in it.source_episodes}
        eids |= {e for it in run.man.items for e, _ in it.covers}
        return sorted(eids)

    def _pool(
        self,
        run: _Run,
        covers: list[tuple[str, int, Action]],
    ) -> tuple[list[tuple[str, Action]], bool]:
        """The recorded observations on the covered channels: per channel, the evidence store's most recent
        :data:`.held_out.MAX_POOL_EPISODES` episodes touching it, in the store's order (independent of the
        pass), then the manifest's other episodes (bounded; whether a cap stopped the read).
        """
        channels = tuple(sorted({a.channel for _, _, a in covers}))
        if channels not in run.pools:
            stored: set[str] = set()
            for ch in channels:
                stored.update(self.ev.episode_ids_since(ch, 0)[-MAX_POOL_EPISODES:])
            eids = sorted(stored, key=self.ev.seq_of)
            eids += [e for e in self._named_episodes(run) if e not in stored]
            run.pools[channels] = pool_actions(eids, self.lookup, set(channels))
        return run.pools[channels]

    def _held_out(
        self,
        run: _Run,
        item: str,
        covers: list[tuple[str, int, Action]],
        seen: list[Action],
        work: Path,
        field_types: dict[str, str],
        input_kind: str | None = None,
        pool: tuple[list[tuple[str, Action]], bool] | None = None,
    ) -> None:
        """Scope is shape: the item must not refuse unseen values of its covered inputs' types.

        Each covered input reaches the item in its declared *input_kind* (None: the kind's convention). A
        field's constancy (identity or format) is judged on *pool*, the recorded observations of the covered
        channels across the evidence store, with whether its read cap was hit.

        A field declared with a semantic type is checked two-sided instead: unseen in-domain values
        accepted, an out-of-domain value refused before any environment call.
        """

        p = plan(
            item,
            covers,
            seen=seen,
            blob=self._blob,
            field_types=field_types,
            input_kind=input_kind,
            pool=None if pool is None else pool[0],
            pool_capped=pool is not None and pool[1],
        )
        verdict = run_plan(item, p, tree=run.c_tree, python=self.python, work=work)
        for f in verdict.refused[:5]:
            run.fail("G2", f"{item} held-out value refused: {f[:80]}", item)
        for text in verdict.failures[:5]:
            run.fail("G2", f"{item} {text}", item)
        for note in verdict.notes:
            run.note(f"G2 held-out values: {item} {note}")

    def _g3(self, run: _Run) -> None:
        man, changed = run.man, set(run.changed)
        new_tests = sorted({t for it in man.items for t in it.tests if t in changed})
        # an unchanged function listed under a skeleton change is held by the regression run and, if its
        # results change, by the behaviour check
        edited = {
            it.item
            for it in man.items
            if it.kind in ("env_function", "function")
            and run.p_bodies.get(it.item, ("", ""))[:2]
            != run.c_bodies.get(it.item, ("", ""))[:2]
        }
        # a clean-up pass may edit a function without a red test when an old test holds it
        cleanup = self._cleanup(run)
        unseen: set[str] = set()
        # already refused here: the behaviour check need not run them
        refused: set[str] = set()
        for it in man.items:
            if it.item in edited and not any(t in changed for t in it.tests):
                if cleanup:
                    unseen.add(it.item)
                else:
                    refused.add(it.item)
                    run.fail(
                        "G3",
                        f"{it.item} has no new or changed test. {RED_TEST_HINT}",
                        it.item,
                    )
        missing = [t for t in new_tests if t not in run.c_files]
        for t in missing:
            run.fail("G3", f"listed test {t} is not in the candidate", _owners(man, t))
        runnable = [t for t in new_tests if t not in missing]
        classic: set[str] = (
            set()
        )  # red on the parent's library with the candidate's test copied in
        if runnable:
            red_tree = run.tmp / "red"
            shutil.copytree(run.p_tree, red_tree)
            # the changed fixtures go in too, so that a test is red on the parent because the behaviour it checks
            # differs there, never because a fixture new in this pass is missing
            fixtures = sorted(p for p in run.changed if p in run.c_files and _is_fixture(p))
            for rel in runnable + man.support + fixtures:
                if rel in run.c_files:
                    (red_tree / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(run.c_tree / rel, red_tree / rel)
            targets = self._example_targets(run)
            for t in runnable:
                on_parent = self._pytest(red_tree, t, qa_env=run.qa_env)
                on_cand = (
                    self._pytest(run.c_tree, t, qa_env=run.qa_env, calls=targets)
                    if targets is not None
                    else self._pytest(run.c_tree, t, qa_env=run.qa_env)
                )
                run.qa_first[t] = on_cand
                # a parent library that hangs on the test is red once the candidate's run is green
                if _red(on_parent) or (on_parent.timed_out and _green(on_cand)):
                    classic.add(t)
                else:
                    # not red on the parent: a keeping test, accepted once green on the candidate; it never stands
                    # in for an edited function's red test
                    self._keeping_note(run, t, edited, on_cand, on_parent)
                if not _green(on_cand):
                    run.output(f"{t} (candidate)", on_cand)
                    run.fail(
                        "G3",
                        f"{t} is not green on the candidate ({_describe(on_cand)}) "
                        f"{_shown(on_cand)}",
                        _owners(man, t),
                    )
        # a function declared with a test that is red on the parent may change behaviour (each edited
        # function is seen changing by at least one such test)
        declared: set[str] = set()
        for it in man.items:
            tests = {t for t in it.tests if t in changed and t in run.c_files}
            if tests & classic:
                declared.add(it.item)
            elif it.item in edited and tests:
                if cleanup:
                    unseen.add(it.item)
                else:
                    refused.add(it.item)
                    run.fail(
                        "G3",
                        f"{it.item} is edited, but none of its tests is red on the parent's library. "
                        f"{RED_TEST_HINT}",
                        it.item,
                    )
        protected = self._suites(run)
        self._behaviour(run, unseen, declared, refused, protected)
        if self.docstring_standard:
            self._examples(run)

    def _example_targets(self, run: _Run) -> list[str] | None:
        """Library mode: the manifest's functions present in the candidate, whose calls G3's candidate runs record;
        None otherwise (nothing is recorded)."""
        if not self._library_checks():
            return None
        return sorted(
            {
                it.item
                for it in run.man.items
                if it.kind == "function" and it.item in run.c_bodies
            },
        )

    def _call_scope(self, run: _Run) -> set[str]:
        """Library mode: the stored functions of the parent in the changed modules and in every module importing
        one, whose name the candidate's module still binds (a ``def``, or an import or assignment such as an alias
        to another function); the behaviour check compares their calls in the parent's tests.
        """
        if run.graph is None:
            run.graph = layout.merge_graphs(
                layout.import_graph(run.p_tree),
                layout.import_graph(run.c_tree),
            )
        changed = {m for p in run.changed if (m := layout.module_of(p)) is not None}
        modules = layout.dependants(run.graph, changed)
        bound: dict[str, dict[str, bool]] = {}

        def still_bound(item: str) -> bool:
            module, name = item.split(":", 1)
            if module not in bound:
                rel = layout.module_path(module)
                bound[module] = (
                    public_bindings(
                        (run.c_tree / rel).read_bytes() if rel in run.c_files else None,
                    )
                    or {}
                )
            return name in bound[module]

        return {
            item
            for item, body in run.p_bodies.items()
            if body[0] == "function"
            and item.split(":", 1)[0] in modules
            and (item in run.c_bodies or still_bound(item))
        }

    def _behaviour(
        self,
        run: _Run,
        unseen: set[str],
        declared: set[str],
        refused: set[str],
        protected: set[str],
    ) -> None:
        """G3's behaviour check: any change in what a stored function returns on recorded inputs needs
        a failing-then-passing test.

        For every channel the pass changes (any file under ``env/<channel>/``), every public function of the
        parent's module (a ``def``, or a name bound by assignment or import that has recorded covers) whose
        name the candidate's module still binds runs confined on both versions (:meth:`_probe`,
        :meth:`_run_probe`), unless *declared* (a test of it in this pass is red on the parent) or already
        *refused*. A function that differs, or cannot be compared, on any input is refused. An edited
        function claiming the clean-up exemption (*unseen*) must also be called by a *protected* parent test
        and is compared on every recorded cover (none unread, at most :data:`.held_out.MAX_OUTPUT_COVERS`).
        Sets ``run.same``. A refused exemption belongs to its item; another refused function to the item
        :meth:`_behaviour_owner` names, else to the pass; a budget run out refuses the pass.

        Library mode: :meth:`_behaviour_library`, on the calls the parent's tests make.
        """
        if self._library_checks():
            self._behaviour_library(run, unseen, declared, refused, protected)
            return
        recorded: dict[str, set[tuple[str, int]]] = {}
        for it, e, i in self.ev.covers():
            recorded.setdefault(it, set()).add((e, i))
        # library-mode CURATE: a declared alias is compared on its own recorded inputs and its
        # target's, so an alias that answers differently is refused even when its old name has no test
        aliases = self._declared_aliases(run)
        for alias, target in aliases.items():
            recorded.setdefault(alias, set()).update(recorded.get(target, set()))
        probes: dict[str, _Probe] = {}
        for item in sorted(unseen):
            if self._held(run, item, protected):
                probes[item] = self._probe(run, item, recorded, strict=True)
            else:
                run.fail(
                    "G3",
                    self._exemption_refused(
                        item,
                        "no parent test that passed on the parent calls it",
                    ),
                    item,
                )
        for item in self._behaviour_targets(run, recorded, self._library_layout):
            if item in probes or item in unseen or item in declared or item in refused:
                continue
            probes[item] = self._probe(run, item, recorded, strict=False)
        for alias, target in sorted(aliases.items()):
            n = (
                len(probes[alias].cases)
                if alias in probes and probes[alias].why is None
                else 0
            )
            if n < 2:
                # fail closed: a declared alias the behaviour check cannot compare on two inputs is not landed
                run.fail(
                    "G3",
                    f"alias {alias} -> {target}: the behaviour check compares an alias on at least 2 recorded "
                    f"inputs (its own and its target's) and found {n}",
                    target if target in {it.item for it in run.man.items} else None,
                )
        total = sum(len(p.cases) for p in probes.values())
        if total > BEHAVIOUR_MAX_CASES:
            run.fail(
                "G3",
                f"behaviour check not completed: {total} recorded inputs of the changed channels' functions "
                f"exceed the {BEHAVIOUR_MAX_CASES} compared in one pass; change fewer channels or functions",
            )
            run.same = set()
            return
        budget = _Budget(BEHAVIOUR_BUDGET_S)
        same: set[str] = set()
        for k, item in enumerate(sorted(probes)):
            pr = probes[item]
            if pr.why is None and not self._run_probe(
                run,
                pr,
                run.tmp / f"same-{k}",
                budget,
            ):
                run.fail(
                    "G3",
                    f"behaviour check not completed: its {BEHAVIOUR_BUDGET_S:g} s budget ran out at {item}",
                )
                break
            why = pr.verdict()
            if why is None:
                same.add(item)
            elif pr.strict:
                run.fail("G3", self._exemption_refused(item, why), item)
            else:
                run.fail(
                    "G3",
                    f"{item} changes behaviour without a test: {why}; any change in what a stored function "
                    "returns on recorded inputs needs a test, listed under it in the manifest, that fails on "
                    "the parent's library and passes on the candidate's",
                    self._behaviour_owner(run, item, pr),
                )
            if pr.skipped:
                run.note(
                    f"G3 behaviour check: {pr.skipped} recorded covers of {item} cannot be read or given "
                    "and were not compared",
                )
        run.same = same

    def _behaviour_library(
        self,
        run: _Run,
        unseen: set[str],
        declared: set[str],
        refused: set[str],
        protected: set[str],
    ) -> None:
        """Library mode's behaviour check (:meth:`.example_checks.ExampleChecks.behaviour`): every stored function of
        the changed modules and their importers (:meth:`_call_scope`) must return, refuse and raise alike on every
        call the parent's tests make of it, on the parent's library and on the candidate's, unless a test of it in
        this pass is red on the parent (*declared*) or it is already *refused*. An edited function claiming the
        clean-up exemption (*unseen*) must be called by a *protected* parent test and is compared strictly.

        A stored function no parent test calls is compared on its stored covers as before when it has covers and
        its parent docstring an ``Input:`` line (a transition for functions stored before tests were their
        examples); with neither its behaviour cannot be compared, and it is refused.
        """
        from .example_checks import ExampleChecks

        ex = ExampleChecks(self, run)
        run.same = set()
        strict: set[str] = set()
        for item in sorted(unseen):
            if self._held(run, item, protected):
                strict.add(item)
            else:
                run.fail(
                    "G3",
                    self._exemption_refused(
                        item,
                        "no parent test that passed on the parent calls it",
                    ),
                    item,
                )
        scope = {
            i
            for i in self._call_scope(run)
            if i not in declared and i not in refused and i not in unseen
        }
        ex.behaviour(
            run.calls_base,
            run.calls_reg,
            scope | strict,
            strict=strict,
            owner=lambda i: self._behaviour_owner(run, i, _Probe(i, False)),
        )
        recorded: dict[str, set[tuple[str, int]]] = {}
        for it, e, i in self.ev.covers():
            recorded.setdefault(it, set()).add((e, i))
        probes: dict[str, _Probe] = {}
        for item in sorted(ex.uncalled):
            if item in strict:
                run.fail(
                    "G3",
                    self._exemption_refused(item, "no test of the parent calls it"),
                    item,
                )
            elif recorded.get(item) and self._parent_input(run, item) is not None:
                run.note(
                    f"G3 behaviour check: {item} has no example call in the parent's tests; compared on its stored "
                    "covers",
                )
                probes[item] = self._probe(run, item, recorded, strict=False)
            else:
                ex.no_comparison(item)
        budget = _Budget(BEHAVIOUR_BUDGET_S)
        for k, item in enumerate(sorted(probes)):
            pr = probes[item]
            if pr.why is None and not self._run_probe(
                run,
                pr,
                run.tmp / f"same-{k}",
                budget,
            ):
                run.fail(
                    "G3",
                    f"behaviour check not completed: its {BEHAVIOUR_BUDGET_S:g} s budget ran out at {item}",
                )
                break
            why = pr.verdict()
            if why is None:
                run.same.add(item)
            else:
                run.fail(
                    "G3",
                    f"{item} changes behaviour without a test: {why}; any change in what a stored function "
                    "returns on recorded inputs needs a test, listed under it in the manifest, that fails on "
                    "the parent's library and passes on the candidate's",
                    self._behaviour_owner(run, item, pr),
                )

    def _declared_aliases(self, run: _Run) -> dict[str, str]:
        """A library-mode CURATE gate's declared aliases (old id -> target) of the checked manifest; {} otherwise."""
        if not (
            self._library_checks()
            and getattr(self.library_mode, "role", "write") == "curate"
        ):
            return {}
        raw = run.manifest_raw if isinstance(run.manifest_raw, dict) else {}
        found = raw.get("aliases")
        if not isinstance(found, dict):
            return {}
        return {
            a: t for a, t in found.items() if isinstance(a, str) and isinstance(t, str)
        }

    @staticmethod
    def _exemption_refused(item: str, why: str) -> str:
        return (
            f"{item} is edited in a clean-up pass without a red test, and {why}; a change of behaviour "
            "needs a test that is red on the parent's library"
        )

    @staticmethod
    def _changed_channels(run: _Run) -> set[str]:
        """``env/<channel>`` of every channel with an added, edited or deleted file."""
        return {
            "/".join(p.split("/")[:2])
            for p in run.changed
            if p.startswith("env/") and p.count("/") >= 2
        }

    @staticmethod
    def _behaviour_targets(
        run: _Run,
        recorded: dict[str, set[tuple[str, int]]],
        library_mode: bool = False,
    ) -> list[str]:
        """The functions G3's behaviour check compares, in order, before :meth:`_behaviour`'s skips.

        **v2:** for each changed channel, every public function of its parent module whose name the candidate's
        module still binds: a ``def``, or a name bound by assignment or import that has recorded covers.

        **Library mode**: the same rule over each changed library module and every module that imports
        one, directly or through others. The scope uses :func:`..layout.dependants` over the import graphs of
        both versions. The graph is kept on the run for :meth:`_behaviour_owner`.
        """
        if library_mode:
            run.graph = layout.merge_graphs(
                layout.import_graph(run.p_tree),
                layout.import_graph(run.c_tree),
            )
            changed = {m for p in run.changed if (m := layout.module_of(p)) is not None}
            scope = [
                (m, layout.module_path(m))
                for m in sorted(layout.dependants(run.graph, changed))
            ]
        else:
            scope = [
                (ch, f"{ch}/__init__.py") for ch in sorted(Gate._changed_channels(run))
            ]
        out: list[str] = []
        for prefix, module in scope:
            before = public_bindings(
                (run.p_tree / module).read_bytes() if module in run.p_files else None,
            )
            after = public_bindings(
                (run.c_tree / module).read_bytes() if module in run.c_files else None,
            )
            for name, is_def in sorted((before or {}).items()):
                item = f"{prefix}:{name}"
                if after is None or name not in after:
                    continue  # deleted: its recorded covers stay covered under G5
                if is_def or item in recorded:
                    out.append(item)
        return out

    @staticmethod
    def _source_channels(acts: list[tuple[str, int, Action]]) -> list[str]:
        """Library mode: the memory channels (``env/<name>``) of a function's recorded covers. They stand for the channel
        its path named in v2 (the source channel moves to the item's record).
        """
        out: set[str] = set()
        for _, _, a in acts:
            kind = getattr(a, "kind", "tool")
            if kind == "shell":
                continue
            ch = env_channel(kind, a.channel)
            if ch:
                out.add(f"env/{ch}")
        return sorted(out)

    @staticmethod
    def _behaviour_owner(run: _Run, item: str, pr: _Probe) -> str | None:
        """The one manifest item a behaviour change of *item* found by :meth:`_behaviour` is attributable to
        (per-item admission); None refuses the pass whole.

        A function reaching this check without a red test is unchanged, or bound by assignment or import (an
        edited function is red-tested, refused or held to the clean-up exemption before it), so the change
        comes from elsewhere in its channel. It is attributable when the pass's only change to that channel's
        module is one environment function in ``items`` whose body it adds or edits: no ``skeleton`` change of
        the channel and no ``deleted`` or ``unlisted`` item there. Refusing that item restores the parent's
        function and refuses its callers (:mod:`.reduction`), and the reduced candidate runs the behaviour
        check again. A parent library that cannot be read is no item's. In library mode the edits that count are
        those in the item's module or in a library module it imports, directly or through others.
        """
        if pr.why == "the parent's library cannot be read":
            return None
        if (
            run.graph is not None
        ):  # library mode: what the item's module imports, transitively
            related = layout.imported_closure(run.graph, item.split(":", 1)[0])
            touched = (*run.man.deleted, *run.man.unlisted)
            if any(s in related for s in run.man.skeleton) or any(
                o.split(":", 1)[0] in related for o in touched
            ):
                return None
            owners = [e for e in Gate._edited(run) if e.split(":", 1)[0] in related]
            return owners[0] if len(owners) == 1 else None
        channel = item.split(":", 1)[0]
        if channel in run.man.skeleton or any(
            other.split(":", 1)[0] == channel
            for other in (*run.man.deleted, *run.man.unlisted)
        ):
            return None
        owners = [e for e in Gate._edited(run) if e.split(":", 1)[0] == channel]
        return owners[0] if len(owners) == 1 else None

    def _examples(self, run: _Run) -> None:
        """Every ``>>>`` example of each new or changed environment function, run as a doctest in one
        confined pytest run on the candidate tree (G3's runner); a failing or unrun example refuses it.
        """
        targets: list[str] = []
        for item in self._edited(run):
            node = self._function_node(run, item)
            if node is not None and docstrings.has_example(
                docstrings.parse(ast.get_docstring(node) or ""),
            ):
                targets.append(item)
        if not targets:
            return
        for item in targets:
            self._fixture_shape(run, item)
        tree = run.tmp / "examples"
        shutil.copytree(run.c_tree, tree)
        (tree / EXAMPLES_TEST).parent.mkdir(parents=True, exist_ok=True)
        (tree / EXAMPLES_TEST).write_text(_examples_source(targets))
        # set and dict orders in an example's output must not vary between runs
        outcome = self._pytest(tree, EXAMPLES_TEST, {"PYTHONHASHSEED": "0"})
        readable = outcome.valid and not outcome.timed_out

        def ran(ids: set[str], n: int) -> bool:
            return any(t.rsplit("::", 1)[-1] == f"test_example_{n}" for t in ids)

        for n, item in enumerate(targets):
            if not readable:
                run.output("docstring examples (candidate)", outcome)
                run.fail(
                    "G3",
                    f"the examples of {item} could not be run ({_describe(outcome)}) "
                    f"{_shown(outcome)}",
                )
            elif ran(outcome.failed, n) or not ran(outcome.passed, n):
                run.output("docstring examples (candidate)", outcome)
                run.fail(
                    "G3",
                    f"an example in the docstring of {item} fails as a doctest "
                    f"{_shown(outcome)}",
                    item,
                )

    def _fixture_shape(self, run: _Run, item: str) -> None:
        """An example's fixture must have the shape of an input *item* was admitted on (its validated
        covers' shapes, :meth:`_record_shapes`); an item without recorded shapes (``env``) is not checked.

        The shape is compared exactly (:func:`.memory_helper.same_shape`, no information floor, so a
        headerless table, a raw grid or a list of scalars passes with an identical-shape fixture), or, for
        keyed shapes, by ``find``'s structural match (:func:`.memory_helper.compare`), which admits a
        fixture that is a partial copy of a recorded record.
        """
        recorded = run.shapes.get(item, ("", []))[1]
        node = self._function_node(run, item)
        if not recorded or node is None:
            return
        channel = item.split("/", 1)[1].split(":", 1)[0]
        paths = [
            p
            for p in docstrings.fixture_paths(
                docstrings.parse(ast.get_docstring(node) or ""),
                channel,
            )
            if p in run.c_files
        ]
        for path in paths:
            data = (run.c_tree / path).read_bytes()
            mine = [
                d for d in (file_shape(path, data), value_shape(data)) if d is not None
            ]
            # a fixture is also a collection of recorded inputs (a JSON list, records by name, one level of
            # wrapping, or JSON lines) whose examples pick one: each element is shaped on its own
            mine += [
                d for d in map(value_shape, _fixture_elements(data)) if d is not None
            ]
            if any(fixture_fits(d, r) for d in mine for r in recorded):
                return
        if paths:
            run.cap(
                "G3",
                f"the examples of {item} read no fixture shaped like an input it covers "
                f"({', '.join(paths[:5])})",
                item,
            )

    def _keeping_note(
        self,
        run: _Run,
        test: str,
        edited: set[str],
        on_cand: PytestOutcome,
        on_parent: PytestOutcome,
    ) -> str:
        """Note a new or changed test that is not red on the parent: a keeping test. It refuses nothing.

        It is accepted once green on the candidate (checked by the caller), and it never counts as an edited
        function's red test. The note names the test, which of three it is (it passes on the parent and records
        existing behaviour; it repairs a test the parent had red, :meth:`_repaired`; or it cannot be compared on the
        parent), and the library functions it exercises (:func:`_exercised`), with those this pass edits.
        """
        # a repaired test also passes on the parent's library once copied there: the repair is named first
        if self._repaired(run, test, on_cand):
            why = "repairs a test the parent had red"
        elif _green(on_parent):
            why = "passes on the parent: it records existing behaviour"
        else:
            why = f"not comparable on the parent ({_describe(on_parent)})"
        try:
            source = (run.c_tree / test).read_bytes()
        except OSError:
            source = b""
        uses = _exercised(source, run.p_bodies, run.c_bodies)
        shown = sorted(uses or ())
        line = f"G3 keeping test: {test} {why}"
        if shown:
            line += f"; it exercises {', '.join(shown[:5])}" + (", …" if len(shown) > 5 else "")
            carried = sorted(edited & set(shown))
            if carried:
                line += (
                    f"; it is not the red test of {', '.join(carried[:5])}, which this pass edits"
                )
        run.note(line)
        return why

    def _repaired(self, run: _Run, test: str, on_cand: PytestOutcome) -> bool:
        """The parent's own *test* file is already red on the parent and the candidate's version repairs it.

        A readable parent report must name a failed test that passes on the candidate; a collection error
        names no test, so it counts for the whole file. (The candidate's run must also be green.) Used
        only for the keeping test's note (:meth:`_keeping_note`).
        """
        if test not in run.p_files:
            return False
        old = self._pytest(run.p_tree, test, qa_env=run.qa_env)
        if not _red(old):
            return False
        return not old.valid or bool(old.failed & on_cand.passed)

    def _regression_tree(self, run: _Run) -> Path | None:
        """The parent tree (its tests and test kit) with only the candidate's channel modules and notes.

        None when it would equal the parent tree. Retired test files are left out.

        Library mode: the candidate's library modules, package inits and notes take the place of the parent's.
        """
        if self._library_layout:
            from .layout import classify

            def library(p: str) -> bool:
                return classify(p) in ("module", "package_init", "note")

        else:

            def library(p: str) -> bool:
                return bool(MODULE_PATH.match(p) or NOTES_PATH.match(p))

        swapped = sorted(
            p
            for p in set(run.p_files) | set(run.c_files)
            if library(p) and run.p_files.get(p) != run.c_files.get(p)
        )
        retired = set(run.man.deleted_tests)
        if not swapped and not retired:
            return None
        tree = run.tmp / "regression"
        shutil.copytree(run.p_tree, tree)
        for rel in swapped:
            if rel in run.c_files:
                (tree / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(run.c_tree / rel, tree / rel)
            else:
                (tree / rel).unlink()
        for rel in retired:
            if (tree / rel).exists():
                (tree / rel).unlink()
        return tree

    def _suites(self, run: _Run) -> set[str]:
        """Per channel, compare against the parent's suite (the baseline).

        A readable baseline protects every test it passed outside retired files. Each must pass in two
        runs:

        * the candidate's own suite, on the candidate tree, which must also be readable and fail only
          tests the baseline failed (a rewritten test file or test kit cannot shrink the net silently);
        * the isolated regression run (:meth:`_regression_tree`): the parent's tests and test kit against
          the candidate's library (a changed test kit or helper cannot mask a broken old contract).

        An unreadable or timed-out baseline protects nothing and skips the regression run. In a channel the
        pass touches, the candidate's suite must then be readable and free of failures (the repair path);
        in an untouched channel, a suite that stays broken is noted as pre-existing. Failures the
        baseline already had and the candidate keeps are noted, not failed. Each run is bounded separately.

        Returns the parent test files with a protected test (a file whose protected tests are lost fails
        here, so on a passing gate each still passes in both runs).

        Library mode (``checks``; "no unexplained test loss"): each package's suite
        ``memory/<package>/tests`` is compared the same way, its protected files are returned as paths from the
        tree's root, and a deleted test file is retired only with a reason in the manifest's ``tests_changed``
        (else its lost tests refuse here). A test of an alias is collected through the alias: listed under the
        alias's target, its failures belong to that item.
        """
        protected: set[str] = set()
        if not run.changed:
            return protected
        library_mode = self._library_checks()
        retired = set(run.man.deleted_tests)
        if library_mode:
            raw = run.manifest_raw if isinstance(run.manifest_raw, dict) else {}
            stated = (
                raw.get("tests_changed")
                if isinstance(raw.get("tests_changed"), dict)
                else {}
            )
            retired = {
                t
                for t in retired
                if isinstance(stated.get(t), str) and stated[t].strip()
            }
        regression = self._regression_tree(run)
        touched = {p.split("/")[1] for p in run.changed if p.startswith("env/")}

        def file_of(t: str, rel: str) -> str:
            return (
                _test_file(t, rel, self._library_layout)
                if library_mode
                else t.split("::", 1)[0]
            )

        def kept(ids: set[str], rel: str) -> set[str]:
            return {t for t in ids if file_of(t, rel) not in retired}

        channels = sorted(
            {
                p.split("/")[1]
                for p in set(run.p_files) | set(run.c_files)
                if p.startswith("env/") and p.split("/")[2:3] == ["tests"]
            },
        )
        suites = [(f"env/{ch}/tests", ch in touched, f"env/{ch}") for ch in channels]
        if library_mode:
            from .layout import classify

            pkgs = sorted(
                {
                    p.split("/")[1]
                    for p in set(run.p_files) | set(run.c_files)
                    if classify(p) in ("test", "test_helper", "fixture")
                },
            )
            changed_pkgs = {
                p.split("/")[1]
                for p in run.changed
                if p.startswith("memory/") and p.count("/") >= 2
            }
            suites += [
                (f"memory/{pkg}/tests", pkg in changed_pkgs, f"memory/{pkg}")
                for pkg in pkgs
            ]
        # library mode: the calls of the stored functions in scope (and of deleted ones) that the parent's tests make,
        # on the parent's library and on the candidate's (behaviour, G3), and of every candidate function in the
        # candidate's suites when items are deleted (their fixtures stay used, G5); example_checks
        scope = sorted(self._call_scope(run)) if library_mode else []
        base_targets = sorted(
            (
                {*scope, *(d for d in run.man.deleted if d in run.p_bodies)}
                if library_mode
                else set()
            ),
        )
        cand_targets = (
            sorted(i for i, b in run.c_bodies.items() if b[0] == "function")
            if library_mode and run.man.deleted
            else []
        )

        def calls_for(rel: str, targets: list[str]) -> dict:
            return {"calls": targets} if targets and rel.startswith("memory/") else {}

        for rel, is_touched, area in suites:
            before: set[str] = set()  # protected: passed on the parent
            known_red: set[str] = set()  # failed on the parent
            broken: str | None = None  # the baseline cannot be read
            if (run.p_tree / rel).is_dir():
                base = self._pytest(
                    run.p_tree,
                    rel,
                    qa_env=run.qa_env,
                    **calls_for(rel, base_targets),
                )
                if (
                    getattr(base, "calls", None) is not None
                    and not base.timed_out
                    and base.valid
                ):
                    run.calls_base[rel] = base.calls
                if base.timed_out or not base.valid:
                    broken = _describe(base)
                    run.note(
                        f"the parent's suite {rel} is unreadable ({broken}): it protects no test "
                        "and its regression run is skipped",
                    )
                else:
                    before, known_red = kept(base.passed, rel), kept(base.failed, rel)
                    protected |= {file_of(t, rel) for t in before}
            after: set[str] = set()
            if (run.c_tree / rel).is_dir():
                suite = self._pytest(
                    run.c_tree,
                    rel,
                    qa_env=run.qa_env,
                    **calls_for(rel, cand_targets),
                )
                if getattr(suite, "calls", None) is not None:
                    run.calls_cand[rel] = suite.calls
                readable = suite.valid and not suite.timed_out
                new_red = sorted(suite.failed - known_red)
                if (
                    broken is not None
                    and not is_touched
                    and not (readable and not suite.failed)
                ):
                    run.note(
                        f"pre-existing: suite {rel} is still not green ({_describe(suite)}); "
                        f"this pass does not touch {area}",
                    )
                elif not readable or new_red:
                    run.output(f"suite {rel} (candidate)", suite)
                    repair = (
                        "; the parent's suite was unreadable, so the candidate's must be green"
                        if broken is not None
                        else ""
                    )
                    # new failures only in test files the manifest's items list are those items' own
                    # (per-item admission); anything else refuses the whole pass
                    owners = [
                        _owners(run.man, _test_file(t, rel, self._library_layout))
                        for t in new_red
                    ]
                    run.fail(
                        "G3",
                        f"suite {rel} has new failures or is unreadable on the candidate "
                        f"({_describe(suite)}; new {new_red[:5]}{repair}) {_shown(suite)}",
                        (
                            sorted({o for found in owners for o in found})
                            if readable and all(owners)
                            else None
                        ),
                    )
                elif suite.failed:
                    run.note(
                        f"pre-existing: {rel} still fails {sorted(suite.failed)[:5]}, as on the parent",
                    )
                after = suite.passed
            lost = sorted(before - after)
            if lost:
                run.fail(
                    "G3",
                    f"the candidate's suite {rel} no longer passes the parent's tests {lost[:5]}",
                )
            if regression is not None and (before or (rel in run.calls_base and scope)):
                reg = self._pytest(
                    regression,
                    rel,
                    qa_env=run.qa_env,
                    **calls_for(rel, scope),
                )
                if getattr(reg, "calls", None) is not None:
                    run.calls_reg[rel] = reg.calls
                if reg.timed_out or not reg.valid:
                    run.output(f"regression run of {rel}", reg)
                    run.fail(
                        "G3",
                        f"the regression run of {rel} is unreadable ({_describe(reg)}) "
                        f"{_shown(reg)}",
                    )
                lost = sorted(before - reg.passed)
                if lost:
                    run.output(f"regression run of {rel}", reg)
                    run.fail(
                        "G3",
                        "the parent's tests no longer pass against the candidate's library: "
                        f"{lost[:5]}",
                    )
        return protected

    @staticmethod
    def _shrinks(run: _Run) -> bool:
        """No measure of :func:`_library_size` grows, and function definitions or their AST nodes fall."""
        before = _library_size(run.p_files, run.p_tree)
        after = _library_size(run.c_files, run.c_tree)
        if before is None or after is None:
            return False
        return after[:2] != before[:2] and all(a <= b for a, b in zip(after, before))

    def _cleanup(self, run: _Run) -> bool:
        """A clean-up pass: it adds no item and the library shrinks.

        Library mode: a CURATE pass is the clean-up pass, by its role, and a WRITE pass never is. The clean-up size
        measure reads v2's channel modules only; in library mode, the behaviour check's strict comparison on the import-graph scope is
        what proves each exempted edit keeps behaviour, and CURATE's growth rule (:meth:`.library_checks.LibraryChecks.curate`)
        bounds the library.
        """
        role = getattr(getattr(self, "library_mode", None), "role", None)
        if role is not None:
            return role == "curate"
        added = any(it.item not in run.p_bodies for it in run.man.items)
        return not added and self._shrinks(run)

    @staticmethod
    def _held(run: _Run, item: str, protected: set[str]) -> bool:
        """A parent test file that passed on the parent, and still passes, calls *item*.

        A call, not an import: :func:`.snapshot.calls_item`, or :func:`.aliases.calls_function` for a library id.
        """
        try:
            from . import layout
        except ImportError:  # a build without the library layout has no library ids
            layout = None
        if layout is not None and layout.FUNCTION_ID.match(item):
            from .aliases import calls_function

            return any(
                rel in run.p_files
                and calls_function(
                    (run.p_tree / rel).read_bytes(),
                    item,
                    layout.package_of_path(rel),
                )
                for rel in sorted(protected)
            )
        return any(
            rel in run.p_files and calls_item((run.p_tree / rel).read_bytes(), item)
            for rel in sorted(protected)
        )

    def _blob(self, sha: str) -> bytes:
        if not self.blobs.has(sha):
            raise KeyError(sha)
        return self.blobs.get(sha)

    def _forms(self, run: _Run, item: str) -> tuple[str | None, str | None] | None:
        """*item*'s declared input form in the parent and in the candidate (None: the kind's convention).

        A candidate name bound by assignment or import has no docstring of its own and keeps the parent's
        form. None when the parent's library cannot be read.

        Library mode: each tree's function docstring ``Input:`` line (the manifest's declared input first, for the
        candidate); v2's report reads only environment functions, which left every library-mode form unknown.
        """
        if self._library_layout:
            # library mode reads no Input: line, except the parent's for the behaviour check's transition path
            form = self._parent_input(run, item)
            return form, form
        if run.p_inputs is None:
            try:
                p_report = items(run.p_tree)
            except ValueError:
                run.p_inputs = False
            else:
                run.p_inputs = {
                    i.item_id: i.input
                    for i in p_report.items
                    if i.kind == "env_function"
                }
        if run.p_inputs is False:
            return None

        def known(v: str | None) -> str | None:
            return v if v in INPUT_KINDS else None

        p_form = known(run.p_inputs.get(item, ""))
        c_docs = self._doc_inputs(run)
        return p_form, (known(c_docs[item]) if item in c_docs else p_form)

    @staticmethod
    def _parent_input(run: _Run, item: str) -> str | None:
        """Library mode: the ``Input:`` line of *item*'s docstring in the parent, for a function stored before tests
        were its examples (the behaviour check's transition path); None when it has none.
        """
        try:
            module = ast.parse((Path(run.p_tree) / layout.item_path(item)).read_bytes())
        except (OSError, SyntaxError, ValueError):
            return None
        name = item.split(":", 1)[1]
        for node in module.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ):
                m = re.search(
                    r"^\s*Input:\s*([a-z]+)\s*$",
                    ast.get_docstring(node) or "",
                    re.M,
                )
                return m.group(1) if m and m.group(1) in INPUT_KINDS else None
        return None  # bound by assignment or import: no docstring of its own

    def _episode_inputs(self, run: _Run, channel: str) -> list[tuple[str, int, Action]]:
        """The recorded actions on *channel* (``env/<name>``) in the pass's own episodes (its manifest's source
        and cover episodes), shell commands aside: at most :data:`MAX_EPISODE_INPUTS`, chosen by hash.
        """
        if channel not in run.episode_inputs:
            name = channel.split("/", 1)[1]
            found: list[tuple[str, int, Action]] = []
            read = 0
            for eid in self._named_episodes(run):
                for i in range(MAX_ACTIONS_PER_EPISODE):
                    if read >= MAX_POOL_ACTIONS:
                        break
                    a = self.lookup(eid, i)
                    read += 1
                    if a is None:
                        break
                    kind = getattr(a, "kind", "tool")
                    if kind != "shell" and env_channel(kind, a.channel) == name:
                        found.append((eid, i, a))
            keep = _sample({(e, i) for e, i, _ in found}, channel, MAX_EPISODE_INPUTS)
            run.episode_inputs[channel] = [c for c in found if (c[0], c[1]) in keep]
        return run.episode_inputs[channel]

    def _probe(
        self,
        run: _Run,
        item: str,
        recorded: dict[str, set[tuple[str, int]]],
        *,
        strict: bool,
    ) -> _Probe:
        """The recorded inputs *item* is compared on, without running anything.

        Its recorded covers (the evidence store's, *recorded* by item, and this manifest's) and the recorded actions of its
        channel in the pass's own episodes (:meth:`_episode_inputs`, so that a repair of an input no cover
        holds is seen), each in the parent's declared input form. Not *strict*: beyond
        :data:`.held_out.MAX_OUTPUT_COVERS` covers a hash-seeded sample, and a cover that cannot be read or
        given is skipped (counted for a note). *strict* (an edited function claiming the clean-up exemption):
        it needs a recorded cover, at most the cap, every one read and given (a recorded rejection that gives
        no input aside). Either way an action that cannot be given in the declared form is not compared, and
        a changed input form, or a parent library that cannot be read, is a difference.
        """
        pr = _Probe(item, strict)
        covers = set(recorded.get(item, ()))
        covers |= {
            (e, i) for it in run.man.items if it.item == item for e, i in it.covers
        }
        if strict and not covers:
            pr.why = "it has no recorded cover to compare its results on"
            return pr
        if strict and len(covers) > MAX_OUTPUT_COVERS:
            pr.why = f"its {len(covers)} recorded covers exceed the {MAX_OUTPUT_COVERS} compared"
            return pr
        covers = _sample(covers, item, MAX_OUTPUT_COVERS)
        forms = self._forms(run, item)
        if forms is None:
            pr.why = "the parent's library cannot be read"
            return pr
        p_form, c_form = forms
        acts: list[tuple[str, int, Action]] = []
        unread: list[list] = []
        for e, i in sorted(covers):
            a = self.lookup(e, i)
            if a is None:
                unread.append([e, i])
            else:
                acts.append((e, i, a))
        if unread and strict:
            pr.why = (
                f"{len(unread)} of its recorded covers cannot be read: {unread[:5]}"
            )
            return pr
        pr.skipped += len(unread)
        given = {(e, i) for e, i, _ in acts}
        if (
            self._library_layout
        ):  # the recorded actions of its covers' channels in this pass's episodes
            pool = [
                c
                for ch in self._source_channels(acts)
                for c in self._episode_inputs(run, ch)
            ]
        else:
            pool = self._episode_inputs(run, item.split(":", 1)[0])
        acts += [c for c in pool if (c[0], c[1]) not in given]
        if any(_form(a, p_form) != _form(a, c_form) for _, _, a in acts):
            pr.why = "its input form changed"
            return pr
        cases, unfit = output_cases(acts, self._blob, p_form)
        rejected = {
            (e, i)
            for e, i, a in acts
            if is_rejection(a) and getattr(a, "kind", "tool") != "shell"
        }
        unfit_covers = [list(c) for c in unfit if c in given and c not in rejected]
        if unfit_covers and strict:
            pr.why = (
                f"{len(unfit_covers)} of its recorded covers cannot be given to it (a shell cover, another "
                f"input form or an unreadable file): {unfit_covers[:5]}"
            )
            return pr
        pr.skipped += len(unfit_covers)
        pr.cases = cases
        pr.covers = {c.cover for c in cases if c.cover in given}
        if strict and not pr.covers:
            pr.why = "none of its recorded covers gives it an input to compare its results on"
        return pr

    def _run_probe(self, run: _Run, pr: _Probe, work: Path, budget: _Budget) -> bool:
        """Run *pr*'s cases confined on the parent's and the candidate's *item*, through G2's runner and
        replay (:func:`.held_out.run_outputs`): the return value (canonical JSON), refusal, raised exception
        class and issued environment calls must be equal. Records the differing and the uncomparable cases;
        False when the *budget* ran out (the check is then not completed).
        """
        if not pr.cases:
            return True
        results = []
        for side, tree in (("parent", run.p_tree), ("candidate", run.c_tree)):
            left = budget.left()
            if left <= 0:
                return False
            results.append(
                run_outputs(
                    pr.item,
                    pr.cases,
                    tree=tree,
                    python=self.python,
                    work=work / side,
                    timeout_s=min(left, OUTPUTS_BUDGET_S),
                ),
            )
        if budget.left() <= 0:
            return False
        before, after = results
        pr.unjudged = sorted(
            c for c in before if before[c] is None or after.get(c) is None
        )
        pr.differ = sorted(
            c for c in before if c not in pr.unjudged and before[c] != after[c]
        )
        return True

    def _g4(self, run: _Run) -> None:
        """The size budget: the v2 index's hard cap, or with ``soft_budget`` a note that hygiene is due past
        it (never a refusal of growth)."""
        if not self.soft_budget:
            try:
                build_index(run.c_tree, budget_tokens=self.budget)
            except IndexOverBudget as exc:
                run.fail("G4", str(exc))
            except ValueError as exc:
                run.fail("G4", f"index not built: {exc}")
            return
        catalogue = self.surfacing == "catalogue"
        what = "catalogue (README and channel lines)" if catalogue else "index"
        try:
            if catalogue:
                size = catalogue_tokens(run.c_tree)
            else:
                size = estimate_tokens(
                    build_index(run.c_tree, budget_tokens=sys.maxsize),
                )
        except ValueError as exc:  # an undecodable notes file, say
            run.fail("G4", f"{'catalogue' if catalogue else 'index'} not built: {exc}")
            return
        if size > self.budget:
            run.note(
                f"G4 hygiene due: the {what} is {size} tokens, over its "
                f"soft budget of {self.budget}; growth is not refused",
            )

    def _g5(self, run: _Run) -> None:
        def size(files: dict, tree: Path) -> int:
            return sum((tree / p).stat().st_size for p in files if MODULE_PATH.match(p))

        if size(run.c_files, run.c_tree) > size(run.p_files, run.p_tree):
            validated = {(eid, idx) for _, eid, idx in run.covers} - run.rejections
            if not validated - self.ev.covered() and not self._shrinks(run):
                run.fail(
                    "G5",
                    "the library grew without covering any new recorded call",
                )
        if self._library_checks():
            # library mode: a deleted function's fixtures stay used by a remaining function's examples
            from .example_checks import ExampleChecks

            parent_calls: dict[str, list] = {}
            for rec in run.calls_base.values():
                for c in rec.calls:
                    parent_calls.setdefault(c.item, []).append(c)
            ExampleChecks(self, run).retained(parent_calls)
            return
        # a deleted function's recorded covers must stay covered by a remaining item
        deleted = set(run.man.deleted)
        recorded = self.ev.covers()
        held = {(e, i) for _, e, i in run.covers}

        changed = self._changed_channels(run)
        scope: set[str] | None = None
        if (
            self.library_mode is not None
            and self.library_mode.layout
            and run.same is not None
        ):
            # library mode: the modules this pass changed, and every module that imports one of them
            from . import layout

            graph = getattr(run, "graph", None)
            if graph is None:
                graph = layout.merge_graphs(
                    layout.import_graph(run.p_tree),
                    layout.import_graph(run.c_tree),
                )
            scope = layout.dependants(
                graph,
                {m for p in run.changed if (m := layout.module_of(p)) is not None},
            )

        def keeps(it: str) -> bool:
            """A kept item still does on its recorded covers what it did: its channel (library mode: its module's
            import-graph scope) is unchanged and so is its body, or G3's behaviour check showed it doing the same. A
            function changed under a red test holds only the covers this manifest lists for it (already in
            ``held``)."""
            if it in deleted or it not in run.c_bodies:
                return False
            if run.same is None:  # a preview runs no G3
                return True
            where = it.split(":", 1)[0]
            if where not in (scope if scope is not None else changed):
                return run.p_bodies.get(it, ("", ""))[:2] == run.c_bodies[it][:2]
            return it in run.same

        held |= {(e, i) for it, e, i in recorded if keeps(it)}
        lost: dict[str, list[tuple[str, int]]] = {}
        for it, e, i in sorted(recorded):
            if it not in deleted or (e, i) in held:
                continue
            action = self.lookup(e, i)
            if action is None:
                run.note(f"G5: {it} covered ({e},{i}), which can no longer be read")
            elif not is_rejection(action) or getattr(action, "kind", "tool") == "shell":
                # a recorded rejection need not be taken over (a shell output is no rejection)
                lost.setdefault(it, []).append((e, i))
        for it, gone in lost.items():
            run.fail(
                "G5",
                f"deleted item {it} covered {len(gone)} recorded inputs that no remaining item covers "
                f"(list them in a remaining item's covers): {[list(c) for c in gone[:5]]}",
            )

    def _g6(self, run: _Run) -> None:
        if run.report_error is not None:
            run.fail("G6", f"items unreadable: {run.report_error}")
        if run.c_report is not None:
            for e in run.c_report.errors:
                run.fail("G6", e)
            changed = set(run.changed)
            for path in sorted(changed & set(run.c_files)):
                if MODULE_PATH.match(path):
                    for name in duplicate_public_defs((run.c_tree / path).read_bytes()):
                        run.fail(
                            "G6",
                            f"{path} defines public function {name} more than once",
                        )
            listed_ids = {it.item for it in run.man.items}
            effects = {}
            for i in run.c_report.items:
                if i.kind != "env_function":
                    continue
                effects[i.item_id] = i.effect
                if i.path in changed and not i.effect:
                    run.fail(
                        "G6",
                        f"{i.item_id} has no Effect: line",
                        i.item_id if i.item_id in listed_ids else None,
                    )
            # structural Effect: an environment function that covers a recorded write call declares it. A
            # library function in run.writes has no Effect: line to check; its effect is derived and recorded
            # instead (library_checks.derive_effect)
            for item in sorted(run.writes & set(effects)):
                if effects[item] != "write":
                    run.fail(
                        "G6",
                        f"{item} covers a recorded write call, so it must declare Effect: write "
                        f"(it declares {effects[item] or 'none'})",
                        item,
                    )
        for p in run.changed:
            if p in run.c_files:
                text = (run.c_tree / p).read_bytes().decode("utf-8", "ignore")
                if KEY_SHAPED.search(text):
                    run.fail("G6", f"key-shaped string in {p}")
        if KEY_SHAPED.search(json.dumps(run.manifest_raw, default=str)):
            run.fail("G6", "key-shaped string in the manifest")
