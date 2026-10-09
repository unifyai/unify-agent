"""The deterministic gate that decides every merge into memory ``main`` (spec §7, D9/D12).

Nothing a consolidation pass writes reaches the working model unless :meth:`Gate.merge` passes it. Both
revisions are resolved to commit SHAs once, and the trees inspected are exactly the committed blobs
(``ls-tree`` + ``cat-file``; no ``git archive``, so no ``.gitattributes`` can hide a file).

Threat model (controller ruling R16)
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
* **Covers** prove that a real recorded observation of the right kind on the item's channel exists (a
  successful tool call, a shell output tail, a file blob, a dialogue observation; :mod:`.admission`), not
  that the item agrees with it: the item's own tests over recorded fixtures check that (spec F3).
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
  ``docstring_standard`` (v2.1; ``UNIFY_MEMORY_V2_DOCSTRINGS=on``), every new or changed environment
  function's docstring meets the lean standard (:mod:`.docstrings`: summary, ``Args:`` naming each
  parameter and the first one's input form, ``Returns:``, ``Raises:`` naming ``MemoryInputError``, an
  ``Example:`` with a ``>>>`` example), and each ``raise MemoryInputError(...)`` in it carries a message of
  at least :data:`.docstrings.MIN_REFUSAL_CHARS` characters; each missing part is its own reason.
* **G2 evidence.** An ``env_function`` covers recorded actions on its own channel, each a real recorded
  observation of its kind (:func:`.admission.cover_problem`): a tool call with status ``ok`` and a
  response, a shell command with an output tail, a file read or write with a blob in the blob store, or a
  dialogue action with an observation; or a recorded environment rejection (status ``error`` with its
  error), never alone. Scope is shape, not observed values (spec F3a): run confined on each covered
  input, passed in its declared ``input`` form (the docstring's ``Input:`` line for an unchanged function
  without one; else the kind's convention), with one value at a time replaced by an unseen value of its
  type (a field with one value across at least 3 recorded observations of its family and field-name set,
  from at least 2 episodes of the whole evidence store, is identity or format and is kept, with a note)
  (:mod:`.held_out`); a declared form a covered input cannot be given in fails; the item
  must not raise its ``MemoryInputError`` before any environment call unless one of its covers is a
  recorded rejection of the same family that varied that field (each such allowance is noted); the reason
  names the field only. A field declared with a semantic type (``field_types``, D21) is checked two-sided:
  unseen in-domain values must be accepted and an out-of-domain value refused. Other exceptions, unused keywords, timeouts and refused baselines are notes.
  Shell covers are skipped with a note. A ``workflow`` is promotable under the job-level rule over its recorded and source episodes.
  An added or changed environment function that replaces a value it computed from its input under a condition
  (:mod:`.analysis.overrides`; the F3 expense case paid an over-cap claim 0 instead of the cap) encodes a
  decision rule its recordings cannot mark wrong: its validated covers must come from at least
  :data:`RULE_EPISODES` episodes, else it is refused with its first such line (a function the analysis cannot
  bound is noted, not refused).
* **G3 red→green and regression.** Every added or changed environment function has a new or changed
  test; each such test passes on the candidate and is red on the parent: it fails on the parent tree with
  the changed tests and the ``support`` helpers copied in (ruling R2; a run there that times out counts as
  red once the candidate's run is green), or, as a test-only repair, it repairs a test file the parent
  already had red (the parent's own version fails on the parent, and a test it failed now passes; a
  collection error counts for the whole file) while importing library functions of which the pass edits
  none. Every edited environment function has at least one test of the first kind, so its change is
  observed by a red run.

  **Behaviour check (D28): any change in what a stored function returns on recorded inputs needs a
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

  A *clean-up pass* (D26: it adds no item and the library shrinks: the channel modules' function
  definitions or the AST nodes of those definitions fall, and neither they nor the nodes of other module
  code grow; comments and docstrings never count) may exempt an edited function from the red run only
  when the change keeps its behaviour: a parent test file that passed on the parent, calls it (an import
  alone does not count) and still passes in both runs below; and the behaviour check above shows it
  doing the same on every recorded cover (none unread or ungiven, at most the cap; a recorded rejection
  that gives no input aside) and on the pass's recorded episode actions. Without it the function needs a
  red test as in any pass (a repair changes behaviour, so it always does); new tests of a clean-up pass
  need only be green. Then, per channel, against the parent's suite (the baseline, ruling R19):

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
* **G4 size budget.** By default (``UNIFY_MEMORY_V2_SOFT_BUDGET=off``, as in v2) :func:`.index.build_index`
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
  the same. A function changed under a red test holds only the covers this manifest lists for it (I1).
* **G6 safety.** No links, executables, submodules, or git, pytest or interpreter configuration files; no
  bytecode or native code, ``__pycache__`` path, ``sitecustomize``/``usercustomize`` under any suffix or root
  directory named like a standard-library or pytest module; and no other root entry than ``env/``,
  ``workflows/`` and the test kit (G1) (:func:`.manifest.unsafe_path`, all refused before anything is
  extracted or run); no key-shaped string in a changed file or the manifest; every public function of a
  changed module declares ``Effect:`` and is defined once; every module parses. **Structural Effect:** an
  environment function in ``items`` that covers a tool call whose recorded effect is ``write``
  (:attr:`.episodes.Action.effect`, the effect the environment declared for the method; never inferred from
  names) declares ``Effect: write``, unless its declared input is a recorded value (``path``, ``text``,
  ``bytes``, ``observation``: it parses what a call returned and makes no call).

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

Under ``surfacing="catalogue"`` (``UNIFY_MEMORY_V2_SURFACING``) a landed merge also freezes the candidate
commit's input-shape snapshot (:mod:`.shape_rows`): per
environment function, the parent's shapes for the same body plus the descriptors of the covers this merge
validated (each covered file's blob, each covered observation), for the export's catalogue
(:mod:`.catalogue`). Computing it never fails the gate. The examples run, like G3's runs, is a drift check
under R16, not a security boundary: module code the runner imports can patch ``doctest`` (M13).

Stage-5 test checks (memory v2.1, :mod:`.qa`) extend G3 when a :class:`.qa.QAConfig` switch is on: seeded
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
from typing import Callable

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
from .gate_v21 import V21Config
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
from .qa import QAChecks, QAConfig, QAEnv
from .redact import KEY_SHAPED
from .sandbox_run import PYTHON, PytestOutcome, run_pytest
from .signals import job_item_status
from .snapshot import (
    calls_item,
    code_size,
    duplicate_public_defs,
    env_references,
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


# The generated doctest runner of G3's examples run (a path the layout admits for no candidate file).
EXAMPLES_TEST = "_memory_examples/test_examples.py"
# Input-shape descriptors computed per item at a merge (the evidence store keeps at most its own cap).
MAX_SHAPE_COVERS = 32

# An environment function that encodes a rule (G2) needs validated covers from at least this many episodes.
RULE_EPISODES = 2

# A check (:meth:`Gate.preview`) examines at most this many covers, naming at most this many episodes.
PREVIEW_MAX_COVERS = 500
PREVIEW_MAX_EPISODES = 32

# G3's behaviour check (D28), per pass: the seconds its confined runs may take and the recorded inputs it may
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


def _describe(o: PytestOutcome) -> str:
    return (
        f"rc={o.returncode} valid={o.valid} timed_out={o.timed_out} "
        f"failed={sorted(o.failed)[:5]} passed={len(o.passed)}"
    )


def _exercised(source: bytes, *bodies: dict) -> set[str] | None:
    """The library functions a test file imports: by id, by channel or all of ``env`` (None: unparsable).

    A channel or ``env`` import stands for every environment function of *bodies* in its scope.
    """
    refs = env_references(source)
    if refs is None:
        return None
    names, channels, everything = refs
    functions = {i for b in bodies for i, v in b.items() if v[0] == "env_function"}
    return set(names) | {
        i
        for i in functions
        if everything or i.split(":", 1)[0].split("/", 1)[1] in channels
    }


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


def _test_file(test_id: str, tests_dir: str) -> str:
    """The test file of a suite's test id, as a path from the tree's root (ids may be relative to *tests_dir*)."""
    path = test_id.split("::", 1)[0]
    return path if path.startswith("env/") else f"{tests_dir}/{path}"


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
    items of a JSON list, the values of a JSON object of records, the same one level down (``{"rows": [...]}``),
    or the lines of a JSON-lines file; ``[]`` for anything else. Bounded and value-free.
    """
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

    out = list(members(top))
    for v in members(top):
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
    """One function's behaviour check (D28): the inputs to run on both versions and what they showed."""

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
    """G3's behaviour-check budget for one pass (D28): wall-clock seconds left."""

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
    # on every compared input (D28). None: not judged (a preview runs no G3).
    same: set[str] | None = None
    # the parent's declared input forms by function, read once (False: its library cannot be read)
    p_inputs: dict[str, str] | bool | None = None
    # changed channel -> recorded actions of the channel in the pass's own episodes (G3's behaviour check)
    episode_inputs: dict[str, list[tuple[str, int, Action]]] = field(
        default_factory=dict,
    )
    # v2.1: the import graph of both versions (G3's behaviour scope; None in v2)
    graph: dict[str, set[str]] | None = None

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
        self.res.reasons.append(KEY_SHAPED.sub(_REDACTED, f"{check}: {reason}"))
        owners = [item] if isinstance(item, str) else list(item or [])
        if not owners:
            self.pass_wide = True
        for owner in owners:
            codes = self.item_fail.setdefault(owner, [])
            if check not in codes:
                codes.append(check)

    def note(self, reason: str) -> None:
        self.notes.append(KEY_SHAPED.sub(_REDACTED, f"note: {reason}"))

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
        v21: V21Config | None = None,
    ) -> None:
        """The v2.1 switches (:mod:`.integration.switch`; each default is the v2 behaviour):
        *docstring_standard* turns on the lean docstring standard (G1) and its examples run (G3) for new or
        changed environment functions; *surfacing* ``"catalogue"`` records and freezes input shapes per
        commit for the export's catalogue; *soft_budget* makes *budget_tokens* G4's soft budget (a note,
        never a refusal) instead of the index's hard cap. *qa* is the stage-5 test checks' configuration
        (:class:`.qa.QAConfig`; None runs none of them). Sol's brief follows the gate's switches.
        *v21* (``UNIFY_MEMORY_V21``, :class:`.gate_v21.V21Config`; None is v2): with ``layout``, covers have
        no channel rule (G2), and G3's behaviour check and per-item reduction are scoped by the import graph
        (D42).
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
        # stage-5 test checks (memory v2.1); the default runs none of them
        self.qa = qa if qa is not None else QAConfig()
        self.v21 = v21

    @property
    def _layout21(self) -> bool:
        """Memory v2.1's library layout rules (D42) are on."""
        v21 = getattr(self, "v21", None)
        return v21 is not None and bool(v21.layout)

    # -- public API ---------------------------------------------------------------------------------------
    def check(self, parent: str, candidate: str, manifest: dict) -> GateResult:
        """The whole gate on *candidate*, without merging; no per-item reduction is tried.

        ``items_refused`` names the items its failures belong to, as :meth:`merge` would refuse them.
        """
        res, _, notes, _, _ = self._check(parent, candidate, manifest)
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
        G2's held-out runs or a workflow's signals (nothing about outcomes, ruling R10), and never writes:
        no commit, git object, evidence or pass row. A clean preview does not mean the gate will pass.
        """
        res = GateResult(True, {c: True for c in CHECKS})
        tmp = Path(tempfile.mkdtemp(prefix="memv2-preview-"))
        run = _Run(res, Manifest(), manifest, "", "", tmp)
        try:
            try:
                run.man = parse_manifest(manifest)
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
            early = self._early(run, refused)
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
                if it.kind == "env_function"
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
    ) -> GateResult:
        """Check, then fast-forward ``main``; evidence is recorded only for a landed candidate.

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
        try:
            res, covers, notes, shapes, reduced = self._check(
                p_sha,
                c_sha,
                manifest,
                reduce_as=pass_id,
            )
            if reduced is not None:
                again, covers2, notes2, shapes2, _ = self._check(
                    p_sha,
                    reduced.sha,
                    reduced.manifest,
                )
                if again.passed:
                    again.reasons = [
                        f"item refused: {r}" for r in res.reasons + reduced.reasons
                    ]
                    again.refused = []
                    res, covers, notes, shapes = again, covers2, notes2, shapes2
                    landed, landed_manifest = reduced.sha, reduced.manifest
                else:
                    res.reasons.append(
                        "per-item admission: the candidate without the refused items "
                        f"({reduced.sha[:12]}) was refused too",
                    )
                    res.reasons += [f"reduced: {r}" for r in again.reasons]
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
                self.mem.fast_forward("main", landed, expected_old=p_sha)
            except GitError as exc:
                res.passed = False
                res.reasons.append(f"merge: {exc}")
        res.reasons.extend(notes)  # after every failure reason, including the merge's
        if res.passed:
            res.merged = landed
            res.items_merged = [it.item for it in parse_manifest(landed_manifest).items]
            for it in parse_manifest(landed_manifest).items:
                for eid in it.source_episodes:
                    self.ev.add_item_evidence(it.item, eid, "source")
            for item, eid, idx in sorted(covers):
                self.ev.add_cover(item, eid, idx)
            if self.surfacing == "catalogue":
                self.ev.write_commit_shapes(landed, shapes)
        else:
            try:
                ids = [it.item for it in parse_manifest(manifest).items]
            except ManifestError:
                ids = []
            res.items_merged = []
            res.items_refused = {i: res.items_refused.get(i) or ["pass"] for i in ids}
        self._record(p_sha, c_sha, pass_id, kind, channel, usd, res)
        return res

    # -- helpers ------------------------------------------------------------------------------------------
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
    ) -> PytestOutcome:
        if qa_env is not None:
            # a stage-5 switch is on or the library's tests use the test kit: memlab and the referenced blobs
            # at /inputs, and a skip for a failed import fails (the kit must never be silently missing)
            return self.pytest(
                target,
                python=self.python,
                ro={tree: "/memory", qa_env.inputs: "/inputs"},
                rw={},
                cwd="/memory",
                timeout_s=_TIMEOUT_S,
                env={**qa_env.env, **(extra_env or {})},
                import_skips_fail=True,
            )
        return self.pytest(
            target,
            python=self.python,
            ro={tree: "/memory"},
            rw={},
            cwd="/memory",
            timeout_s=_TIMEOUT_S,
            env={**_PYTEST_ENV, **(extra_env or {})},
        )

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
        tmp = Path(tempfile.mkdtemp(prefix="memv2-gate-"))
        run = _Run(res, Manifest(), manifest, parent, candidate, tmp)
        reduced = None
        try:
            if self._prepare(run):
                self._g1(run)
                self._g2(run)
                # stage 5: the test kit when a switch is on or the library's tests use it (else nothing)
                qa = QAChecks(self, run)
                if qa.prepare():
                    qa.kit()
                    if self.qa.on:
                        qa.static(self.lookup)
                self._g3(run)
                self._g4(run)
                self._g5(run)
                self._g6(run)
                if self.qa.on:
                    # last: only a candidate the rest of the gate accepts, or, under per-item admission,
                    # one refused item by item only, so that a stage-5 refusal joins the reduction
                    qa.dynamic(item_scoped=reduce_as is not None)
                if reduce_as is not None and not res.passed:
                    reduced = self._reduce(run, reduce_as)
            snapshot = (
                self._snapshot(run)
                if res.passed and self.surfacing == "catalogue"
                else {}
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        res.items_refused = {i: list(c) for i, c in run.item_fail.items()}
        passed = res.passed
        return res, (run.covers if passed else set()), run.notes, snapshot, reduced

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
                v21=self._layout21,
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
            run.man = parse_manifest(run.manifest_raw)
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
        early = self._early(run, refused)
        if early:
            run.stop(early)
            return False
        run.p_tree = materialise(self.mem, run.p_files, run.tmp / "parent")
        run.c_tree = materialise(self.mem, run.c_files, run.tmp / "cand")
        self._read_trees(run)
        return True

    @staticmethod
    def _early(run: _Run, refused: list[str]) -> list[tuple[str, str]]:
        """What must never be extracted or run (refused entries, forbidden files, layout); sets ``changed``."""
        early = [("G6", f"the candidate holds {r}") for r in refused[:10]]
        early += [
            ("G6", f"forbidden file {p}") for p in sorted(run.c_files) if forbidden(p)
        ]
        early += [
            ("G1", f"file {p} is outside the layout (no submodules or root modules)")
            for p in sorted(run.c_files)
            if not layout_allowed(p)
        ]
        # Compiled code, start-up hooks and root entries the layout above admits (review I4); one reason each.
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

    @staticmethod
    def _read_trees(run: _Run) -> None:
        run.p_bodies, run.c_bodies = item_bodies(run.p_tree), item_bodies(run.c_tree)
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
                declared.add(item_path(eid))
        for eid in man.unlisted:
            if self._unlisted_ok(run, eid):
                declared.add(item_path(eid))
        for t in man.deleted_tests:
            self._retired_ok(run, t)
        self._pins(run)
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
            if not it.source_episodes:
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

    def _retired_ok(self, run: _Run, test: str) -> None:
        """A retired test file is removed and imports only items this manifest deletes."""
        if test not in run.p_files or test in run.c_files:
            run.fail("G1", f"deleted test {test} is not removed by the candidate")
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
        for n, it in enumerate(run.man.items):
            if it.kind == "env_function":
                if not it.covers:
                    run.fail("G2", f"{it.item} covers no recorded action", it.item)
                valid: list[tuple[str, int, Action]] = []
                for eid, idx in it.covers:
                    action = lookup(eid, idx)
                    problem = cover_problem(
                        action,
                        it.channel,
                        self.blobs.has,
                        channel_rule=not self._layout21,
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
                    if self.surfacing == "catalogue" or self.docstring_standard:
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
            )
            return snapshot_rows(run.c_tree, prev, run.shapes)
        except (
            Exception
        ):  # noqa: BLE001 - shapes are a catalogue aid, never a gate reason
            return snapshot_rows(run.c_tree, {}, run.shapes)

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
        """Scope is shape (spec F3a): the item must not refuse unseen values of its covered inputs' types.

        Each covered input reaches the item in its declared *input_kind* (None: the kind's convention). A
        field's constancy (identity or format) is judged on *pool*, the recorded observations of the covered
        channels across the evidence store, with whether its read cap was hit.

        A field declared with a semantic type (D21) is checked two-sided instead: unseen in-domain values
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
        # results change, by the behaviour check (D28)
        edited = {
            it.item
            for it in man.items
            if it.kind == "env_function"
            and run.p_bodies.get(it.item, ("", ""))[:2]
            != run.c_bodies.get(it.item, ("", ""))[:2]
        }
        # a clean-up pass may edit a function without a red test when an old test holds it (D26)
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
                    run.fail("G3", f"{it.item} has no new or changed test", it.item)
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
            for rel in runnable + man.support:
                if rel in run.c_files:
                    (red_tree / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(run.c_tree / rel, red_tree / rel)
            for t in runnable:
                on_parent = self._pytest(red_tree, t, qa_env=run.qa_env)
                on_cand = self._pytest(run.c_tree, t, qa_env=run.qa_env)
                run.qa_first[t] = on_cand
                # a parent library that hangs on the test is red once the candidate's run is green
                if _red(on_parent) or (on_parent.timed_out and _green(on_cand)):
                    classic.add(t)
                elif not cleanup:
                    self._test_only_repair(run, t, edited, on_cand, on_parent)
                if not _green(on_cand):
                    run.fail(
                        "G3",
                        f"{t} is not green on the candidate ({_describe(on_cand)}) "
                        f"{on_cand.output[-300:]}",
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
                        f"{it.item} is edited, but none of its tests is red on the parent's library",
                        it.item,
                    )
        protected = self._suites(run)
        self._behaviour(run, unseen, declared, refused, protected)
        if self.docstring_standard:
            self._examples(run)

    def _behaviour(
        self,
        run: _Run,
        unseen: set[str],
        declared: set[str],
        refused: set[str],
        protected: set[str],
    ) -> None:
        """G3's behaviour check (D28): any change in what a stored function returns on recorded inputs needs
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
        """
        recorded: dict[str, set[tuple[str, int]]] = {}
        for it, e, i in self.ev.covers():
            recorded.setdefault(it, set()).add((e, i))
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
        for item in self._behaviour_targets(run, recorded, self._layout21):
            if item in probes or item in unseen or item in declared or item in refused:
                continue
            probes[item] = self._probe(run, item, recorded, strict=False)
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
        v21: bool = False,
    ) -> list[str]:
        """The functions G3's behaviour check compares (D28), in order, before :meth:`_behaviour`'s skips.

        **v2:** for each changed channel, every public function of its parent module whose name the candidate's
        module still binds: a ``def``, or a name bound by assignment or import that has recorded covers.

        **v2.1** (spec §4.1, D42): the same rule over each changed library module and every module that imports
        one, directly or through others. The scope uses :func:`..layout.dependants` over the import graphs of
        both versions. The graph is kept on the run for :meth:`_behaviour_owner`.
        """
        if v21:
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
        """v2.1: the memory channels (``env/<name>``) of a function's recorded covers. They stand for the channel
        its path named in v2 (spec §4.1: the source channel moves to the item's record, P5).
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
        check again. A parent library that cannot be read is no item's. Under v2.1 the edits that count are
        those in the item's module or in a library module it imports, directly or through others.
        """
        if pr.why == "the parent's library cannot be read":
            return None
        if (
            run.graph is not None
        ):  # v2.1 (D42): what the item's module imports, transitively
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
                run.fail(
                    "G3",
                    f"the examples of {item} could not be run ({_describe(outcome)}) "
                    f"{outcome.output[-300:]}",
                )
            elif ran(outcome.failed, n) or not ran(outcome.passed, n):
                run.fail(
                    "G3",
                    f"an example in the docstring of {item} fails as a doctest "
                    f"{outcome.output[-300:]}",
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
            run.fail(
                "G3",
                f"the examples of {item} read no fixture shaped like an input it covers "
                f"({', '.join(paths[:5])})",
                item,
            )

    def _test_only_repair(
        self,
        run: _Run,
        test: str,
        edited: set[str],
        on_cand: PytestOutcome,
        on_parent: PytestOutcome,
    ) -> bool:
        """A test that is not classically red still counts if it is a test-only repair (else G3 fails).

        It must import at least one library function (same analysis as for retired tests), none of which
        this pass edits, and :meth:`_repaired` must hold. Otherwise a function change could ride a repaired
        test and go unobserved by any red run.
        """
        uses = _exercised((run.c_tree / test).read_bytes(), run.p_bodies, run.c_bodies)
        carried = sorted(edited & uses) if uses else []
        if uses and not carried and self._repaired(run, test, on_cand):
            return True
        why = (
            "already pass on the parent (not red)"
            if _green(on_parent)
            else f"are not red on the parent ({_describe(on_parent)})"
        )
        if carried:
            why += (
                f"; nor can it count as a repair of a red test, since the pass edits "
                f"{carried[:5]}, which it exercises"
            )
        run.fail("G3", f"the tests in {test} {why}", _owners(run.man, test))
        return False

    def _repaired(self, run: _Run, test: str, on_cand: PytestOutcome) -> bool:
        """The parent's own *test* file is already red on the parent and the candidate's version repairs it.

        A readable parent report must name a failed test that passes on the candidate; a collection error
        names no test, so it counts for the whole file. (The candidate's run must also be green.) Used
        only for test-only repairs (:meth:`_test_only_repair`).
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
        """
        swapped = sorted(
            p
            for p in set(run.p_files) | set(run.c_files)
            if (MODULE_PATH.match(p) or NOTES_PATH.match(p))
            and run.p_files.get(p) != run.c_files.get(p)
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
        """Per channel, compare against the parent's suite (the baseline; ruling R19).

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
        """
        protected: set[str] = set()
        if not run.changed:
            return protected
        retired = set(run.man.deleted_tests)
        regression = self._regression_tree(run)
        touched = {p.split("/")[1] for p in run.changed if p.startswith("env/")}

        def kept(ids: set[str]) -> set[str]:
            return {t for t in ids if t.split("::", 1)[0] not in retired}

        channels = sorted(
            {
                p.split("/")[1]
                for p in set(run.p_files) | set(run.c_files)
                if p.startswith("env/") and p.split("/")[2:3] == ["tests"]
            },
        )
        for ch in channels:
            rel = f"env/{ch}/tests"
            before: set[str] = set()  # protected: passed on the parent
            known_red: set[str] = set()  # failed on the parent
            broken: str | None = None  # the baseline cannot be read
            if (run.p_tree / rel).is_dir():
                base = self._pytest(run.p_tree, rel, qa_env=run.qa_env)
                if base.timed_out or not base.valid:
                    broken = _describe(base)
                    run.note(
                        f"the parent's suite {rel} is unreadable ({broken}): it protects no test "
                        "and its regression run is skipped",
                    )
                else:
                    before, known_red = kept(base.passed), kept(base.failed)
                    protected |= {t.split("::", 1)[0] for t in before}
            after: set[str] = set()
            if (run.c_tree / rel).is_dir():
                suite = self._pytest(run.c_tree, rel, qa_env=run.qa_env)
                readable = suite.valid and not suite.timed_out
                new_red = sorted(suite.failed - known_red)
                if (
                    broken is not None
                    and ch not in touched
                    and not (readable and not suite.failed)
                ):
                    run.note(
                        f"pre-existing: suite {rel} is still not green ({_describe(suite)}); "
                        f"this pass does not touch env/{ch}",
                    )
                elif not readable or new_red:
                    repair = (
                        "; the parent's suite was unreadable, so the candidate's must be green"
                        if broken is not None
                        else ""
                    )
                    # new failures only in test files the manifest's items list are those items' own
                    # (per-item admission); anything else refuses the whole pass
                    owners = [_owners(run.man, _test_file(t, rel)) for t in new_red]
                    run.fail(
                        "G3",
                        f"suite {rel} has new failures or is unreadable on the candidate "
                        f"({_describe(suite)}; new {new_red[:5]}{repair}) {suite.output[-300:]}",
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
            if regression is not None and before:
                reg = self._pytest(regression, rel, qa_env=run.qa_env)
                if reg.timed_out or not reg.valid:
                    run.fail(
                        "G3",
                        f"the regression run of {rel} is unreadable ({_describe(reg)}) "
                        f"{reg.output[-300:]}",
                    )
                lost = sorted(before - reg.passed)
                if lost:
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
        """A clean-up pass (D26): it adds no item and the library shrinks."""
        added = any(it.item not in run.p_bodies for it in run.man.items)
        return not added and self._shrinks(run)

    @staticmethod
    def _held(run: _Run, item: str, protected: set[str]) -> bool:
        """A parent test file that passed on the parent, and still passes, calls *item*.

        A call, not an import: :func:`.snapshot.calls_item`.
        """
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
        """
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
        """The recorded inputs *item* is compared on, without running anything (D26, D28).

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
            self._layout21
        ):  # the recorded actions of its covers' channels in this pass's episodes (D42)
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
        # a deleted function's recorded covers must stay covered by a remaining item (D26)
        deleted = set(run.man.deleted)
        recorded = self.ev.covers()
        held = {(e, i) for _, e, i in run.covers}

        changed = self._changed_channels(run)

        def keeps(it: str) -> bool:
            """A kept item still does on its recorded covers what it did (I1, D28): its channel is unchanged,
            or G3's behaviour check showed it doing the same. A function changed under a red test holds only
            the covers this manifest lists for it (already in ``held``)."""
            if it in deleted or it not in run.c_bodies:
                return False
            if run.same is None:  # a preview runs no G3
                return True
            if it.split(":", 1)[0] not in changed:
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
            # structural Effect: a function that covers a recorded write call is a writer
            for item in sorted(run.writes):
                if effects.get(item, "write") != "write":
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
