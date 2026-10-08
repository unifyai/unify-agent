"""The deterministic gate that decides every merge into memory ``main`` (spec §7, D9/D12).

Nothing a consolidation pass writes reaches the working model unless :meth:`Gate.merge` passes it. Both
revisions are resolved to commit SHAs once, and the trees inspected are exactly the committed blobs
(``ls-tree`` + ``cat-file``; no ``git archive``, so no ``.gitattributes`` can hide a file).

Threat model (controller ruling R16)
------------------------------------
The gate guards against a careless or mistaken consolidator, not a malicious one; malice is contained by
the runtime confinement of every run (:mod:`.sandbox_run`). Its rules are structural: they pin what a pass
may touch and keep every old test passing against the new library. Documented limits:

* **Test adequacy** is not judged: a weak test that is red on the parent and green on the candidate passes.
* **Effect: lines** are checked for presence, not truthfulness.
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
  ``Input:`` line.
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
  for a timed-out red run as above. Each run is bounded at 300 s.
* **G4 index budget.** :func:`build_index` of the candidate fits the budget.
* **G5 description length.** If ``env/*/__init__.py`` grew, a cover G2 validated is new to the evidence
  and is a successful observation (a recorded rejection cover, status ``error``, is not new coverage), or
  the library shrinks by the clean-up measure above. Every recorded input a deleted item covered (except
  recorded rejections) is covered by a remaining item: one of this pass's validated covers, or a recorded
  cover of a kept item whose channel the pass leaves unchanged or that G3's behaviour check showed doing
  the same. A function changed under a red test holds only the covers this manifest lists for it (I1).
* **G6 safety.** No links, executables, submodules, or git, pytest or interpreter configuration files; no
  key-shaped string in a changed file or the manifest; every public function of a changed module declares
  ``Effect:`` and is defined once; every module parses.

:meth:`Gate.preview` runs the cheap, read-only part (the manifest, G1, G2's covers, G4 to G6) on an
uncommitted tree, for the consolidator's ``check`` tool; it never decides or records a merge.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .admission import cover_problem, is_rejection
from .blobs import BlobStore
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
from .index import IndexOverBudget, build_index
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
)
from .memory_repo import ItemsReport, items
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
    # functions of the changed channels that G3's behaviour check showed doing what their parent versions did
    # on every compared input (D28). None: not judged (a preview runs no G3).
    same: set[str] | None = None
    # the parent's declared input forms by function, read once (False: its library cannot be read)
    p_inputs: dict[str, str] | bool | None = None
    # changed channel -> recorded actions of the channel in the pass's own episodes (G3's behaviour check)
    episode_inputs: dict[str, list[tuple[str, int, Action]]] = field(
        default_factory=dict,
    )

    def fail(self, check: str, reason: str) -> None:
        # reasons are stored in the evidence store; test output in them is model-controlled
        self.res.checks[check] = False
        self.res.passed = False
        if check not in self.res.refused:
            self.res.refused.append(check)
        self.res.reasons.append(KEY_SHAPED.sub(_REDACTED, f"{check}: {reason}"))

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
    ) -> None:
        self.mem, self.ev, self.blobs = memory, evidence, blobs
        self.python, self.budget, self.reader_promotes = (
            python,
            budget_tokens,
            reader_promotes,
        )
        # Fail closed: without an episodes reader no covered call can be confirmed.
        self.lookup = action_lookup or (lambda eid, i: None)
        self.pytest = pytest_runner

    # -- public API ---------------------------------------------------------------------------------------
    def check(self, parent: str, candidate: str, manifest: dict) -> GateResult:
        res, _, notes = self._check(parent, candidate, manifest)
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
            if len(covers) > PREVIEW_MAX_COVERS or len(episodes) > PREVIEW_MAX_EPISODES:
                run.fail(
                    "G2",
                    f"the manifest names {len(covers)} covers over {len(episodes)} episodes; a check "
                    f"examines at most {PREVIEW_MAX_COVERS} covers over {PREVIEW_MAX_EPISODES} episodes",
                )
            else:
                memo: dict[tuple[str, int], Action | None] = {}

                def lookup(eid: str, idx: int) -> Action | None:
                    if (eid, idx) not in memo:
                        memo[(eid, idx)] = self.lookup(eid, idx)
                    return memo[(eid, idx)]

                self._g2(run, preview=True, lookup=lookup)
                self._g5(run)  # needs G2's validated covers
            self._g4(run)
            self._g6(run)
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
        """Check, then fast-forward ``main``; evidence is recorded only for a landed candidate."""
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
        try:
            res, covers, notes = self._check(p_sha, c_sha, manifest)
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
                self.mem.fast_forward("main", c_sha, expected_old=p_sha)
            except GitError as exc:
                res.passed = False
                res.reasons.append(f"merge: {exc}")
        res.reasons.extend(notes)  # after every failure reason, including the merge's
        if res.passed:
            for it in parse_manifest(manifest).items:
                for eid in it.source_episodes:
                    self.ev.add_item_evidence(it.item, eid, "source")
            for item, eid, idx in sorted(covers):
                self.ev.add_cover(item, eid, idx)
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

    def _pytest(self, tree: Path, target: str) -> PytestOutcome:
        return self.pytest(
            target,
            python=self.python,
            ro={tree: "/memory"},
            rw={},
            cwd="/memory",
            timeout_s=_TIMEOUT_S,
            env=dict(_PYTEST_ENV),
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
        if not res.passed and patch_ok:
            try:
                patch = self.blobs.put(self.mem.diff(parent, candidate).encode())
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
            },
        )

    # -- the checks ---------------------------------------------------------------------------------------
    def _check(
        self,
        parent: str,
        candidate: str,
        manifest: dict,
    ) -> tuple[GateResult, set[tuple[str, str, int]], list[str]]:
        """The result (failure reasons only), the validated covers if it passed, and the ``note:`` lines."""
        res = GateResult(True, {c: True for c in CHECKS})
        tmp = Path(tempfile.mkdtemp(prefix="memv2-gate-"))
        run = _Run(res, Manifest(), manifest, parent, candidate, tmp)
        try:
            if self._prepare(run):
                self._g1(run)
                self._g2(run)
                self._g3(run)
                self._g4(run)
                self._g5(run)
                self._g6(run)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return res, (run.covers if res.passed else set()), run.notes

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
        run.changed = sorted(
            p
            for p in set(run.p_files) | set(run.c_files)
            if run.p_files.get(p) != run.c_files.get(p)
        )
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
        for it in man.items:
            if cb.get(it.item, ("",))[0] != it.kind:
                run.fail("G1", f"{it.item} ({it.kind}) is not in the candidate")
            if not it.source_episodes:
                run.fail("G1", f"{it.item} names no source episode")
            for eid in it.source_episodes:
                if not self.ev.episode_exists(eid):
                    run.fail("G1", f"{it.item} cites unknown episode {eid}")

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
                run.fail("G1", f"{it.item} has more than one Input: line")
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
                    )
            elif doc[it.item] != it.input:
                run.fail(
                    "G1",
                    f"{it.item} declares input {it.input} but its docstring's Input: line says "
                    f"{doc[it.item] or 'missing'}",
                )

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
                    run.fail("G2", f"{it.item} covers no recorded action")
                valid: list[tuple[str, int, Action]] = []
                for eid, idx in it.covers:
                    action = lookup(eid, idx)
                    problem = cover_problem(action, it.channel, self.blobs.has)
                    if problem is not None:
                        run.fail("G2", f"{it.item} covers ({eid},{idx}), {problem}")
                    else:
                        run.covers.add((it.item, eid, idx))
                        valid.append((eid, idx, action))
                        if (
                            is_rejection(action)
                            and getattr(action, "kind", "tool") != "shell"
                        ):
                            run.rejections.add((eid, idx))
                if valid and all(
                    is_rejection(a) and getattr(a, "kind", "tool") != "shell"
                    for _, _, a in valid
                ):
                    run.fail("G2", f"{it.item} covers only recorded rejections")
                elif valid and preview:
                    # the structural input-form check only: no held-out perturbation
                    declared = self._doc_inputs(run).get(it.item, "")
                    form = it.input or (declared if declared in INPUT_KINDS else None)
                    for why in _unfit_forms(valid, form)[:5]:
                        run.fail("G2", f"{it.item} {why}")
                elif valid:
                    if seen is None:
                        seen = seen_actions(self._named_episodes(run), self.lookup)
                    declared = self._doc_inputs(run).get(it.item, "")
                    self._held_out(
                        run,
                        it.item,
                        valid,
                        seen,
                        run.tmp / f"held-out-{n}",
                        it.field_types,
                        it.input or (declared if declared in INPUT_KINDS else None),
                        self._pool(run, valid),
                    )
            elif it.kind == "workflow" and not preview:
                status = job_item_status(
                    it.item,
                    _WithSources(self.ev, it.item, it.source_episodes),
                    reader_promotes=self.reader_promotes,
                )
                if status != "promotable":
                    run.fail("G2", f"{it.item} is {status}")

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
            run.fail("G2", f"{item} held-out value refused: {f[:80]}")
        for text in verdict.failures[:5]:
            run.fail("G2", f"{item} {text}")
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
                    run.fail("G3", f"{it.item} has no new or changed test")
        missing = [t for t in new_tests if t not in run.c_files]
        for t in missing:
            run.fail("G3", f"listed test {t} is not in the candidate")
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
                on_parent = self._pytest(red_tree, t)
                on_cand = self._pytest(run.c_tree, t)
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
                    )
        protected = self._suites(run)
        self._behaviour(run, unseen, declared, refused, protected)

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
        Sets ``run.same``; a budget run out refuses the pass.
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
                )
        for channel in sorted(self._changed_channels(run)):
            module = f"{channel}/__init__.py"
            before = public_bindings(
                (run.p_tree / module).read_bytes() if module in run.p_files else None,
            )
            after = public_bindings(
                (run.c_tree / module).read_bytes() if module in run.c_files else None,
            )
            for name, is_def in sorted((before or {}).items()):
                item = f"{channel}:{name}"
                if (
                    item in probes
                    or item in unseen
                    or item in declared
                    or item in refused
                ):
                    continue
                if after is None or name not in after:
                    continue  # deleted: its recorded covers stay covered under G5
                if is_def or item in recorded:
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
                run.fail("G3", self._exemption_refused(item, why))
            else:
                run.fail(
                    "G3",
                    f"{item} changes behaviour without a test: {why}; any change in what a stored function "
                    "returns on recorded inputs needs a test, listed under it in the manifest, that fails on "
                    "the parent's library and passes on the candidate's",
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
        run.fail("G3", f"the tests in {test} {why}")
        return False

    def _repaired(self, run: _Run, test: str, on_cand: PytestOutcome) -> bool:
        """The parent's own *test* file is already red on the parent and the candidate's version repairs it.

        A readable parent report must name a failed test that passes on the candidate; a collection error
        names no test, so it counts for the whole file. (The candidate's run must also be green.) Used
        only for test-only repairs (:meth:`_test_only_repair`).
        """
        if test not in run.p_files:
            return False
        old = self._pytest(run.p_tree, test)
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
                base = self._pytest(run.p_tree, rel)
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
                suite = self._pytest(run.c_tree, rel)
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
                    run.fail(
                        "G3",
                        f"suite {rel} has new failures or is unreadable on the candidate "
                        f"({_describe(suite)}; new {new_red[:5]}{repair}) {suite.output[-300:]}",
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
                reg = self._pytest(regression, rel)
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
        acts += [
            c
            for c in self._episode_inputs(run, item.split(":", 1)[0])
            if (c[0], c[1]) not in given
        ]
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
        try:
            build_index(run.c_tree, budget_tokens=self.budget)
        except IndexOverBudget as exc:
            run.fail("G4", str(exc))
        except ValueError as exc:
            run.fail("G4", f"index not built: {exc}")

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
            for i in run.c_report.items:
                if i.kind == "env_function" and i.path in changed and not i.effect:
                    run.fail("G6", f"{i.item_id} has no Effect: line")
        for p in run.changed:
            if p in run.c_files:
                text = (run.c_tree / p).read_bytes().decode("utf-8", "ignore")
                if KEY_SHAPED.search(text):
                    run.fail("G6", f"key-shaped string in {p}")
        if KEY_SHAPED.search(json.dumps(run.manifest_raw, default=str)):
            run.fail("G6", "key-shaped string in the manifest")
