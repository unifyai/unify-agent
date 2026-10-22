"""The deterministic gate that decides every merge into memory ``main``.

Nothing a consolidation pass writes reaches the working model unless :meth:`Gate.merge` passes it. Both
revisions are resolved to commit SHAs once, and the trees inspected are exactly the committed blobs
(``ls-tree`` + ``cat-file``; no ``git archive``, so no ``.gitattributes`` can hide a file).

The gate runs no code (rev 2.1, :mod:`.gate_policy`): no item, test, recipe, reader, suite or recorded episode is
run to check, classify or rank anything. It reads files, the manifest and the evidence store. A check that would
need a run records ``not run: <why>`` on the label of the items it concerns (:meth:`_Run.not_run`), or a note for
the pass; running code is a test the writer states and runs itself (``tools.test``).

Threat model
------------------------------------
The gate guards against a careless or mistaken consolidator, not a malicious one; what the writer's own code does is
contained by the confinement of the writer's box. Its rules are structural: they pin what a pass may touch. A
finding about an item's quality (G2 and G3) is recorded on its label and refuses nothing (:func:`.gate_policy.hard`,
:meth:`_Run.fail`); only the hard checks refuse.

The checks (library layout, :mod:`.library_checks`):

* **G1 provenance and scope** (hard). The manifest is well formed and its ids strict (:mod:`.manifest`); the
  candidate descends from the parent; every file has mode 100644 and sits where the layout admits it; every changed
  file is declared; every added, changed or removed item is listed; every item names source episodes that exist;
  no commit writes a path the harness reserves for its generated files. Every new or changed fixture re-derives from
  the recording's bytes (:func:`.fixtures.verify`); a parsed fixture (re-deriving it runs its reader) and a replay
  case file (verifying it replays recorded code) are not run and are labelled. The links, ``When to use`` sections
  and folder README files are checked for presence and form.
* **G2 evidence** (labels). A function's typed covers name a recorded cell or work-tree diff that defines a function;
  an ``episode`` cover is not re-run. A procedure names a recipe and the cases it is for, each read from its case
  file (a failed source episode is a finding); the recipe is not run, and its record says so (``ran: false``). Each
  function's examples (the calls its listed tests make) are not run. Its effect is derived from its recorded covers
  (:func:`.library_checks.derive_effect`).
* **G3 tests** (labels). An edited function needs a new or changed test listed under it, and every listed test must be
  in the candidate; tests are plain pytest, and a test deleted or weakened needs a reason in ``tests_changed``. The
  listed tests, each package's suite and every stored function whose module the pass changes (or that imports one)
  are not run, and each is labelled.
* **G4 size.** Over its soft budget, the index gets a ``note: G4 hygiene due``; growth is never refused.
* **G5 items** (hard). A pass does not reduce the number of items, and does not drop an alias.
* **G6 safety** (hard). No links, executables, submodules, or git, pytest or interpreter configuration files; no
  bytecode or native code; no key-shaped string in a changed file or the manifest; every module parses and defines
  each public function once.

**Evidence levels.** The gate finds no level above ``unchecked`` (it runs nothing): every manifest function lands at
``unchecked`` with its findings (``level_reasons``), or with :data:`.gate_policy.NOT_RUN_GATE` when it has none.
Outcomes after use, never the gate, give ``confirmed-by-outcomes``.

Per-item admission (:meth:`Gate.merge`): a hard failure that belongs to manifest items refuses those items (and what
depends on them, :mod:`.reduction`) instead of the pass, when no failure is pass-wide; the reduced candidate passes
the whole gate again before it lands.

:meth:`Gate.preview` runs the cheap part (the manifest, G1, G2's covers, G4 to G6 and the static checks) on an
uncommitted tree, for the writer's ``check`` tool; it never decides or records a merge. A landed merge also freezes
the candidate's input-shape snapshot (:mod:`.shape_rows`) for the catalogue; computing it never fails the gate.
Stage 5 (:mod:`.qa`) adds static checks of the tests (the test kit stays test-side; fixture size, replay fidelity
and cuts of truncated recordings behind their switches), tagged ``[qa:<check>]``.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import docstrings, layout
from .admission import cover_problem, is_rejection
from .blobs import BlobStore
from .catalogue import body_digest, reserved
from .episodes import Action
from .evidence import EvidenceStore
from .gitio import GitError, Repo
from .cover_forms import unfit_reason as _unfit_reason
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
from .index import build_index, estimate_tokens
from .gate_policy import NOT_RUN_GATE
from .shape_rows import descriptors, shapes_at, snapshot_rows
from .memory_repo import ItemsReport, items
from .item_states import UNCHECKED as _LOWEST_LEVEL
from .qa import QAChecks, QAConfig
from .redact import KEY_SHAPED
from .signals import job_item_status
from .snapshot import (
    code_size,
    duplicate_public_defs,
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
_REDACTED = "<redacted:key-shaped>"
# G3's per-function finding (an edited function with no new or changed test): what would satisfy the rule. The gate
# runs no test, so the writer runs it itself.
RED_TEST_HINT = (
    "A function you edit needs at least one new or changed test listed under it. The gate does not run it: run it "
    "yourself, and see it fail on the parent's version and pass on yours."
)


# Input-shape descriptors computed per item at a merge (the evidence store keeps at most its own cap).
MAX_SHAPE_COVERS = 32

# An environment function that encodes a rule (G2) needs validated covers from at least this many episodes.
RULE_EPISODES = 2

# A check (:meth:`Gate.preview`) examines at most this many covers, naming at most this many episodes.
PREVIEW_MAX_COVERS = 500
PREVIEW_MAX_EPISODES = 32

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
    # the item verification records (library_checks)
    verification: dict[str, dict] = field(default_factory=dict)
    # the candidate's renames; a landed merge records them (EvidenceStore.record_curations)
    curations: list[dict] = field(default_factory=list)
    # library layout: one ``{"item", "adopts"}`` row per landed item, the proposals (staged files, helper output,
    # pool entries) its manifest entry says it came from; empty when nothing landed
    provenance: list[dict] = field(default_factory=list)
    # library mode: the fixture provenance entries G1 verified (path -> (sha256, entry)), persisted on merge
    fixtures_verified: dict[str, tuple[str, dict]] = field(default_factory=dict)
    # each refused item's own reason lines, as they stand in ``reasons`` (a failure attributed to items); the
    # writer pass's repair rounds compare them round to round (:func:`.repair.refusals`)
    item_reasons: dict[str, list[str]] = field(default_factory=dict)
    # library mode: each manifest function's evidence level (:mod:`.item_states`; the gate, which runs no code, gives
    # ``unchecked``) and the findings that set it (``level_reasons``: evidence findings, never refusals)
    levels: dict[str, str] = field(default_factory=dict)
    level_reasons: dict[str, list[str]] = field(default_factory=dict)


def _failed(reason: str, check: str | None = None) -> GateResult:
    return GateResult(
        False,
        {c: False for c in CHECKS},
        [KEY_SHAPED.sub(_REDACTED, reason)],
        [check] if check else [],
    )


# --- outcome rules -----------------------------------------------------------------------------------------


def _unchanged(run: Any, it: Any) -> bool:
    """A listed function whose body (and kind) is the parent's: the pass lists it only to change its tests."""
    return (
        it.kind in ("env_function", "function")
        and it.item in run.p_bodies
        and it.item in run.c_bodies
        and run.p_bodies[it.item][:2] == run.c_bodies[it.item][:2]
    )


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


def _unfit_forms(
    covers: list[tuple[str, int, Action]],
    input_kind: str | None,
) -> list[str]:
    """Why the covered inputs cannot be given in the declared *input_kind* (:func:`.cover_forms.unfit_reason`): the
    covers with status ``ok``, not a rejection, not shell, each once, deduplicated."""
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


#: The checks whose findings about an item are quality results: G2 (evidence: the
#: recipe's cases, the unseen values, the grounding of its calls, its covers) and G3 (its tests, its red test, its
#: behaviour, its examples). G1 (layout, provenance, the manifest, links), G5 (items and coverage kept) and G6
#: (safety) stay hard, as do the library's other tests (a finding no item owns).
LABELLED_CHECKS = frozenset({"G2", "G3"})
#: What makes a G2 or G3 finding hard all the same: syntax, an import that fails, a missing path, the gate's own kit.
_HARD_MARKS = (
    "does not parse",
    "SyntaxError",
    "IndentationError",
    "ImportError",
    "ModuleNotFoundError",
    "cannot be imported",
    "could not be imported",
    "is not in the candidate",
    "[library_mode:plain]",
    "key-shaped",
)


def _not_run(run: Any, item: str) -> str:
    """Why the gate did not run *item* (:func:`.gate_policy.not_run_reason`, from its candidate source)."""
    from .gate_policy import not_run_reason

    body = run.c_bodies.get(item)
    return not_run_reason(body[1] if body is not None and body[0] == "function" else None)


def labelled(check: str, reason: str) -> bool:
    """Whether a finding of *check* about an item is a quality result its label carries
    rather than a refusal: G2 and G3, except syntax, imports that fail, missing paths and the gate's own kit."""
    return check in LABELLED_CHECKS and not any(m in str(reason) for m in _HARD_MARKS)


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
    # item -> (body digest, input-shape descriptors of its validated covers), recorded on a landed merge
    shapes: dict[str, tuple[str, list[dict]]] = field(default_factory=dict)
    # environment functions that cover a recorded write call (structural Effect; G6)
    writes: set[str] = field(default_factory=set)
    # per-item admission: item -> the checks that refused it, and whether any failure is pass-wide
    item_fail: dict[str, list[str]] = field(default_factory=dict)
    pass_wide: bool = False
    # per item, what the checks measured (library_checks, qa)
    verification: dict[str, dict] = field(default_factory=dict)
    # the candidate's renames (LibraryChecks.aliases_kept)
    curations: list[dict] = field(default_factory=list)
    # library mode: the import graph of both versions (the scope of G3's not-run labels and of per-item reduction)
    graph: dict[str, set[str]] | None = None
    # library mode: the fixture provenance entries G1 verified in this pass (path -> (sha256, entry)), persisted on
    # merge
    fixtures_verified: dict[str, tuple[str, dict]] = field(default_factory=dict)
    # library mode: evidence findings set an item's evidence level instead of refusing it (:meth:`cap`)
    levels_on: bool = False
    # a quality finding about an item is recorded on its label and
    # refuses nothing (:meth:`fail`, :func:`labelled`); the hard checks refuse as before
    labels_on: bool = False
    # item -> the quality findings recorded on its label (in order, each once)
    label_findings: dict[str, list[str]] = field(default_factory=dict)
    # item -> the evidence findings that set its level (in order, each once)
    level_reasons: dict[str, list[str]] = field(default_factory=dict)
    # (:mod:`.gate_policy`) every finding but the hard ones labels its item or notes the pass (:meth:`check`; a
    # preview keeps its G2 and G3 labels only, :func:`labelled`)
    labels_only: bool = False

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
        owners = [item] if isinstance(item, str) else list(item or [])
        if self.labels_only:
            from .gate_policy import hard

            if not hard(check, reason, owners):
                # rev 2: a finding labels its items (or notes the pass) and refuses nothing
                line = KEY_SHAPED.sub(_REDACTED, f"{check}: {reason}")
                for owner in owners:
                    found = self.label_findings.setdefault(owner, [])
                    if line not in found:
                        found.append(line)
                who = ", ".join(owners) if owners else "the pass"
                self.note(f"label: {who}: {line.splitlines()[0][:300]}")
                return
        if self.labels_on and owners and labelled(check, reason):
            # a quality finding about these items: their label carries it, and nothing is refused
            line = KEY_SHAPED.sub(_REDACTED, f"{check}: {reason}")
            for owner in owners:
                found = self.label_findings.setdefault(owner, [])
                if line not in found:
                    found.append(line)
            self.note(f"label: {', '.join(owners)}: {line.splitlines()[0][:300]}")
            return
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

    def not_run(self, item: str | list[str] | None, why: str) -> None:
        """Rev 2 (:mod:`.gate_policy`): a check that did not run on *item*, recorded on its label (``not run: <why>``)
        and noted; it refuses nothing."""
        owners = [item] if isinstance(item, str) else list(item or [])
        line = KEY_SHAPED.sub(_REDACTED, why if why.startswith("not run") else f"not run: {why}")
        for owner in owners:
            found = self.label_findings.setdefault(owner, [])
            if line not in found:
                found.append(line)
        self.note(f"{', '.join(owners) if owners else 'the pass'}: {line[:300]}")

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
        budget_tokens: int = 4000,
        reader_promotes: bool = False,
        action_lookup: Callable[[str, int], Action | None] | None = None,
        qa: QAConfig | None = None,
        library_mode: LibraryConfig,
        segments_dir: Path | str | None = None,
    ) -> None:
        """*library_mode* is the library's checks' configuration (:class:`.library_checks.LibraryConfig`): covers have
        no channel rule (G2), G3's not-run labels and per-item reduction are scoped by the import graph, and the
        library gate checks (:mod:`.library_checks`) and stage 5's static checks run. Every finding but the hard ones
        is a label on its item, and the gate runs no code (:mod:`.gate_policy`, rev 2.1). *budget_tokens* is G4's soft budget (a note, never a refusal). *qa* is the stage-5 test
        checks' configuration (:class:`.qa.QAConfig`). *segments_dir* (``UNIFY_MEMORY=in_episode``):
        the state dir whose segment records (``<episode>.s<n>``) an item may cite as its source while their episode
        is still open; None: only recorded episodes.
        """
        self.mem, self.ev, self.blobs = memory, evidence, blobs
        self.budget, self.reader_promotes = budget_tokens, reader_promotes
        # Fail closed: without an episodes reader no covered call can be confirmed.
        self.lookup = action_lookup or (lambda eid, i: None)
        from .qa import library_config

        self.qa = library_config(qa if qa is not None else QAConfig())
        self.library_mode = library_mode
        self.segments_dir = Path(segments_dir) if segments_dir is not None else None

    # -- public API ---------------------------------------------------------------------------------------
    def check(
        self,
        parent: str,
        candidate: str,
        manifest: dict,
    ) -> GateResult:
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
        form; at most :data:`PREVIEW_MAX_COVERS` covers over :data:`PREVIEW_MAX_EPISODES` episodes, each looked
        up once), G4, G5 and G6. Never G3 or a workflow's signals (nothing about outcomes), and never writes:
        no commit, git object, evidence or pass row. A clean preview does not mean the gate will pass.
        """
        res = GateResult(True, {c: True for c in CHECKS})
        tmp = Path(tempfile.mkdtemp(prefix="memory-preview-"))
        run = _Run(res, Manifest(), manifest, "", "", tmp)
        run.levels_on = True
        run.labels_on = True
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
            early = self._early(run, refused, True)
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
            self._library_mode(run).static()
            qa = QAChecks(self, run)
            qa.kit()  # library code never uses the test kit (static)
            if bounded:
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
        items (an item's hard G1 checks, such as its fixtures' provenance, and its G6 ``Effect:`` checks; under the
        labels-only gate a G2 or G3 finding labels its item instead), those items are refused with
        everything that depends on them (:mod:`.reduction`). The candidate reduced to the other items is
        committed as a child of *parent* and the whole gate runs on it once more; it lands if it passes,
        and otherwise the pass is refused whole. A pass-wide failure (layout, unsafe or forbidden files,
        a malformed manifest, undeclared changes, G4, G5, key-shaped strings, unparsable modules) refuses the whole pass at once. One round; no search over
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
            # typed covers persist beside covers, for the behaviour check and deletion (G5)
            from .procedures import cover_raw

            for it in landed_items:
                for c in it.typed_covers:
                    self.ev.add_typed_cover(
                        it.item,
                        c.episode,
                        json.dumps(cover_raw(c), sort_keys=True),
                    )
            # fixture provenance this pass verified, for later passes (fixtures.ledger_for)
            for path, (sha, entry) in sorted(res.fixtures_verified.items()):
                self.ev.add_fixture_provenance(path, sha, entry, landed)
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
        if res.verification:
            # the item verification records, for the item records; measured, never a refusal
            self.ev.add_pass_notes(
                pass_id,
                [
                    "library_mode-verification "
                    + json.dumps(res.verification, sort_keys=True, default=str),
                ],
            )
        if res.passed and res.curations:
            # the pass's renames, which carry an item's uid to its new id (the item records)
            self.ev.record_curations(pass_id, res.merged, res.curations)
        return res

    # -- helpers ------------------------------------------------------------------------------------------
    def _parse_manifest(self, manifest: object) -> Manifest:
        """The manifest under the library layout (:func:`.manifest.parse_manifest`); an item may state its label
        fields."""
        return parse_manifest(manifest, labels=True)

    def _library_mode(self, run: _Run):
        """The library checks of *run* (:mod:`.library_checks`)."""
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
        tmp = Path(tempfile.mkdtemp(prefix="memory-gate-"))
        run = _Run(res, Manifest(), manifest, parent, candidate, tmp)
        run.levels_on = True
        run.labels_on = True
        reduced = None
        try:
            if self._prepare(run):
                run.labels_only = True
                v = self._library_mode(run)
                self._g1(run)
                v.g1()
                self._g2(run)
                v.g2()
                # stage 5's static checks (the test kit stays test-side; fixture size, replay fidelity, cuts)
                qa = QAChecks(self, run)
                qa.kit()
                qa.static(self.lookup)
                self._g3(run)
                v.g3()
                v.g4()
                v.aliases_kept()
                self._g5(run)
                self._g6(run)
                if reduce_as is not None and not res.passed:
                    reduced = self._reduce(run, reduce_as)
            snapshot = self._snapshot(run) if res.passed else {}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        res.items_refused = {i: list(c) for i, c in run.item_fail.items()}
        self._levels(run)
        self._labels(run)
        res.verification = {i: dict(r) for i, r in sorted(run.verification.items())}
        res.curations = [dict(r) for r in run.curations]
        res.fixtures_verified = dict(run.fixtures_verified)
        passed = res.passed
        return res, (run.covers if passed else set()), run.notes, snapshot, reduced

    @staticmethod
    def _labels(run: _Run) -> None:
        """Each manifest item's label inputs in its verification record, from which
        the item records build its label (:mod:`.item_label`): what the writer stated (``stated``) and the quality
        findings recorded instead of refusals (``label_findings``)."""
        for it in run.man.items:
            rec = run.verification.setdefault(it.item, {})
            rec["stated"] = dict(getattr(it, "stated", {}) or {})
            rec["label_findings"] = list(run.label_findings.get(it.item, []))

    @staticmethod
    def _levels(run: _Run) -> None:
        """Library mode: each manifest function's evidence level, with the findings :meth:`_Run.cap` recorded.

        The gate runs no code (rev 2.1), so it never finds that a function reproduces its case: every function it
        checks is ``unchecked``, with its findings, or with :data:`.gate_policy.NOT_RUN_GATE` when it has none. Both
        go in the gate result and in the function's verification record (``level``, ``level_reasons``), from which the
        item records take them; outcomes after use, never the gate, give ``confirmed-by-outcomes``.
        """
        from .item_states import UNCHECKED

        res = run.res
        for it in run.man.items:
            if it.kind != "function" or it.item not in run.c_bodies:
                continue
            found = list(run.level_reasons.get(it.item, [])) or [f"not run: {NOT_RUN_GATE}"]
            level = UNCHECKED
            res.levels[it.item] = level
            res.level_reasons[it.item] = found
            rec = run.verification.setdefault(it.item, {})
            rec["level"] = level
            rec["level_reasons"] = found

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
        early = self._early(run, refused, True)
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
        if isinstance(raw.get("fixtures"), dict):
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
                if not (self.ev.episode_exists(eid) or self._segment_recorded(eid)):
                    run.fail("G1", f"{it.item} cites unknown episode {eid}", it.item)

    def _segment_recorded(self, eid: str) -> bool:
        """Whether *eid* is a segment of an open episode (``<episode>.s<n>``) whose record the harness wrote
        (``UNIFY_MEMORY=in_episode``: :mod:`.segments`); always False without a segments dir."""
        root = getattr(self, "segments_dir", None)
        if root is None:
            return False
        from . import segments

        return segments.is_segment_name(eid) and segments.snapshot_path(root, eid).is_file()

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
        from . import layout

        return layout.item_path(item)

    def _retired_ok(self, run: _Run, test: str) -> None:
        """A retired test file is removed and imports only items this manifest deletes.

        Library mode: removed only; whether it may go is the ``tests_changed`` reason rule (:mod:`.library_checks`,
        and :meth:`_suites`).
        """
        if test not in run.p_files or test in run.c_files:
            run.fail("G1", f"deleted test {test} is not removed by the candidate")
            return
        return

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
        """*preview* (:meth:`preview`) checks the covers and their declared input form only, and no workflow
        signals.

        *lookup* replaces the gate's action lookup (the preview's memoised one).
        """
        lookup = lookup or self.lookup
        library = True
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
                        channel_rule=False,
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
                    # the structural input-form check only
                    for why in _unfit_forms(valid, form)[:5]:
                        run.fail("G2", f"{it.item} {why}", it.item)
                elif valid:
                    # the catalogue's shapes
                    self._record_shapes(
                        run,
                        it.item,
                        valid,
                        it.input or (declared if declared in INPUT_KINDS else None),
                    )
                    # unseen values of its covered inputs would be a rerun of the item: the gate runs no code
                    run.not_run(it.item, f"G2 held-out values: {NOT_RUN_GATE}")
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
            return snapshot_rows(
                run.c_tree,
                prev,
                run.shapes,
            )
        except (
            Exception
        ):  # noqa: BLE001 - shapes are a catalogue aid, never a gate reason
            return snapshot_rows(
                run.c_tree,
                {},
                run.shapes,
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

    def _g3(self, run: _Run) -> None:
        """G3 on the candidate's library, from its files only: an edited function needs a new or changed test listed
        under it, and every listed test must be in the candidate.

        The gate runs no code (rev 2.1, :data:`.gate_policy.NOT_RUN_GATE`): a listed test, a package's suite and a
        stored function whose module the pass changes, or that imports one (:meth:`_call_scope`), are not run, and
        each is recorded on its owners' labels (``not run: <why>``); a suite is noted.
        """
        man, changed = run.man, set(run.changed)
        new_tests = sorted({t for it in man.items for t in it.tests if t in changed})
        edited = {
            it.item
            for it in man.items
            if it.kind in ("env_function", "function")
            and run.p_bodies.get(it.item, ("", ""))[:2]
            != run.c_bodies.get(it.item, ("", ""))[:2]
        }
        refused: set[str] = set()
        for it in man.items:
            if it.item in edited and not any(t in changed for t in it.tests):
                refused.add(it.item)
                run.fail("G3", f"{it.item} has no new or changed test. {RED_TEST_HINT}", it.item)
        for t in new_tests:
            if t not in run.c_files:
                run.fail("G3", f"listed test {t} is not in the candidate", _owners(man, t))
            else:
                run.not_run(_owners(man, t), f"G3 {t}: {NOT_RUN_GATE}")
        self._suites(run)
        for item in sorted(self._call_scope(run) - refused):
            run.not_run(item, f"G3 behaviour on its recorded calls: {_not_run(run, item)}")

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

    @staticmethod
    def _suites(run: _Run) -> None:
        """The library's test suites (``memory/<package>/tests``) are not run (rev 2.1, the gate runs no code): each
        suite of the parent or the candidate that holds a test module is noted, when the pass changes anything."""
        if not run.changed:
            return
        from .layout import classify

        suites = sorted(
            {
                "/".join(p.split("/")[:2]) + "/tests"
                for p in set(run.p_files) | set(run.c_files)
                if p.startswith("memory/") and classify(p) == "test"
            },
        )
        for rel in suites:
            run.note(f"suite {rel} not run: {NOT_RUN_GATE}")

    @staticmethod
    def _shrinks(run: _Run) -> bool:
        """No measure of :func:`_library_size` grows, and function definitions or their AST nodes fall."""
        before = _library_size(run.p_files, run.p_tree)
        after = _library_size(run.c_files, run.c_tree)
        if before is None or after is None:
            return False
        return after[:2] != before[:2] and all(a <= b for a, b in zip(after, before))

    def _blob(self, sha: str) -> bytes:
        if not self.blobs.has(sha):
            raise KeyError(sha)
        return self.blobs.get(sha)

    def _g4(self, run: _Run) -> None:
        """The size budget: a note that hygiene is due past it (never a refusal of growth)."""
        try:
            size = estimate_tokens(build_index(run.c_tree, budget_tokens=sys.maxsize))
        except ValueError as exc:  # an undecodable notes file, say
            run.fail("G4", f"index not built: {exc}")
            return
        if size > self.budget:
            run.note(
                f"G4 hygiene due: the index is {size} tokens, over its "
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
