"""The memory v2.1 gate checks (spec §9.1, §8.2 rule 1, §13.2, §13.4), run by :class:`.gate.Gate` with ``v21``.

With ``Gate(v21=None)`` (the default) nothing here runs and the gate is v2's. With a :class:`V21Config`, the gate
(``checks`` on, which needs P3's v2.1 layout, ``layout``) adds these checks:

* **G1:**
  - scope: only ``memory/`` and ``notes/``, never the harness-generated paths;
  - links: no dangling ``uses:``/``Notes:`` target;
  - fixture provenance: every new or changed file under ``tests/data/`` came through ``fixture()`` and re-derives
    from the recording (:mod:`.fixtures`).
* **G2:**
  - typed covers: an actor cell or work-tree diff that defines a function, or an ``episode`` cover verified by its
    procedure runner (:mod:`.procedures`);
  - the cross-episode held-out check, for inputs with nothing to perturb;
  - the literal lint (:mod:`.code_lint`).
* **G3:**
  - plain pytest: no test-kit import;
  - per changed function, at least one exact assertion against a trust 1–3 value on a recorded fixture, and no
    self-reimplementing oracle (:mod:`.quality`);
  - no test deleted or weakened without a reason in the manifest's ``tests_changed``;
  - for WRITE, no fewer tests.
  - a stored function's episode covers re-run on the parent and the candidate: green to red is a behaviour change
    (Amendment C; the covers are persisted at merge, :meth:`.evidence.EvidenceStore.add_typed_cover`).
* **G4** never refuses growth: over the index view budget it notes that CURATE is due.
* **G5:**
  - for WRITE, no fewer items (removal belongs to CURATE);
  - a deleted function's typed episode covers stay covered by a remaining item (Amendment C).

The stage-5 checks run with drawn inputs ``strict`` and mutation gated at :data:`.qa.MUTATION_MIN`
(:func:`.qa.v21_config`). Measures go to the item verification record (``GateResult.verification``). The merge
also appends that record to the pass row as one ``v21-verification`` line, for P5's item records.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .admission import is_rejection
from .batch_map import actor_functions
from .code_lint import LINT_MIN, input_value, literal_problems, varying_values
from .episodes import Action, Episode
from .fixtures import DATA_PATH, DRAWN_DIR, HashIndex, module_path, package_dir, verify
from .gitio import Repo
from .held_out import (
    OUTPUTS_BUDGET_S,
    _doc,
    _unfit_reason,
    family,
    output_cases,
    run_outputs,
)
from .catalogue import estimate_tokens
from . import layout
from .aliases import forwards, uses_name
from .layout import classify, discover
from .library_index import INDEX_VIEW_TOKENS, build_links, render_index
from .procedures import (
    PROCEDURE_S,
    cover_raw,
    parse_cover,
    run_procedure,
    snapshot_files,
)
from .qa import builtin_error
from .qa_static import truncated
from .sandbox_run import SandboxResult, run_confined
from .redact import KEY_SHAPED
from .quality import (
    TEST_FILE,
    count_items,
    exact_assertions,
    inventory,
    recorded_literals,
    test_changes,
)
from .testkit import imports_kit

ROLES = ("write", "curate")
CROSS_CASES = (
    16  # recorded inputs from other episodes per function (cross-episode held-out)
)
# Amendment C: the procedure re-runs of the behaviour check, bounded like v2's (held_out.OUTPUTS_BUDGET_S)
BEHAVIOUR_MAX_PROCEDURES = 16
OUTPUT_VIEW_BYTES = 4000  # the head of a test run's output kept in memory and quoted (spec P5: explicit, marked)
_KEY_SHAPED_BYTES = re.compile(KEY_SHAPED.pattern.encode())


def keep_output(outcome: Any, spool: Path, blobs: Any) -> None:
    """Spec P5 under v2.1: a test run's whole output (stdout, then stderr, as v2 joins them; each spooled up to
    :data:`.sandbox_run.SPOOL_MAX_BYTES`, marked past it) is kept as a blob, with key-shaped strings redacted line
    by line. The outcome's ``output`` becomes a head-first view of at most :data:`OUTPUT_VIEW_BYTES` bytes, marked
    with the whole size and the blob id. Nothing is kept when the run printed nothing.
    """
    combined = spool / "combined"
    with open(combined, "wb") as out:
        for name in ("stdout", "stderr"):
            part = spool / name
            if part.is_file():
                with open(part, "rb") as f:
                    for line in f:
                        out.write(_KEY_SHAPED_BYTES.sub(b"<redacted:key-shaped>", line))
    n = combined.stat().st_size
    if n == 0:
        return
    sha = blobs.put_file(combined)
    with open(combined, "rb") as f:
        head = f.read(OUTPUT_VIEW_BYTES)
    text = head.decode("utf-8", "replace")
    if n > len(head):
        outcome.output = (
            text + f"\n[… shown bytes 0–{len(head)} of {n}; full output: blob {sha}]"
        )
    else:
        outcome.output = text + f"\n[full output: blob {sha}]"
    outcome.output_blob = sha


def _no_episode(eid: str) -> Episode | None:
    return None


def _no_snapshot(sha: str, dest: Path) -> Path | None:
    return None


def _index_tokens(tree: Path) -> int:
    """The tokens of the ``INDEX.md`` the harness would generate for *tree* (P3's renderer and estimator)."""
    lib = discover(tree)
    return estimate_tokens(render_index(lib, build_links(lib)))


@dataclass(frozen=True)
class V21Config:
    """The memory v2.1 gate (Amendment E: ``Gate(v21=V21Config(...))``; None is v2).

    *layout*: P3's v2.1 library layout. *checks*: this module's checks and stage 5 in v2.1 mode, off until the
    consolidation builds the gate with :func:`config_for`. *checker_visible*: the bed shows the actor its
    checker's verdict, so a checker signal may verify a dialogue procedure (P5's ``switch.checker_visible``,
    Amendments A3 and D).
    """

    layout: bool = True
    checks: bool = False
    role: str = "write"
    episodes: Callable[[str], Episode | None] = _no_episode
    worktree_files: Callable[[str, Path], Path | None] = _no_snapshot
    index_tokens: int = INDEX_VIEW_TOKENS
    cross_cases: int = CROSS_CASES
    lint_min: int = LINT_MIN
    runner: Callable[..., SandboxResult] = run_confined
    checker_visible: bool = False

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, not {self.role!r}")


def config_for(
    stores: Any,
    lookup: Any,
    role: str = "write",
    checker_visible: bool = False,
) -> V21Config:
    """The v2.1 gate configuration of a run's stores (:mod:`.integration.consolidate`), checks on."""

    def episodes(eid: str) -> Episode | None:
        try:
            return lookup.episode(eid)
        except Exception:  # noqa: BLE001 - unknown or unreadable: no episode
            return None

    git = Path(stores.paths.worktree_git)
    reader = snapshot_files(Repo(git)) if (git / "HEAD").is_file() else _no_snapshot
    return V21Config(
        checks=True,
        role=role,
        episodes=episodes,
        worktree_files=reader,
        checker_visible=checker_visible,
    )


class V21Checks:
    def __init__(self, gate: Any, run: Any) -> None:
        self.gate, self.run, self.cfg = gate, run, gate.v21
        self._episodes: dict[str, Episode | None] = {}

    # -- shared ---------------------------------------------------------------------------------------------
    def ep(self, eid: object) -> Episode | None:
        if not isinstance(eid, str):
            return None
        if eid not in self._episodes:
            try:
                self._episodes[eid] = self.cfg.episodes(eid)
            except Exception:  # noqa: BLE001 - unreadable: no episode
                self._episodes[eid] = None
        return self._episodes[eid]

    def _functions(self) -> list[Any]:
        return [it for it in self.run.man.items if it.kind == "function"]

    def _edited(self) -> list[Any]:
        run = self.run
        raw = run.manifest_raw if isinstance(run.manifest_raw, dict) else {}
        declared = raw.get("aliases")
        # P6: a CURATE pass's declared aliases are forwarding stubs that D28 compares strictly, not new code to test
        forwarding = (
            set(declared)
            if self.cfg.role == "curate" and isinstance(declared, dict)
            else set()
        )
        return [
            it
            for it in self._functions()
            if it.item not in forwarding
            and it.item in run.c_bodies
            and run.p_bodies.get(it.item, ("", ""))[:2] != run.c_bodies[it.item][:2]
        ]

    def _record(self, item: str) -> dict:
        return self.run.verification.setdefault(item, {})

    @staticmethod
    def _provenance(it: Any) -> list[str]:
        return sorted(
            {
                *it.source_episodes,
                *(e for e, _ in it.covers),
                *(c.episode for c in getattr(it, "typed_covers", [])),
            },
        )

    def _owners(self, path: str) -> list[str] | None:
        """The items a test-side *path* belongs to: those listing it, else the functions of its package."""
        listed = [it.item for it in self.run.man.items if path in it.tests]
        if listed:
            return listed
        pkg = path.split("/tests/", 1)[0]
        found = [it.item for it in self._functions() if package_dir(it.item) == pkg]
        return found or None

    def _covers(self, it: Any) -> list[tuple[str, int, Action]]:
        out = []
        for eid, idx in it.covers:
            a = self.gate.lookup(eid, idx)
            if (
                a is not None
                and a.status == "ok"
                and not is_rejection(a)
                and getattr(a, "kind", "tool") != "shell"
            ):
                out.append((eid, idx, a))
        return out

    def static(self) -> None:
        """The checks that run no code, for :meth:`.gate.Gate.preview`."""
        self.g1()
        self._plain()
        self._test_changes()
        self.curate()

    # -- G1 -------------------------------------------------------------------------------------------------
    def g1(self) -> None:
        run = self.run
        for p in run.changed:
            if p in run.c_files and classify(p) in (None, "generated"):
                run.fail(
                    "G1",
                    f"file {p} is outside memory/ and notes/, or is generated by the harness (memory/__init__.py, "
                    "INDEX.md, links.json)",
                )
        self._links()
        self._fixtures()

    def _links(self) -> None:
        run = self.run
        try:
            dangling = build_links(discover(run.c_tree))["dangling"]
        except (ValueError, OSError) as exc:
            run.fail("G1", f"links cannot be read: {exc}"[:300])
            return
        listed = {it.item for it in run.man.items}
        for d in dangling:
            run.fail(
                "G1",
                f"dangling link: {d['from']} links to {d['to']} ({d['why']}), which the candidate does not hold",
                d["from"] if d["from"] in listed else None,
            )

    def _fixtures(self) -> None:
        run = self.run
        raw = run.manifest_raw if isinstance(run.manifest_raw, dict) else {}
        entries = raw.get("fixtures") if isinstance(raw.get("fixtures"), dict) else {}
        index: HashIndex | None = None
        for p in run.changed:
            if (
                p not in run.c_files
                or not p.startswith("memory/")
                or "/tests/data/" not in p
            ):
                continue
            owners = self._owners(p)
            if not DATA_PATH.match(p) or p.split("/tests/data/", 1)[1].startswith(
                DRAWN_DIR,
            ):
                run.fail(
                    "G1",
                    f"fixture {p} is not a plain file name under tests/data/",
                    owners,
                )
                continue
            data = (run.c_tree / p).read_bytes()
            problem = verify(p, data, entries.get(p), self.ep, self.gate._blob)
            if problem is None:
                continue
            if entries.get(p) is None:
                if index is None:
                    index = HashIndex()
                    for eid in sorted(
                        {e for it in run.man.items for e in self._provenance(it)},
                    ):
                        ep = self.ep(eid)
                        if ep is not None:
                            index.add(ep, self.gate._blob)
                hits = index.find(data)
                if hits:
                    problem += f" (its bytes equal source {hits[0][1]} of {hits[0][0]}: make it with fixture())"
            run.fail("G1", f"fixture provenance: {problem}", owners)

    # -- G2 -------------------------------------------------------------------------------------------------
    def g2(self) -> None:
        n = 0
        for it in self._functions():
            for cover in getattr(it, "typed_covers", []):
                self._typed_cover(it, cover, n)
                n += 1
        for it in self._edited():
            self._cross_episode(it)
            self._lint(it)

    def _typed_cover(self, it: Any, cover: Any, n: int) -> None:
        run = self.run
        ep = self.ep(cover.episode)
        if ep is None:
            run.fail(
                "G2",
                f"{it.item} covers episode {cover.episode}, which cannot be read",
                it.item,
            )
            return
        if cover.type == "cell":
            if not any(
                f["source"] == "cell" and f["cell"] == cover.index
                for f in actor_functions(ep)
            ):
                run.fail(
                    "G2",
                    f"{it.item} covers cell {cover.index} of {cover.episode}, which defines no Python function",
                    it.item,
                )
        elif cover.type == "diff":
            if not any(f["source"] == "diff" for f in actor_functions(ep)):
                run.fail(
                    "G2",
                    f"{it.item} covers the work-tree diff of {cover.episode}, which adds no Python function",
                    it.item,
                )
        elif cover.type == "episode":
            out = run_procedure(
                it.item,
                cover,
                ep=ep,
                tree=run.c_tree,
                python=self.gate.python,
                work=run.tmp / f"procedure-{n}",
                blob=self.gate._blob,
                worktree_files=self.cfg.worktree_files,
                signals=self.gate.ev.signals_for,
                runner=self.cfg.runner,
                checker_visible=self.cfg.checker_visible,
            )
            self._record(it.item).setdefault("procedures", []).append(
                {"episode": cover.episode, "runner": cover.runner, "ok": out.ok},
            )
            for note in out.notes:
                run.note(f"G2 procedure: {it.item} on {cover.episode}: {note}")
            if not out.ok:
                run.fail(
                    "G2",
                    f"{it.item} procedure on {cover.episode} ({cover.runner}): {out.reason}",
                    it.item,
                )

    def _cross_episode(self, it: Any) -> None:
        """Spec §9.1: an input with nothing to perturb (plain text) also runs on recorded inputs of its family
        from episodes outside its provenance. It must return or raise MemoryInputError, never another exception.
        """
        run, form = self.run, it.input
        covers = self._covers(it)
        flat = [c for c in covers if _doc(c[2], self.gate._blob, form) is None]
        if not flat:
            return
        prov = set(self._provenance(it))
        fams = {family(a) for _, _, a in flat}
        pool, _ = self.gate._pool(run, covers)
        chosen: dict[str, tuple[str, int, Action]] = {}
        for k, (eid, a) in enumerate(pool):
            if eid in prov or family(a) not in fams:
                continue
            if (
                a.status != "ok"
                or is_rejection(a)
                or _unfit_reason(a, form) is not None
                or truncated(a)
            ):
                continue
            chosen[hashlib.sha256(f"{it.item}\0{eid}\0{k}".encode()).hexdigest()] = (
                eid,
                k,
                a,
            )
        picked = [chosen[h] for h in sorted(chosen)[: self.cfg.cross_cases]]
        record = self._record(it.item)
        record["cross_episode"] = {"ran": 0, "ok": 0}
        if not picked:
            run.note(
                f"G2 cross-episode: {it.item} has no recorded input of its family outside its provenance",
            )
            return
        cases, _ = output_cases(picked, self.gate._blob, form)
        work = run.tmp / f"cross-{hashlib.sha256(it.item.encode()).hexdigest()[:12]}"
        got = run_outputs(
            it.item,
            cases,
            tree=run.c_tree,
            python=self.gate.python,
            work=work,
            runner=self.cfg.runner,
            timeout_s=OUTPUTS_BUDGET_S,
        )
        ran = ok = 0
        raised: dict[str, int] = {}
        for value in got.values():
            if value is None:
                continue
            ran += 1
            outcome, result, _ = value
            if outcome in ("handled", "refused"):
                ok += 1
            else:
                name = builtin_error(result) or "another exception"
                raised[name] = raised.get(name, 0) + 1
        record["cross_episode"] = {"ran": ran, "ok": ok}
        if raised:
            kinds = ", ".join(f"{k} ({v})" for k, v in sorted(raised.items()))
            run.fail(
                "G2",
                f"{it.item} raises on {ran - ok} of {ran} recorded inputs of its family from episodes outside its "
                f"provenance ({kinds}); it must return or raise MemoryInputError",
                it.item,
            )
        if ran < len(cases):
            run.note(
                f"G2 cross-episode: {it.item}: {len(cases) - ran} input(s) not judged (timed out or not JSON)",
            )

    def _lint(self, it: Any) -> None:
        run = self.run
        values = [
            input_value(a, it.input, self.gate._blob) for _, _, a in self._covers(it)
        ]
        values += [
            c.params
            for c in getattr(it, "typed_covers", [])
            if c.type == "episode" and c.params
        ]
        if len(values) < 2:
            run.note(
                f"G2 literal lint: {it.item} has fewer than 2 recorded inputs, so nothing varies yet",
            )
            return
        try:
            source = (run.c_tree / module_path(it.item)).read_bytes()
        except OSError:
            return  # an absent module is refused elsewhere
        name = it.item.split(":", 1)[1]
        for line, what in literal_problems(
            source,
            name,
            varying_values(values),
            lint_min=self.cfg.lint_min,
        )[:5]:
            run.fail(
                "G2",
                f"{it.item} line {line}: a {what} literal equals a value that varies across its recorded inputs; "
                "make it a parameter",
                it.item,
            )

    # -- G3 -------------------------------------------------------------------------------------------------
    def g3(self) -> None:
        self._plain()
        self._test_changes()
        for it in self._edited():
            self._exact(it)
        self._procedure_behaviour()

    def _plain(self) -> None:
        run = self.run
        for p in sorted(set(run.changed) & set(run.c_files)):
            if p.startswith("memory/") and "/tests/" in p and p.endswith(".py"):
                if imports_kit((run.c_tree / p).read_bytes()):
                    run.fail(
                        "G3",
                        f"[v21:plain] {p} imports the gate's test kit (memlab or the pin plugin); a library's tests "
                        "run with plain pytest",
                        self._owners(p),
                    )

    def _test_changes(self) -> None:
        run = self.run

        def read(
            files: dict,
            tree: Path,
            keep: Callable[[str], bool],
        ) -> dict[str, bytes]:
            return {p: (tree / p).read_bytes() for p in files if keep(p)}

        is_test = lambda p: bool(TEST_FILE.match(p))  # noqa: E731
        before = inventory(read(run.p_files, run.p_tree, is_test))
        after = inventory(read(run.c_files, run.c_tree, is_test))
        raw = run.manifest_raw if isinstance(run.manifest_raw, dict) else {}
        stated = (
            raw.get("tests_changed")
            if isinstance(raw.get("tests_changed"), dict)
            else {}
        )

        def reason(tid: str) -> str | None:
            for key in (tid, tid.split("::", 1)[0]):
                r = stated.get(key)
                if isinstance(r, str) and r.strip():
                    return r
            return None

        deleted, weakened = test_changes(before, after)
        for tid, what in [
            *((t, "deleted") for t in deleted),
            *((t, "weakened (fewer or looser assertions)") for t in weakened),
        ]:
            if reason(tid) is None:
                path = tid.split("::", 1)[0]
                run.fail(
                    "G3",
                    f"test {tid} is {what} without a stated reason (the manifest's tests_changed)",
                    self._owners(path) if path in run.c_files else None,
                )
        if self.cfg.role != "write":
            return
        if len(after) < len(before):
            run.fail(
                "G3",
                f"a WRITE pass may not reduce the number of tests ({len(before)} to {len(after)}); removals and "
                "merges belong to CURATE",
            )
        is_item = lambda p: (
            p.startswith("memory/") and p.endswith(".py")
        ) or (  # noqa: E731
            p.startswith("notes/") and p.endswith(".md")
        )
        n_p = count_items(read(run.p_files, run.p_tree, is_item))
        n_c = count_items(read(run.c_files, run.c_tree, is_item))
        if n_c < n_p:
            run.fail(
                "G5",
                f"a WRITE pass may not reduce the number of items ({n_p} to {n_c}); removals and merges belong "
                "to CURATE",
            )

    def _exact(self, it: Any) -> None:
        run = self.run
        record = self._record(it.item)
        record["tests"] = sorted(it.tests)
        name = it.item.split(":", 1)[1]
        pkg = package_dir(it.item)
        fixtures = {p for p in run.c_files if p.startswith(pkg + "/tests/data/")}
        recorded = recorded_literals(
            [e for e in map(self.ep, self._provenance(it)) if e is not None],
            self.gate._blob,
        )
        try:
            function_source = (run.c_tree / module_path(it.item)).read_bytes()
        except OSError:
            return  # an absent module is refused elsewhere
        total = 0
        for t in it.tests:
            if t not in run.c_files:
                continue
            for e in exact_assertions(
                (run.c_tree / t).read_bytes(),
                function_source,
                name,
                fixtures,
                recorded,
                t,
            ):
                if e.oracle:
                    run.fail(
                        "G3",
                        f"[v21:oracle] {t} line {e.line} computes its expected value with {it.item}'s own "
                        "expression; take the expected value from a recorded fixture",
                        it.item,
                    )
                elif e.on_fixture and e.trusted:
                    total += 1
        record["exact_assertions"] = total
        if total == 0:
            run.fail(
                "G3",
                f"[v21:exact] {it.item}'s tests hold no exact assertion of its result, on a recorded fixture, "
                "against a trust 1-3 value (a recorded response, a value a successful episode used, or one "
                "recorded across episodes)",
                it.item,
            )

    # -- G4 -------------------------------------------------------------------------------------------------
    def g4(self) -> None:
        run = self.run
        try:
            size = _index_tokens(run.c_tree)
        except (ValueError, OSError) as exc:
            run.fail("G4", f"index not built: {exc}"[:300])
            return
        if size > self.cfg.index_tokens:
            run.curate_due = True
            run.note(
                f"G4 CURATE due: the index is {size} tokens, over its {self.cfg.index_tokens}-token view; "
                "growth is not refused",
            )

    # -- Amendment C: procedures in the behaviour check and in deletion ---------------------------------------
    # -- CURATE (P6; spec §10.4, §9.1) --------------------------------------------------------------------------
    def curate(self) -> None:
        """CURATE's own rules (spec §10.4), static: they read the manifest and both trees and run nothing.

        A WRITE pass may not declare aliases or retirements (removals and merges belong to CURATE, §9.1). A
        CURATE pass:
        - states why in one line (the ``Why:`` trailer);
        - forwards each declared alias, an item of the parent, to a function of the candidate that is not itself an
          alias (:func:`.aliases.forwards`);
        - gives every deleted item a reason: an alias, or a one-line ``retired`` reason;
        - lists an old name's tests under its alias's target, which they keep testing through the alias (§9.1);
        - keeps a live alias for one more pass: one added by the pass that made the parent stays;
        - does not grow the library: the items that are not aliases do not increase.
        The aliases, retirements and dropped aliases go to ``run.curations``; :meth:`.gate.Gate.merge` records them
        for a landed pass.
        """
        run = self.run
        raw = run.manifest_raw if isinstance(run.manifest_raw, dict) else {}
        aliases, retired = raw.get("aliases") or {}, raw.get("retired") or {}
        if self.cfg.role != "curate":
            if aliases or retired:
                run.fail(
                    "G1",
                    "aliases and retirements belong to CURATE; a WRITE pass may not declare them",
                )
            return
        if not all(
            isinstance(d, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in d.items())
            for d in (aliases, retired)
        ):
            run.fail("G1", "aliases and retired map item ids to strings")
            return
        why = raw.get("why")
        if not isinstance(why, str) or not why.strip() or "\n" in why.strip():
            run.fail("G1", "a CURATE pass states why in one line (the manifest's why)")
            why = ""
        why = why.strip()
        p_lib, c_lib = layout.discover(run.p_tree), layout.discover(run.c_tree)
        p_ids = {f.item_id for f in p_lib.functions}
        c_ids = {f.item_id for f in c_lib.functions}
        p_mods, c_mods = layout.library_modules(run.p_tree), layout.library_modules(
            run.c_tree,
        )
        listed = {it.item: it for it in run.man.items}
        live = dict(self.gate.ev.aliases())
        rows: list[dict] = []
        for alias, target in sorted(aliases.items()):
            problem = self._alias_problem(
                alias,
                target,
                set(aliases) | set(live),
                p_ids | set(live),
                c_ids,
                c_mods,
            )
            if problem is not None:
                run.fail("G1", problem, target if target in listed else None)
                continue
            rows.append(
                {"item": alias, "action": "alias", "target": target, "reason": why},
            )
            self._alias_tests(alias, target, listed, p_mods)
        deleted = set(run.man.deleted)
        for item in sorted(deleted - set(aliases)):
            if not retired.get(item, "").strip():
                run.fail(
                    "G5",
                    f"deleted item {item} needs a reason: keep its name as an alias (aliases) or retire it with a "
                    "one-line reason (retired)",
                )
        for item, reason in sorted(retired.items()):
            if item not in deleted:
                run.fail("G1", f"retired {item} is not in deleted")
            elif reason.strip():
                rows.append(
                    {
                        "item": item,
                        "action": "retire",
                        "target": None,
                        "reason": " ".join(reason.split())[:300],
                    },
                )
        for alias, row in sorted(live.items()):
            if (
                alias in aliases
                or self._forwards(run.c_tree, alias, c_mods) is not None
            ):
                continue  # still forwarded, or declared again
            if row.get("commit") == run.parent:
                run.fail(
                    "G5",
                    f"alias {alias} was added by the pass that made the parent ({run.parent[:12]}); it stays for "
                    "one more pass",
                )
            else:
                rows.append(
                    {
                        "item": alias,
                        "action": "drop_alias",
                        "target": row.get("target"),
                        "reason": why,
                    },
                )
        aliased = set(aliases) | set(live)
        n_p = len((p_ids | {n.item_id for n in p_lib.notes}) - aliased)
        n_c = len((c_ids | {n.item_id for n in c_lib.notes}) - aliased)
        if n_c > n_p:
            run.fail(
                "G5",
                f"a CURATE pass may not grow the library ({n_p} to {n_c} items that are not aliases); add a function "
                "only as the target that old names become aliases of",
            )
        run.curations = rows

    def _alias_problem(
        self,
        alias: str,
        target: str,
        aliases: set[str],
        known: set[str],
        c_ids: set[str],
        c_mods: set[str],
    ) -> str | None:
        if not (layout.FUNCTION_ID.match(alias) and layout.FUNCTION_ID.match(target)):
            return f"alias {alias} -> {target}: both must be function ids (memory.<package>.<module>:<name>)"
        if target == alias or target in aliases:
            return f"alias {alias} -> {target}: an alias forwards to an item that is not itself an alias"
        if target not in c_ids:
            return (
                f"alias {alias} -> {target}: the candidate holds no function {target}"
            )
        if alias not in known:
            return f"alias {alias}: the parent has no item {alias} to keep working"
        if self._forwards(self.run.c_tree, alias, c_mods) != target:
            name, callee = alias.split(":", 1)[1], target.split(":", 1)[1]
            return (
                f"alias {alias}: its module does not forward {name} to {target} (bind it to the function, import it "
                f"under that name, or make it a one-line wrapper that returns {callee}(...))"
            )
        return None

    def _alias_tests(
        self,
        alias: str,
        target: str,
        listed: dict,
        p_mods: set[str],
    ) -> None:
        """Spec §9.1: the parent's tests of *alias* that the candidate keeps are collected through the alias, so they
        are listed under *target* (they test it, and a failure in them belongs to it).
        """
        run = self.run
        tests = set(listed[target].tests) if target in listed else set()
        for p in sorted(run.p_files):
            if layout.classify(p) != "test" or p not in run.c_files or p in tests:
                continue
            if uses_name(
                (run.p_tree / p).read_bytes(),
                layout.package_of_path(p),
                alias,
                p_mods,
            ):
                run.fail(
                    "G3",
                    f"the tests of {alias} ({p}) keep testing it through its alias: list them under {target} in items",
                    target if target in listed else None,
                )

    @staticmethod
    def _forwards(tree: Path, item: str, modules: set[str]) -> str | None:
        """What the candidate's module of *item* forwards *item*'s name to (None: not forwarded, or no module)."""
        module, name = item.split(":", 1)
        try:
            source = (Path(tree) / layout.module_path(module)).read_bytes()
        except OSError:
            return None
        found = forwards(source, module, modules)
        return None if found is None else found.get(name)

    def _stored_typed(self) -> dict[str, list[Any]]:
        """The typed covers recorded at merge, by item (unreadable rows are skipped)."""
        out: dict[str, list[Any]] = {}
        for item, _, raw in self.gate.ev.typed_covers():
            try:
                c = parse_cover(json.loads(raw))
            except ValueError:
                continue
            if not isinstance(c, tuple):
                out.setdefault(item, []).append(c)
        return out

    def _procedure_behaviour(self) -> None:
        """Amendment C, D28: a stored function's ``episode`` covers re-run on the parent's and the candidate's
        library when the pass changes a library module. A procedure green on the parent and red on the
        candidate is a behaviour change: the function now does its recorded job differently.

        Every runner compares with the same recording, so "a different confirmed effect" is green to red. The
        scope is P3's (D42): the stored functions of each changed module and of every module importing one
        (``run.graph``, built by the gate's own behaviour check, :meth:`.gate.Gate._behaviour_targets`), and
        the owner is :meth:`.gate.Gate._behaviour_owner`'s. Without a graph (a run that built none) every stored
        procedure is in scope and the owner is the pass's one edited function, else the pass.
        """
        run = self.run
        if not any(classify(p) in ("module", "package_init") for p in run.changed):
            return
        graph = getattr(run, "graph", None)
        scope: set[str] | None = None
        if graph is not None:
            from . import layout

            changed = {m for p in run.changed if (m := layout.module_of(p)) is not None}
            scope = layout.dependants(graph, changed)
        edited = {it.item for it in self._edited()}
        targets = [
            (item, c)
            for item, covers in sorted(self._stored_typed().items())
            if item in run.p_bodies
            and item in run.c_bodies
            and item not in edited
            and item not in run.item_fail
            and (scope is None or item.split(":", 1)[0] in scope)
            for c in covers
            if c.type == "episode"
        ]
        if not targets:
            return
        if len(targets) > BEHAVIOUR_MAX_PROCEDURES:
            run.fail(
                "G3",
                f"behaviour check not completed: {len(targets)} stored procedures exceed the "
                f"{BEHAVIOUR_MAX_PROCEDURES} re-run in one pass; change fewer modules",
            )
            return
        fallback = sorted(edited)[0] if len(edited) == 1 else None

        def owner_of(item: str) -> str | None:
            if graph is None:
                return fallback
            from types import SimpleNamespace

            from .gate import Gate

            return Gate._behaviour_owner(run, item, SimpleNamespace(why=None))

        started = time.monotonic()
        for n, (item, cover) in enumerate(targets):
            ep = self.ep(cover.episode)
            if ep is None:
                run.note(
                    f"G3 behaviour check: {item}'s procedure on {cover.episode} cannot be read; not compared",
                )
                continue
            outcome = []
            for side, tree in (("parent", run.p_tree), ("candidate", run.c_tree)):
                left = OUTPUTS_BUDGET_S - (time.monotonic() - started)
                if left <= 1:
                    run.fail(
                        "G3",
                        f"behaviour check not completed: its {OUTPUTS_BUDGET_S:g} s procedure budget ran out "
                        f"at {item}",
                    )
                    return
                outcome.append(
                    run_procedure(
                        item,
                        cover,
                        ep=ep,
                        tree=tree,
                        python=self.gate.python,
                        work=run.tmp / f"procedure-behaviour-{n}-{side}",
                        blob=self.gate._blob,
                        worktree_files=self.cfg.worktree_files,
                        signals=self.gate.ev.signals_for,
                        runner=self.cfg.runner,
                        timeout_s=min(PROCEDURE_S, left),
                        checker_visible=self.cfg.checker_visible,
                    ),
                )
                if side == "parent" and not outcome[0].ok:
                    break  # red on the parent already: nothing the candidate can break
            if len(outcome) == 2 and not outcome[1].ok:
                run.fail(
                    "G3",
                    f"{item} changes behaviour without a test: its procedure on {cover.episode} ({cover.runner}) "
                    f"passes on the parent's library and fails on the candidate's ({outcome[1].reason}); any "
                    "change in what a stored function does on its recorded work needs a test that fails on the "
                    "parent's library and passes on the candidate's",
                    owner_of(item),
                )

    def g5(self) -> None:
        """Amendment C, D26: a deleted function's typed ``episode`` covers stay covered by a remaining item.

        P6 Amendment B (D28, as P4 Amendment C): a remaining item vouches for a deleted item's procedure only when
        :func:`.procedures.run_procedure` gives on the candidate's library the same outcome (``ok``) as the deleted
        item gave on the parent's. The holders tried are the remaining items that list or store the same cover and,
        for a deleted item that is an alias (declared by this CURATE pass, or live), its target: an alias forwarding a
        procedure is held by that re-run. Bounded like the behaviour check, never silently.
        """
        run = self.run
        deleted = set(run.man.deleted)
        if not deleted:
            return

        def key(c: Any) -> str:
            return json.dumps(
                {k: v for k, v in cover_raw(c).items() if k not in ("params", "paths")},
                sort_keys=True,
            )

        stored = self._stored_typed()
        holders: dict[str, list[str]] = {}
        for it in self._functions():
            if it.item in deleted:
                continue
            for c in getattr(it, "typed_covers", []):
                if c.type == "episode":
                    holders.setdefault(key(c), []).append(it.item)
        for item, covers in stored.items():
            if item in deleted or item not in run.c_bodies:
                continue
            for c in covers:
                if c.type == "episode":
                    holders.setdefault(key(c), []).append(item)
        targets = self._alias_targets()
        started = time.monotonic()
        tried = 0
        for item in sorted(deleted & set(stored)):
            lost = []
            for c in stored[item]:
                if c.type != "episode":
                    continue
                cands = [targets[item]] if item in targets else []
                cands += [
                    h for h in sorted(set(holders.get(key(c), []))) if h not in cands
                ]
                cands = [h for h in cands if h in run.c_bodies and h not in deleted]
                if not cands:
                    lost.append(c.episode)
                    continue
                tried += 1
                if tried > BEHAVIOUR_MAX_PROCEDURES:
                    run.fail(
                        "G5",
                        f"take-over check not completed: more than {BEHAVIOUR_MAX_PROCEDURES} deleted procedures "
                        "to re-run in one pass; delete fewer items",
                    )
                    return
                verdict = self._reproduced(item, c, cands, started, tried)
                if verdict is None:
                    return  # the budget ran out (already refused)
                if not verdict:
                    lost.append(c.episode)
            if lost:
                run.fail(
                    "G5",
                    f"{item} is deleted, but its procedures on {', '.join(sorted(set(lost))[:5])} are reproduced by "
                    "no remaining item; a kept function must take over each recorded job with the same outcome",
                    item,
                )

    def _alias_targets(self) -> dict[str, str]:
        """Deleted name -> the function it forwards to: this CURATE pass's declared aliases and the live ones."""
        out: dict[str, str] = {}
        try:
            live = dict(self.gate.ev.aliases())
        except AttributeError:
            live = {}
        for a, row in live.items():
            if isinstance(row, dict) and isinstance(row.get("target"), str):
                out[a] = row["target"]
        raw = self.run.manifest_raw if isinstance(self.run.manifest_raw, dict) else {}
        declared = raw.get("aliases")
        if self.cfg.role == "curate" and isinstance(declared, dict):
            out.update(
                {
                    a: t
                    for a, t in declared.items()
                    if isinstance(a, str) and isinstance(t, str)
                },
            )
        return out

    def _reproduced(
        self,
        item: str,
        cover: Any,
        cands: list[str],
        started: float,
        n: int,
    ) -> bool | None:
        """Whether one of *cands* gives on the candidate the outcome *item* gave on the parent for *cover*
        (P6 Amendment B). None when the budget ran out (refused)."""
        run = self.run
        ep = self.ep(cover.episode)
        if ep is None:
            run.note(
                f"G5 take-over check: {item}'s procedure on {cover.episode} cannot be read; not reproduced",
            )
            return False

        def outcome(who: str, tree: Path, side: str) -> Any:
            left = OUTPUTS_BUDGET_S - (time.monotonic() - started)
            if left <= 1:
                run.fail(
                    "G5",
                    f"take-over check not completed: its {OUTPUTS_BUDGET_S:g} s procedure budget ran out at {item}",
                )
                return None
            return run_procedure(
                who,
                cover,
                ep=ep,
                tree=tree,
                python=self.gate.python,
                work=run.tmp / f"procedure-takeover-{n}-{side}",
                blob=self.gate._blob,
                worktree_files=self.cfg.worktree_files,
                signals=self.gate.ev.signals_for,
                runner=self.cfg.runner,
                timeout_s=min(PROCEDURE_S, left),
                checker_visible=self.cfg.checker_visible,
            )

        before = outcome(item, run.p_tree, "parent")
        if before is None:
            return None
        for k, who in enumerate(cands):
            after = outcome(who, run.c_tree, f"candidate-{k}")
            if after is None:
                return None
            if bool(after.ok) == bool(before.ok):
                return True
        return False
