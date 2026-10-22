"""The library-mode gate checks, run by :class:`.gate.Gate` with a :class:`LibraryConfig`. Nothing here runs code
(rev 2.1, :mod:`.gate_policy`): a check that would need a run labels the items it concerns ``not run: <why>``.

* **G1:**
  - scope: only ``memory/`` and ``notes/``, never the harness-generated paths;
  - when to use: every new or changed function and note carries its writer-written ``When to use`` section, with
    its four parts (:mod:`.when_to_use`; shape only, never wording), and every folder the pass changes holds a
    ``README.md`` with some text;
  - links: no dangling ``uses:``/``Notes:`` target;
  - fixture provenance: every new or changed file under ``tests/data/`` came through ``fixture()`` and re-derives
    from the recording's bytes (:mod:`.fixtures`); a parsed fixture (its reader would run) and a replay case file
    (its recorded code would run) are labelled, not run.
* **G2:**
  - typed covers: an actor cell or work-tree diff that defines a function; an ``episode`` cover is labelled, not
    re-run;
  - a procedure (``role: procedure``) carries its recipe and the cases it lists, each read from the recording; a
    source case counts only from an episode that did not fail, and no procedure lands from a failed episode. The
    recipe is not run: its verification record says so (``recipe``: ``ran: false`` and why);
  - each function's examples (the calls its listed tests make) are labelled, not run.
* **G3:**
  - plain pytest: no test-kit import;
  - per changed function, at least one exact assertion against a trust 1–3 value on a recorded fixture, and no
    self-reimplementing oracle (:mod:`.quality`);
  - no test deleted or weakened without a reason in the manifest's ``tests_changed``;
  - for WRITE, no fewer tests.
* **G4** never refuses growth: over the index view budget it notes the index's size.
* **G5:** no fewer items, and no alias dropped.

Measures go to the item verification record (``GateResult.verification``). The merge also appends that record to the
pass row as one ``library_mode-verification`` line, for the item records.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .admission import is_rejection
from .batch_map import actor_functions
from .episodes import Action, Episode
from .fixtures import DATA_PATH, DRAWN_DIR, HashIndex, module_path, package_dir, readers_of, verify
from .fixed_answer import returns_a_value
from .gate_policy import NOT_RUN_GATE, not_run_reason
from .catalogue import estimate_tokens
from . import layout
from . import when_to_use
from .aliases import forwards, uses_name
from .layout import discover, misplaced
from .library_index import INDEX_VIEW_TOKENS, build_links, render_index
from .procedures import (
    parse_cover,
)
from .quality import (
    TEST_FILE,
    count_items,
    exact_assertions,
    inventory,
    recorded_literals,
    test_changes,
)
from .testkit import imports_kit

# a function's replay case file under tests/data/ (fixtures.REPLAY_SUFFIX), verified by replay, not by its bytes
REPLAY_SUFFIX = ".replay.jsonl"
def _no_episode(eid: str) -> Episode | None:
    return None


def _tool_effect(effects: set[str]) -> str:
    """The effect of a set of recorded tool-call effects: any write writes; an undeclared one says nothing."""
    if "write" in effects:
        return "write"
    return "read" if effects and effects <= {"read"} else "unknown"


def derive_effect(
    typed_covers: list[Any],
    episode: Callable[[str], Episode | None],
) -> str | None:
    """A library function's effect, derived from the procedures it reproduces: ``write``, ``read`` or ``unknown``.

    Structural, from the recording only; the writer declares nothing (no docstring line):
    - an ``episode`` cover run by the ``tool`` runner repeats the episode's tool calls, so it writes when the
      episode recorded a write call, and is unknown when the episode's calls declare no effect;
    - an ``episode`` cover run by the ``worktree`` runner changes files: it writes;
    - an ``episode`` cover run by another runner (a dialogue) records no call it makes: it reads.
    The strongest wins: ``write``, then ``unknown``, then ``read``. None for a function with no ``episode`` cover:
    its examples are calls its tests make, which record no call it makes itself, so no effect is guessed.
    """
    found: set[str] = set()
    for c in typed_covers:
        if getattr(c, "type", None) != "episode":
            continue
        found.add("read")
        if c.runner == "worktree":
            found.add("write")
        elif c.runner == "tool":
            ep = episode(c.episode)
            effects = {
                str(getattr(a, "effect", "unknown"))
                for a in (ep.actions if ep is not None else [])
                if getattr(a, "kind", "tool") == "tool"
            }
            found.add(_tool_effect(effects))
    for effect in ("write", "unknown", "read"):
        if effect in found:
            return effect
    return None


def _index_tokens(tree: Path, examples: dict[str, dict] | None = None) -> int:
    """The tokens of the ``INDEX.md`` the harness would generate for *tree* (the layout's renderer and estimator),
    with each function's example line when *examples* (item -> its recorded example) holds one the tree still
    quotes."""
    from .example_view import example_of, tree_reader

    lib = discover(tree)
    records = {item: {"example": ex} for item, ex in (examples or {}).items()}
    return estimate_tokens(
        render_index(
            lib,
            build_links(lib),
            example_of=example_of(records, tree_reader(tree)),
        ),
    )


@dataclass(frozen=True)
class LibraryConfig:
    """The library checks' configuration (``Gate(library_mode=LibraryConfig(...))``): how to read an episode, and
    the index view's budget."""

    episodes: Callable[[str], Episode | None] = _no_episode
    index_tokens: int = INDEX_VIEW_TOKENS


def config_for(
    stores: Any,
    lookup: Any,
) -> LibraryConfig:
    """The library checks' configuration for a run's stores (:mod:`.integration.consolidate`)."""

    def episodes(eid: str) -> Episode | None:
        try:
            return lookup.episode(eid)
        except Exception:  # noqa: BLE001 - unknown or unreadable: no episode
            return None

    return LibraryConfig(episodes=episodes)


def cap(run: Any, check: str, reason: str, item: Any) -> None:
    """An evidence finding: it sets *item*'s evidence level (``run.cap``, library mode) instead of refusing it."""
    found = getattr(run, "cap", None)
    (found if callable(found) else run.fail)(check, reason, item)


class LibraryChecks:
    def __init__(self, gate: Any, run: Any) -> None:
        self.gate, self.run, self.cfg = gate, run, gate.library_mode
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
        return [
            it
            for it in self._functions()
            if it.item in run.c_bodies
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
        self.aliases_kept()

    # -- G1 -------------------------------------------------------------------------------------------------
    def g1(self) -> None:
        run = self.run
        for p in run.changed:
            if p in run.c_files and (why := misplaced(p)) is not None:
                run.fail("G1", f"file {p}: {why}")
        self._links()
        self._fixtures()
        self._when_to_use()
        self._readmes()

    def _when_to_use(self) -> None:
        """Every new or changed function and note carries its "when to use" section, its four parts present."""
        run = self.run
        functions = [(it.item, "function") for it in self._edited()]
        notes = [
            (it.item, "note")
            for it in run.man.items
            if it.kind == "note"
            and it.item in run.c_bodies
            and run.p_bodies.get(it.item) != run.c_bodies[it.item]
        ]
        for item, kind in functions + notes:
            text = run.c_bodies[item][1]
            if kind == "function":
                try:
                    node = ast.parse(text).body[0]
                except (SyntaxError, ValueError, IndexError):
                    continue  # an unreadable definition is refused elsewhere
                problems = when_to_use.function_problems(ast.get_docstring(node) or "")
            else:
                problems = when_to_use.note_problems(text)
            for problem in problems:
                run.fail("G1", f"{item} {problem}", item)

    def _readmes(self) -> None:
        """Every library folder the pass changes, and that still holds a file, holds a ``README.md`` with text."""
        run = self.run
        folders = sorted({f for p in run.changed if (f := when_to_use.folder(p)) is not None})
        for folder in folders:
            if not any(p.startswith(folder + "/") for p in run.c_files):
                continue  # the pass emptied it
            readme = when_to_use.readme_of(folder)
            data = None
            if readme in run.c_files:
                try:
                    data = (Path(run.c_tree) / readme).read_bytes()
                except OSError:
                    data = None
            why = when_to_use.readme_problem(data)
            if why is None:
                continue
            reason = (
                f"folder {folder}/ {why}: each library folder holds a short {when_to_use.README} saying what it "
                "holds and how its items relate"
            )
            items = [it.item for it in run.man.items if when_to_use.folder(layout.item_path(it.item)) == folder]
            for item in items or [None]:
                run.fail("G1", reason, item)

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
        replays: list[str] = []
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
            if readers_of(entries.get(p)):
                # re-deriving a parsed fixture would run its reader: the gate runs no code
                run.not_run(owners, f"G1 parsed fixture {p}: re-deriving it would run its reader; {NOT_RUN_GATE}")
                continue
            if p.endswith(REPLAY_SUFFIX):
                # a replay case file is verified by replaying its instances, which runs recorded code
                replays.append(p)
                continue
            problem = verify(p, data, entries.get(p), self.ep, self.gate._blob)
            if problem is None:
                if isinstance(entries.get(p), dict):
                    # verified this pass from the recording's bytes, persisted on merge
                    run.fixtures_verified[p] = (hashlib.sha256(data).hexdigest(), dict(entries[p]))
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
        if replays:
            owners = sorted({o for p in replays for o in (self._owners(p) or [])})
            run.not_run(owners or None, f"G1 replay case files: {NOT_RUN_GATE}")

    # -- G2 -------------------------------------------------------------------------------------------------
    def effects(self) -> None:
        """Record each manifest function's effect (:func:`derive_effect`) in its verification record, from which
        the item record and the index take it. Measured, never a refusal."""
        for it in self._functions():
            effect = derive_effect(list(getattr(it, "typed_covers", [])), self.ep)
            if effect is not None:
                self._record(it.item)["effect"] = effect

    def g2(self) -> None:
        self.effects()
        stored: dict[str, list[Any]] | None = None
        run = self.run
        for it in self._functions():
            unchanged = (
                it.item in run.p_bodies
                and it.item in run.c_bodies
                and run.p_bodies[it.item][:2] == run.c_bodies[it.item][:2]
            )
            for cover in getattr(it, "typed_covers", []):
                if unchanged:
                    # a function listed only to change its tests: the typed covers it landed with were checked
                    # then, and its code did not change; a new cover is checked as usual
                    if stored is None:
                        stored = self._stored_typed()
                    if cover in stored.get(it.item, []):
                        continue
                self._typed_cover(it, cover)
        self._recipes()

    # -- the generalisation check ---------------------------------------------------------------------------
    def _episode_failed(self, eid: str) -> bool:
        """Whether episode *eid*'s outcome is failed: a signal the regime lets the run observe contradicts the value
        it handed back (a checker verdict only when the actor saw it; :func:`.trust.signal_outcome`). An episode with
        no such signal (none recorded, or a regime without one) has not failed."""
        from .captures import final_value_signals
        from .trust import signal_outcome

        ep = self.ep(eid)
        if ep is None:
            return False
        try:
            sigs = list(self.gate.ev.signals_for(eid) or [])
        except Exception:  # noqa: BLE001 - no readable signal: nothing contradicts it
            sigs = []
        regime = str(getattr(ep, "regime", "") or "none")
        out = signal_outcome(final_value_signals(sigs, ep), regime)
        return out["verdict"] == "rejected" or out["end_outcome"] == "contrary"

    def _recipes(self) -> None:
        """The generalisation check (module docstring, G2) on every procedure of the manifest."""
        from .fixtures import read_case

        run = self.run
        listed = {it.item for it in run.man.items}
        for it in self._functions():
            if getattr(it, "role", None) != "procedure":
                continue
            recipe = getattr(it, "recipe", None)
            owners = [it.item] + ([recipe] if recipe and recipe != it.item and recipe in listed else [])
            if not recipe:
                run.fail(
                    "G2",
                    f"{it.item} is a procedure without a recipe: add apply(request) to its module, the whole job on "
                    "the request it is handed with every parameter read from it, and name it in the entry's recipe",
                    it.item,
                )
                continue
            if run.c_bodies.get(recipe, ("",))[0] != "function":
                run.fail("G2", f"{it.item}: its recipe {recipe} is not a function of the candidate's library", owners)
                continue
            failed = sorted(e for e in set(it.source_episodes) if self._episode_failed(e))
            if failed:
                run.fail(
                    "G2",
                    f"{it.item}: its source episode {', '.join(failed)} failed; a failed episode never lands a "
                    "procedure (land a note of what was tried and what broke)",
                    owners,
                )
            cases: list[tuple[str, dict]] = []
            for path in it.cases:
                if path not in run.c_files:
                    run.fail("G2", f"{it.item}: case {path} is not in the candidate", owners)
                    continue
                case, problem = read_case(path, (run.c_tree / path).read_bytes(), self.ep, self.gate._blob)
                if problem is not None:
                    run.fail("G2", f"{it.item}: case {problem}", owners)
                    continue
                if case["kind"] == "source" and self._episode_failed(case["episode"]):
                    run.fail(
                        "G2",
                        f"{it.item}: case {path} is the source case of {case['episode']}, which failed: the value it "
                        "handed back is not a known outcome",
                        owners,
                    )
                    continue
                cases.append((path, case))
            if not it.cases:
                run.fail(
                    "G2",
                    f"{it.item}: its recipe has no case to run on: list in cases the source case (its request with "
                    "the value it handed back) and every input the request or the environment showed with its "
                    "output, each made with fixture(..., expected=...)",
                    owners,
                )
            # the gate runs no recipe (rev 2.1: no code at all), so its label says why
            why = self._not_run(it.item)
            run.not_run(owners, f"G2 recipe {recipe}: {why}")
            self._record(it.item)["recipe"] = {"item": recipe, "cases": len(it.cases), "ran": False, "why": why}

    def _typed_cover(self, it: Any, cover: Any) -> None:
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
                cap(
                    run,
                    "G2",
                    f"{it.item} covers cell {cover.index} of {cover.episode}, which defines no Python function",
                    it.item,
                )
        elif cover.type == "diff":
            if not any(f["source"] == "diff" for f in actor_functions(ep)):
                cap(
                    run,
                    "G2",
                    f"{it.item} covers the work-tree diff of {cover.episode}, which adds no Python function",
                    it.item,
                )
        elif cover.type == "episode":
            # re-running the procedure on its recorded episode runs code: the gate runs none
            run.not_run(it.item, f"G2 episode cover {cover.episode}: {NOT_RUN_GATE}")
            self._record(it.item).setdefault("procedures", []).append(
                {"episode": cover.episode, "runner": cover.runner, "ok": None, "ran": False},
            )

    # -- G3 -------------------------------------------------------------------------------------------------
    def g3(self) -> None:
        self._plain()
        self._test_changes()
        for it in self._edited():
            self._exact(it)
        # a function's examples are the calls its listed tests make: the gate runs none of them
        for it in self._functions():
            if it.item in self.run.c_bodies:
                self.run.not_run(it.item, f"G2 examples: {self._not_run(it.item)}")

    def _not_run(self, item: str) -> str:
        """Why the gate did not run *item* (:func:`.gate_policy.not_run_reason`, from its candidate source)."""
        body = self.run.c_bodies.get(item)
        return not_run_reason(body[1] if body is not None and body[0] == "function" else None)

    def _plain(self) -> None:
        run = self.run
        for p in sorted(set(run.changed) & set(run.c_files)):
            if p.startswith("memory/") and "/tests/" in p and p.endswith(".py"):
                if imports_kit((run.c_tree / p).read_bytes()):
                    run.fail(
                        "G3",
                        f"[library_mode:plain] {p} imports the gate's test kit (memlab or the pin plugin); a library's tests "
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
        if len(after) < len(before):
            run.fail(
                "G3",
                f"a pass may not reduce the number of tests ({len(before)} to {len(after)})",
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
                f"a pass may not reduce the number of items ({n_p} to {n_c})",
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
                    cap(
                        run,
                        "G3",
                        f"[library_mode:oracle] {t} line {e.line} computes its expected value with {it.item}'s own "
                        "expression; take the expected value from a recorded fixture",
                        it.item,
                    )
                elif e.on_fixture and e.trusted:
                    total += 1
        record["exact_assertions"] = total
        # a function that hands back no value has no result to assert: its outcome is what it did, and its tests
        # check that instead (the calls it makes, the files it writes), in any world
        if total == 0 and returns_a_value(function_source, name):
            cap(
                run,
                "G3",
                f"[library_mode:exact] {it.item}'s tests hold no exact assertion of its result, on a recorded fixture, "
                "against a trust 1-3 value (a recorded response, a value a successful episode used, or one "
                "recorded across episodes). A form that passes: both sides come from fixtures made with `fixture` "
                "from the recording, `assert f(load(fixture_in)) == load(fixture_out)`; the expected value is never "
                "computed by the function or its expression",
                it.item,
            )

    # -- G4 -------------------------------------------------------------------------------------------------
    def g4(self) -> None:
        run = self.run
        try:
            # the example lines the index will show count too (the examples check, in G3, recorded them)
            examples = {
                item: v["examples"].get("example")
                for item, v in getattr(run, "verification", {}).items()
                if isinstance(v, dict) and isinstance(v.get("examples"), dict)
            }
            size = _index_tokens(run.c_tree, examples)
        except (ValueError, OSError) as exc:
            run.fail("G4", f"index not built: {exc}"[:300])
            return
        if size > self.cfg.index_tokens:
            run.note(f"G4: the index is {size} tokens, over its {self.cfg.index_tokens}-token view; growth is not refused")

    # -- procedures in the behaviour check and in deletion ---------------------------------------
    # -- aliases and retirements --------------------------------------------------------------------
    def aliases_kept(self) -> None:
        """A pass changes no item's name by alias or retirement, and keeps every live alias forwarding: static, it
        reads the manifest and both trees and runs nothing. The pass's renames go to ``run.curations``;
        :meth:`.gate.Gate.merge` records them for a landed pass."""
        run = self.run
        raw = run.manifest_raw if isinstance(run.manifest_raw, dict) else {}
        aliases, retired = raw.get("aliases") or {}, raw.get("retired") or {}
        live = dict(self.gate.ev.aliases())
        run.curations = self._renames(raw)
        if aliases or retired:
            run.fail("G1", "a pass may not declare aliases or retirements")
        c_mods = layout.library_modules(run.c_tree)
        for alias in sorted(live):
            if self._forwards(run.c_tree, alias, c_mods) is None:
                run.fail("G5", f"alias {alias} is no longer forwarded; a pass may not drop an alias")

    def _renames(self, raw: dict) -> list[dict]:
        """The manifest's ``renames`` (old item id -> new item id), any role: the item keeps its identity (its
        record's ``uid``, use and lineage) under the new id. Refused: an old id that is not an item of the parent,
        a new id that is not an item of the candidate or is already an item of the parent (merging two items is an
        alias), and two old ids renamed to one new id. Returns the ``rename`` rows the gate records on landing.
        """
        run = self.run
        renames = raw.get("renames") or {}
        if not isinstance(renames, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in renames.items()
        ):
            run.fail("G1", "renames maps old item ids to new item ids (strings)")
            return []
        if not renames:
            return []
        p_lib, c_lib = layout.discover(run.p_tree), layout.discover(run.c_tree)
        p_ids = {f.item_id for f in p_lib.functions} | {n.item_id for n in p_lib.notes}
        c_ids = {f.item_id for f in c_lib.functions} | {n.item_id for n in c_lib.notes}
        rows: list[dict] = []
        taken: dict[str, str] = {}
        for old, new in sorted(renames.items()):
            if old not in p_ids:
                run.fail("G1", f"rename of {old}: it is not an item of the parent")
            elif new not in c_ids:
                run.fail(
                    "G1",
                    f"rename of {old} to {new}: {new} is not an item of the candidate",
                )
            elif new in p_ids:
                run.fail(
                    "G1",
                    f"rename of {old} to {new}: {new} is already an item of the parent; a rename may not merge two "
                    "items",
                )
            elif new in taken:
                run.fail(
                    "G1",
                    f"rename of {old} to {new}: {taken[new]} is also renamed to {new}",
                )
            else:
                taken[new] = old
                rows.append(
                    {"item": old, "action": "rename", "target": new, "reason": ""},
                )
        return rows

    def _parent_chain(self) -> set[str]:
        """The parent and the commits under it reached through harness status commits only, so a
        status commit between the alias's pass and this one does not end its "one more pass".
        """
        run = self.run
        out = {run.parent}
        mem = getattr(self.gate, "mem", None)
        if mem is None:
            return out
        from .memory_writer import is_status_commit

        c = run.parent
        for _ in range(200):
            try:
                if not is_status_commit(mem, c):
                    break
                c = mem.run("rev-parse", f"{c}^").strip()
            except Exception:  # noqa: BLE001 - an unreadable commit ends the walk
                break
            out.add(c)
        return out

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
        """The parent's tests of *alias* that the candidate keeps are collected through the alias, so they
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
