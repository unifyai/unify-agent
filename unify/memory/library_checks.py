"""The library-mode gate checks, run by :class:`.gate.Gate` with ``library_mode``.

With ``Gate(library_mode=None)`` (the default) nothing here runs and the gate is v2's. With a :class:`LibraryConfig`, the gate
(``checks`` on, which needs the library layout, ``layout``) adds these checks:

* **G1:**
  - scope: only ``memory/`` and ``notes/``, never the harness-generated paths;
  - when to use: every new or changed function and note carries its writer-written ``When to use`` section, with
    its four parts (:mod:`.when_to_use`; shape only, never wording), and every folder the pass changes holds a
    ``README.md`` with some text;
  - links: no dangling ``uses:``/``Notes:`` target;
  - fixture provenance: every new or changed file under ``tests/data/`` came through ``fixture()`` and re-derives
    from the recording (:mod:`.fixtures`).
* **G2:**
  - typed covers: an actor cell or work-tree diff that defines a function, or an ``episode`` cover verified by its
    procedure runner (:mod:`.procedures`);
  - the generalisation check: a procedure (``role: procedure``) carries its recipe, and the recipe, run confined,
    gives the expected output on every case it lists, each re-derived from the recording (:mod:`.recipes`); a
    source case (its request with the value it handed back) counts only from an episode that did not fail, and no
    procedure lands from a failed episode. Each failing case refuses the procedure and its recipe with the case and
    the difference report; the counts go to the procedure's verification record (``recipe``);
  - the cross-episode held-out check, for inputs with nothing to perturb;
  - the literal lint (:mod:`.code_lint`).
* **G3:**
  - plain pytest: no test-kit import;
  - per changed function, at least one exact assertion against a trust 1–3 value on a recorded fixture, and no
    self-reimplementing oracle (:mod:`.quality`);
  - no test deleted or weakened without a reason in the manifest's ``tests_changed``;
  - for WRITE, no fewer tests.
  - a stored function's episode covers re-run on the parent and the candidate: green to red is a behaviour change
    (the covers are persisted at merge, :meth:`.evidence.EvidenceStore.add_typed_cover`).
* **G4** never refuses growth: over the index view budget it notes the index's size.
* **G5:**
  - no fewer items;
  - a deleted function's typed episode covers stay covered by a remaining item.

The stage-5 checks run with drawn inputs ``strict`` and mutation gated at :data:`.qa.MUTATION_MIN`
(:func:`.qa.library_config`). Measures go to the item verification record (``GateResult.verification``). The merge
also appends that record to the pass row as one ``library_mode-verification`` line, for the item records.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .admission import is_rejection
from .batch_map import actor_functions
from .code_lint import LINT_MIN
from .episodes import Action, Episode
from .fixtures import DATA_PATH, DRAWN_DIR, HashIndex, module_path, package_dir, verify
from .gitio import Repo
from .held_out import OUTPUTS_BUDGET_S
from .catalogue import estimate_tokens
from . import layout
from . import when_to_use
from .aliases import forwards, uses_name
from .layout import classify, discover, misplaced
from .library_index import INDEX_VIEW_TOKENS, build_links, render_index
from .procedures import (
    PROCEDURE_S,
    cover_raw,
    parse_cover,
    run_procedure,
    snapshot_files,
)
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

CROSS_CASES = (
    16  # recorded inputs from other episodes per function (cross-episode held-out)
)
# the procedure re-runs of the behaviour check, bounded like v2's (held_out.OUTPUTS_BUDGET_S)
BEHAVIOUR_MAX_PROCEDURES = 16
# a function's replay case file under tests/data/ (fixtures.REPLAY_SUFFIX), verified by replay, not by its bytes
REPLAY_SUFFIX = ".replay.jsonl"
OUTPUT_VIEW_BYTES = (
    4000  # the head of a test run's output kept in memory and quoted (explicit, marked)
)
_KEY_SHAPED_BYTES = re.compile(KEY_SHAPED.pattern.encode())


def keep_output(outcome: Any, spool: Path, blobs: Any) -> None:
    """In library mode, a test run's whole output (stdout, then stderr, as v2 joins them; each spooled up to
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


def _no_snapshot(sha: str, dest: Path) -> Path | None:
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
    """The library checks' configuration (``Gate(library_mode=LibraryConfig(...))``): how to read an episode and a
    recorded work tree, and the checks' bounds."""

    episodes: Callable[[str], Episode | None] = _no_episode
    worktree_files: Callable[[str, Path], Path | None] = _no_snapshot
    index_tokens: int = INDEX_VIEW_TOKENS
    cross_cases: int = CROSS_CASES
    lint_min: int = LINT_MIN
    runner: Callable[..., SandboxResult] = run_confined


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

    git = Path(stores.paths.worktree_git)
    reader = snapshot_files(Repo(git)) if (git / "HEAD").is_file() else _no_snapshot
    return LibraryConfig(episodes=episodes, worktree_files=reader)


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
        replays: dict[str, bytes] = {}
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
            if getattr(run, "policy", None) is not None and self._readers_of(entries.get(p)):
                # rev 2: re-deriving a parsed fixture would run its reader; the gate runs none
                run.not_run(owners, f"G1 parsed fixture {p}: re-deriving it would run its reader")
                continue
            if p.endswith(REPLAY_SUFFIX):
                replays[p] = (
                    data  # verified by replaying its instances again, not by its bytes
                )
                continue
            started = time.monotonic()
            problem = verify(
                p,
                data,
                entries.get(p),
                self.ep,
                self.gate._blob,
                parse=self._parse,
            )
            self._parsed(p, entries.get(p), problem, time.monotonic() - started)
            if problem is None:
                if isinstance(entries.get(p), dict):
                    # verified this pass: the examples checks' provenance, persisted on merge
                    run.fixtures_verified[p] = (
                        hashlib.sha256(data).hexdigest(),
                        self._current(entries[p]),
                    )
                    self._reads_for(p, entries[p])
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
            self._replays(raw, replays)
        self._readers_changed()

    # -- stored readers (parsed fixtures) -------------------------------------------------------------------
    def _reader_allowed(self, reader: str) -> str | None:
        """A reader a fixture names is admitted in this pass (a manifest function) or already landed (the parent's),
        and is in the candidate; None when it may be used, else why not (a G1 reason)."""
        run = self.run
        if run.c_bodies.get(reader, ("",))[0] != "function":
            return f"reader {reader} is not a function of the candidate's library"
        listed = any(it.item == reader and it.kind == "function" for it in run.man.items)
        if not listed and reader not in run.p_bodies:
            return f"reader {reader} is neither listed in this pass's manifest nor stored in the library"
        return None

    def _parse(self, reader: str, text: str) -> tuple[str, str]:
        """Run *reader* confined on the candidate's library (:func:`.procedures.reader_parse`), once per text."""
        if getattr(self, "_parser", None) is None:
            from .procedures import reader_parse

            self._parser = reader_parse(
                self.run.c_tree,
                python=self.gate.python,
                work=self.run.tmp / "readers",
                runner=self.cfg.runner,
                allowed=self._reader_allowed,
            )
        return self._parser(reader, text)

    @staticmethod
    def _readers_of(entry: object) -> list[str]:
        from .fixtures import readers_of

        return readers_of(entry)

    def _current(self, entry: dict) -> dict:
        """*entry* with each named reader's digest as the candidate has it (the version that re-derived it)."""
        from .procedures import reader_digest

        def fix(row: dict) -> dict:
            if isinstance(row.get("parser"), str):
                return {**row, "parser_sha256": reader_digest(self.run.c_tree, row["parser"])}
            return row

        if isinstance(entry.get("lines"), list):
            return {**entry, "lines": [fix(r) if isinstance(r, dict) else r for r in entry["lines"]]}
        return fix(dict(entry))

    def _parsed(self, path: str, entry: object, problem: str | None, seconds: float) -> None:
        """The gate result's evidence fields of a parsed fixture G1 re-derived (``GateResult.parsed``)."""
        from .procedures import reader_digest

        readers = self._readers_of(entry)
        res = getattr(self.run, "res", None)
        if not readers or res is None or not hasattr(res, "parsed"):
            return
        res.parsed[path] = {
            "parser": readers,
            "parser_sha256": {r: reader_digest(self.run.c_tree, r) for r in readers},
            "rederived": problem is None,
            "parse_seconds": round(max(0.0, seconds), 3),
        }

    def _reads_for(self, path: str, entry: object) -> None:
        """Record on each reader *entry* names that it produced the fixture at *path* (``reads_for``). Only a
        reader this pass lists gets a verification record; another keeps the record it landed with."""
        listed = {it.item for it in self.run.man.items}
        for reader in self._readers_of(entry):
            if reader not in listed:
                continue
            found = self._record(reader).setdefault("reads_for", [])
            if path not in found:
                found.append(path)
                found.sort()

    def _readers_changed(self) -> None:
        """Every unchanged parsed fixture of the candidate whose reader's code changed in this pass is re-derived
        with the reader as it now is; one whose bytes no longer re-derive is refused (re-make it), so a reader
        cannot change what its fixtures say unseen."""
        from .fixtures import REMAKE
        from .procedures import reader_digest

        run = self.run
        if getattr(run, "policy", None) is not None:
            run.note("G1: parsed fixtures of changed readers not re-derived (rev 2: re-deriving runs the reader)")
            return
        modules = {layout.module_of(p) for p in run.changed}
        if not any(modules):
            return
        stored = getattr(self.gate.ev, "fixture_provenance", None)
        if not callable(stored):
            return
        changed = set(run.changed)
        for p in sorted(run.c_files):
            if p in changed or not p.startswith("memory/") or "/tests/data/" not in p:
                continue
            try:
                data = (run.c_tree / p).read_bytes()
            except OSError:
                continue
            entry = stored(p, hashlib.sha256(data).hexdigest())
            readers = self._readers_of(entry)
            if not readers:
                continue
            rows = entry.get("lines") if isinstance(entry.get("lines"), list) else [entry]
            recorded = {
                r.get("parser"): r.get("parser_sha256") for r in rows if isinstance(r, dict) and r.get("parser")
            }
            if all(reader_digest(run.c_tree, r) == recorded.get(r) for r in readers):
                self._reads_for(p, entry)
                continue
            started = time.monotonic()
            problem = verify(p, data, entry, self.ep, self.gate._blob, parse=self._parse)
            self._parsed(p, entry, problem, time.monotonic() - started)
            if problem is None:
                run.fixtures_verified[p] = (hashlib.sha256(data).hexdigest(), self._current(entry))
                self._reads_for(p, entry)
                continue
            if REMAKE not in problem:
                problem += f"; {REMAKE}"
            run.fail(
                "G1",
                f"fixture provenance: its reader changed in this pass: {problem}",
                self._owners(p),
            )

    def _replays(self, raw: dict, replays: dict[str, bytes]) -> None:
        """New or changed replay case files: their instances are replayed again (:func:`.replay_verify.reverify`);
        a file whose every line verifies is verified provenance this pass, any other is refused.
        """
        from . import replay_verify

        run = self.run
        policy = getattr(run, "policy", None)
        if policy is not None and not policy.replays:
            from .gate_policy import NOT_RUN_REPLAYS

            owners = sorted({o for p in replays for o in (self._owners(p) or [])})
            run.not_run(owners or None, f"G1 replay case files: {NOT_RUN_REPLAYS}")
            return
        replaces_of = {
            r["item"]: r["replaces"]
            for r in (raw.get("items") if isinstance(raw.get("items"), list) else [])
            if isinstance(r, dict)
            and isinstance(r.get("item"), str)
            and isinstance(r.get("replaces"), list)
        }
        scratch = run.tmp / "replay-reverify"
        scratch.mkdir(parents=True, exist_ok=True)
        found, reasons = replay_verify.reverify(
            self.ep,
            replays,
            replaces_of,
            run.c_tree,
            scratch=scratch,
        )
        for p, entry in sorted(found.items()):
            run.fixtures_verified[p] = (hashlib.sha256(replays[p]).hexdigest(), entry)
        owners = sorted({o for p in replays for o in (self._owners(p) or [])})
        for r in reasons:
            run.fail("G1", r, owners or None)

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
        n = 0
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
                    # a function listed only to change its tests: the typed covers it landed with judge its own
                    # code, which did not change, so they are not run again; a new cover is checked as usual
                    if stored is None:
                        stored = self._stored_typed()
                    if cover in stored.get(it.item, []):
                        if cover.type == "episode":
                            kept = getattr(run, "typed_kept", None)
                            if isinstance(kept, dict):
                                kept.setdefault(it.item, []).append(cover.episode)
                        continue
                self._typed_cover(it, cover, n)
                n += 1
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
        from .recipes import check_cases

        run = self.run
        listed = {it.item for it in run.man.items}
        for k, it in enumerate(self._functions()):
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
            policy = getattr(run, "policy", None)
            if policy is not None:
                ok, why = policy.item(it.item)
                if ok:
                    ok, why = policy.item(recipe)
                if not ok:
                    # rev 2: an impure procedure's recipe is never run by the gate
                    run.not_run(owners, f"G2 recipe {recipe}: {why}")
                    self._record(it.item)["recipe"] = {"item": recipe, "cases": len(it.cases), "ran": False,
                                                       "why": why}  # fmt: skip
                    continue
            reports = check_cases(
                recipe,
                [c for _, c in cases],
                tree=run.c_tree,
                python=self.gate.python,
                work=run.tmp / f"recipe-{k}",
                runner=self.cfg.runner,
            )
            kinds = {kind: 0 for kind in ("source", "shown")}
            for _, case in cases:
                kinds[case["kind"]] = kinds.get(case["kind"], 0) + 1
            self._record(it.item)["recipe"] = {
                "item": recipe,
                "cases": len(it.cases),
                "passed": sum(1 for r in reports if r is None),
                "kinds": kinds,
            }
            for (path, case), report in zip(cases, reports):
                if report is not None:
                    run.fail(
                        "G2",
                        f"{it.item}: its recipe {recipe} does not give the expected output of case {path} (a "
                        f"{case['kind']} case of {case['episode']}):\n{report}",
                        owners,
                    )

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
            policy = getattr(run, "policy", None)
            if policy is not None and not policy.replays:
                from .gate_policy import NOT_RUN_REPLAYS

                run.not_run(it.item, f"G2 episode cover {cover.episode}: {NOT_RUN_REPLAYS}")
                self._record(it.item).setdefault("procedures", []).append(
                    {"episode": cover.episode, "runner": cover.runner, "ok": None, "ran": False},
                )
                return
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
            )
            self._record(it.item).setdefault("procedures", []).append(
                {"episode": cover.episode, "runner": cover.runner, "ok": out.ok},
            )
            for note in out.notes:
                run.note(f"G2 procedure: {it.item} on {cover.episode}: {note}")
            if not out.ok:
                cap(
                    run,
                    "G2",
                    f"{it.item} procedure on {cover.episode} ({cover.runner}): {out.reason}",
                    it.item,
                )

    # -- G3 -------------------------------------------------------------------------------------------------
    def g3(self) -> None:
        self._plain()
        self._test_changes()
        for it in self._edited():
            self._exact(it)
        self._procedure_behaviour()
        # last: the examples checks read G3's candidate runs (example_checks)
        from .example_checks import ExampleChecks

        ExampleChecks(self.gate, self.run).judge()

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
        if total == 0:
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

    def _procedure_behaviour(self) -> None:
        """A stored function's ``episode`` covers re-run on the parent's and the candidate's
        library when the pass changes a library module. A procedure green on the parent and red on the
        candidate is a behaviour change: the function now does its recorded job differently.

        Every runner compares with the same recording, so "a different confirmed effect" is green to red. The
        scope is the import graph's: the stored functions of each changed module and of every module importing one
        (``run.graph``, built by the gate's own behaviour check, :meth:`.gate.Gate._call_scope`), and
        the owner is :meth:`.gate.Gate._behaviour_owner`'s. Without a graph (a run that built none) every stored
        procedure is in scope and the owner is the pass's one edited function, else the pass.
        """
        run = self.run
        if not any(classify(p) in ("module", "package_init") for p in run.changed):
            return
        policy = getattr(run, "policy", None)
        if policy is not None and not policy.replays:
            run.note("G3: stored procedures' episode covers not re-run (rev 2: recorded replays are off)")
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
                # fail closed, as _reproduced does: a stored procedure that cannot be compared refuses
                run.fail(
                    "G3",
                    f"behaviour check not completed: {item}'s procedure on {cover.episode} cannot be read",
                    owner_of(item),
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
        """A deleted function's typed ``episode`` covers stay covered by a remaining item.

        As in the behaviour check, a remaining item vouches for a deleted item's procedure only when
        :func:`.procedures.run_procedure` gives on the candidate's library the same outcome (``ok``) as the deleted
        item gave on the parent's. The holders tried are the remaining items that list or store the same cover and,
        for a deleted item that is a live alias, its target: an alias forwarding a
        procedure is held by that re-run. Bounded like the behaviour check, never silently.
        """
        run = self.run
        deleted = set(run.man.deleted)
        if not deleted:
            return
        policy = getattr(run, "policy", None)
        if policy is not None and not policy.replays:
            run.note("G5: deleted procedures' episode covers not re-run (rev 2: recorded replays are off)")
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
        """Deleted name -> the function it forwards to: the live aliases."""
        out: dict[str, str] = {}
        try:
            live = dict(self.gate.ev.aliases())
        except AttributeError:
            live = {}
        for a, row in live.items():
            if isinstance(row, dict) and isinstance(row.get("target"), str):
                out[a] = row["target"]
        return out

    def _reproduced(
        self,
        item: str,
        cover: Any,
        cands: list[str],
        started: float,
        n: int,
    ) -> bool | None:
        """Whether one of *cands* gives on the candidate the outcome *item* gave on the parent for *cover*.
        None when the budget ran out (refused)."""
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
