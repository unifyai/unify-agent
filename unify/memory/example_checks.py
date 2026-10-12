"""A library function's examples, judged by the gate: the calls its listed tests make on it.

The gate does not call a library function itself. Each candidate run of a listed test records, through the
observe-only plugin of :mod:`.example_calls`, every call of the manifest's functions made from test-side code. This
module judges those records:

- a function needs a call in its listed tests (``[examples:none]``) that returns on arguments loaded from fixtures
  made with ``fixture`` or computed in the test from one (``[examples:grounded]``); scalar literal arguments are
  allowed and recorded;
- an example call may return or raise ``MemoryInputError``, nothing else (``[examples:raised]``);
- a record that is incomplete judges nothing (``[examples:record]``, fail closed), and a function the plugin cannot
  observe is refused (``[examples:unobserved]``);
- the fixtures' provenance, the cross-episode rule, the literal lint and the rule check come from
  :mod:`.example_evidence`; the check with unseen fixture values from :mod:`.example_perturb`; the quoted example
  and the docstring's ``Example:`` line from :mod:`.example_view`; argument shapes from :mod:`.shape_rows`. Each is
  injected (``ExampleChecks(gate, run, evidence=..., perturb=..., view=..., shapes=..., ledger=...)``), so the
  checks are tested with fakes.

Behaviour of stored functions (G3) is compared on the calls the parent's tests make, on the parent's library and on
the candidate's (:meth:`ExampleChecks.behaviour`); a deleted function's fixtures must stay used by a remaining
function's examples (:meth:`ExampleChecks.retained`).

Reasons name ids, paths, line numbers, field names and counts, never a recorded value.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import time
from pathlib import Path
from typing import Any, Callable

from .example_calls import CallKey, CallRecord, ExampleCall, unresolved_kind

#: How many places (file:line or call keys) one reason names.
MAX_NAMED = 5
#: The seconds of one gate run's checks with unseen fixture values when the perturbation module names none.
DEFAULT_PERTURB_BUDGET_S = 240.0
GROUNDED_KINDS = ("copied", "derived")


def _module(name: str) -> Any:
    return importlib.import_module(f"{__package__}.{name}")


def _where(c: ExampleCall) -> str:
    return f"{c.file}:{c.line}"


def _places(calls: list[ExampleCall]) -> str:
    seen: list[str] = []
    for c in sorted(calls, key=lambda c: c.key):
        w = _where(c)
        if w not in seen:
            seen.append(w)
    shown = ", ".join(seen[:MAX_NAMED])
    return shown + (", …" if len(seen) > MAX_NAMED else "")


def key_text(key: CallKey) -> str:
    """A call key as a reason names it: ``<test>[<param>=<index>,…]#<ordinal>``."""
    idx = ",".join(f"{k}={v}" for k, v in key.indices)
    return f"{key.test}{f'[{idx}]' if idx else ''}#{key.ordinal}"


def merge_records(records: list[CallRecord | None]) -> CallRecord:
    """One record of several runs: complete only when every run's record is."""
    out = CallRecord(loaded=bool(records) and all(r is not None and r.loaded for r in records))
    out.ended = bool(records) and all(r is not None and r.ended for r in records)
    for r in records:
        if r is None:
            continue
        out.calls.extend(r.calls)
        out.unreadable += r.unreadable
        out.dropped += r.dropped
        out.unresolved += [u for u in r.unresolved if u not in out.unresolved]
        out.unobservable = out.unobservable or r.unobservable
    return out


def incomplete_why(rec: CallRecord | None) -> str | None:
    """None for a complete record, else the counts a reason names."""
    if rec is not None and rec.complete:
        return None
    if rec is None or not rec.loaded:
        return "0 unreadable lines, 0 calls not recorded, unobservable: the recorder did not load"
    why = f"{rec.unreadable} unreadable lines, {rec.dropped} calls not recorded"
    if rec.unobservable is not None:
        why += f", unobservable: {rec.unobservable}"
    elif not rec.ended:
        why += ", unobservable: the record has no end line"
    return why


def record_reason(test: str, why: str) -> str:
    return f"[examples:record] the call record of {test} is incomplete ({why}); its examples are not judged"


def docstring_of(tree: Path, item: str) -> str:
    """The docstring of *item*'s ``def`` in *tree* ("" when it has none or is not a ``def``)."""
    from .fixtures import module_path

    try:
        mod = ast.parse((Path(tree) / module_path(item)).read_bytes())
    except (OSError, SyntaxError, ValueError):
        return ""
    name = item.split(":", 1)[1]
    for node in mod.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_docstring(node) or ""
    return ""


class ExampleChecks:
    """The examples checks of one gate run (*gate*: :class:`.gate.Gate`, *run*: its ``_Run``)."""

    def __init__(
        self,
        gate: Any,
        run: Any,
        *,
        evidence: Any = None,
        perturb: Any = None,
        view: Any = None,
        shapes: Any = None,
        ledger: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gate, self.run, self.clock = gate, run, clock
        self._given = {
            "example_evidence": evidence,
            "example_perturb": perturb,
            "example_view": view,
            "shape_rows": shapes,
            "fixtures": ledger,
        }
        self._calls: dict[str, list[ExampleCall]] | None = None
        self.records: dict[str, CallRecord | None] = {}  # listed test file -> its candidate run's record
        self.incomplete: set[str] = set()  # items whose examples are not judged
        self.unresolved: dict[str, str] = {}  # item -> what it is
        self.uncalled: set[str] = set()  # behaviour: stored functions no parent test calls
        self._episodes: dict[str, Any] = {}

    def part(self, name: str) -> Any:
        if self._given.get(name) is None:
            self._given[name] = _module(name)
        return self._given[name]

    # -- the record -------------------------------------------------------------------------------------------
    def functions(self) -> list[Any]:
        return [
            it for it in self.run.man.items if it.kind == "function" and it.item in self.run.c_bodies
        ]

    def targets(self) -> list[str]:
        """The manifest's function items present in the candidate."""
        return sorted({it.item for it in self.functions()})

    def listed(self, it: Any) -> list[str]:
        return [t for t in it.tests if t in self.run.c_files]

    def record(self) -> dict[str, list[ExampleCall]]:
        """Item -> its example calls in its listed tests. A new or changed listed test's record comes from G3's
        candidate run (``run.qa_first``); an unchanged listed test runs once more on the candidate."""
        if self._calls is not None:
            return self._calls
        run, targets = self.run, self.targets()
        out: dict[str, list[ExampleCall]] = {t: [] for t in targets}
        tests = sorted({t for it in self.functions() for t in self.listed(it)})
        for t in tests:
            o = run.qa_first.get(t)
            rec = getattr(o, "calls", None) if o is not None else None
            if rec is None:
                o = self.gate._pytest(run.c_tree, t, qa_env=getattr(run, "qa_env", None), calls=targets)
                rec = o.calls
            self.records[t] = rec
        for it in self.functions():
            for t in self.listed(it):
                rec = self.records.get(t)
                why = incomplete_why(rec)
                if why is not None:
                    self.incomplete.add(it.item)
                    continue
                for entry in rec.unresolved:
                    item, kind = unresolved_kind(entry)
                    if item == it.item:
                        self.unresolved[it.item] = kind
                keys = {c.key for c in out[it.item]}
                out[it.item] += [c for c in rec.calls if c.item == it.item and c.key not in keys]
        for t in tests:
            why = incomplete_why(self.records.get(t))
            if why is not None:
                owners = [it.item for it in self.functions() if t in it.tests]
                self.run.fail("G3", record_reason(t, why), owners or None)
        self._calls = out
        return out

    # -- G2 (judged after G3's runs) --------------------------------------------------------------------------
    def judge(self) -> None:
        """Every manifest function's examples: B-rules here, the evidence, perturbation, view and shape parts called
        with the record."""
        calls = self.record()
        perturb = self.part("example_perturb")
        deadline = self.clock() + float(getattr(perturb, "PERTURB_BUDGET_S", DEFAULT_PERTURB_BUDGET_S))
        for it in self.functions():
            if it.item in self.incomplete:
                continue
            if it.item in self.unresolved:
                self.run.fail(
                    "G2",
                    f"[examples:unobserved] {it.item} cannot be observed when called (it is not a Python function: "
                    f"{self.unresolved[it.item]}); define it with def",
                    it.item,
                )
                continue
            mine = calls.get(it.item, [])
            if not mine:
                self.run.fail(
                    "G2",
                    f"[examples:none] {it.item}: none of its listed tests calls it; its examples are the calls its "
                    "tests make on fixtures made with fixture",
                    it.item,
                )
                continue
            self._judge(it, sorted(mine, key=lambda c: c.key), deadline)

    def _verification(self, item: str) -> dict:
        return self.run.verification.setdefault(item, {})

    def _episode(self, eid: str) -> Any:
        if eid not in self._episodes:
            try:
                cfg = getattr(self.gate, "library_mode", None)
                self._episodes[eid] = cfg.episodes(eid) if cfg is not None else None
            except Exception:  # noqa: BLE001 - unreadable: no episode
                self._episodes[eid] = None
        return self._episodes[eid]

    def _package_episodes(self, item: str) -> list[str]:
        from .fixtures import package_dir

        pkg = package_dir(item)
        out: set[str] = set()
        for it in self.run.man.items:
            if it.kind == "function" and package_dir(it.item) == pkg:
                out |= set(it.source_episodes)
                out |= {c.episode for c in getattr(it, "typed_covers", [])}
        return sorted(out)

    def _data_files(self, item: str) -> dict[str, bytes]:
        from .fixtures import package_dir

        prefix = f"{package_dir(item)}/tests/data/"
        out: dict[str, bytes] = {}
        for p in sorted(self.run.c_files):
            if p.startswith(prefix):
                try:
                    out[p] = (self.run.c_tree / p).read_bytes()
                except OSError:
                    continue
        return out

    def _ledger(self, item: str) -> tuple[dict[str, dict], dict[str, dict]]:
        """(path -> provenance entry, path -> views) of the fixtures of *item*'s package."""
        fx, run = self.part("fixtures"), self.run
        files = self._data_files(item)
        in_pass = {p: e for p, e in dict(getattr(run, "fixtures_verified", {}) or {}).items() if p in files}
        stored = getattr(self.gate.ev, "fixture_provenance", None)
        ledger, notes = fx.ledger_for(
            files,
            in_pass,
            stored if callable(stored) else (lambda path, sha: None),
            self._episode,
            self.gate._blob,
            self._package_episodes(item),
        )
        for n in notes:
            run.note(n)
        views = {p: fx.value_views(p, files[p], ledger.get(p)) for p in files if p in ledger}
        return ledger, views

    def _pool(self, fixtures: list[Any]) -> tuple[list, bool]:
        fx = self.part("fixtures")
        covers = []
        for f in fixtures:
            found = fx.source_action(f.entry, self._episode)
            if found is not None:
                covers.append(found)
        if not covers:
            return [], False
        return self.gate._pool(self.run, covers)

    def _run_tests(self, item: str) -> Callable[[Path, list[str]], Any]:
        def run_tests(tree: Path, tests: list[str]) -> Any:
            from .sandbox_run import PytestOutcome

            outs = [
                self.gate._pytest(Path(tree), t, qa_env=getattr(self.run, "qa_env", None), calls=[item])
                for t in tests
            ]
            merged = PytestOutcome(valid=all(o.valid for o in outs) and bool(outs))
            for o in outs:
                merged.passed |= o.passed
                merged.failed |= o.failed
                merged.skipped |= o.skipped
                merged.timed_out = merged.timed_out or o.timed_out
                merged.returncode = merged.returncode or o.returncode
            merged.calls = merge_records([o.calls for o in outs])
            return merged

        return run_tests

    @staticmethod
    def _structured(arg: Any) -> bool:
        from .memory_helper import value_shape

        if arg.path is not None or arg.cut:
            return True
        try:
            return value_shape(arg.json) is not None
        except Exception:  # noqa: BLE001 - a value without a shape is a scalar or text
            return False

    def _judge(self, it: Any, calls: list[ExampleCall], deadline: float) -> None:
        run, item = self.run, it.item
        evidence, perturb = self.part("example_evidence"), self.part("example_perturb")
        rec = self._verification(item)
        by_key = {c.key: c for c in calls}
        ledger, views = self._ledger(item)
        inputs = perturb.input_fixtures(item, calls, views, ledger, run.c_tree)
        pool, capped = self._pool(inputs)
        from .held_out import seen_actions

        verdict = perturb.check(
            item,
            calls=calls,
            fixtures=inputs,
            tree=run.c_tree,
            tests=self.listed(it),
            run_tests=self._run_tests(item),
            pool=pool,
            pool_capped=capped,
            seen=seen_actions(self.gate._named_episodes(run), self.gate.lookup),
            blob=self.gate._blob,
            field_types=dict(getattr(it, "field_types", None) or {}),
            work=run.tmp / f"perturb-{hashlib.sha256(item.encode()).hexdigest()[:12]}",
            deadline=deadline,
        )
        derived = dict(getattr(verdict, "derived", {}) or {})
        ground = {c.key: evidence.grounding(c, views, derived.get(c.key, {})) for c in calls}
        accepted = [c for c in calls if c.outcome == "returned"]
        grounded: set[CallKey] = set()
        literal: set[str] = set()
        for c in accepted:
            g = ground.get(c.key, {})
            kinds = {a.name: getattr(g.get(a.name), "kind", "literal") for a in c.args}
            if not any(k in GROUNDED_KINDS for k in kinds.values()):
                continue
            if all(kinds[a.name] in GROUNDED_KINDS or not self._structured(a) for a in c.args):
                grounded.add(c.key)
                literal |= {a.name for a in c.args if kinds[a.name] not in GROUNDED_KINDS}
        if not grounded:
            run.fail(
                "G2",
                f"[examples:grounded] {item}: no call of it in its tests returns on fixture values ({len(calls)} "
                f"calls; each passes only literal values: {_places(calls)}); load an argument from a fixture made "
                "with fixture, or compute it in the test from one",
                item,
            )
        raised = [c for c in calls if c.outcome == "raised"]
        named: set[tuple[str | None, str]] = set()
        for c in raised:
            if (c.raised, _where(c)) in named or len(named) >= MAX_NAMED:
                continue
            named.add((c.raised, _where(c)))
            run.fail(
                "G2",
                f"[examples:raised] {item} raises {c.raised or 'another exception'} on its example call at "
                f"{_where(c)}; a function returns or raises MemoryInputError",
                item,
            )
        self._perturbation(item, verdict, by_key, rec)
        typed = {
            p.get("episode")
            for p in rec.get("procedures", [])
            if isinstance(p, dict) and p.get("ok") and isinstance(p.get("episode"), str)
        }
        reasons, fields = evidence.evidence_check(item, calls, ground, typed)
        for r in reasons:
            run.fail("G2", r, item)
        if self._edited(item):
            self._lint(item, evidence.lint_values(calls, ground))
            self._rule(item, evidence.rule_episodes({k: ground[k] for k in grounded}))
            self._docstring(it, calls, grounded)
        view = self.part("example_view")
        doc = docstring_of(run.c_tree, item)
        example = view.pick_example(item, calls, grounded, view.docstring_example(doc), run.c_tree)
        body = run.c_bodies.get(item)
        if body is not None:
            from .catalogue import body_digest

            found = self.part("shape_rows").arg_descriptors(calls, run.c_tree, note=run.note)
            if found:
                run.shapes[item] = (body_digest(body[1]), found)
        examples = rec.setdefault("examples", {})
        examples.update(
            {
                "calls": len(calls),
                "accepted": len(accepted),
                "refused": sum(1 for c in calls if c.outcome == "refused"),
                "raised": len(raised),
                "grounded_calls": len(grounded),
                "episodes": list(fields.get("episodes", [])),
                "fixtures": list(fields.get("fixtures", [])),
                "literal_args": sorted(literal),
                "example": example,
            },
        )

    def _perturbation(self, item: str, verdict: Any, by_key: dict[CallKey, ExampleCall], rec: dict) -> None:
        run = self.run

        def where(key: CallKey) -> str:
            c = by_key.get(key)
            return _where(c) if c is not None else key_text(key)

        for key, fixture, fld in list(getattr(verdict, "refused", []))[:MAX_NAMED]:
            run.fail(
                "G2",
                f"[examples:perturbed-refused] {item} refuses fixture values it accepted unchanged: with {fld} of "
                f"{fixture} replaced by an unseen value of its type, its call at {where(key)} raises "
                "MemoryInputError; refuse inputs by their shape, not by values the recording happened to hold",
                item,
            )
        for key, fixture, fld, name in list(getattr(verdict, "special", []))[:MAX_NAMED]:
            run.fail(
                "G2",
                f"[examples:special-cased] {item} handles {fixture} only for its recorded values: with {fld} "
                f"replaced by an unseen value of its type, its call at {where(key)} raises {name} where it "
                "returned before",
                item,
            )
        for why in list(getattr(verdict, "unjudged", []))[:MAX_NAMED]:
            run.fail(
                "G2",
                f"[examples:unjudged] {item}: the check with unseen fixture values did not complete ({why}); a "
                "function is admitted only once it is judged",
                item,
            )
        for note in getattr(verdict, "notes", []):
            run.note(f"G2 examples: {item}: {note}")
        changes = list(getattr(verdict, "type_changes", None) or [])
        perturbation = {k: getattr(verdict, k, 0) for k in ("runs", "fields", "judged", "unstable")}
        perturbation["type_changes"] = changes
        examples = rec.setdefault("examples", {})
        examples["perturbation"] = perturbation
        examples["type_changes"] = len(changes)
        if changes:
            fixtures = sorted({str(c.get("fixture")) for c in changes if isinstance(c, dict) and c.get("fixture")})
            run.note(
                f"G2 examples: {item}: {len(changes)} calls return None or another type with unseen fixture "
                f"values ({', '.join(fixtures[:MAX_NAMED]) or 'no fixture named'}); recorded for review",
            )

    def _edited(self, item: str) -> bool:
        run = self.run
        return item in run.c_bodies and run.p_bodies.get(item, ("", ""))[:2] != run.c_bodies[item][:2]

    def _lint(self, item: str, values: list[Any]) -> None:
        from .code_lint import LINT_MIN, literal_problems, varying_values
        from .fixtures import module_path

        run = self.run
        if len(values) < 2:
            run.note(f"G2 literal lint: {item} has fewer than 2 recorded inputs, so nothing varies yet")
            return
        try:
            source = (run.c_tree / module_path(item)).read_bytes()
        except OSError:
            return  # an absent module is refused elsewhere
        cfg = getattr(self.gate, "library_mode", None)
        lint_min = getattr(cfg, "lint_min", LINT_MIN)
        found = literal_problems(source, item.split(":", 1)[1], varying_values(values), lint_min=lint_min)
        for line, what in found[:MAX_NAMED]:
            run.fail(
                "G2",
                f"{item} line {line}: a {what} literal equals a value that varies across its recorded inputs; "
                "make it a parameter",
                item,
            )

    def _rule(self, item: str, episodes: set[str]) -> None:
        """An added or changed function that replaces a computed value under a condition needs examples from
        ``RULE_EPISODES`` episodes."""
        from .gate import RULE_EPISODES

        run = self.run
        found = [i for i in (run.c_report.items if run.c_report else []) if i.item_id == item]
        if not found:
            return  # an absent item (or an unreadable module) is refused elsewhere
        info = found[0]
        if info.rule_unchecked:
            run.note(f"G2 rule check: {item} is too large to analyse; not checked")
        elif info.rule_line and len(episodes) < RULE_EPISODES:
            run.fail(
                "G2",
                f"{item} replaces a value it computed from its input under a condition (line {info.rule_line} of "
                f"{info.path}); a function that encodes such a rule needs examples from at least {RULE_EPISODES} "
                "episodes",
                item,
            )

    def _docstring(self, it: Any, calls: list[ExampleCall], grounded: set[CallKey]) -> None:
        view = self.part("example_view")
        doc = docstring_of(self.run.c_tree, it.item)
        for problem in view.docstring_problems(it.item, doc, list(it.tests), calls, grounded):
            self.run.fail("G1", problem, it.item)

    # -- G3: behaviour of stored functions ------------------------------------------------------------------
    def behaviour(
        self,
        base: dict[str, CallRecord],
        reg: dict[str, CallRecord],
        scope: set[str],
        *,
        strict: set[str] | frozenset[str] = frozenset(),
        owner: Callable[[str], str | None] | None = None,
    ) -> None:
        """Compare every call of a *scope* function in the parent's tests on the parent's library (*base*, suite ->
        record) and on the candidate's (*reg*): outcome, ``result_sha`` and ``raised`` must be equal, and a
        ``result_sha`` of null on either side is uncomparable (fail closed). *strict* functions claim the clean-up
        exemption and are refused with its text. *owner* names the item a difference belongs to (None: the pass).
        Sets ``run.same`` to the scope functions compared equal on at least one call, and :attr:`uncalled` to those
        no parent test calls."""
        run = self.run
        before: dict[str, dict[CallKey, ExampleCall]] = {}
        after: dict[str, dict[CallKey, ExampleCall]] = {}
        for rel in sorted(base):
            b, r = base[rel], reg.get(rel)
            why = incomplete_why(b) or (incomplete_why(r) if rel in reg else None)
            if why is not None:
                # fail closed: a stored function's calls in this suite cannot be compared
                run.fail("G3", record_reason(rel, why), None)
                continue
            for c in b.calls if b is not None else []:
                if c.item in scope:
                    before.setdefault(c.item, {})[c.key] = c
            for c in r.calls if r is not None else []:
                if c.item in scope:
                    after.setdefault(c.item, {})[c.key] = c
        same: set[str] = set()
        for item in sorted(scope):
            mine = before.get(item, {})
            if not mine:
                self.uncalled.add(item)
                continue
            theirs = after.get(item, {})
            differ = []
            for key, c in sorted(mine.items()):
                o = theirs.get(key)
                if (
                    o is None
                    or (c.outcome, c.raised) != (o.outcome, o.raised)
                    or c.result_sha != o.result_sha
                    or (c.outcome == "returned" and (c.result_sha is None or o.result_sha is None))
                ):
                    differ.append(key)
            if not differ:
                same.add(item)
                continue
            keys = ", ".join(key_text(k) for k in differ[:MAX_NAMED]) + (", …" if len(differ) > MAX_NAMED else "")
            what = (
                f"{len(differ)} of {len(mine)} calls of it in the parent's tests return, refuse or raise differently "
                f"on the candidate's library ({keys})"
            )
            if item in strict:
                run.fail(
                    "G3",
                    f"{item} is edited in a clean-up pass without a red test, and {what}; a change of behaviour "
                    "needs a test that is red on the parent's library",
                    item,
                )
            else:
                run.fail(
                    "G3",
                    f"{item} changes behaviour without a test: {what}; any change in what a stored function does "
                    "needs a test, listed under it, that fails on the parent's library and passes on the candidate's",
                    owner(item) if owner is not None else None,
                )
        run.same = (run.same or set()) | same

    def no_comparison(self, item: str) -> None:
        """A stored function in a changed module that neither a parent test calls nor a stored cover holds."""
        self.run.fail(
            "G3",
            f"{item} is in a changed module but no test of the parent calls it and it has no stored cover; its "
            "behaviour cannot be compared, so change it only with a test listed under it that fails on the parent's "
            "library",
            item if item in {it.item for it in self.run.man.items} else None,
        )

    # -- G5: a deleted function's fixtures stay used -----------------------------------------------------------
    @staticmethod
    def ran_on(calls: list[ExampleCall]) -> set[str]:
        """The fixtures a set of example calls ran on: the files their test modules opened under tests/data/ and
        the fixtures a path argument names."""
        out: set[str] = set()
        for c in calls:
            out |= {p for p in c.opened if "/tests/data/" in p}
            out |= {a.path for a in c.args if a.path is not None and "/tests/data/" in a.path}
        return out

    def retained(self, parent_calls: dict[str, list[ExampleCall]]) -> None:
        """Each deleted function's fixtures (from the parent's calls, *parent_calls*: item -> calls) must be used by
        a remaining function's examples on the candidate (its suites' records, ``run.calls_cand``)."""
        run = self.run
        deleted = set(run.man.deleted)
        kept_calls = [
            c
            for rec in (getattr(run, "calls_cand", {}) or {}).values()
            if rec is not None
            for c in rec.calls
            if c.item not in deleted
        ]
        used = self.ran_on(kept_calls)
        for item in sorted(deleted):
            lost = sorted(self.ran_on(parent_calls.get(item, [])) - used)
            if not lost:
                continue
            shown = ", ".join(lost[:MAX_NAMED]) + (", …" if len(lost) > MAX_NAMED else "")
            run.fail(
                "G5",
                f"{item} is deleted, but the fixtures its examples ran on ({shown}) are used by no remaining "
                "function's examples; a kept function's tests must take each over",
            )
