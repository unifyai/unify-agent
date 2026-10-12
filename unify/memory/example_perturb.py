"""The memorisation check of a library function: its tests rerun on fixtures holding unseen values.

A library function's examples are the calls its tests make on values loaded from recorded fixtures
(:mod:`.example_calls`). A function that only handles the values the recording happened to hold (an ``if`` on a
recorded id, a lookup table keyed by recorded inputs) passes its own tests all the same. This check replaces the
values of the fixtures the tests load with type-valid values no recording holds (the perturbations of
:mod:`.held_out`, which keep every structure: keys, columns, list lengths, the header, formats) and reruns the
function's listed tests, confined, with its calls recorded.

The tests' own assertions fail legitimately on changed inputs, so pass or fail is not the signal. The check
judges at the call boundary: each call of the function is matched across runs by its :class:`.CallKey`, and a
call the function accepted (returned on) in an unperturbed run is judged in a run where one of its arguments
changed, or where a path argument names a changed fixture:

* now refused (``MemoryInputError``): the function refuses values, not shapes; allowed only for a field that a
  recorded rejection behind the fixture varied (:func:`.held_out._exemptions`), and noted;
* now raising anything else: the function special-cases the recorded values;
* now returning ``None`` or another top-level type: noted as a suspected special case and recorded in
  :attr:`PerturbVerdict.type_changes` (values there are for review only and never enter a reason or note);
* returning the same result under every perturbation of at least three fields: noted as insensitive.

An argument whose digest changed is *derived* from the fixture changed in that run, the link the cross-episode
evidence reads (:attr:`PerturbVerdict.derived`).

**Fields.** Each input fixture is parsed as :func:`.held_out._file_doc` parses a recorded file; an assembled
``.jsonl`` file has only the fields under ``[].input`` perturbed. Field values, constancy (identity or format
fields with one value across the recorded observations of their family and shape are kept) and declared
semantic types (``field_types`` keys ``<fixture path>#<field>``, judged two-sided: unseen in-domain values must
be accepted, out-of-domain values may be refused) come from :func:`.held_out.constancy` and
:func:`.held_out.field_cases`, seeded by SHA-256 of the item, the fixture path and the field. A fixture with no
recorded action behind it has no pool, so none of its fields is kept constant.

**Consistent replacement.** One run's changes form a map from each old value to its new value, applied to every
input fixture wherever the same leaf value occurs, so a value joining two fixtures (an id) stays joined.

**Runs** (at most :data:`PERTURB_MAX_RUNS` per function; one deadline for every function of a gate run). The
candidate tree is copied, the changed bytes written, and the function's listed test files that open an input
fixture run with its calls recorded. Run 0 is the control (unchanged values): a call whose outcome or result
digest differs from the gate's own record is unstable and not judged. Run 1 changes every chosen field of every
input fixture at once. A finding is then narrowed, per fixture and then per field in hashed order, until the
field is named or the runs are spent (the reason then names the candidates together). Out-of-domain values of
declared fields run last. Running out before the control and group runs leave the function unjudged, which
refuses it (fail closed).

Reasons and notes carry item ids, paths, field names, line numbers and counts, never a recorded value.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import fixtures as _fixtures
from . import held_out
from .admission import is_rejection
from .episodes import Action, Episode
from .example_calls import CallKey, CallRecord, ExampleCall
from .sandbox_run import PytestOutcome

PERTURB_MAX_RUNS = 12  # pytest runs per item (control, group, narrowing)
PERTURB_BUDGET_S = 240.0  # every item of one gate run
MAX_INPUT_FIXTURES = 8  # per item, chosen by SHA-256 of (item, path)
MIN_INSENSITIVE_FIELDS = 3  # a call unchanged under at least this many changed fields is noted as insensitive
VALUE_RECORD_BYTES = 1024  # a value recorded in a type change past this canonical size is kept as its digest
_NAMED = 5  # ids named per note or reason; the rest are counted

RunTests = Callable[
    [Path, list[str]], PytestOutcome
]  # (tree, test files) -> outcome with .calls


@dataclass(frozen=True)
class InputFixture:
    path: str
    data: bytes
    entry: dict
    family: (
        tuple[str, str, str] | None
    )  # held_out.family of fixtures.source_action, else None
    action: Action | None = (
        None  # the recorded action behind the fixture (a copied file), else None
    )


@dataclass
class PerturbVerdict:
    refused: list[tuple[CallKey, str, str]] = field(
        default_factory=list
    )  # (call, fixture, field)
    special: list[tuple[CallKey, str, str, str]] = field(
        default_factory=list,
    )  # (call, fixture, field, raised class name)
    derived: dict[CallKey, dict[str, set[str]]] = field(
        default_factory=dict,
    )  # call -> param -> fixtures whose perturbation changed it
    notes: list[str] = field(default_factory=list)
    unjudged: list[str] = field(
        default_factory=list
    )  # why the check did not complete, as reason text
    runs: int = 0
    fields: int = 0
    judged: int = 0
    unstable: int = 0
    # suspected special cases (a call returning None or another type), with the values for review only
    type_changes: list[dict] = field(default_factory=list)
    # every refusal text: values replaced and refused or raising, and an unjudged check
    reasons: list[str] = field(default_factory=list)


# --- which fixtures ------------------------------------------------------------------------------------------


def _rank(*parts: str) -> bytes:
    return hashlib.sha256("\0".join(parts).encode()).digest()


def _test_file(key: CallKey) -> str:
    return key.test.split("::", 1)[0]


def input_fixtures(
    item: str,
    calls: list[ExampleCall],
    views: dict[str, dict[str, list[Any]]],
    ledger: dict[str, dict],
    tree: Path,
    *,
    load: Callable[[str], Episode | None] | None = None,
    source_action: Callable[..., tuple[str, int, Action] | None] | None = None,
    notes: list[str] | None = None,
) -> list[InputFixture]:
    """The ledger fixtures of *item*'s package that its example calls load: at most :data:`MAX_INPUT_FIXTURES`.

    A fixture is an input fixture when an argument of a call copies one of its *views* (path to digest to
    views), names it as a path, or when the call's test module opened it (an argument computed from it is
    found by the check). With *load*, each fixture's recorded action comes from *source_action*
    (:func:`.fixtures.source_action` by default) and gives its family; an assembled file has one only when every
    line's action shares it. Notes go to *notes*.
    """
    notes = notes if notes is not None else []
    prefix = _fixtures.package_dir(item) + "/tests/data/"
    found: set[str] = set()
    for c in calls:
        for p in c.opened:
            if p in ledger:
                found.add(p)
        for a in c.args:
            if a.path is not None and a.path in ledger:
                found.add(a.path)
            if a.sha is not None:
                found |= {p for p in ledger if a.sha in views.get(p, {})}
    paths = sorted(
        (
            p
            for p in found
            if p.startswith(prefix)
            and not p[len(prefix) :].startswith(_fixtures.DRAWN_DIR)
        ),
        key=lambda p: _rank(item, p),
    )
    if len(paths) > MAX_INPUT_FIXTURES:
        notes.append(
            f"{item}: {len(paths) - MAX_INPUT_FIXTURES} input fixtures past the first {MAX_INPUT_FIXTURES} "
            "(by a hash of item and path) are not perturbed",
        )
        paths = paths[:MAX_INPUT_FIXTURES]
    if source_action is None and load is not None:
        source_action = getattr(_fixtures, "source_action", None)
    out: list[InputFixture] = []
    for p in paths:
        data = _fixtures._read_plain(Path(tree) / p)
        if data is None:
            notes.append(
                f"{item}: input fixture {p} is not a plain file; its values are not perturbed"
            )
            continue
        entry = ledger[p]
        fam: tuple[str, str, str] | None = None
        action: Action | None = None
        if load is not None and source_action is not None:
            rows = entry.get("lines") if entry.get("kind") == "assembled" else [entry]
            got = [source_action(r, load) for r in (rows or []) if isinstance(r, dict)]
            fams = {held_out.family(g[2]) for g in got if g is not None}
            if got and all(g is not None for g in got) and len(fams) == 1:
                fam = fams.pop()
                action = got[0][2] if entry.get("kind") != "assembled" else None
        out.append(InputFixture(p, data, entry, fam, action))
    return out


# --- documents and replacement -------------------------------------------------------------------------------


class _InputOnly:
    """An assembled file's document with only the fields under ``[].input``."""

    def __init__(self, doc: Any) -> None:
        self.doc = doc

    @staticmethod
    def _keep(loc: Any) -> bool:
        return isinstance(loc, tuple) and len(loc) >= 2 and loc[1] == "input"

    def fields(self) -> dict:
        return {n: v for n, v in self.doc.fields().items() if self._keep(v[0])}

    def siblings(self, loc: tuple) -> list:
        return [(s, v) for s, v in self.doc.siblings(loc) if self._keep(s)]

    def render(self, changes: list = ()) -> Any:
        return self.doc.render(changes)


def _inner(doc: Any) -> Any:
    return doc.doc if isinstance(doc, _InputOnly) else doc


def _leaves(doc: Any) -> list[tuple[Any, Any]]:
    """Every scalar leaf of *doc* as (locator, value), in document order."""
    d = _inner(doc)
    out: list[tuple[Any, Any]] = []
    if isinstance(d, held_out._Json):
        found: list = []
        held_out._walk(d.obj, (), "", found)
        out = [(steps, v) for _, steps, v in found]
    elif isinstance(d, held_out._Table):
        start = 1 if d.header else 0
        for i, row in enumerate(d.rows[start:], start=start):
            out += [((i, j), cell) for j, cell in enumerate(row) if cell.strip()]
    elif isinstance(d, held_out._Yaml):
        for n, line in enumerate(d.lines):
            for (_, span), value, _ in (
                held_out._Yaml(line, d.encoding).fields().values()
            ):
                out.append(((n, span), value))
    if isinstance(doc, _InputOnly):
        out = [(loc, v) for loc, v in out if _InputOnly._keep(loc)]
    return out


def _value_at(doc: Any, loc: Any) -> Any:
    d = _inner(doc)
    if isinstance(d, held_out._Json):
        node = d.obj
        for s in loc:
            node = node[s]
        return node
    if isinstance(d, held_out._Table):
        return d.rows[loc[0]][loc[1]]
    n, (a, b) = loc
    return d.lines[n][a:b]


def _join(v: Any) -> str | None:
    """The key under which equal leaf values of two fixtures are replaced together (None: never joined)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return "n:" + str(v)
    if isinstance(v, float):
        return "n:" + repr(v)
    if isinstance(v, str) and v:
        return "n:" + str(int(v)) if held_out._INT_TEXT.match(v) else "s:" + v
    return None


def _like(old: Any, new: Any, text: bool) -> Any:
    """*new* written as the leaf it replaces: text in a table or YAML file, an integer for an integer."""
    if text or (isinstance(old, str) and not isinstance(new, str)):
        return new if isinstance(new, str) else str(new)
    if (
        isinstance(old, int)
        and not isinstance(old, bool)
        and isinstance(new, str)
        and held_out._INT_TEXT.match(new)
    ):
        return int(new)
    return new


@dataclass
class _Doc:
    fx: InputFixture
    doc: Any
    fam: tuple
    cases: list  # held_out.FieldCase, one per field, not out-of-domain, in hashed field order
    outs: list  # out-of-domain cases of declared fields
    exempt: list = field(
        default_factory=list
    )  # fields a recorded rejection behind the fixture varied


def _render(docs: list[_Doc], chosen: list[tuple[_Doc, Any]]) -> dict[str, bytes]:
    """The changed bytes of every input fixture for the cases *chosen*, replaced consistently."""
    mapping: dict[str, Any] = {}
    for d, fc in chosen:
        for loc, new in fc.changes:
            key = _join(_value_at(d.doc, loc))
            if key is not None and key not in mapping:
                mapping[key] = new
    out: dict[str, bytes] = {}
    for d in docs:
        text = not isinstance(_inner(d.doc), held_out._Json)
        changes = []
        for loc, v in _leaves(d.doc):
            key = _join(v)
            if key is not None and key in mapping:
                changes.append((loc, _like(v, mapping[key], text)))
        if changes:
            out[d.fx.path] = d.doc.render(changes)
    return out


# --- judging -------------------------------------------------------------------------------------------------


@dataclass
class _Finding:
    kind: str  # "refused" | "raised" | "type"
    call: ExampleCall  # the perturbed run's call


def _value_record(type_name: str | None, value: Any, sha: str | None) -> dict:
    text = _fixtures.canonical(value)
    if value is None or len(text.encode("utf-8")) > VALUE_RECORD_BYTES:
        if sha is None and value is not None:
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return {"type": type_name, "sha": sha}
    return {"type": type_name, "json": value}


def _call_id(key: CallKey) -> dict:
    return {"test": key.test, "indices": dict(key.indices), "ordinal": key.ordinal}


def _key_text(key: CallKey) -> str:
    idx = ",".join(f"{k}={v}" for k, v in key.indices)
    return f"{key.test}[{idx}]#{key.ordinal}" if idx else f"{key.test}#{key.ordinal}"


def _one_of(names: list[str]) -> str:
    names = sorted(set(names))
    if len(names) == 1:
        return names[0]
    shown = ", ".join(n[:80] for n in names[:_NAMED])
    more = f" and {len(names) - _NAMED} more" if len(names) > _NAMED else ""
    return f"one of {shown}{more}"


def _unjudged(item: str, why: str) -> str:
    return (
        f"[examples:unjudged] {item}: the check with unseen fixture values did not complete ({why}); "
        "a function is admitted only once it is judged"
    )


def _moved(before: ExampleCall, after: ExampleCall) -> list[str]:
    """The parameters whose argument digest differs between two runs of one call."""
    old = {a.name: a for a in before.args}
    return [
        a.name
        for a in after.args
        if a.name in old
        and a.sha != old[a.name].sha
        and not (a.sha is None and old[a.name].sha is None)
    ]


def _names_changed(call: ExampleCall, changed: set[str]) -> bool:
    return any(a.path is not None and a.path in changed for a in call.args)


class _Check:
    def __init__(
        self, item: str, base: dict[CallKey, ExampleCall], deadline: float
    ) -> None:
        self.item, self.base, self.deadline = item, base, deadline
        self.unmatched = 0
        self.judged = 0
        # call -> (fixture, field) pairs changed in runs where it was affected, and whether its result changed
        self.exposure: dict[CallKey, set[tuple[str, str]]] = {}
        self.moved: set[CallKey] = set()

    def judge(
        self,
        rec: CallRecord,
        changed: set[str],
        origins: list[str],
        fields: list[tuple[str, str]],
        derived: dict[CallKey, dict[str, set[str]]] | None,
        ambiguous: set[tuple[CallKey, str]],
        opened: dict[CallKey, set[str]],
    ) -> dict[CallKey, _Finding]:
        """Findings of one run.

        *changed* are the fixtures whose bytes changed (a joined value changes a fixture whose fields were not
        chosen), *origins* those whose fields were chosen. A changed argument is derived from the one origin
        its test module opened (to *derived*), else it is *ambiguous*.
        """
        got = {c.key: c for c in rec.calls if c.item == self.item}
        findings: dict[CallKey, _Finding] = {}
        for key, before in self.base.items():
            after = got.get(key)
            if after is None:
                self.unmatched += 1
                continue
            moved = _moved(before, after)
            if not moved and not _names_changed(after, changed):
                continue
            self.judged += 1
            sources = [p for p in origins if p in opened.get(key, set())] or list(
                origins
            )
            for name in moved:
                if derived is None:
                    continue
                if len(sources) == 1:
                    derived.setdefault(key, {}).setdefault(name, set()).add(sources[0])
                else:
                    ambiguous.add((key, name))
            if after.outcome == "refused":
                findings[key] = _Finding("refused", after)
            elif after.outcome == "raised":
                findings[key] = _Finding("raised", after)
            else:
                self.exposure.setdefault(key, set()).update(fields)
                if after.result_sha != before.result_sha or after.result_sha is None:
                    self.moved.add(key)
                if after.result_type != before.result_type or (
                    after.result_type == "NoneType" and before.result_type != "NoneType"
                ):
                    findings[key] = _Finding("type", after)
        return findings


def check(
    item: str,
    *,
    calls: list[ExampleCall],
    fixtures: list[InputFixture],
    tree: Path,
    tests: list[str],
    run_tests: RunTests,
    pool: list[tuple[str, Action]],
    pool_capped: bool,
    seen: list[Action],
    blob: Callable[[str], bytes],
    field_types: dict[str, str],
    work: Path,
    deadline: float,
) -> PerturbVerdict:
    """Rerun *item*'s listed *tests* on its input *fixtures* with unseen values, judging its calls.

    *calls* are the gate's record of the item's example calls on the candidate *tree*; *run_tests* runs test
    files on a tree with the item's calls recorded; *pool* the recorded observations (episode id, action) of the
    fixtures' families and *pool_capped* whether a read cap stopped it; *seen* every recorded action of the
    episodes the manifest names; *blob* reads a recorded blob; *field_types* the item's declared types
    (``<fixture path>#<field>``); *work* a fresh directory for the tree copies; *deadline* the
    :func:`time.monotonic` time the gate's budget ends.
    """
    v = PerturbVerdict()
    paths = {f.path for f in fixtures}
    for k in sorted(field_types):
        if k.split("#", 1)[0] not in paths:
            v.notes.append(
                f"{item}: field_types names {k[:120]}, which is not an input fixture's field; not checked",
            )
    if not fixtures:
        v.notes.append(
            f"{item}: no input fixture to change; its examples are not checked with unseen values"
        )
        return v

    # documents and families
    docs: list[_Doc] = []
    for f in fixtures:
        doc = held_out._file_doc(f.path, f.data)
        if doc is None:
            v.notes.append(
                f"{f.path} is not JSON, JSON lines, CSV or YAML; its values are not perturbed",
            )
            continue
        if f.entry.get("kind") == "assembled":
            doc = _InputOnly(doc)
        if f.family is None:
            fam: tuple = ("fixture", f.path, "")
            v.notes.append(
                f"{f.path} has no recorded action behind it; none of its fields is kept as identity or format",
            )
        elif f.family[0] == "tool" and not str(f.entry.get("source", "")).startswith(
            ("args:", "request:")
        ):
            fam = ("tool_response", f.family[1], f.family[2])
        else:
            fam = f.family
        docs.append(_Doc(f, doc, fam, [], []))

    # stats, strings and constancy, as the held-out plan builds them
    strings: set[str] = set()
    for a in seen + [a for _, a in pool]:
        for part in (a.args, a.kwargs, a.response, a.error):
            held_out._strings(part, strings)
    stats: dict[tuple, list] = {}
    for d in docs:
        for name, (_, _, values) in d.doc.fields().items():
            stats.setdefault((d.fam, name), []).extend(values)
            held_out._strings(values, strings)
    families = {d.fam for d in docs}

    def ranged(fam: tuple, doc: Any) -> None:
        for name, (_, _, values) in doc.fields().items():
            stats.setdefault((fam, name), []).extend(values)

    for a in seen + [a for _, a in pool]:
        if getattr(a, "kind", "tool") != "tool":
            continue
        if isinstance(a.kwargs, dict) and held_out.family(a) in families:
            ranged(held_out.family(a), held_out._Json(dict(a.kwargs), lambda o: o))
        rfam = ("tool_response", a.channel, a.method)
        if rfam in families and a.status == "ok":
            od = held_out._observation_doc(a.response)
            if od is not None:
                ranged(rfam, od)
    responses = {f for f in families if f[0] == "tool_response"}
    c_env, notes_env = held_out.constancy(
        pool, families - responses, blob, None, pool_capped=pool_capped
    )
    c_obs, notes_obs = held_out.constancy(
        pool, responses, blob, "observation", pool_capped=pool_capped
    )
    for n in notes_env + [n for n in notes_obs if n not in notes_env]:
        v.notes.append(f"{item}: {n}")

    def constant(fam: tuple, shape: frozenset, name: str) -> bool:
        return (c_obs if fam[0] == "tool_response" else c_env)(fam, shape, name)

    if any(isinstance(_inner(d.doc), held_out._Yaml) for d in docs):
        strings |= held_out._YAML_LITERALS
    exempt_cache: dict[str, set[str]] = {}

    def exempt(d: _Doc) -> set[str]:
        if d.fx.path not in exempt_cache:
            names: set[str] = set()
            a = d.fx.action
            if a is not None and is_rejection(a):
                p = held_out.Plan()
                held_out._exemptions(p, [("", 0, a)], {}, seen, blob)
                varied = p.exempt.get(held_out.family(a), set())
                args = str(d.fx.entry.get("source", "")).startswith("args:")
                names = {f"kwargs.{n}" for n in varied} if args else set(varied)
                v.notes += [f"{item}: {d.fx.path}: {n}" for n in p.notes]
            exempt_cache[d.fx.path] = names
        return exempt_cache[d.fx.path]

    for d in docs:
        declared = {
            k.split("#", 1)[1]: t
            for k, t in field_types.items()
            if k.split("#", 1)[0] == d.fx.path and "#" in k
        }
        cases, notes = held_out.field_cases(
            item,
            (d.fx.path, 0),
            d.doc,
            d.fam,
            stats=stats,
            strings=strings,
            constant=constant,
            field_types=declared,
        )
        v.notes += [f"{item}: {d.fx.path}: {n}" for n in notes]
        first: dict[str, Any] = {}
        for fc in cases:
            if fc.side == "out":
                d.outs.append(fc)
            else:
                first.setdefault(fc.name, fc)
        chosen = sorted(first.values(), key=lambda fc: _rank(item, d.fx.path, fc.name))
        # a field a recorded rejection behind the fixture varied may be refused: it runs on its own
        d.cases = [fc for fc in chosen if fc.name not in exempt(d)]
        d.exempt = [fc for fc in chosen if fc.name in exempt(d)]
    parsed = list(docs)  # a joined value is replaced in every parsed input fixture, changed fields or not
    docs = [d for d in docs if d.cases or d.outs or d.exempt]
    v.fields = sum(len(d.cases) + len(d.exempt) for d in docs)
    if not docs:
        v.notes.append(
            f"{item}: no field of its input fixtures can take an unseen value; nothing to judge"
        )
        return v
    docs.sort(key=lambda d: _rank(item, d.fx.path))
    by_path = {d.fx.path: d for d in docs}

    # the test files that open an input fixture
    files = sorted(
        {
            _test_file(c.key)
            for c in calls
            if set(c.opened) & set(by_path) or any(a.path in by_path for a in c.args)
        }
        & set(tests),
    )
    if not files:
        v.notes.append(
            f"{item}: none of its listed tests opens an input fixture; nothing to judge"
        )
        return v
    opened = {c.key: set(c.opened) | {a.path for a in c.args if a.path} for c in calls}

    def run(changed: dict[str, bytes]) -> CallRecord | str:
        if v.runs >= PERTURB_MAX_RUNS or time.monotonic() >= deadline:
            return "budget"
        dest = Path(work) / str(v.runs)
        v.runs += 1
        try:
            shutil.copytree(tree, dest, symlinks=True)
            for p, data in changed.items():
                target = dest / p
                if target.is_symlink():
                    target.unlink()
                target.write_bytes(data)
            outcome = run_tests(dest, files)
        finally:
            shutil.rmtree(dest, ignore_errors=True)
        if outcome.timed_out or not outcome.valid:
            return "failed"
        if outcome.calls is None or not outcome.calls.complete:
            return "record"
        return outcome.calls

    def fail(why: str) -> PerturbVerdict:
        text = _unjudged(item, why)
        v.unjudged.append(text)
        v.reasons.append(text)
        return v

    # run 0: the control
    control = run({})
    if control == "budget":
        return fail("the budget ran out before the control run")
    if control == "failed":
        return fail("the control run failed")
    if control == "record":
        return fail("the call record of the control run is incomplete")
    assert isinstance(control, CallRecord)
    mine = {c.key: c for c in calls if c.item == item and _test_file(c.key) in files}
    again = {c.key: c for c in control.calls if c.item == item}
    unstable = {
        k
        for k, c in mine.items()
        if k not in again
        or (again[k].outcome, again[k].result_sha) != (c.outcome, c.result_sha)
    }
    v.unstable = len(unstable)
    if unstable:
        v.notes.append(
            f"{len(unstable)} calls of {item} differ between two unperturbed runs; not judged"
        )
    base = {
        k: c
        for k, c in again.items()
        if k in mine and k not in unstable and c.outcome == "returned"
    }
    if not base:
        v.notes.append(
            f"{item}: no call of it returned in both unperturbed runs; nothing to judge"
        )
        return v
    judge = _Check(item, base, deadline)

    # run 1: the group run
    every = [(d, fc) for d in docs for fc in d.cases]
    ambiguous: set[tuple[CallKey, str]] = set()
    group_fields = [(d.fx.path, fc.name) for d, fc in every]
    group: dict[CallKey, _Finding] = {}
    if every:
        rec = run(_render(parsed, every))
        if rec == "budget":
            return fail("the budget ran out before the run with every field changed")
        if not isinstance(rec, CallRecord):
            return fail(
                "the call record of the run with every field changed is incomplete"
                if rec == "record"
                else "the run with every field changed did not finish"
            )
        group = judge.judge(
            rec,
            set(_render(parsed, every)),
            [d.fx.path for d in docs if d.cases],
            group_fields,
            v.derived,
            ambiguous,
            opened,
        )
        if judge.judged == 0 and judge.unmatched:
            v.judged = judge.judged
            return fail(
                f"no call of it could be matched across runs ({judge.unmatched} unmatched)"
            )

    # narrowing: per fixture, then per field
    narrowed: dict[
        CallKey, tuple[str, str, ExampleCall, str]
    ] = {}  # key -> (fixture, field, call, kind)
    pending_fixture: dict[CallKey, list[str]] = {}
    out_of_runs = False
    tried: dict[
        str, set[str]
    ] = {}  # fixture -> fields run alone (a finding not narrowed is in none of them)
    per_fixture: dict[str, dict[CallKey, _Finding]] = {}
    with_cases = [d for d in docs if d.cases]
    if len(with_cases) == 1:
        per_fixture[with_cases[0].fx.path] = group
    elif group or ambiguous:
        order = sorted(
            with_cases,
            key=lambda d: (
                not any(p == d.fx.path for k in group for p in opened.get(k, ())),
                _rank(item, d.fx.path),
            ),
        )
        for d in order:
            chosen = [(d, fc) for fc in d.cases]
            rec = run(_render(parsed, chosen))
            if not isinstance(rec, CallRecord):
                out_of_runs = True
                break
            per_fixture[d.fx.path] = judge.judge(
                rec,
                set(_render(parsed, chosen)),
                [d.fx.path],
                [(d.fx.path, fc.name) for fc in d.cases],
                v.derived,
                set(),
                opened,
            )
    if ambiguous:
        unattributed = {(k, n) for k, n in ambiguous if n not in v.derived.get(k, {})}
        if unattributed:
            v.notes.append(
                f"{len(unattributed)} arguments of {item}'s calls changed with several fixtures changed at once "
                "and could not be attributed to one; they count as not derived",
            )

    def same(a: _Finding, b: _Finding | None) -> bool:
        return b is not None and (a.kind == "type") == (b.kind == "type")

    for key, finding in group.items():
        hits = [p for p, got in per_fixture.items() if same(finding, got.get(key))]
        pending_fixture[key] = hits or [d.fx.path for d in with_cases]
    # field by field within each fixture a finding is pinned to, violations first
    todo = sorted(
        {
            (pending_fixture[k][0], group[k].kind == "type")
            for k in group
            if len(pending_fixture[k]) == 1
        },
        key=lambda t: (t[1], _rank(item, t[0])),
    )
    for path, _ in todo:
        d = by_path[path]
        keys = [k for k in group if pending_fixture[k] == [path] and k not in narrowed]
        if not keys:
            continue
        if len(d.cases) == 1:
            for k in keys:
                got = per_fixture.get(path, group).get(k, group[k])
                narrowed[k] = (path, d.cases[0].name, got.call, got.kind)
            continue
        for fc in d.cases:
            keys = [k for k in keys if k not in narrowed]
            if not keys:
                break
            rendered = _render(parsed, [(d, fc)])
            rec = run(rendered)
            if not isinstance(rec, CallRecord):
                out_of_runs = True
                break
            got = judge.judge(
                rec, set(rendered), [path], [(path, fc.name)], None, set(), opened
            )
            tried.setdefault(path, set()).add(fc.name)
            for k in keys:
                if same(group[k], got.get(k)):
                    narrowed[k] = (path, fc.name, got[k].call, got[k].kind)
        if out_of_runs:
            break
    if out_of_runs:
        v.notes.append(
            f"{item}: the runs ran out while narrowing a finding to one field; the reason names the candidates",
        )

    # verdicts
    # (call, fixture, field, perturbed call, kind, the field's case when it is named, from exempt fields)
    found: list[tuple[CallKey, str, str, ExampleCall, str, Any, bool]] = []
    for key in sorted(group):
        finding = group[key]
        if key in narrowed:
            path, fname, after, kind = narrowed[key]
            case = next(fc for fc in by_path[path].cases if fc.name == fname)
            found.append((key, path, fname, after, kind, case, False))
            continue
        cands = pending_fixture[key]
        fname = _one_of(
            [
                fc.name
                for p in cands
                for fc in by_path[p].cases
                if fc.name not in tried.get(p, set())
            ]
            or [fc.name for p in cands for fc in by_path[p].cases],
        )
        found.append(
            (key, _one_of(cands), fname, finding.call, finding.kind, None, False)
        )
    ex = [(d, fc) for d in docs for fc in d.exempt]
    if ex:
        rendered = _render(parsed, ex)
        rec = run(rendered)
        ex_paths = sorted({d.fx.path for d, _ in ex})
        ex_names = _one_of([fc.name for _, fc in ex])
        if not isinstance(rec, CallRecord):
            v.notes.append(
                f"{item}: the fields a recorded rejection varied ({ex_names}) were not run; the runs ran out",
            )
        else:
            fields_run = [(d.fx.path, fc.name) for d, fc in ex]
            got = judge.judge(
                rec, set(rendered), ex_paths, fields_run, None, set(), opened
            )
            for key in sorted(got):
                case = ex[0][1] if len(ex) == 1 else None
                found.append(
                    (
                        key,
                        _one_of(ex_paths),
                        ex_names,
                        got[key].call,
                        got[key].kind,
                        case,
                        True,
                    )
                )

    grouped: dict[tuple[str, str, str, str], list[CallKey]] = {}
    allowed: list[str] = []
    for key, path, fname, after, kind, case, exempted in found:
        if kind == "refused" and exempted:
            note = f"refusal of {fname[:80]} of {path} by {item} allowed by a recorded rejection"
            if note not in allowed:
                allowed.append(note)
            continue
        if kind == "refused":
            v.refused.append((key, path, fname))
            grouped.setdefault(("refused", path, fname, ""), []).append(key)
        elif kind == "raised":
            name = after.raised or "another exception"
            v.special.append((key, path, fname, name))
            grouped.setdefault(("raised", path, fname, name), []).append(key)
        else:
            before = base[key]
            entry: dict[str, Any] = {
                "item": item,
                "call": _call_id(key),
                "file": before.file,
                "line": before.line,
                "fixture": path,
                "field": fname,
                "before": _value_record(
                    before.result_type, before.result_json, before.result_sha
                ),
                "after": _value_record(
                    after.result_type, after.result_json, after.result_sha
                ),
                "field_before": None,
                "field_after": None,
            }
            if case is not None:
                old_value = _value_at(by_path[path].doc, case.changes[0][0])
                new_value = case.changes[0][1]
                entry["field_before"] = _value_record(
                    type(old_value).__name__, old_value, None
                )
                entry["field_after"] = _value_record(
                    type(new_value).__name__, new_value, None
                )
            v.type_changes.append(entry)
            grouped.setdefault(("type", path, fname, ""), []).append(key)
    v.notes += allowed
    for (kind, path, fname, name), keys in grouped.items():
        c = base[keys[0]]
        if kind == "refused":
            text = (
                f"[examples:perturbed-refused] {item} refuses fixture values it accepted unchanged: with "
                f"{fname} of {path} replaced by an unseen value of its type, its call at {c.file}:{c.line} "
                "raises MemoryInputError; refuse inputs by their shape, not by values the recording happened "
                "to hold"
            )
            v.reasons.append(text)
        elif kind == "raised":
            text = (
                f"[examples:special-cased] {item} handles {path} only for its recorded values: with {fname} "
                f"replaced by an unseen value of its type, its call at {c.file}:{c.line} raises {name} where "
                "it returned before"
            )
            v.reasons.append(text)
        else:
            shown = ", ".join(_key_text(k) for k in keys[:_NAMED])
            more = f" and {len(keys) - _NAMED} more" if len(keys) > _NAMED else ""
            v.notes.append(
                f"special-case suspected: {item} returns None or another type with {fname} of {path} replaced "
                f"by an unseen value ({len(keys)} calls: {shown}{more}, first at {c.file}:{c.line})",
            )

    # out-of-domain values of declared fields: a refusal is expected, so it is never a finding
    outs = [(d, fc) for d in docs for fc in d.outs]
    rounds: dict[int, list[tuple[_Doc, Any]]] = {}
    for d in docs:
        seen_names: dict[str, int] = {}
        for fc in d.outs:
            n = seen_names.get(fc.name, 0)
            seen_names[fc.name] = n + 1
            rounds.setdefault(n, []).append((d, fc))
    accepted_out: set[str] = set()
    for n in sorted(rounds):
        chosen = rounds[n]
        rendered = _render(parsed, chosen)
        rec = run(rendered)
        if not isinstance(rec, CallRecord):
            names = sorted({f"{fc.name} of {d.fx.path}" for d, fc in outs})
            v.notes.append(
                f"{item}: the out-of-domain values of declared fields were not all checked "
                f"({_one_of(names)}); the runs ran out",
            )
            break
        got = {c.key: c for c in rec.calls if c.item == item}
        for key, before in base.items():
            after = got.get(key)
            if after is None or after.outcome != "returned":
                continue
            if _moved(before, after) or _names_changed(after, set(rendered)):
                accepted_out.add(
                    _one_of([f"{fc.name[:80]} of {d.fx.path}" for d, fc in chosen])
                )
    if accepted_out:
        v.notes.append(
            f"{item} accepts out-of-domain values of its declared fields "
            f"{', '.join(sorted(accepted_out)[:_NAMED])}; a declared type's out-of-domain values are expected to be "
            "refused",
        )

    # insensitive calls
    flat = [
        k
        for k, exposed in judge.exposure.items()
        if len(exposed) >= MIN_INSENSITIVE_FIELDS
        and k not in judge.moved
        and k not in group
    ]
    if flat:
        shown = ", ".join(_key_text(k) for k in sorted(flat)[:_NAMED])
        more = f" and {len(flat) - _NAMED} more" if len(flat) > _NAMED else ""
        v.notes.append(
            f"insensitive: {len(flat)} calls of {item} return the same result under every change of at least "
            f"{MIN_INSENSITIVE_FIELDS} fields ({shown}{more})",
        )
    if judge.unmatched:
        v.notes.append(
            f"{judge.unmatched} call matches of {item} were not found in a run with changed values (the test's "
            "control flow changed); not judged",
        )
    v.judged = judge.judged
    if v.judged == 0:
        v.notes.append(
            f"{item}: no call of it took a changed value; nothing was judged"
        )
    return v
