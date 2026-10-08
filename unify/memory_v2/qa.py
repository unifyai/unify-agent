"""QA-level test checks in the consolidation gate (memory v2.1 stage 5; plan "Stage 5", spec G3).

G3 asks that each new or changed environment function has a test that is red on the parent and green on the
candidate. It does not judge whether the test is any good: one fixture read many times, a stand-in environment
that ignores its arguments, an assertion that would hold whatever the function did. These checks judge test
quality, each behind its own switch (all off by default; :class:`QAConfig`). With every switch off, ``on`` is
False, the gate makes exactly the calls it made before, and nothing here runs.

**Seeded random fixtures** (``UNIFY_MEMORY_V2_QA_FIXTURES``: ``on`` or ``strict``). Inputs are drawn, never
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

**Mutation testing** (``UNIFY_MEMORY_V2_QA_MUTATION``; ``UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL``, default 0.5).
Up to :data:`MAX_MUTANTS` mutants per function (:mod:`.mutation`, chosen by the seed) each run the function's
new or changed tests, confined, at most :data:`MUTANT_S` each. A mutant is killed when a test is not green. A
surviving mutant whose probe outputs equal the original's on every informative row (the function returned,
refused or raised) of the covers, the drawn inputs and the covers' structural negatives
(:func:`.mutation.negatives`: a field dropped, retyped, emptied or added) is likely equivalent and leaves the
count; the negatives keep a mutant that drops a shape guard from looking equivalent. A kill share below the
threshold refuses the pass, naming operator kinds and lines. When survivors remain but the probe got no
informative row (every input missed the replay or was unbound), the check is "not judged": a refusal under
``strict``, else a note. First the tests run on the module re-printed unchanged (the control): if that is not
green, nothing is judged.

**Determinism** (``UNIFY_MEMORY_V2_QA_DETERMINISM``). Every gate pytest run gets ``PYTHONHASHSEED=0``,
``TZ=UTC`` and the plugin :mod:`.pin` (``-p _memv2_pin``: a stepping clock from a fixed epoch, ``random`` seeded
before each test). Each new or changed test file green in G3 runs once more under the other pin variant
(another epoch, random seed and hash seed); different outcomes refuse it, so a test can neither be flaky nor
depend on a pinned value. A second run that times out is noted, not judged.

**The test kit** (:mod:`.testkit`). Every gate pytest run mounts the kit (``memlab``, the pin plugin, the
blobs the tests name) read-only at ``/inputs`` when a switch is on **or** the library's tests use it, so a
stored library never depends on a switch setting; the same kit is in Sol's box and the actor's export. New
or changed library code that uses it, and tests importing what it does not provide or naming ``/inputs``,
are refused (``[qa:kit]``).

**Replay fidelity** (``UNIFY_MEMORY_V2_QA_REPLAY``). A new or changed test that passes an environment-taking
function a stand-in of its own instead of ``memlab.replay.RecordedEnv``, while the function has recorded
calls, is refused (:func:`.qa_static.stand_ins`, structural).

**Fixture size** (``UNIFY_MEMORY_V2_QA_FIXTURE_SIZE``). A new or changed file under ``env/<channel>/tests/``, the
test kit or a support file larger than :data:`FIXTURE_MAX_BYTES` is refused: recorded payloads are referenced
by blob id (:func:`.inputs.blob`). The cause of the 1.1 MB Crafter fixture (offline sweep, 300k) was a brief
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
recorded value (R10). All reasons are filed under G3 with a ``[qa:<check>]`` tag.

Rejected alternatives: rewriting the consolidator's fixtures to the drawn inputs (expected values would no
longer match, so only property tests survive); a pytest plugin wrapping library functions to watch their
arguments (it changes the code under test); ``sitecustomize`` for the pins (shadowed by any in the venv);
mutating by text substitution (cannot be kept syntactically valid). Known limits: an input's "shape" is its
field-name set, so a refusal of a same-named but differently typed value counts as a false refusal; the stand-in
check resolves only literal data flow (an environment built behind a helper's parameter or an attribute is
``unknown``, never refused); the truncation check sees only ``assert`` statements of the test source (an
expected value kept in a data file is not seen); module code under test shares the probe's process (ruling
R16: careless, not malicious).
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from . import mutation, qa_static, testkit
from .admission import is_rejection
from .analysis import shapes as _shapes
from .episodes import Action
from .held_out import (
    _FORMS,
    _blob_sha,
    _doc,
    _doc_family,
    _file_name,
    _form,
    _key,
    _read_results,
    _unfit_reason,
)
from .integration import switch
from .manifest import INPUT_KINDS, TESTKIT, TESTS_DIR, item_path
from .sandbox_run import SandboxResult, run_confined

SAMPLE_K = 8  # drawn inputs per function
MAX_COVERS = 8  # covers per function the probe also runs (for mutant equivalence)
MAX_MUTANTS = 8  # mutants per function
RUN_S = 120.0  # one stage-5 pytest run or probe
MUTANT_S = 60.0  # one mutant's test run
BUDGET_S = 900.0  # every dynamic stage-5 check of one gate run together
FLOOR_S = 5.0  # no run starts with less of the budget left
PROBE_CASE_S = 2  # one call in the probe
FIXTURE_MAX_BYTES = 64 * 1024
MIN_KILL = Decimal("0.5")
# Under the fixture-size switch, an exported response at least 1/RESPONSE_BLOB_SHARE of the fixture bound also
# becomes a blob (1024 bytes at the default 64 KiB bound): a test file then holds at least 64 inline copies of
# responses under the threshold, and the brief's "handful" (8 or so) takes at most 1/8 of the bound. Dialogue
# observations are capped at 4000 characters by the recorder, so a capped Crafter screen (about 4 KB, which the
# old fixed 4096-byte threshold just missed) is always a blob; a blob reference costs 66 bytes of JSON, so the
# 1.1 MB Crafter fixture, if it held ~275 screens of ~4 KB (inferred from its size), would take ~18 KB as
# references.
RESPONSE_BLOB_SHARE = 64
RESPONSE_BLOB_BYTES = FIXTURE_MAX_BYTES // RESPONSE_BLOB_SHARE
MAX_CONTEXT = 64  # recorded calls an ``env`` input's replay answers
MAX_ROW_BYTES = 1024**2
MAX_SAMPLES_BYTES = 16 * 1024**2
MAX_INPUT_FILE_BYTES = testkit.MAX_INPUT_FILE_BYTES
PIN_MODULE = testkit.PIN_MODULE
PINNED_ENV = {"PYTHONHASHSEED": "0", "TZ": "UTC"}
PIN_VARIANT_ENV = "MEMV2_PIN"  # :data:`.pin.VARIANT_ENV`
# probe outcomes that say something about the function (equivalence is judged on these only)
INFORMATIVE = frozenset({"handled", "refused", "error"})
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
        """``UNIFY_MEMORY_V2_QA_FIXTURES=strict``: what the checks could not judge refuses the pass."""
        return self.fixtures == "strict"

    @property
    def response_blob_bytes(self) -> int:
        """The size from which Sol's export writes a response as a blob (relative to the fixture bound)."""
        return max(1, self.fixture_max_bytes // RESPONSE_BLOB_SHARE)

    @classmethod
    def from_settings(cls, settings: Any) -> "QAConfig":
        """The switches as the run's settings hold them (validated again; unset means off)."""

        def get(name: str) -> Any:
            return getattr(settings, name, "")

        return cls(
            fixtures=switch.parse_qa_fixtures(get(switch.QA_FIXTURES)),
            mutation=switch.parse_qa_mutation(get(switch.QA_MUTATION)) == "on",
            min_kill=Decimal(switch.parse_qa_min_kill(get(switch.QA_MIN_KILL))),
            determinism=switch.parse_qa_determinism(get(switch.QA_DETERMINISM)) == "on",
            replay=switch.parse_qa_replay(get(switch.QA_REPLAY)) == "on",
            fixture_size=switch.parse_qa_fixture_size(get(switch.QA_FIXTURE_SIZE))
            == "on",
        )


def seed_of(candidate: str) -> bytes:
    """The stage-5 seed: SHA-256 of the candidate commit (reproducible, unknown before the commit)."""
    return hashlib.sha256(f"memory-v2-qa\0{candidate}".encode()).digest()


def pytest_env(cfg: QAConfig, variant: int = 0) -> dict[str, str]:
    """The environment of every gate pytest run while the kit is mounted (a switch on, or the tests use it).

    Under determinism the pins of *variant* (:mod:`.pin`): 0 for every run but the determinism rerun, which
    uses 1 (another epoch, random seed and hash seed).
    """
    env = {
        "PYTHONPATH": "/memory:/inputs",
        "PYTEST_ADDOPTS": "-c /dev/null --import-mode=importlib",
    }
    if cfg.determinism:
        env["PYTEST_ADDOPTS"] += f" -p {PIN_MODULE}"
        env.update(PINNED_ENV)
        env["PYTHONHASHSEED"] = str(variant)
        env[PIN_VARIANT_ENV] = str(variant)
    return env


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


def _usable(a: Action | None, form: str | None) -> bool:
    if a is None:
        return False
    kind = getattr(a, "kind", "tool")
    return (
        kind != "shell"
        and a.status == "ok"
        and not is_rejection(a)
        and _form(a, form) in _FORMS.get(kind, ())
        and _unfit_reason(a, form) is None
    )


def _rank(seed: bytes, *parts: str) -> bytes:
    return hashlib.sha256(seed + "\0".join(parts).encode()).digest()


def draw(
    item: str,
    covers: list[tuple[str, int, Action]],
    pool: list[tuple[str, Action]],
    *,
    form: str | None,
    blob: Callable[[str], bytes],
    seed: bytes,
    k: int = SAMPLE_K,
) -> tuple[list[Row], list[str]]:
    """The covers (at most :data:`MAX_COVERS`) and up to *k* drawn inputs of *item*'s family, with notes.

    *pool* is G2's store-wide pool of recorded observations on the covered channels (episode id, action);
    *form* the item's declared input form (None: each kind's convention).
    """
    notes: list[str] = []
    usable = [(eid, idx, a) for eid, idx, a in covers if _usable(a, form)]
    usable.sort(key=lambda c: _rank(seed, item, c[0], str(c[1])))
    usable = usable[:MAX_COVERS]
    if not usable:
        notes.append(
            "no cover it can be called on (a shell cover, or none in its form); nothing drawn",
        )
        return [], notes

    def shape(a: Action) -> tuple | None:
        try:
            d = _doc(a, blob, form)
        except (OSError, ValueError, KeyError):
            return None
        if d is None:
            return None
        if _form(a, form) == "env":
            # a call's fields are its keyword names; the function also reads the response, so its field
            # names are part of the shape (a refused response of another shape is a precondition, not a
            # false refusal)
            r = a.response
            return (
                frozenset(d.fields()),
                frozenset(map(str, r)) if isinstance(r, dict) else type(r).__name__,
            )
        return (frozenset(d.fields()),)

    families = {_doc_family(a, form) for _, _, a in usable}
    shapes = {shape(a) for _, _, a in usable} - {None}
    taken = {_content(a, form) for _, _, a in usable}
    by_eid: dict[str, list[Action]] = {}
    for eid, a in pool:
        by_eid.setdefault(eid, []).append(a)
    found: dict[str, tuple[str, Action]] = {}
    cut = 0
    for eid, a in pool:
        if not _usable(a, form) or _doc_family(a, form) not in families:
            continue
        if qa_static.truncated(a):
            cut += 1
            continue
        key = _content(a, form)
        if key not in taken and key not in found:
            found[key] = (eid, a)
    ranked = sorted(found, key=lambda key: _rank(seed, item, key))

    def row(role: str, eid: str, a: Action, n: int) -> Row | None:
        f = _form(a, form) or "observation"
        context: list[Action] = []
        if getattr(a, "kind", "tool") == "tool" and f == "env":
            others = [
                b
                for b in by_eid.get(eid, [])
                if b is not a
                and getattr(b, "kind", "tool") == "tool"
                and b.status in ("ok", "error")
            ]
            context = [a] + others[: MAX_CONTEXT - 1]  # the drawn call answers first
        r = Row(role, a, context, role == "cover" or shape(a) in shapes, f)
        if getattr(a, "kind", "tool") == "worktree":
            sha = _blob_sha(a)
            try:
                data = blob(sha) if sha is not None else None
            except (OSError, KeyError, ValueError):
                data = None
            if data is None or len(data) > MAX_INPUT_FILE_BYTES:
                return None
            r.file = (_file_name(a, n), data)
            if f == "text":
                r.text = _shapes.decode(data)[1]
        return r

    rows: list[Row] = []
    for eid, _, a in usable:
        r = row("cover", eid, a, len(rows))
        if r is not None:
            rows.append(r)
    covers_rows = list(rows)
    drawn = 0
    for key in ranked:
        if drawn >= k:
            break
        eid, a = found[key]
        r = row("sample", eid, a, len(rows))
        if r is not None:
            rows.append(r)
            drawn += 1
    rows += negative_rows(item, covers_rows, seed)
    if cut:
        notes.append(f"{cut} truncated recording(s) of its family left out of the draw")
    if not drawn:
        notes.append("no recorded input of its family beyond its covers to draw")
    return rows, notes


def negative_rows(item: str, covers: list[Row], seed: bytes) -> list[Row]:
    """Structurally broken copies of *item*'s cover rows (:func:`.mutation.negatives`), for the mutants'
    equivalence only: an ``env`` cover's recorded response broken (the replay answers the broken response), an
    observation or text input broken itself, a file input emptied."""
    if not covers:
        return []
    if covers[0].form in ("path", "bytes"):
        c = covers[0]
        name = c.file[0] if c.file is not None else "file"
        return [Row("negative", c.action, [], False, c.form, file=(name, b""))]

    def recorded(c: Row) -> Any:
        if c.form == "text":
            return c.text if c.text is not None else c.action.response
        return c.action.response

    out: list[Row] = []
    for i, _, broken in mutation.negatives([recorded(c) for c in covers], seed, item):
        c = covers[i]
        if c.form == "env":
            a = dataclasses.replace(c.action, response=broken)
            out.append(Row("negative", a, [a, *c.context[1:]], False, "env"))
        else:
            out.append(
                Row(
                    "negative",
                    c.action,
                    [],
                    False,
                    c.form,
                    value=broken,
                    has_value=True,
                ),
            )
    return out


def _action_dict(a: Action) -> dict:
    return {
        "cell": a.cell,
        "channel": a.channel,
        "method": a.method,
        "args": list(a.args),
        "kwargs": dict(a.kwargs or {}),
        "response": a.response,
        "status": a.status,
        "effect": a.effect,
        "error": a.error,
        "kind": getattr(a, "kind", "tool"),
    }


def write_samples(
    dest: Path,
    items: dict[str, list[Row]],
) -> tuple[dict[str, list[dict]], int]:
    """``dest/samples.json``, ``dest/files/<n>/<name>`` and ``dest/probe.py``; the rows written, the dropped count."""
    (dest / "files").mkdir(parents=True)
    data: dict = {"items": {}}
    total = dropped = n = 0
    written: dict[str, list[dict]] = {}
    for item, rows in items.items():
        out: list[dict] = []
        for r in rows:
            row: dict = {
                "id": n,
                "role": r.role,
                "form": r.form,
                "covered_shape": r.covered_shape,
                "action": _action_dict(r.action),
            }
            if r.context:
                row["context"] = [_action_dict(c) for c in r.context]
            if r.text is not None:
                row["text"] = r.text
            if r.has_value:
                row["value"] = r.value
            size = len(json.dumps(row, default=str))
            if size > MAX_ROW_BYTES or total + size > MAX_SAMPLES_BYTES:
                dropped += 1
                continue
            if r.file is not None:
                name, payload = r.file
                (dest / "files" / str(n)).mkdir()
                (dest / "files" / str(n) / name).write_bytes(payload)
                row["file"] = f"/qa/files/{n}/{name}"
            out.append(row)
            total += size
            n += 1
        written[item] = out
        data["items"][item] = {"rows": out}
    (dest / "samples.json").write_text(json.dumps(data, default=str))
    (dest / "probe.py").write_text(PROBE)
    return written, dropped


# --- the probe -----------------------------------------------------------------------------------------------

# Runs inside the box: /memory (the tree), /inputs (memlab, the pin), /qa (samples.json, files/), /out.
PROBE = r"""
import hashlib, importlib, inspect, json, signal, sys
sys.path[:0] = ["/memory", "/inputs"]
sys.dont_write_bytecode = True
import _memv2_pin as pin
pin.install()
from memlab.inputs import RecordedInput
from memlab.replay import RecordedError, ReplayMiss

class Timeout(BaseException): pass
state = {"timed_out": False}
def _alarm(*_):
    state["timed_out"] = True
    signal.alarm(1)  # re-armed: a function that catches the timeout is stopped again
    raise Timeout()
signal.signal(signal.SIGALRM, _alarm)

item, per_case = sys.argv[1], int(sys.argv[2])
out = open("/out/results.jsonl", "w")
try:
    rows = json.load(open("/qa/samples.json"))["items"][item]["rows"]
    module, _, function = item.partition(":")
    fn = getattr(importlib.import_module(module.replace("/", ".")), function)
    params = list(inspect.signature(fn).parameters.values())
except BaseException as exc:
    out.write(json.dumps({"fatal": type(exc).__name__}) + "\n"); sys.exit(0)
# Arguments beyond the first come from recordings only: the input's own recorded keyword arguments, else the
# first cover's that recorded the name. A required parameter no recording names is never filled in: the row is
# "unbound" (inconclusive), never called with a made-up value.
def _recorded(row):
    a = row["action"]
    return (a.get("kwargs") or {}) if a.get("kind", "tool") == "tool" else {}
from_covers = {}
for row in rows:
    if row.get("role") == "cover":
        for k, v in _recorded(row).items(): from_covers.setdefault(k, v)
for row in rows:
    res = {"id": row["id"]}
    pin.reset()
    try:
        first = row["value"] if "value" in row else RecordedInput(row).value(row.get("form"))
    except BaseException:
        res["outcome"] = "unbuilt"
        out.write(json.dumps(res) + "\n"); out.flush(); continue
    recorded = _recorded(row)
    kwargs, unbound = {}, False
    for p in params[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD): continue
        if p.name in recorded and p.kind != p.POSITIONAL_ONLY: kwargs[p.name] = recorded[p.name]
        elif p.default is not p.empty: continue
        elif p.name in from_covers and p.kind != p.POSITIONAL_ONLY: kwargs[p.name] = from_covers[p.name]
        else: unbound = True
    if unbound:
        res["outcome"] = "unbound"
        out.write(json.dumps(res) + "\n"); out.flush(); continue
    state["timed_out"] = False
    signal.alarm(per_case)
    try:
        try:
            result = fn(first, **kwargs)
        finally:
            signal.alarm(0)
        res["outcome"] = "handled"
        try:
            text = json.dumps(result, sort_keys=True, default=repr)
        except BaseException:
            text = "unserialisable:" + type(result).__name__
        res["digest"] = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]
    except Timeout:
        res["outcome"] = "timeout"
    except BaseException as exc:
        names = [c.__name__ for c in type(exc).__mro__]
        if "MemoryInputError" in names: res["outcome"] = "refused"
        elif isinstance(exc, ReplayMiss): res["outcome"] = "miss"
        elif isinstance(exc, RecordedError): res["outcome"] = "recorded_error"
        else:
            res["outcome"] = "error"
            res["raised"] = type(exc).__name__
    signal.alarm(0)
    if state["timed_out"]:
        res["outcome"] = "timeout"
    out.write(json.dumps(res) + "\n"); out.flush()
"""

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


def informative(base: dict[int, dict] | None) -> list[int]:
    """The probe rows that say something about the function: it returned, refused or raised.

    A replay miss, a recorded error, an unbuilt or unbound input and a timeout say nothing about it (the
    probe's convention or the recording failed, not the function), so equivalence is never judged on them.
    """
    return [i for i, r in (base or {}).items() if r.get("outcome") in INFORMATIVE]


def same_outputs(base: dict[int, dict] | None, other: dict[int, dict] | None) -> bool:
    """Whether a mutant's probe rows equal the original's on every informative input (at least one)."""
    if not base or other is None:
        return False
    judged = informative(base)
    if not judged:
        return False
    return all(
        i in other and _signature(base[i]) == _signature(other[i]) for i in judged
    )


# --- the checks ----------------------------------------------------------------------------------------------


def _names(items: list[str]) -> str:
    more = f" and {len(items) - _NAMED} more" if len(items) > _NAMED else ""
    return ", ".join(items[:_NAMED]) + more


def _imports_inputs(source: bytes, kit: bytes | None) -> bool:
    """Whether a test file (or the test kit it imports) imports ``memlab.inputs``."""

    def direct(src: bytes | None) -> tuple[bool, bool]:
        if src is None:
            return False, False
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError):
            return False, False
        found = kit_used = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "memlab.inputs" or (
                    node.module == "memlab"
                    and any(a.name == "inputs" for a in node.names)
                ):
                    found = True
                if node.module.split(".")[0] == TESTKIT[:-3]:
                    kit_used = True
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "memlab.inputs":
                        found = True
                    if a.name.split(".")[0] == TESTKIT[:-3]:
                        kit_used = True
        return found, kit_used

    found, kit_used = direct(source)
    return found or (kit_used and direct(kit)[0])


class _BudgetSpent(Exception):
    """The stage-5 time budget ran out (noted once; the remaining dynamic checks are not judged)."""


class QAChecks:
    """The stage-5 checks of one gate run (*run* is the gate's :class:`~.gate._Run`)."""

    def __init__(self, gate: Any, run: Any) -> None:
        self.gate, self.run, self.cfg = gate, run, gate.qa
        self.seed = seed_of(run.candidate)
        self.started: float | None = None

    # -- shared -------------------------------------------------------------------------------------------
    def _fail(self, tag: str, reason: str) -> None:
        self.run.fail("G3", f"[qa:{tag}] {reason}")

    def _note(self, tag: str, reason: str) -> None:
        self.run.note(f"[qa:{tag}] {reason}")

    def _timeout(self, cap: float, what: str) -> float:
        """The timeout for the next run (at most *cap*); :class:`_BudgetSpent` once the budget is spent.

        Fail closed: a pass the budget left unjudged is refused (``[qa:budget]``), never merged unjudged.
        """
        if self.started is None:
            self.started = self.cfg.clock()
        left = self.cfg.budget_s - (self.cfg.clock() - self.started)
        if left < FLOOR_S:
            self._fail(
                "budget",
                f"the {self.cfg.budget_s:.0f} s stage-5 budget ran out at {what}: it and the stage-5 "
                "checks after it are unjudged, and an unjudged pass is not merged",
            )
            raise _BudgetSpent(what)
        return min(cap, left)

    def _pytest(
        self,
        tree: Path,
        target: str,
        timeout_s: float,
        ro: dict[Path, str] | None = None,
        rw: dict[Path, str] | None = None,
        env: dict[str, str] | None = None,
    ):
        qa_env = self.run.qa_env
        return self.gate.pytest(
            target,
            python=self.gate.python,
            ro={tree: "/memory", qa_env.inputs: "/inputs", **(ro or {})},
            rw=dict(rw or {}),
            cwd="/memory",
            timeout_s=timeout_s,
            env=dict(env if env is not None else qa_env.env),
        )

    def _edited(self) -> list[Any]:
        run = self.run
        return [
            it
            for it in run.man.items
            if it.kind == "env_function"
            and run.p_bodies.get(it.item, ("", ""))[:2]
            != run.c_bodies.get(it.item, ("", ""))[:2]
        ]

    def _tests(self, it: Any) -> list[str]:
        changed = set(self.run.changed)
        return sorted(t for t in it.tests if t in changed and t in self.run.c_files)

    def _form(self, it: Any) -> str | None:
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
        """Refuse new or changed library code that uses the kit, test imports the kit does not provide, and
        tests that name the ``/inputs`` path (each would work in one place a library's tests run, not all).

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
                self._fail(
                    "kit",
                    f"{p} line {line} imports {what}, which the test kit (version "
                    f"{testkit.KIT_VERSION}) does not provide",
                )
            for line in testkit.names_inputs_path(source):
                self._fail(
                    "kit",
                    f"{p} line {line} names a path under /inputs; read recorded blobs through "
                    "memlab.inputs.blob(<id>), which works wherever the library's tests run",
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
                            "function or literal of their own); test it through memlab.replay.RecordedEnv "
                            "over its recorded calls",
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
                        )

    # -- after G3: determinism, drawn inputs, mutants -----------------------------------------------------
    def dynamic(self) -> None:
        """Determinism, drawn inputs and mutants, in that order, within one time budget."""
        run, cfg = self.run, self.cfg
        if not (cfg.determinism or cfg.draws):
            return
        if not run.res.passed:
            self._note(
                "skipped",
                "dynamic stage-5 checks not run: the candidate is already refused",
            )
            return
        if run.qa_env is None:  # prepare mounts the kit whenever a switch is on
            return
        try:
            self._dynamic()
        except _BudgetSpent:
            pass  # noted by _timeout

    def _dynamic(self) -> None:
        run, cfg = self.run, self.cfg
        if cfg.determinism:
            self._determinism()
        edited = self._edited()
        if not cfg.draws or not edited:
            return
        rows: dict[str, list[Row]] = {}
        for it in edited:
            covers = self._covers(it.item, self.gate.lookup)
            if not covers:
                continue
            pool, _ = self.gate._pool(run, covers)

            def blob(sha: str) -> bytes:
                if not self.gate.blobs.has(sha):
                    raise KeyError(sha)
                return self.gate.blobs.get(sha)

            got, notes = draw(
                it.item,
                covers,
                pool,
                form=self._form(it),
                blob=blob,
                seed=self.seed,
                k=cfg.sample_k,
            )
            for n in notes:
                self._note("fixtures", f"{it.item}: {n}")
            if got:
                rows[it.item] = got
        samples = run.tmp / "qa-samples"
        written, dropped = write_samples(samples, rows)
        if dropped:
            self._note(
                "fixtures",
                f"{dropped} drawn input(s) over the size bounds left out",
            )
        base: dict[str, dict[int, dict] | None] = {}
        for it in edited:
            if written.get(it.item):
                base[it.item] = self._probe(it.item, run.c_tree, samples)
        if cfg.fixtures:
            for it in edited:
                if written.get(it.item):
                    self._fixtures(it, written[it.item], base.get(it.item), samples)
        if cfg.mutation:
            for it in edited:
                self._mutants(
                    it,
                    samples if written.get(it.item) else None,
                    base.get(it.item),
                )

    def _probe(self, item: str, tree: Path, samples: Path) -> dict[int, dict] | None:
        to = self._timeout(self.cfg.run_s, f"the probe of {item}")
        work = Path(tempfile.mkdtemp(prefix="qa-probe-", dir=self.run.tmp))
        return probe(
            item,
            tree,
            self.run.qa_env,
            samples,
            python=self.gate.python,
            work=work,
            timeout_s=to,
            runner=self.cfg.runner,
        )

    def _determinism(self) -> None:
        from .gate import _green

        run = self.run
        for t in sorted(run.qa_first):
            first = run.qa_first[t]
            if not _green(first):
                continue
            to = self._timeout(self.cfg.run_s, f"the determinism rerun of {t}")
            # the second run under the other pins: a test passing only on the pinned values is caught too
            second = self._pytest(
                run.c_tree,
                t,
                to,
                env=pytest_env(self.cfg, variant=1),
            )
            if second.timed_out:
                self._note("determinism", f"the rerun of {t} timed out; not judged")
                continue
            differ = (first.passed ^ second.passed) | (first.failed ^ second.failed)
            if differ or (first.valid, first.returncode) != (
                second.valid,
                second.returncode,
            ):
                self._fail(
                    "determinism",
                    f"{t} gives different outcomes on two runs with the clock, hash seed and random pinned "
                    f"to different values ({len(differ)} test(s) differ): it depends on the clock, "
                    "randomness or hash order, or is flaky",
                )

    def _fixtures(
        self,
        it: Any,
        rows: list[dict],
        base: dict[int, dict] | None,
        samples: Path,
    ) -> None:
        from .gate import _green

        drawn = [r for r in rows if r["role"] == "sample"]
        if not drawn:
            return
        item = it.item
        covers = [r["id"] for r in rows if r["role"] == "cover"]
        if base is None:
            self._note(
                "fixtures",
                f"{item} could not be loaded for the probe; drawn inputs not judged",
            )
        elif not any((base.get(i) or {}).get("outcome") == "handled" for i in covers):
            # the probe's call convention (the arguments it can take from the recordings) does not reach
            # the function on its own covers, so its verdicts on the drawn inputs would be about the probe
            self._note(
                "fixtures",
                f"{item} returns on none of its covers under the probe's call (arguments from the "
                "recordings only); drawn inputs not judged",
            )
        else:
            refused = crashed = inconclusive = allowed = timeouts = 0
            raised: dict[str, int] = {}
            for r in drawn:
                got = base.get(r["id"])
                outcome = got.get("outcome") if got else None
                if outcome == "refused":
                    if r["covered_shape"]:
                        refused += 1
                    else:
                        allowed += 1
                elif outcome == "error":
                    crashed += 1
                    name = builtin_error(got.get("raised")) or "another exception"
                    raised[name] = raised.get(name, 0) + 1
                elif outcome == "timeout":
                    timeouts += 1
                elif outcome != "handled":
                    inconclusive += 1
            n = len(drawn)
            if crashed:
                kinds = ", ".join(f"{k} ({v})" for k, v in sorted(raised.items()))
                self._fail(
                    "fixtures",
                    f"{item} raises on {crashed} of {n} drawn recorded inputs of its family ({kinds}); it "
                    "must return or raise MemoryInputError",
                )
            if refused:
                self._fail(
                    "fixtures",
                    f"{item} refuses {refused} of {n} drawn recorded inputs shaped like its covers, which "
                    "the environment accepted",
                )
            if allowed:
                self._note(
                    "fixtures",
                    f"{item} refuses {allowed} of {n} drawn inputs whose fields differ from its covers' "
                    "(a precondition)",
                )
            if inconclusive or timeouts:
                self._note(
                    "fixtures",
                    f"{item}: {inconclusive} drawn input(s) inconclusive (a call the replay did not record, "
                    "an argument no recording names, or no input in its form) and "
                    f"{timeouts} past the {PROBE_CASE_S} s limit",
                )
        # the function's own tests on the drawn inputs
        kit = self._kit()
        tests = [
            t
            for t in self._tests(it)
            if _imports_inputs((self.run.c_tree / t).read_bytes(), kit)
        ]
        read: set[int] = set()
        for t in tests:
            to = self._timeout(self.cfg.run_s, f"the sample run of {t}")
            reads = Path(tempfile.mkdtemp(prefix="qa-reads-", dir=self.run.tmp))
            outcome = self._pytest(
                self.run.c_tree,
                t,
                to,
                ro={samples: "/qa"},
                rw={reads: "/qa-out"},
            )
            if not _green(outcome):
                self._fail(
                    "fixtures",
                    f"{t} is not green with the drawn inputs appended ({len(outcome.failed)} test "
                    f"case(s) fail, timed_out={outcome.timed_out})",
                )
            for row in _read_results(reads / "reads.jsonl"):
                node = row.get("test")
                if (
                    row.get("item") == item
                    and isinstance(row.get("sample"), int)
                    and isinstance(node, str)
                    and node.split("::", 1)[0] in self._tests(it)
                ):
                    read.add(row["sample"])
        unread = [r["id"] for r in drawn if r["id"] not in read]
        if unread:
            text = (
                f"{item}'s own tests read {len(drawn) - len(unread)} of {len(drawn)} drawn inputs (a test "
                "parametrised over memlab.inputs.inputs reads them)"
            )
            if self.cfg.fixtures == "strict":
                self._fail("fixtures", text + "; every drawn input must be exercised")
            else:
                self._note("fixtures", text)

    def _mutants(
        self,
        it: Any,
        samples: Path | None,
        base: dict[int, dict] | None,
    ) -> None:
        from .gate import _green

        run, cfg, item = self.run, self.cfg, it.item
        tests = self._tests(it)
        if not tests:
            return  # G3 refuses a new or changed function without a new or changed test
        if not all(t in run.qa_first and _green(run.qa_first[t]) for t in tests):
            self._note(
                "mutation",
                f"{item}'s new or changed tests are not all green in G3; mutants not judged",
            )
            return
        rel = item_path(item)
        function = item.split(":", 1)[1]
        source = (run.c_tree / rel).read_text(encoding="utf-8", errors="replace")
        chosen = mutation.choose(
            mutation.sites(source, function),
            self.seed,
            item,
            cfg.max_mutants,
        )
        if not chosen:
            self._note("mutation", f"{item} has no mutation site")
            return
        tree = run.tmp / "qa-mutant"
        if not tree.exists():
            shutil.copytree(run.c_tree, tree)
        target = tree / rel
        original = target.read_bytes()

        def tests_green(text: str) -> bool:
            target.write_text(text, encoding="utf-8")
            for t in tests:
                to = self._timeout(cfg.mutant_s, f"the mutants of {item}")
                if not _green(self._pytest(tree, t, to)):
                    return False
            return True

        try:
            control = mutation.reprint(source)
            ok = tests_green(control) if control is not None else False
            if not ok:
                self._note(
                    "mutation",
                    f"{item}'s tests are not green on its module re-printed unchanged; mutants not judged",
                )
                return
            killed: list[mutation.Site] = []
            survived: list[tuple[mutation.Site, str]] = []
            for site in chosen:
                text = mutation.apply(source, function, site)
                if text is None or text == control:
                    continue
                if tests_green(text):
                    survived.append((site, text))
                else:
                    killed.append(site)
            equivalent: list[mutation.Site] = []
            if survived and samples is not None and base:
                if not informative(base):
                    # every probe row missed the replay, or could not be built or bound: equivalence cannot
                    # be judged, so neither can the kill share (never a pass on the survivors' behalf)
                    why = (
                        f"{item}: mutation not judged: {len(survived)} of {len(killed) + len(survived)} "
                        "mutants survive and the probe got no output of the function on any recorded or "
                        "structurally broken input (replay misses, unbound arguments)"
                    )
                    if cfg.strict:
                        self._fail("mutation", why)
                    else:
                        self._note("mutation", why)
                    return
                for site, text in survived:
                    target.write_text(text, encoding="utf-8")
                    other = self._probe(item, tree, samples)
                    if same_outputs(base, other):
                        equivalent.append(site)
            judged = len(killed) + len(survived) - len(equivalent)
            if judged <= 0:
                self._note(
                    "mutation",
                    f"{item}: no mutant changed its outputs on the recorded inputs; not judged",
                )
                return
            lost = [s for s, _ in survived if s not in equivalent]
            if Decimal(len(killed)) < cfg.min_kill * judged:
                where = _names([f"{s.op} at line {s.line}" for s in lost])
                self._fail(
                    "mutation",
                    f"{item}'s tests kill {len(killed)} of {judged} mutants that change its outputs "
                    f"(threshold {cfg.min_kill}; {len(equivalent)} likely equivalent left out); "
                    f"surviving: {where}",
                )
            else:
                self._note(
                    "mutation",
                    f"{item}'s tests kill {len(killed)} of {judged} mutants ({len(equivalent)} likely "
                    "equivalent left out)",
                )
        finally:
            target.write_bytes(original)


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
# every old text present in :data:`.sol_pass.SOL_SYSTEM`, so the brief cannot drift from these rewrites.
REWRITES: tuple[tuple[str, str, str], ...] = (
    (
        "on",
        'env={"PYTHONPATH": "/memory", "PYTHONDONTWRITEBYTECODE": "1"})',
        'env={"PYTHONPATH": "/memory:/inputs", "PYTHONDONTWRITEBYTECODE": "1"})',
    ),
    (
        "replay",
        "may build the fake environment",
        "may hold shared helpers; build environments with memlab.replay.RecordedEnv",
    ),
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
    """The paragraph Sol's brief gains for the stage-5 checks that are on ("" when none is)."""
    if not cfg.on:
        return ""
    lines = [
        "The gate also checks test quality. Run tests with PYTHONPATH=/memory:/inputs (memlab importable), as "
        "the gate then does; tests may import memlab.replay and memlab.inputs, which the gate provides (the "
        "same test kit wherever the library's tests run). Read a recorded blob with memlab.inputs.blob(<id>), "
        "never by its /inputs path. Library code outside tests never imports memlab: the working model "
        "imports the library without it.",
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
            "- Replay: test a function that takes the environment through memlab.replay.RecordedEnv(<recorded "
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
    """Sol's system prompt with the stage-5 rewrites and paragraph (*base* unchanged when no switch is on)."""
    if not cfg.on:
        return base
    text = base
    for when, old, new in REWRITES:
        if (cfg.on if when == "on" else bool(getattr(cfg, when))) and old in text:
            text = text.replace(old, new)
    return text + "\n" + brief(cfg) + "\n"
