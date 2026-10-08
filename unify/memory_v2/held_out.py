"""G2's held-out-value check: scope is shape, not observed values (spec §F3a, D20).

An environment function may refuse an input (raise its module's ``MemoryInputError``) only for its shape:
types, columns or fields, required keys, the call's identity, formats. It must not refuse a value merely
because no recording showed it. The gate checks this by calling the item, confined, on each covered
observation with one value at a time replaced by a type-valid value no recording holds, and refusing the
item if it then refuses an input it accepted unchanged.

Perturbations come from types alone, never from task ids, stream positions, benchmark names or words.

**A field with one value is identity or format, never perturbed.** The gate reads the recorded observations
of the item's families from the whole evidence store (:func:`pool_actions`: the episodes the manifest names
and the most recent :data:`MAX_POOL_EPISODES` episodes touching each covered channel, read in the store's
order before the manifest's own, at most :data:`MAX_POOL_ACTIONS` actions), not only the covers the pass
chose. When a read cap stops the pool, or the file parse cap leaves a family's file unread, no field (of
that family) is kept: an unread observation could vary it. A family is a tool call's
``channel.method`` (its keywords, or its ``ok`` responses for an item taking observations), a dialogue
``channel.method``, or a work-tree channel and path family (at most :data:`MAX_HOST_PARSES` files parsed).
Among the observations of a cover's family *with the same set of field names* (key paths, columns, keyword
paths), a field is identity or format when it holds one value across at least
:data:`MIN_CONSTANT_OBSERVATIONS` observations from at least :data:`MIN_CONSTANT_EPISODES` episodes: a
message-kind tag (``"type": "SubmitFeedback"`` in every message of that shape), a schema version, a constant
currency. Perturbing it would perturb what the input is, so it is kept, for strings, numbers and booleans
alike, and a note names the field (never a value). Observations are counted by content (a file's blob, a
call's keywords, a response or an observation, canonically): the same file read in two episodes is one
observation, credited to the first episode that showed it. With less support, or two or more values, the
field is perturbed. Matching field-name sets keep a tag that separates message shapes constant within each shape,
while a value field that varies among same-shape observations elsewhere in the store is still perturbed, so
choosing few or similar covers does not hide a whitelist. Tool keywords count calls of every status, so a
value only a rejected call used makes the field vary (the conservative side); responses count ``ok`` calls:

* a string keeps its literal format parts and varies the rest within its character classes. The literal
  parts are the longest common prefix and suffix of the field's distinct observed values, cut back to
  boundaries between character-class runs (letters, digits, each other character) in every value: ids
  ``E1243`` and ``E1488`` keep ``E``; ``ENG-0042``, ``OPS-0043`` and ``SAL-0044`` share nothing, so the prefix
  varies. In the varying part letters stay letters of
  the same case and digits stay digits (hex-like parts stay hex); every other character is kept, so a
  value's non-alphanumeric skeleton never changes. The result differs from the original and is absent
  from every recorded value the gate sees (the episodes the manifest names, the pool and the covered files);
* an integer, or a decimal string, goes beyond the field's observed range in its own direction: a
  non-negative value past the maximum of the field's non-negative values (``2·max + span + k``), a negative
  one past the minimum of its negative values (``2·min − span − k``). The value's sign is kept, so a sign
  convention (credits negative) survives in a mixed-sign field; a zero (or ``-0.00``) goes the way of the
  field's other values (non-positive when they hold a negative and no positive). A decimal string keeps
  its number of places;
* an ISO date (or the date part of a datetime) moves past the observed span by ``span + k`` days, and
  every other date of the same record (the same table row, or the same JSON object) moves by the same
  offset, so order relations within a record (``start <= end``) hold. Numbers change one at a time: an
  item that checks a row sum or a cross-row total against a perturbed number can still be flagged (a
  known limit);
* booleans, nulls, empty strings, float forms other than ``d.d`` and strings without ASCII letters or
  digits are left alone. ``k`` (1–9) and every random choice come from SHA-256 of (item id, episode, action, field).

Where values are replaced, structure (keys, columns, list lengths, the header) is unchanged:

* **tool**: the recorded call's keyword values; the item gets a replay that answers only the recorded
  call, so a perturbed call misses. A miss is fine (the item tried the call); only a refusal raised before
  any environment call is a violation;
  declared as taking an ``observation``, the leaves of the recorded response instead (a field family of
  its own: a rejected call exempts none of them);
* **worktree**: cells of a CSV/TSV table (one per column, the header kept), leaves of a JSON document or
  JSON lines, scalar ``key: value`` lines of YAML; other formats are not perturbed;
* **dialogue**: leaves of a JSON observation (a dict or list, or a string holding one);
* **shell**: skipped, with a note (not supported yet).

Calling convention: the first parameter gets the covered input in the item's declared ``input`` form
(:data:`.manifest.INPUT_KINDS`): ``env``, a replay answering the recorded tool call (its keywords are
perturbed); ``observation``, a tool call's recorded response or a dialogue observation (its leaves are
perturbed); ``path``, a path to the (perturbed) file; ``text``, the file's decoded text, or a dialogue
observation that is a string; ``bytes``, the file's raw bytes. A cover whose kind cannot give the declared
form (a file as ``env``, say) checks nothing and is noted. Without a declaration (an unchanged item
from before declarations), each kind's convention, as ``memory_v2_offline/later_use.py``: tool ``env``,
worktree ``path``, dialogue ``observation``. A later parameter takes the
recorded call's keyword of the same name (tool), else its default, else ``""``. A field case whose keyword
the function does not take is not run, and is noted. Each covered input is also run unperturbed, through
the same serialisation (the baseline); a cover whose baseline is refused checks nothing and is noted.
A declared form that a cover cannot give fails G2 (it would check nothing and tell the working model a
wrong form): a form the cover's kind lacks (a dialogue item declared ``path``), ``text`` for a dialogue
observation that is not a string, ``observation`` for a tool response that is not JSON. A cover that gives
its form but nothing to perturb (a plain-text observation, a file of another format) is noted. A
baseline that fails otherwise (any other error) still lets its perturbed cases be judged: a refusal there
means the changed value turned the run into a refusal.

Only a ``MemoryInputError`` is judged. A held-out value that raises ``KeyError``, ``AssertionError`` or
``ValueError`` before any environment call, on a cover whose baseline was handled, is noted (a likely
whitelist under another exception). A call past :data:`PER_CASE_S` is stopped: the alarm raises a
``BaseException`` and is re-armed if the item catches it; the case counts as an error and is noted.

**Exemptions are narrow and visible.** A cover that is a recorded environment rejection
(:func:`.admission.is_rejection`) lets the item refuse perturbed values only of the fields the rejected
action varied: for a tool call, the keyword paths whose rejected value no accepted call of the same
``channel.method`` (in the episodes the gate sees, or among the covers) used; for a rejected file write, the fields
whose after-values include a value the action's own recorded ``blob_before`` lacks in that field, compared
by content (a refused write that records the unchanged file, reorders rows or appends a copy varies
nothing). When that cannot be told (no accepted call to compare,
before or after missing or equal, a dialogue rejection), the rejection exempts nothing and a note says so.
A tool rejection can still vary a field by coincidence (a rate limit on a call with a new value exempts
that field); every allowed refusal is noted. Every refusal an exemption allows is noted, naming the item and field.

**Known limits.**

* Tool items taking the environment: response values and positional arguments are never perturbed; only
  keywords whose name is a parameter of the function are; a check placed after the environment call is
  invisible behind the replay miss.
* A field with one recorded value is never perturbed, so a guard on it (an id equal to the one recorded)
  is not seen; with few covers, more fields hold one value.
* Numbers change one at a time, so a row-sum or cross-row total check can be flagged; dates move together
  only within a record, so a cross-row order (rows sorted by date) can be flagged.
* Fixed value domains (months 1–12, probabilities 0–1) are perturbed like any range (open question for
  the lead).
* ``"1."``, ``".5"`` and exponent forms are skipped as floats; non-ASCII letters are kept unchanged, so a
  string without ASCII letters or digits is not perturbed.
* Field names are keys, columns or keyword paths; where a recording keys objects by data (ids as keys, a
  first data row taken for a header) the name in a reason is a recorded value (only key-shaped strings are
  redacted).
* Under R16 (careless, not malicious) the check is not hidden from the item: the replay's type, the
  ``/cases`` paths and the results file in the same process are visible to code that looks for them.

**Same results (D26).** The gate also uses this runner and its replay, without perturbation, to show that an
edited function still does what its parent version did (:func:`output_cases`, :func:`run_outputs`): each
recorded input runs on both versions, and what each returns (canonical JSON), refuses or raises, and which
environment calls it issues, are compared on the host. A recorded tool rejection is replayed as the
recorded error. A return value that is not JSON, a timeout or a case that did not run cannot be compared.
Under R16 the comparison is not hidden from the item either.

Bounds: :data:`MAX_COVERS_PER_ITEM` covers (chosen by a hash of item and cover),
:data:`MAX_FIELDS_PER_COVER` fields per cover, :data:`PER_CASE_S` per call and :data:`ITEM_BUDGET_S` per
item, all in one confined process (:func:`.sandbox_run.run_confined`).
"""

from __future__ import annotations

import copy
import csv
import datetime
import hashlib
import io
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from .admission import is_rejection
from .analysis import shapes as _shapes
from .episodes import Action
from .fingerprint import path_family
from .manifest import SEMANTIC_TYPES
from .sandbox_run import SandboxResult, run_confined

MAX_COVERS_PER_ITEM = 8
# :func:`run_outputs` (the gate's same-results check of an edited function): covers compared, seconds per tree
MAX_OUTPUT_COVERS = 128
OUTPUTS_BUDGET_S = 120.0
MAX_FIELDS_PER_COVER = 16
PER_CASE_S = 2
ITEM_BUDGET_S = 60.0
MAX_ACTIONS_PER_EPISODE = 10_000
RESULTS_MAX_BYTES = 4 * 1024**2
_MAX_DEPTH = 8
_MAX_LEAVES = 5000
_ATTEMPTS = 32
# The pool of recorded observations a field's constancy is judged on (see the module docstring).
MAX_POOL_EPISODES = (
    64  # the most recent episodes per covered channel, besides the manifest's
)
MAX_POOL_ACTIONS = 20_000  # actions read for the pool
MAX_HOST_PARSES = 256  # recorded files parsed on the host for the pool
MIN_CONSTANT_OBSERVATIONS = 3
MIN_CONSTANT_EPISODES = 2

# The input forms each kind of cover can give (:data:`.manifest.INPUT_KINDS`), and each kind's convention
# for an item that declares none. Shell covers are not checked at all.
_FORMS = {
    "tool": ("env", "observation"),
    "worktree": ("path", "text", "bytes"),
    "dialogue": ("observation", "text"),
}
_DEFAULT_FORM = {"tool": "env", "worktree": "path", "dialogue": "observation"}

_INT_TEXT = re.compile(r"^[+-]?(?:0|[1-9]\d*)\Z")
_DECIMAL_TEXT = re.compile(r"^[+-]?\d+\.\d+\Z")
_DATE_PART = re.compile(r"^\d{4}-\d{2}-\d{2}")
_LOWER, _UPPER, _DIGITS, _HEX = (
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    "abcdef",
)
# YAML's own scalar literals (syntax, not content): never produced unquoted, never perturbed.
_YAML_LITERALS = frozenset(
    {"true", "false", "yes", "no", "on", "off", "null", "~", "y", "n"},
)


# --- values ------------------------------------------------------------------------------------------------


def seed_of(item: str, eid: str, idx: int, name: str) -> bytes:
    """The deterministic seed of one perturbation: never the clock."""
    return hashlib.sha256(f"{item}\0{eid}\0{idx}\0{name}".encode()).digest()


def _stream(seed: bytes, n: int) -> bytes:
    out, i = b"", 0
    while len(out) < n:
        out += hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        i += 1
    return out[:n]


def _run_class(c: str) -> str:
    if c.isascii() and c.isalpha():
        return "a"
    if c.isascii() and c.isdigit():
        return "0"
    return c


def _boundary(v: str, k: int) -> bool:
    return k <= 0 or k >= len(v) or _run_class(v[k - 1]) != _run_class(v[k])


def literal_affixes(observed: list) -> tuple[str, str]:
    """The literal prefix and suffix every distinct observed string shares, cut to class-run boundaries."""
    vals = sorted({v for v in observed if isinstance(v, str)})
    if len(vals) < 2:
        return "", ""
    k = len(os.path.commonprefix(vals))
    while k and not all(_boundary(v, k) for v in vals):
        k -= 1
    rest = [v[k:] for v in vals]
    m = len(os.path.commonprefix([r[::-1] for r in rest]))
    while m and not all(_boundary(r, len(r) - m) for r in rest):
        m -= 1
    return vals[0][:k], (rest[0][len(rest[0]) - m :] if m else "")


def fresh_string(
    original: str,
    seed: bytes,
    seen: set[str],
    observed: list = (),
) -> str | None:
    """*original* with its varying part's ASCII letters and digits replaced within their classes.

    The literal prefix and suffix the field's *observed* values share (:func:`literal_affixes`) stay;
    the result is changed and unseen, or None.
    """
    pre, suf = literal_affixes(list(observed) + [original])
    if not (original.startswith(pre) and original.endswith(suf)) or len(pre) + len(
        suf,
    ) > len(original):
        pre, suf = "", ""
    middle = original[len(pre) : len(original) - len(suf)]
    if not any(c.isascii() and c.isalnum() for c in middle):
        return None
    letters = [c for c in middle if c.isascii() and c.isalpha()]
    hexlike = (
        bool(letters)
        and any(c.isdigit() for c in middle)
        and all(c.lower() in _HEX for c in letters)
    )
    lower = _HEX if hexlike else _LOWER
    upper = _HEX.upper() if hexlike else _UPPER
    for attempt in range(_ATTEMPTS):
        rnd = _stream(seed + b"/" + attempt.to_bytes(2, "big"), len(middle))
        out = [pre]
        for c, b in zip(middle, rnd):
            if "a" <= c <= "z":
                out.append(lower[b % len(lower)])
            elif "A" <= c <= "Z":
                out.append(upper[b % len(upper)])
            elif "0" <= c <= "9":
                out.append(_DIGITS[b % 10])
            else:
                out.append(c)
        out.append(suf)
        cand = "".join(out)
        if cand != original and cand not in seen:
            return cand
    return None


def beyond_number(
    values: list[Decimal],
    seed: bytes,
    own: Decimal | None = None,
) -> Decimal:
    """A number past the observed range in *own*'s direction, scaled and shifted; its sign is kept.

    A non-negative *own* goes past the maximum of the non-negative observed values (``2·max + span + k``),
    a negative one past the minimum of the negative ones (``2·min − span − k``), so a sign convention
    (credits negative, debits positive) holds. A zero (``-0.00`` too) takes the side of the field's other
    values: non-positive when they hold a negative and no positive, else non-negative. Without *own*, the
    direction is the range's maximum's.
    """
    if own is None:
        own = max(values)
    if own == 0:
        negative = any(v < 0 for v in values) and not any(v > 0 for v in values)
    else:
        negative = own < 0
    side = [v for v in values if (v <= 0 if negative else v >= 0)] or [own]
    hi, lo = max(side), min(side)
    span, k = hi - lo, 1 + seed[0] % 9
    return 2 * lo - span - k if negative else 2 * hi + span + k


def _beyond_date(values: list[datetime.date], seed: bytes) -> datetime.date | None:
    hi, lo = max(values), min(values)
    step = datetime.timedelta(days=(hi - lo).days + 1 + seed[0] % 9)
    for base, sign in ((hi, 1), (lo, -1)):
        try:
            return base + sign * step
        except OverflowError:
            continue
    return None


def _date(text: object) -> datetime.date | None:
    if not isinstance(text, str) or not _DATE_PART.match(text):
        return None
    try:
        if len(text) > 10:
            datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return datetime.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _text_kind(text: str) -> str:
    if _INT_TEXT.match(text):
        return "int"
    if _DECIMAL_TEXT.match(text):
        return "decimal"
    if _date(text) is not None:
        return "date"
    v = _shapes.value_type(text)
    if v in ("float", "bool", "empty"):
        return "skip"  # exponent forms, boolean literals, blanks
    return "str"


def _number(v: object) -> Decimal | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return Decimal(repr(v)) if isinstance(v, float) else Decimal(v)


def perturb_value(value: Any, observed: list, seed: bytes, seen: set[str]) -> Any:
    """A type-valid value no recording shows, from *value*'s type and the field's *observed* values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        own = _number(value)
        if own is None:
            return None
        nums = [n for n in (_number(v) for v in observed) if n is not None]
        new = beyond_number(nums + [own], seed, own)
        return int(new) if isinstance(value, int) else float(new)
    if not isinstance(value, str):
        return None
    kind = _text_kind(value)
    same = [v for v in observed if isinstance(v, str) and _text_kind(v) == kind]
    if kind in ("int", "decimal"):
        try:
            new = beyond_number(
                [Decimal(v) for v in same + [value]],
                seed,
                Decimal(value),
            )
        except InvalidOperation:
            return None
        if kind == "int":
            return str(int(new))
        places = len(value.split(".", 1)[1])
        return format(new.quantize(Decimal(1).scaleb(-places)), "f")
    if kind == "date":
        dates = [d for d in (_date(v) for v in same + [value]) if d is not None]
        new_date = _beyond_date(dates, seed)
        return None if new_date is None else new_date.isoformat() + value[10:]
    if kind == "str":
        return fresh_string(value, seed, seen, same)
    return None


_PADDED_TEXT = re.compile(r"^0\d+\Z")


def _numeric_form(value: Any) -> tuple[str, int] | None:
    """How a number is written, or None.

    ("int" | "float" | "int_text" | "int_pad" | "dec_text", n): decimal places, or the width of a
    zero-padded integer text (``"09"``), which is written back with the same width.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return ("int", 0)
    if isinstance(value, float):
        return ("float", 2) if math.isfinite(value) else None
    if isinstance(value, str) and _PADDED_TEXT.match(value):
        return ("int_pad", len(value))
    if isinstance(value, str) and _INT_TEXT.match(value):
        return ("int_text", 0)
    if isinstance(value, str) and _DECIMAL_TEXT.match(value):
        return ("dec_text", len(value.split(".", 1)[1]))
    return None


def _write_like(form: tuple[str, int], v: Decimal) -> Any:
    kind, places = form
    integral = v == v.to_integral_value()
    if kind == "int":
        return int(v) if integral else float(v)
    if kind == "float":
        return float(v)
    if kind == "int_pad" and integral:
        return ("-" if v < 0 else "") + str(abs(int(v))).zfill(places)
    if kind in ("int_text", "int_pad") and integral:
        return str(int(v))
    q = Decimal(1).scaleb(-(places if kind == "dec_text" else 2))
    return format(v.quantize(q), "f")


def _pick(seed: bytes, n: int) -> int:
    return int.from_bytes(seed[:8], "big") % n


def _as_decimal(v: Any) -> Decimal:
    return Decimal(repr(v)) if isinstance(v, float) else Decimal(str(v))


@dataclass
class Declared:
    """What a declared field's check runs: in-domain values (to accept) and out-of-domain values (to refuse).

    ``why`` is None, or: ``"all_seen"`` (every in-domain value is recorded already: only ``outs`` run),
    ``"format"`` (the value is not written as a value of the type, e.g. ``"40%"`` or null: nothing runs and
    the field keeps the untagged shape check), ``"contradicts"`` (a value of the type outside its domain,
    or a fraction for an integer type: the declaration contradicts the recording).
    """

    ins: list = field(default_factory=list)
    outs: list = field(default_factory=list)
    why: str | None = None


def declared_values(
    sem: str,
    value: Any,
    observed: list,
    seed: bytes,
    seen: set[str],
) -> Declared:
    """For a field declared *sem*: unseen in-domain values and out-of-domain values, written like *value*.

    In-domain values for a bounded type are the lowest and highest unseen values of its domain grid (every
    integer, also for a number written as an integer; otherwise interior points at the value's own
    decimal precision, or 0.01 for a float) and one picked by *seed*; for an unbounded one, a value past
    the observed maximum and one unseen interior value. Out-of-domain values sit on both sides of a
    bounded domain (``low - 1`` and ``high + 1`` for integers, a tenth of the span past each end for
    numbers) and at ``-1`` below a zero bound. A currency code's are two letters and two letters with a
    digit, which upper-casing cannot repair.
    """
    kind, low, high = SEMANTIC_TYPES[sem]
    if kind == "code3":
        if not (isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value)):
            return Declared(why="format")
        taken = set(seen) | {v for v in observed if isinstance(v, str)}
        ins: list = []
        for i in range(2):
            new = fresh_string(value, seed + bytes([i]), taken | set(ins))
            if new is not None:
                ins.append(new)
        base = (ins[0] if ins else value)[:2]
        return Declared(
            ins,
            [base, base + str(seed[1] % 10)],
            None if ins else "all_seen",
        )
    form = _numeric_form(value)
    if form is None:
        return Declared(why="format")
    own = _as_decimal(value)
    if kind == "int" and own != own.to_integral_value():
        return Declared(why="contradicts")
    if (low is not None and own < low) or (high is not None and own > high):
        return Declared(why="contradicts")
    # every recorded value of the field the gate sees must fit the type, not only this cover's first one
    for v in observed:
        if _numeric_form(v) is None:
            continue
        x = _as_decimal(v)
        if (
            (kind == "int" and x != x.to_integral_value())
            or (low is not None and x < low)
            or (high is not None and x > high)
        ):
            return Declared(why="contradicts")
    obs = {_as_decimal(v) for v in observed if _numeric_form(v) is not None}
    # integers stay integers (the written type is shape) unless a number's domain has no interior integer
    integer = kind == "int" or (
        form[0] in ("int", "int_text", "int_pad") and (high is None or high - low >= 2)
    )
    # a decimal string's own precision; 0.01 for floats and for integers written into a fractional domain
    places = form[1] if form[0] == "dec_text" else 2
    step = Decimal(1) if integer else Decimal(1).scaleb(-places)
    if high is not None:
        if integer:
            grid = [Decimal(i) for i in range(low, high + 1)]
            outs = [Decimal(low - 1), Decimal(high + 1)]
        else:
            count = min(int((high - low) / step), 10_000)
            step = (Decimal(high) - low) / count
            grid = [low + step * i for i in range(1, count)]
            tenth = Decimal(high - low) / 10
            outs = [low - tenth, high + tenth]
        unseen = [g for g in grid if g not in obs]
        picks = (
            [unseen[0], unseen[-1], unseen[_pick(seed, len(unseen))]] if unseen else []
        )
    else:
        top = max(obs | {own, Decimal(0)})
        picks = [2 * top + 1 + seed[0] % 9]
        inner = [g for g in (top * (_pick(seed[8:], 97) + 1) / 98,) if 0 < g < top]
        picks += [
            g.quantize(step) if not integer else g.to_integral_value() for g in inner
        ]
        picks = [g for g in picks if g not in obs]
        outs = [Decimal(-1)]
    ins = []
    for g in picks:
        w = _write_like(form, g)
        if w not in ins:
            ins.append(w)
    return Declared(
        ins,
        [_write_like(form, g) for g in outs],
        None if ins else "all_seen",
    )


def shift_date(text: str, days: int) -> str | None:
    """*text* (an ISO date, or a datetime's date part) moved by *days*, the rest kept; None if not a date."""
    d = _date(text) if isinstance(text, str) and _text_kind(text) == "date" else None
    if d is None:
        return None
    try:
        return (d + datetime.timedelta(days=days)).isoformat() + text[10:]
    except OverflowError:
        return None


# --- documents: where the values sit -----------------------------------------------------------------------


def _walk(obj: Any, steps: tuple, disp: str, out: list, depth: int = 0) -> None:
    if len(out) >= _MAX_LEAVES or depth > _MAX_DEPTH:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, steps + (k,), f"{disp}.{k}" if disp else str(k), out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(v, steps + (i,), disp + "[]", out, depth + 1)
    else:
        out.append((disp or ".", steps, obj))


def _replace(obj: Any, changes: list[tuple[tuple, Any]]) -> Any:
    new = copy.deepcopy(obj)
    for steps, value in changes:
        if not steps:
            return value
        node = new
        for s in steps[:-1]:
            node = node[s]
        node[steps[-1]] = value
    return new


# A document's ``render(changes)`` takes [(locator, new value), ...] (empty: the baseline);
# ``siblings(locator)`` lists the other scalar fields of the same record as (locator, value).


class _Json:
    """A JSON-like value; *dump* turns a (possibly perturbed) copy into the payload."""

    def __init__(self, obj: Any, dump: Callable[[Any], Any]) -> None:
        self.obj, self.dump = obj, dump

    def fields(self) -> dict[str, tuple[Any, Any, list]]:
        leaves: list = []
        _walk(self.obj, (), "", leaves)
        out: dict[str, tuple[Any, Any, list]] = {}
        for disp, steps, value in leaves:
            if disp not in out:
                out[disp] = (steps, value, [])
            out[disp][2].append(value)
        return out

    def render(self, changes: list = ()) -> Any:
        return self.dump(_replace(self.obj, list(changes)))

    def siblings(self, loc: tuple) -> list[tuple[tuple, Any]]:
        if not loc:
            return []
        node = self.obj
        for s in loc[:-1]:
            node = node[s]
        if not isinstance(node, dict):
            return []
        return [
            (loc[:-1] + (k,), v)
            for k, v in node.items()
            if k != loc[-1] and not isinstance(v, (dict, list, tuple))
        ]


class _Table:
    """A delimited table; one cell per column is perturbed, the header and every row's width kept."""

    def __init__(
        self,
        text: str,
        delimiter: str,
        header: bool,
        encoding: str,
        crlf: bool,
    ):
        self.rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        self.delimiter, self.header, self.encoding = delimiter, header, encoding
        self.lineterminator = "\r\n" if crlf else "\n"

    def fields(self) -> dict[str, tuple[Any, Any, list]]:
        body_start = 1 if self.header else 0
        width = max((len(r) for r in self.rows), default=0)
        out: dict[str, tuple[Any, Any, list]] = {}
        for j in range(width):
            name = (
                self.rows[0][j].strip()
                if self.header and j < len(self.rows[0])
                else f"#{j + 1}"
            )
            cells = [
                (i, r[j])
                for i, r in enumerate(self.rows[body_start:], start=body_start)
                if j < len(r) and r[j].strip()
            ]
            if cells and name not in out:
                out[name] = ((cells[0][0], j), cells[0][1], [c for _, c in cells])
        return out

    def siblings(self, loc: tuple) -> list[tuple[tuple, Any]]:
        i, j = loc
        return [((i, c), cell) for c, cell in enumerate(self.rows[i]) if c != j]

    def render(self, changes: list = ()) -> bytes:
        rows = [list(r) for r in self.rows]
        for (i, j), value in changes:
            rows[i][j] = value
        buf = io.StringIO()
        w = csv.writer(
            buf,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
        )
        for r in rows:
            w.writerow(r)
        return buf.getvalue().encode(self.encoding, errors="replace")


_YAML_LINE = re.compile(
    r"^(?P<ind>[ ]*)(?P<dash>-[ ]+)?(?P<key>[A-Za-z0-9_][^:#\n]*?):(?:[ \t]+(?P<val>.*?))?[ \t]*\Z",
)


class _Yaml:
    """Block-style YAML, line by line: scalar ``key: value`` lines are the fields, by key path."""

    def __init__(self, text: str, encoding: str) -> None:
        self.lines, self.encoding = text.split("\n"), encoding

    def fields(self) -> dict[str, tuple[Any, Any, list]]:
        out: dict[str, tuple[Any, Any, list]] = {}
        stack: list[tuple[int, str]] = []
        for n, line in enumerate(self.lines):
            m = _YAML_LINE.match(line)
            if m is None:
                continue
            level = len(m["ind"]) + len(m["dash"] or "")
            while stack and stack[-1][0] >= level:
                stack.pop()
            key = m["key"].strip()
            path = ".".join([k for _, k in stack] + [key])
            raw = m["val"] or ""
            if not raw:
                stack.append((level, key))
                continue
            start = m.start("val")
            quote = raw[0] if raw[0] in "'\"" else ""
            if quote:
                end = raw.find(quote, 1)
                if end < 0:
                    continue
                inner, span = raw[1:end], (start + 1, start + end)
            else:
                cut = raw.find(" #")
                inner = raw if cut < 0 else raw[:cut].rstrip()
                span = (start, start + len(inner))
                if inner[:1] in tuple("|>&*!{[%@`") or inner.lower() in _YAML_LITERALS:
                    continue
            if path not in out:
                out[path] = ((n, span), inner, [])
            out[path][2].append(inner)
        return out

    def siblings(self, loc: Any) -> list:
        return []  # line by line, records are not tracked: a YAML date moves alone

    def render(self, changes: list = ()) -> bytes:
        lines = list(self.lines)
        for (n, (a, b)), value in changes:
            lines[n] = lines[n][:a] + value + lines[n][b:]
        return "\n".join(lines).encode(self.encoding, errors="replace")


def _file_doc(path: str, data: bytes) -> _Json | _Table | _Yaml | None:
    if len(data) > _shapes.PARSE_LIMIT:
        return None
    s = _shapes.shape(path, data)
    encoding = s.get("encoding")
    if not isinstance(encoding, str):
        return None
    _, text = _shapes.decode(data)
    fmt = s.get("format")
    if fmt == "csv" and isinstance(s.get("delimiter"), str):
        return _Table(
            text,
            s["delimiter"],
            bool(s.get("header")),
            encoding,
            b"\r\n" in data,
        )
    if fmt == "yaml":
        return _Yaml(text, encoding)
    if fmt in ("json", "jsonl"):
        try:
            if fmt == "jsonl":
                rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
                return _Json(
                    rows,
                    lambda o: "".join(
                        json.dumps(r, ensure_ascii=False) + "\n" for r in o
                    ).encode(encoding, errors="replace"),
                )
            obj = json.loads(text)
        except ValueError:
            return None
        indent = 1 if "\n" in text.strip() else None
        return _Json(
            obj,
            lambda o: json.dumps(o, ensure_ascii=False, indent=indent).encode(
                encoding,
                errors="replace",
            ),
        )
    return None


# --- the plan ----------------------------------------------------------------------------------------------


def family(a: Action) -> tuple[str, str, str]:
    """The family a rejection exempts: ``channel.method``, or the worktree channel and path family."""
    kind = getattr(a, "kind", "tool")
    if kind == "worktree":
        path = a.args[0] if a.args and isinstance(a.args[0], str) else ""
        return (kind, a.channel, path_family(path))
    return (kind, a.channel, a.method)


@dataclass
class Case:
    cover: tuple[str, int]
    field: str | None  # None: the baseline (the covered input, unperturbed)
    family: tuple[str, str, str]
    kind: str
    payload: dict
    param: str | None = None  # tool: the keyword the perturbation sits in
    file: bytes | None = None  # worktree: the file's bytes, mounted at payload["path"]
    name: str = ""
    side: str | None = (
        None  # a declared field's "in" (must be accepted) or "out" (must be refused)
    )
    semantic: str | None = None  # its declared type


@dataclass
class Plan:
    cases: list[Case] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # declared input forms a cover cannot give, as reason text (G2 fails on each)
    unfit: list[str] = field(default_factory=list)
    # family -> the fields a covered rejection of that family varied (see the module docstring)
    exempt: dict[tuple[str, str, str], set[str]] = field(default_factory=dict)
    # declared fields whose recorded values are not of their declared type: field -> type
    mismatched: dict[str, str] = field(default_factory=dict)


def _strings(value: Any, out: set[str], depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        return
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str):
                out.add(k)
            _strings(v, out, depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _strings(v, out, depth + 1)


def _file_name(a: Action, n: int) -> str:
    path = a.args[0] if a.args and isinstance(a.args[0], str) else ""
    base = path.rsplit("/", 1)[-1]
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,99}", base):
        return base
    return f"file{n}{_shapes.extension(path)[:16]}"


def _blob_sha(a: Action) -> str | None:
    """The recorded file a work-tree action shows: a write's after-blob, else its before-blob."""
    r = a.response if isinstance(a.response, dict) else {}
    sha = r.get("blob_after") if a.method == "write" else r.get("blob_before")
    sha = sha or r.get("blob_before") or r.get("blob_after")
    return sha if isinstance(sha, str) else None


def _blob_of(a: Action, blob: Callable[[str], bytes]) -> bytes | None:
    sha = _blob_sha(a)
    if sha is None:
        return None
    try:
        return blob(sha)
    except (OSError, KeyError, ValueError):
        return None


def _form(a: Action, input_kind: str | None) -> str | None:
    """The form *a*'s covered input reaches the item in: the declared one, else the kind's convention."""
    return input_kind or _DEFAULT_FORM.get(getattr(a, "kind", "tool"))


def _observation_doc(obs: Any) -> _Json | None:
    """A JSON observation (a dict or list, or a string holding one); a string stays a string when rendered."""
    if isinstance(obs, str):
        try:
            parsed = json.loads(obs)
        except ValueError:
            return None
        if isinstance(parsed, (dict, list)):
            return _Json(parsed, lambda o: json.dumps(o, ensure_ascii=False))
        return None
    return _Json(obs, lambda o: o) if isinstance(obs, (dict, list)) else None


def _doc_family(a: Action, input_kind: str | None) -> tuple[str, str, str]:
    """A cover's field family: :func:`family`, except a tool response (its fields are not the call's)."""
    if getattr(a, "kind", "tool") == "tool" and _form(a, input_kind) == "observation":
        return ("tool_response", a.channel, a.method)
    return family(a)


def _doc(
    a: Action,
    blob: Callable[[str], bytes],
    input_kind: str | None = None,
) -> _Json | _Table | _Yaml | None:
    """What is perturbed for *a*'s covered input in its form; None if nothing (or the form does not fit)."""
    kind = getattr(a, "kind", "tool")
    form = _form(a, input_kind)
    if form not in _FORMS.get(kind, ()):
        return None
    if kind == "tool":
        if form == "env":
            return _Json(dict(a.kwargs or {}), lambda o: o)
        return _observation_doc(a.response)
    if kind == "dialogue":
        if form == "text" and not isinstance(a.response, str):
            return None
        return _observation_doc(a.response)
    if kind == "worktree":
        data = _blob_of(a, blob)
        path = a.args[0] if a.args and isinstance(a.args[0], str) else ""
        return None if data is None else _file_doc(path, data)
    return None


def _same_record(d: Any, loc: Any, value: Any, new: Any) -> list[tuple[Any, Any]]:
    """For a perturbed date, every other date of its record moved by the same offset (else nothing)."""
    before, after = _date(value), _date(new)
    if before is None or after is None or not isinstance(value, str):
        return []
    days = (after - before).days
    out = []
    for sloc, sval in d.siblings(loc):
        moved = shift_date(sval, days) if isinstance(sval, str) else None
        if moved is not None:
            out.append((sloc, moved))
    return out


def _key(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def _rejected_change(r: Action, blob: Callable[[str], bytes]) -> set[str] | None:
    """The fields a rejected file write changed: its own before and after blobs, compared by content.

    A field varied when the after-file holds a value of it that the before-file does not hold anywhere in
    that field, so reordering rows, appending a copy or deleting rows varies nothing. None (nothing can be
    attributed) when either blob is missing or unreadable, they are equal, they do not parse as one format,
    or their fields (columns, key paths) differ.
    """
    resp = r.response if isinstance(r.response, dict) else {}
    path = r.args[0] if r.args and isinstance(r.args[0], str) else ""
    shas = [resp.get(k) for k in ("blob_before", "blob_after")]
    if not all(isinstance(h, str) for h in shas) or shas[0] == shas[1]:
        return None
    try:
        before, after = (blob(h) for h in shas)
    except (OSError, KeyError, ValueError):
        return None
    if before == after:
        return None
    docs = [_file_doc(path, data) for data in (before, after)]
    if any(d is None for d in docs) or type(docs[0]) is not type(docs[1]):
        return None
    fields = [{n: {_key(x) for x in v[2]} for n, v in d.fields().items()} for d in docs]
    if set(fields[0]) != set(fields[1]):
        return None
    return {n for n, after_values in fields[1].items() if after_values - fields[0][n]}


def _exemptions(
    out: Plan,
    covers: list[tuple[str, int, Action]],
    chosen: dict[tuple[str, int], Action],
    seen: list[Action],
    blob: Callable[[str], bytes],
) -> None:
    """Fill ``out.exempt``: per rejection cover, the fields it varied against accepted observations."""
    for _, _, r in covers:
        kind = getattr(r, "kind", "tool")
        if kind == "shell" or not is_rejection(r):
            continue
        fam = family(r)
        rejected: dict[str, list] = {}
        accepted: dict[str, set[str]] = {}
        known = False
        if kind == "tool" and isinstance(r.kwargs, dict):
            rejected = {
                n: v[2] for n, v in _Json(r.kwargs, lambda o: o).fields().items()
            }
            for a in seen + [a for a in chosen.values()]:
                if (
                    getattr(a, "kind", "tool") == "tool"
                    and a.status == "ok"
                    and family(a) == fam
                    and isinstance(a.kwargs, dict)
                ):
                    known = True
                    for n, v in _Json(a.kwargs, lambda o: o).fields().items():
                        accepted.setdefault(n, set()).update(_key(x) for x in v[2])
            what = f"a recorded rejection of {r.channel}.{r.method}"
        elif kind == "worktree":
            varied = _rejected_change(r, blob)
            if varied is None:
                out.notes.append(
                    "a recorded file rejection does not record a changed file it can attribute "
                    "(before and after missing, unreadable, equal, or of another format or "
                    "other fields); it exempts nothing",
                )
            elif not varied:
                out.notes.append(
                    "a recorded file rejection changes no field; it exempts nothing",
                )
            out.exempt.setdefault(fam, set()).update(varied or set())
            continue
        else:
            what = f"a recorded {kind} rejection"
        if not known or not rejected:
            out.notes.append(
                f"{what} cannot be compared with an accepted one; it exempts nothing",
            )
            continue
        varied = {
            n
            for n, vals in rejected.items()
            if any(_key(x) not in accepted.get(n, set()) for x in vals)
        }
        if not varied:
            out.notes.append(
                f"{what} varies no field of an accepted one; it exempts nothing",
            )
        out.exempt.setdefault(fam, set()).update(varied)


def _unfit_reason(a: Action, input_kind: str | None) -> str | None:
    """Why *a*'s covered input cannot be given in the declared form, or None (also when none is declared)."""
    if input_kind is None:
        return None
    kind = getattr(a, "kind", "tool")
    if input_kind not in _FORMS.get(kind, ()):
        return f"declares input {input_kind}, which a {kind} cover cannot give"
    if kind == "dialogue" and input_kind == "text" and not isinstance(a.response, str):
        return "declares input text, which a dialogue cover with a non-text observation cannot give"
    if (
        kind == "tool"
        and input_kind == "observation"
        and _observation_doc(a.response) is None
    ):
        return "declares input observation, which a tool cover without a JSON response cannot give"
    return None


def pool_actions(
    episode_ids: list[str],
    lookup: Callable[[str, int], Action | None],
    channels: set[str],
) -> tuple[list[tuple[str, Action]], bool]:
    """(episode id, action) of *episode_ids* on *channels*, reading at most :data:`MAX_POOL_ACTIONS` actions.

    The second value is True when the cap stopped the read.
    """
    out: list[tuple[str, Action]] = []
    read = 0
    for eid in episode_ids:
        for i in range(MAX_ACTIONS_PER_EPISODE):
            if read >= MAX_POOL_ACTIONS:
                return out, True
            a = lookup(eid, i)
            read += 1
            if a is None:
                break
            if a.channel in channels:
                out.append((eid, a))
    return out, False


def plan(
    item: str,
    covers: list[tuple[str, int, Action]],
    *,
    seen: list[Action],
    blob: Callable[[str], bytes],
    field_types: dict[str, str] | None = None,
    input_kind: str | None = None,
    pool: list[tuple[str, Action]] | None = None,
    pool_capped: bool = False,
) -> Plan:
    """The cases that check *item* on its covered observations (see the module docstring).

    *covers* are the item's validated covers; *seen* every recorded action of the episodes the gate sees;
    *blob* reads a recorded file blob; *field_types* the item's declared semantic types (D21);
    *input_kind* its declared input form (None: each kind's convention); *pool* the recorded observations
    (episode id, action) a field's constancy is judged on, holding the covers (None: the covers, plus
    *seen* under an unknown episode id ``""``, which never counts toward the episodes); *pool_capped*
    whether a read cap stopped the pool, in which case no field is kept as constant (fail safe).
    """
    out = Plan()
    field_types = dict(field_types or {})
    declared_seen: set[str] = set()
    unwritten: dict[str, str] = (
        {}
    )  # declared fields not written as a value of their type
    all_seen: dict[str, str] = {}  # declared fields whose whole domain is recorded
    typed_any: set[str] = set()  # declared fields some cover checked as their type
    shell = sum(1 for _, _, a in covers if getattr(a, "kind", "tool") == "shell")
    if shell:
        out.notes.append(f"skips {shell} shell cover(s) (not supported yet)")
    chosen: dict[tuple[str, int], Action] = {}
    for eid, idx, a in covers:
        kind = getattr(a, "kind", "tool")
        if kind != "shell" and a.status == "ok" and not is_rejection(a):
            chosen.setdefault((eid, idx), a)
    fit: list[tuple[str, int]] = []
    for c, a in chosen.items():
        why = _unfit_reason(a, input_kind)
        if why is None:
            fit.append(c)
        elif why not in out.unfit:
            out.unfit.append(why)
    ranked = sorted(
        fit,
        key=lambda c: hashlib.sha256(f"{item}\0{c[0]}\0{c[1]}".encode()).digest(),
    )
    order = ranked[:MAX_COVERS_PER_ITEM]
    docs = {c: _doc(chosen[c], blob, input_kind) for c in order}
    empty = sum(1 for d in docs.values() if d is None)
    if empty and input_kind is not None:
        out.notes.append(
            f"{empty} cover(s) give nothing to perturb as {input_kind} (not JSON, or a file of "
            "another format); not checked",
        )
    docs = {c: d for c, d in docs.items() if d is not None}
    dfam = {c: _doc_family(chosen[c], input_kind) for c in docs}
    _exemptions(out, covers, chosen, seen, blob)

    strings: set[str] = set()
    if (
        pool is None
    ):  # the covers, and the seen actions under an unknown episode (never counted as one)
        pool = [(eid, a) for eid, _, a in covers]
        pool += [("", a) for a in seen if not any(a is b for _, _, b in covers)]
    pool = list(pool)
    for a in seen + [a for _, a in pool]:
        for part in (a.args, a.kwargs, a.response, a.error):
            _strings(part, strings)
    stats: dict[tuple, list] = {}
    for c, d in docs.items():
        for name, (_, _, values) in d.fields().items():
            stats.setdefault((dfam[c], name), []).extend(values)
            _strings(values, strings)
    families = set(dfam.values())

    def ranged(fam: tuple, d: Any) -> None:
        for name, (_, _, values) in d.fields().items():
            stats.setdefault((fam, name), []).extend(values)

    # a tool field's range is every recorded call (or ok response) of the same method: named episodes and pool
    for a in seen:
        if getattr(a, "kind", "tool") != "tool":
            continue
        if isinstance(a.kwargs, dict) and family(a) in families:
            ranged(family(a), _Json(a.kwargs, lambda o: o))
        rfam = ("tool_response", a.channel, a.method)
        if rfam in families and a.status == "ok":
            d = _observation_doc(a.response)
            if d is not None:
                ranged(rfam, d)
    # constancy: the pool's distinct observations (by content) by family and field-name set, each with the
    # first episode that showed it and its fields' values
    shapes: dict[tuple, dict[str, tuple[str, dict[str, set[str]]]]] = {}
    parses = 0
    parse_capped: set[tuple] = set()  # families with a file the parse cap left unread
    for eid, a in pool:
        kind = getattr(a, "kind", "tool")
        if kind == "shell":
            continue
        if kind == "tool" and _form(a, input_kind) == "env":
            fam = family(a)
            if fam not in families or not isinstance(a.kwargs, dict):
                continue
            d = _Json(dict(a.kwargs), lambda o: o)
            ranged(fam, d)
            content = "kwargs:" + _key(a.kwargs)
        elif kind == "tool":
            fam = ("tool_response", a.channel, a.method)
            if fam not in families or a.status != "ok":
                continue
            d = _observation_doc(a.response)
            if d is not None:
                ranged(fam, d)
            content = "response:" + _key(a.response)
        else:
            fam = family(a)
            if fam not in families or a.status != "ok" or is_rejection(a):
                continue
            if kind == "worktree":
                if parses >= MAX_HOST_PARSES:
                    parse_capped.add(fam)
                    continue
                parses += 1
            content = (
                f"blob:{_blob_sha(a)}"
                if kind == "worktree"
                else "observation:" + _key(a.response)
            )
            d = _doc(a, blob, input_kind)
        if d is None:
            continue
        fields = d.fields()
        seen_contents = shapes.setdefault((fam, frozenset(fields)), {})
        if (
            content not in seen_contents
        ):  # identical content in two episodes is one observation
            seen_contents[content] = (
                eid,
                {n: {_key(v) for v in vals} for n, (_, _, vals) in fields.items()},
            )
    if pool_capped:
        out.notes.append(
            "the recorded-observation pool stopped at its read cap; no field is kept as identity or "
            "format",
        )
    if parse_capped:
        out.notes.append(
            f"constancy parsed its cap of {MAX_HOST_PARSES} recorded files; no field of a family with "
            "unread files is kept as identity or format",
        )

    def constant(fam: tuple, shape: frozenset, name: str) -> bool:
        if pool_capped or fam in parse_capped:
            return False  # an unread observation could vary the field: fail safe
        obs = list(shapes.get((fam, shape), {}).values())
        values: set[str] = set()
        for _, by_name in obs:
            values |= by_name.get(name, set())
        return (
            len(obs) >= MIN_CONSTANT_OBSERVATIONS
            and len({eid for eid, _ in obs if eid}) >= MIN_CONSTANT_EPISODES
            and len(values) == 1
        )

    kept: list[str] = []  # fields kept as identity or format, for the note
    if any(isinstance(d, _Yaml) for d in docs.values()):
        strings |= _YAML_LITERALS

    for c, d in docs.items():  # in the hashed cover order
        a = chosen[c]
        kind, fam, form = getattr(a, "kind", "tool"), dfam[c], _form(a, input_kind)
        perturbed: list[tuple] = []
        fields = list(d.fields().items())
        shape = frozenset(name for name, _ in fields)
        typed: set[str] = set()
        # declared fields first, outside the field limit
        for name, (loc, value, _) in fields:
            sem = field_types.get(name)
            if sem is None:
                continue
            declared_seen.add(name)
            got = declared_values(
                sem,
                value,
                stats.get((fam, name), []),
                seed_of(item, c[0], c[1], name),
                strings,
            )
            if got.why == "contradicts":
                out.mismatched[name] = sem
                typed.add(name)
                continue
            if got.why == "format":
                unwritten.setdefault(name, sem)
                continue  # the field keeps the untagged shape check below
            if got.why == "all_seen":
                all_seen.setdefault(name, sem)
            perturbed += [(name, loc, [(loc, v)], "in", sem) for v in got.ins]
            perturbed += [(name, loc, [(loc, v)], "out", sem) for v in got.outs]
            typed.add(name)
        typed_any.update(typed)
        checked = 0
        for name, (loc, value, _) in fields:
            if name in typed:
                continue
            if constant(fam, shape, name):
                if name not in kept:
                    kept.append(name)
                continue
            if checked >= MAX_FIELDS_PER_COVER:
                break
            seed = seed_of(item, c[0], c[1], name)
            new = perturb_value(value, stats.get((fam, name), []), seed, strings)
            if new is not None:
                same = _same_record(d, loc, value, new)
                perturbed.append((name, loc, [(loc, new)] + same, None, None))
                checked += 1
        if not perturbed:
            continue
        for name, loc, changes, side, sem in [(None, None, [], None, None)] + perturbed:
            n = len(out.cases)
            case = Case(c, name, fam, kind, {}, side=side, semantic=sem)
            rendered = d.render(changes)
            if kind == "tool" and form == "env":
                case.payload = {
                    "kind": kind,
                    "form": form,
                    "action": {
                        "channel": a.channel,
                        "method": a.method,
                        "args": list(a.args),
                        "kwargs": dict(a.kwargs),
                        "response": a.response,
                    },
                    "kwargs": rendered,
                }
                case.param = None if loc is None else str(loc[0])
            elif (
                kind == "tool"
            ):  # the recorded response; later parameters take the call's keywords
                case.payload = {
                    "kind": kind,
                    "form": form,
                    "kwargs": dict(a.kwargs or {}),
                    "observation": rendered,
                }
            elif kind == "worktree" and form == "text":
                case.payload = {
                    "kind": kind,
                    "form": form,
                    "observation": _shapes.decode(rendered)[1],
                }
            elif kind == "worktree":
                case.name = _file_name(a, n)
                case.file = rendered
                case.payload = {
                    "kind": kind,
                    "form": form,
                    "path": f"/cases/files/{n}/{case.name}",
                }
            else:
                case.payload = {"kind": kind, "form": form, "observation": rendered}
            out.cases.append(case)
    if kept:
        out.notes.append(
            "fields with one value across the recorded observations of their shape, kept as identity "
            "or format and "
            "not perturbed: " + ", ".join(k[:80] for k in kept[:_NAMED]) + _more(kept),
        )
    unwritten = {n: t for n, t in unwritten.items() if n not in typed_any}
    for name, sem in list(unwritten.items())[:5]:
        out.notes.append(
            f"declared {sem} field {name[:80]} is not written as a {sem} value in a covered "
            "input; it keeps the untagged shape check",
        )
    for name, sem in list(all_seen.items())[:5]:
        out.notes.append(
            f"declared {sem} field {name[:80]}: every in-domain value is already recorded; "
            "only out-of-domain values are checked",
        )
    missing = sorted(set(field_types) - declared_seen)
    if missing:
        out.notes.append(
            "declared fields not in any covered input, not checked: "
            + ", ".join(m[:80] for m in missing[:5])
            + (f" and {len(missing) - 5} more" if len(missing) > 5 else ""),
        )
    return out


# --- the confined run --------------------------------------------------------------------------------------

# Runs inside the box: /memory (the tree, read-only), /cases (cases.json, files/), /out. Standard library only.
_RUNNER = r"""
import hashlib, importlib, inspect, json, signal, sys
sys.path.insert(0, "/memory")
sys.dont_write_bytecode = True

class ReplayMiss(LookupError): pass
class RecordedError(RuntimeError): pass

def _key(channel, method, args, kwargs):
    return json.dumps([channel, method, list(args), dict(sorted(kwargs.items()))], sort_keys=True, default=str)

class _Channel:
    def __init__(self, env, name): self._env, self._name = env, name
    def __getattr__(self, method):
        if method.startswith("_"): raise AttributeError(method)
        return lambda *a, **k: self._env._serve(self._name, method, a, k)

class Replay:
    def __init__(self, a):
        self._k = _key(a["channel"], a["method"], a["args"], a["kwargs"]); self._r = a["response"]; self._calls = 0
        self._error = a.get("status") == "error"; self._issued = hashlib.sha256()
    def __getattr__(self, ch):
        if ch.startswith("_"): raise AttributeError(ch)
        return _Channel(self, ch)
    def _serve(self, channel, method, args, kwargs):
        self._calls += 1
        key = _key(channel, method, args, kwargs)
        self._issued.update(key.encode("utf-8", "surrogatepass") + b"\0")
        if key != self._k:
            raise ReplayMiss(channel + "." + method)
        if self._error:  # a recorded rejection (outputs mode only; the held-out cases are never one)
            raise RecordedError(channel + "." + method)
        return json.loads(json.dumps(self._r))

def _digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()

class Timeout(BaseException): pass
state = {"timed_out": False}
def _alarm(*_):
    state["timed_out"] = True
    signal.alarm(1)  # re-armed: an item that catches the timeout is stopped again
    raise Timeout()
signal.signal(signal.SIGALRM, _alarm)
_NOTED = ("KeyError", "AssertionError", "ValueError")

spec = json.load(open("/cases/cases.json"))
outputs = bool(spec.get("outputs"))  # record what each case returned (run_outputs), not only whether it refused
out = open("/out/results.jsonl", "w")
try:
    fn = getattr(importlib.import_module(spec["module"]), spec["function"])
    params = list(inspect.signature(fn).parameters.values())
except BaseException as exc:
    out.write(json.dumps({"fatal": type(exc).__name__}) + "\n"); sys.exit(0)
for case in spec["cases"]:
    kind, replay = case["kind"], None
    form = case.get("form") or {"tool": "env", "worktree": "path"}.get(kind, "observation")
    if form == "env": first = replay = Replay(case["action"])
    elif form == "path": first = case["path"]
    elif form == "bytes":
        with open(case["path"], "rb") as fh: first = fh.read()
    else: first = case["observation"]  # an observation, or a text the host decoded
    recorded = (case.get("kwargs") or {}) if kind == "tool" else {}
    kwargs, names = {}, set()
    for p in params[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD): continue
        names.add(p.name)
        if p.name in recorded: kwargs[p.name] = recorded[p.name]
        elif p.default is not p.empty: continue
        else: kwargs[p.name] = ""
    row = {"id": case["id"]}
    if case.get("param") is not None and case["param"] not in names:
        row["outcome"] = "unused"
        out.write(json.dumps(row) + "\n"); out.flush(); continue
    state["timed_out"] = False
    signal.alarm(spec["per_case_s"])
    try:
        try:
            result = fn(first, **kwargs); row["outcome"] = "handled"
            if outputs:
                try:
                    row["digest"] = _digest(result)
                except Exception:  # not JSON: cannot be compared
                    row["digest"] = None
        finally:
            signal.alarm(0)
    except Timeout:
        row["outcome"] = "error"
    except BaseException as exc:
        names_ = [c.__name__ for c in type(exc).__mro__]
        row["outcome"] = "refused" if "MemoryInputError" in names_ else "error"
        if row["outcome"] == "error":
            row["raised"] = next((n for n in _NOTED if n in names_), None)
            if outputs:
                row["error_class"] = (type(exc).__module__ + "." + type(exc).__qualname__)[:200]
    signal.alarm(0)
    if state["timed_out"]:
        row["outcome"], row["timeout"] = "error", True
    row["calls"] = replay._calls if replay is not None else 0
    if outputs:
        row["issued"] = replay._issued.hexdigest() if replay is not None else None
    out.write(json.dumps(row) + "\n"); out.flush()
"""


# The exception classes the box may report for a noted error. The host keeps a reported name only when it is
# one of these (an identifier of bounded length), so no text the item controls reaches a note.
_NOTED_ERRORS = ("KeyError", "AssertionError", "ValueError")
_NAMED = 5  # fields named per note; the rest are counted


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def _error_name(name: object) -> str | None:
    """*name* if it is an identifier of bounded length naming one of :data:`_NOTED_ERRORS`, else None."""
    if isinstance(name, str) and _IDENTIFIER.fullmatch(name) and name in _NOTED_ERRORS:
        return name
    return None


def _more(items: Any) -> str:
    return f" and {len(items) - _NAMED} more" if len(items) > _NAMED else ""


@dataclass
class Verdict:
    refused: list[str] = field(
        default_factory=list,
    )  # fields refused without an exempting rejection
    notes: list[str] = field(default_factory=list)
    # declared-type failures (D21), as reason text naming the type and field, never a value
    failures: list[str] = field(default_factory=list)


def _read_results(path: Path) -> list[dict]:
    """The box's result rows: never following a link, a regular file of bounded size, bad lines dropped."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return []
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > RESULTS_MAX_BYTES:
            return []
        data = b""
        while len(data) <= RESULTS_MAX_BYTES:
            chunk = os.read(fd, RESULTS_MAX_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    rows = []
    for line in data.decode("utf-8", "replace").splitlines():
        try:
            row = json.loads(line)
        except (
            ValueError,
            RecursionError,
        ):  # a nesting too deep to parse is a bad line too
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _fail(verdict: Verdict, c: Case, what: str) -> None:
    text = f"declared {c.semantic} field {(c.field or '')[:80]} {what}"
    if text not in verdict.failures:
        verdict.failures.append(text)


def _run_box(
    item: str,
    cases: list[Case],
    *,
    tree: Path,
    python: Path,
    work: Path,
    runner: Callable[..., SandboxResult],
    timeout_s: float,
    outputs: bool = False,
) -> list[dict]:
    """Run *cases* on *item* of the memory *tree* in one confined process; the result rows it wrote."""
    module, _, function = item.partition(":")
    cases_dir, out_dir = work / "cases", work / "out"
    (cases_dir / "files").mkdir(parents=True)
    out_dir.mkdir()
    spec_cases = []
    for n, c in enumerate(cases):
        if c.file is not None:
            (cases_dir / "files" / str(n)).mkdir()
            (cases_dir / "files" / str(n) / c.name).write_bytes(c.file)
        spec_cases.append({**c.payload, "id": n, "param": c.param})
    spec: dict[str, Any] = {
        "module": module.replace("/", "."),
        "function": function,
        "per_case_s": PER_CASE_S,
        "cases": spec_cases,
    }
    if outputs:
        spec["outputs"] = True
    (cases_dir / "cases.json").write_text(json.dumps(spec, default=str))
    (cases_dir / "run.py").write_text(_RUNNER)
    runner(
        [str(python), "-I", "/cases/run.py"],
        ro={tree: "/memory", cases_dir: "/cases"},
        rw={out_dir: "/out"},
        cwd="/memory",
        timeout_s=timeout_s,
    )
    return _read_results(out_dir / "results.jsonl")


def run_plan(
    item: str,
    p: Plan,
    *,
    tree: Path,
    python: Path,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
) -> Verdict:
    """Run *p*'s cases on *item* of the memory *tree* in one confined process and judge them."""
    verdict = Verdict(notes=list(p.notes), failures=list(p.unfit))
    for name, sem in p.mismatched.items():
        verdict.failures.append(
            f"declared {sem} field {name[:80]} does not match its recorded values",
        )
    if not p.cases:
        return verdict
    rows = _run_box(
        item,
        p.cases,
        tree=tree,
        python=python,
        work=work,
        runner=runner,
        timeout_s=ITEM_BUDGET_S,
    )
    fatal = next((r["fatal"] for r in rows if "fatal" in r), None)
    if fatal is not None:
        verdict.notes.append(
            f"could not be loaded for the held-out value check ({str(fatal)[:60]})",
        )
        return verdict
    by_id = {
        r["id"]: r
        for r in rows
        if isinstance(r.get("id"), int) and 0 <= r["id"] < len(p.cases)
    }
    if len(by_id) < len(p.cases):
        verdict.notes.append(
            f"ran {len(by_id)} of {len(p.cases)} held-out value cases within "
            f"{ITEM_BUDGET_S:.0f} s",
        )
    baseline: dict[tuple[str, int], str] = {}
    refused_own = timeouts = 0
    allowed: list[str] = []
    out_skipped: dict[str, str | None] = (
        {}
    )  # declared fields whose out-of-domain cases were not judged
    out_judged: set[str] = set()
    raised: dict[str, str] = {}
    unused: list[str] = []
    for n, c in enumerate(p.cases):
        r = by_id.get(n)
        if r is None:
            continue
        timeouts += bool(r.get("timeout"))
        if c.field is None:
            baseline[c.cover] = r.get("outcome")
            refused_own += r.get("outcome") == "refused"
            continue
        outcome, before_env = r.get("outcome"), r.get("calls") == 0
        judged = baseline.get(c.cover) not in (None, "refused")
        if outcome == "unused":
            if c.field not in unused:
                unused.append(c.field)
        elif c.side == "out":
            # judged only on a cover the item handled unchanged
            if baseline.get(c.cover) != "handled":
                out_skipped.setdefault(c.field, c.semantic)
                continue
            out_judged.add(c.field)
            if not (outcome == "refused" and before_env):
                _fail(verdict, c, "does not refuse out-of-domain values")
        elif c.side == "in" and outcome == "refused" and before_env and judged:
            if c.field in p.exempt.get(c.family, set()):
                if c.field not in allowed:
                    allowed.append(c.field)
            else:
                _fail(verdict, c, "refuses in-domain values")
        elif (
            outcome == "error"
            and before_env
            and baseline.get(c.cover) == "handled"
            and _error_name(r.get("raised")) is not None
        ):
            raised.setdefault(c.field, r["raised"])
        elif outcome == "refused" and before_env and judged:
            if c.field in p.exempt.get(c.family, set()):
                if c.field not in allowed:
                    allowed.append(c.field)
            elif c.field not in verdict.refused:
                verdict.refused.append(c.field)
    for f, sem in list((f, t) for f, t in out_skipped.items() if f not in out_judged)[
        :_NAMED
    ]:
        verdict.notes.append(
            f"declared {sem} field {f[:80]}: the out-of-domain check did not run (no covered "
            "input of it ran cleanly unchanged)",
        )
    for f in allowed[:_NAMED]:
        verdict.notes.append(
            f"refusal of {f[:80]} allowed by a covered recorded rejection",
        )
    if len(allowed) > _NAMED:
        verdict.notes.append(
            f"and {len(allowed) - _NAMED} more refusals allowed by a covered recorded rejection",
        )
    if raised:
        listed = ", ".join(f"{f[:80]} ({e})" for f, e in list(raised.items())[:_NAMED])
        verdict.notes.append(
            f"held-out values raised before any environment call on {listed}{_more(raised)}; "
            "only MemoryInputError is judged",
        )
    if unused:
        verdict.notes.append(
            "perturbed keywords not taken under the same name, not checked: "
            + ", ".join(f[:80] for f in unused[:_NAMED])
            + _more(unused),
        )
    if timeouts:
        verdict.notes.append(
            f"{timeouts} held-out case(s) passed the {PER_CASE_S} s limit and were stopped; "
            "counted as errors",
        )
    if refused_own:
        verdict.notes.append(
            f"refuses {refused_own} of its own covered inputs as re-serialised; "
            "those covers are not checked",
        )
    return verdict


def output_cases(
    covers: list[tuple[str, int, Action]],
    blob: Callable[[str], bytes],
    input_kind: str | None,
) -> tuple[list[Case], list[tuple[str, int]]]:
    """Each cover's recorded input, unperturbed, in the item's input form, for :func:`run_outputs`.

    The calling convention is G2's (see the module docstring): ``env`` gets a replay of the recorded call
    (a recorded rejection raises), ``observation`` the recorded response or observation as recorded,
    ``path``/``bytes`` the recorded file, ``text`` its decoded text. The second value lists the covers that
    cannot be given: a shell cover, a form the cover's kind cannot give, or a file blob that cannot be read.
    """
    cases: list[Case] = []
    unfit: list[tuple[str, int]] = []
    for eid, idx, a in covers:
        kind, form = getattr(a, "kind", "tool"), _form(a, input_kind)
        if (
            kind == "shell"
            or form not in _FORMS.get(kind, ())
            or _unfit_reason(a, input_kind) is not None
        ):
            unfit.append((eid, idx))
            continue
        c = Case((eid, idx), None, family(a), kind, {})
        if kind == "tool" and form == "env":
            c.payload = {
                "kind": kind,
                "form": form,
                "action": {
                    "channel": a.channel,
                    "method": a.method,
                    "args": list(a.args),
                    "kwargs": dict(a.kwargs or {}),
                    "response": a.response,
                    "status": a.status,
                },
                "kwargs": dict(a.kwargs or {}),
            }
        elif kind == "tool":
            c.payload = {
                "kind": kind,
                "form": form,
                "kwargs": dict(a.kwargs or {}),
                "observation": a.response,
            }
        elif kind == "worktree":
            data = _blob_of(a, blob)
            if data is None:
                unfit.append((eid, idx))
                continue
            if form == "text":
                c.payload = {
                    "kind": kind,
                    "form": form,
                    "observation": _shapes.decode(data)[1],
                }
            else:
                c.name, c.file = _file_name(a, len(cases)), data
                c.payload = {
                    "kind": kind,
                    "form": form,
                    "path": f"/cases/files/{len(cases)}/{c.name}",
                }
        else:
            c.payload = {"kind": kind, "form": form, "observation": a.response}
        cases.append(c)
    return cases, unfit


_HEX64 = re.compile(r"[0-9a-f]{64}")


def run_outputs(
    item: str,
    cases: list[Case],
    *,
    tree: Path,
    python: Path,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
) -> dict[tuple[str, int], tuple[str, str, str | None] | None]:
    """What *item* of the memory *tree* does on each case (:func:`output_cases`), run confined, by cover.

    ``(outcome, result, calls)``: ``handled`` with the SHA-256 of its return value as canonical JSON
    (sorted keys), ``refused`` (its ``MemoryInputError``) with an empty result, or ``error`` with the
    exception's class; *calls* is the SHA-256 of the environment calls it issued (``env`` form), else
    None. None when the case cannot be compared: it did not run within :data:`OUTPUTS_BUDGET_S`, timed
    out, returned a value that is not JSON, or the module could not be loaded. Only digests and class
    names leave the box, and the host only compares them.
    """
    out: dict[tuple[str, int], tuple[str, str, str | None] | None] = {
        c.cover: None for c in cases
    }
    if not cases:
        return out
    rows = _run_box(
        item,
        cases,
        tree=tree,
        python=python,
        work=work,
        runner=runner,
        timeout_s=OUTPUTS_BUDGET_S,
        outputs=True,
    )
    if any("fatal" in r for r in rows):
        return out
    by_id = {
        r["id"]: r
        for r in rows
        if isinstance(r.get("id"), int) and 0 <= r["id"] < len(cases)
    }
    for n, c in enumerate(cases):
        r = by_id.get(n)
        if r is None or r.get("timeout"):
            continue
        outcome, calls = r.get("outcome"), r.get("issued")
        if outcome == "handled":
            result = r.get("digest")
            if not (isinstance(result, str) and _HEX64.fullmatch(result)):
                continue
        elif outcome == "refused":
            result = ""
        elif outcome == "error" and isinstance(r.get("error_class"), str):
            result = r["error_class"][:200]
        else:
            continue
        if calls is not None and not (
            isinstance(calls, str) and _HEX64.fullmatch(calls)
        ):
            continue
        out[c.cover] = (outcome, result, calls)
    return out


def seen_actions(
    episode_ids: list[str],
    lookup: Callable[[str, int], Action | None],
) -> list[Action]:
    """Every recorded action of *episode_ids*, through the gate's action lookup (bounded per episode)."""
    out: list[Action] = []
    for eid in episode_ids:
        for i in range(MAX_ACTIONS_PER_EPISODE):
            a = lookup(eid, i)
            if a is None:
                break
            out.append(a)
    return out
