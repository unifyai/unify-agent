"""Trust ramp (``UNIFY_STORE_TRUST=ramp``): a stored function earns trust by working when it is reused.

``UNIFY_STORE_VERIFY`` checks a function once, before it is stored, and remembers the pass only in the
process that ran it. After that nothing watches the function: in past AppWorld runs 14 of 36 calls to a
stored function raised (a login that returned 401, an attribute of ``Primitives`` that does not exist, a
package that would not install), and the function stayed in the library looking exactly like one that works.
This switch keeps a trust record per stored function in ``function_trust`` and updates it on every reuse:

- **states**: ``probation`` (new, changed, or not yet seen enough), ``trusted`` and ``quarantined``;
- **evidence**: a call through the sandbox boundary, a proxy or ``execute_function`` that returns is a pass,
  one that raises is a failure (exception type and message, truncated). Passes are counted with the sha256
  of the call's arguments, so the record knows over how many distinct inputs the function has worked;
- **promotion**: at the thresholds legacy Unify used, by effect class -- a function that only reads (no
  environment method it can reach, directly or through the stored functions it calls, is labelled other
  than ``read``) after 3 passes over 2 distinct inputs, one that can change anything after 5 over 3;
- **demotion**: any failure quarantines the function until its source or the source of a function it calls
  changes, or it is overwritten, which puts it back on probation with its counts cleared.

Evidence is recorded only for the stored version: a call of code loaded before the stored source changed
says nothing about the current one and is ignored. With the switch unset nothing here is imported by the
call paths and nothing is written.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

from unify import db

logger = logging.getLogger(__name__)

PROBATION = "probation"
TRUSTED = "trusted"
QUARANTINED = "quarantined"
STATES = (PROBATION, TRUSTED, QUARANTINED)

READ_ONLY = "read_only"
CHANGES = "changes"
PROMOTION = {READ_ONLY: (3, 2), CHANGES: (5, 3)}
"""Passes and distinct inputs needed for ``trusted``, per effect class (legacy Unify's thresholds)."""

REASON_LIMIT = 300
"""Characters of a failure's ``Type: message`` that are kept."""

MAX_INPUT_HASHES = 32
"""Distinct input hashes kept per function; enough to tell 1, 2 and 3 apart with room to spare."""


def enabled() -> bool:
    from unify.settings import SETTINGS

    return str(getattr(SETTINGS, "UNIFY_STORE_TRUST", "") or "") == "ramp"


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trust:
    """One stored function's trust record, as it stands for the stored version."""

    function_id: int
    name: str
    state: str
    source_hash: str
    dependency_hash: str
    effect_class: str
    passes: int = 0
    failures: int = 0
    distinct_inputs: int = 0
    clean_uses: int = 0
    last_failure: Optional[str] = None
    input_hashes: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class _Stored:
    """What the store says about a function now: its hashes and effect class."""

    function_id: int
    name: str
    source_hash: str
    dependency_hash: str
    effect_class: str


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable(value: Any) -> Any:
    """A JSON stand-in for a value JSON cannot hold: its repr, or its type when the repr is an address."""
    text = repr(value)
    return f"<{type(value).__name__}>" if " at 0x" in text else text


def input_hash(arguments: Mapping[str, Any]) -> str:
    """The sha256 of a call's arguments bound to parameter names (``f(1)`` and ``f(x=1)`` hash alike)."""
    try:
        text = json.dumps(dict(arguments), sort_keys=True, default=_stable)
    except Exception:  # noqa: BLE001 - unsortable keys and the like
        text = _stable(dict(arguments))
    return sha256(text)


def failure_reason(error: Any) -> str:
    """``Type: message`` of an exception, or the last line of a traceback, truncated."""
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        lines = [line.strip() for line in str(error).strip().splitlines()]
        text = next((line for line in reversed(lines) if line), "failed")
    return text if len(text) <= REASON_LIMIT else text[: REASON_LIMIT - 3] + "..."


def _effect_class(primitive_deps: Iterable[str]) -> str:
    """``read_only`` when every environment method reachable is labelled ``read``; else ``changes``.

    A ``primitives.*`` reference whose effect is not known counts as a change.
    """
    from .function_manager import FunctionManager

    for dep in primitive_deps:
        labels = FunctionManager._candidate_effects([dep]) or ["write"]
        if any(label != "read" for label in labels):
            return CHANGES
    return READ_ONLY


def _stored(function_id: int) -> Optional[_Stored]:
    """The function's current hashes: its source, and the sources of the stored functions it calls
    (transitively); ``None`` if it is not stored."""
    row = db.query_one(
        "SELECT name, implementation, depends_on FROM functions WHERE function_id = ?",
        (int(function_id),),
    )
    if row is None:
        return None
    callees: dict[str, str] = {}
    primitive_deps: set[str] = set()
    queue = list(db.loads(row["depends_on"]) or [])
    seen = {row["name"]}
    while queue:
        dep = queue.pop(0)
        if not isinstance(dep, str) or not dep or dep in seen:
            continue
        seen.add(dep)
        if "." in dep:
            if dep.startswith("primitives."):
                primitive_deps.add(dep)
            continue
        callee = db.query_one(
            "SELECT implementation, depends_on FROM functions WHERE name = ?",
            (dep,),
        )
        if callee is None:
            callees[dep] = "missing"
            continue
        callees[dep] = sha256(callee["implementation"])
        queue.extend(db.loads(callee["depends_on"]) or [])
    return _Stored(
        function_id=int(function_id),
        name=row["name"],
        source_hash=sha256(row["implementation"]),
        dependency_hash=sha256(json.dumps(sorted(callees.items()))),
        effect_class=_effect_class(sorted(primitive_deps)),
    )


def _fresh(stored: _Stored) -> Trust:
    return Trust(
        function_id=stored.function_id,
        name=stored.name,
        state=PROBATION,
        source_hash=stored.source_hash,
        dependency_hash=stored.dependency_hash,
        effect_class=stored.effect_class,
    )


def _from_row(row: Mapping[str, Any], stored: _Stored) -> Trust:
    hashes = tuple(db.loads(row["input_hashes"]) or ())
    return Trust(
        function_id=stored.function_id,
        name=stored.name,
        state=row["state"],
        source_hash=row["source_hash"],
        dependency_hash=row["dependency_hash"],
        effect_class=stored.effect_class,
        passes=int(row["passes"]),
        failures=int(row["failures"]),
        distinct_inputs=int(row["distinct_inputs"]),
        clean_uses=int(row["clean_uses"]),
        last_failure=row["last_failure"],
        input_hashes=hashes,
    )


def _write(trust: Trust) -> None:
    db.execute(
        "INSERT OR REPLACE INTO function_trust (function_id, state, source_hash,"
        " dependency_hash, effect_class, passes, failures, input_hashes,"
        " distinct_inputs, clean_uses, last_failure, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trust.function_id,
            trust.state,
            trust.source_hash,
            trust.dependency_hash,
            trust.effect_class,
            trust.passes,
            trust.failures,
            db.dumps(list(trust.input_hashes)),
            trust.distinct_inputs,
            trust.clean_uses,
            trust.last_failure,
            db.now_iso(),
        ),
    )


def _current(stored: _Stored) -> tuple[Trust, bool]:
    """The record for the stored version and whether it had to be restarted (a changed source or callee)."""
    row = db.query_one(
        "SELECT * FROM function_trust WHERE function_id = ?",
        (stored.function_id,),
    )
    if row is None:
        return _fresh(stored), False
    if (
        row["source_hash"] != stored.source_hash
        or row["dependency_hash"] != stored.dependency_hash
    ):
        return _fresh(stored), True
    return _from_row(row, stored), False


def trust(function_id: int) -> Optional[Trust]:
    """The function's trust record for its stored version (``None`` if it is not stored).

    A record kept for another source or other callees is restarted on probation here, and the restart is
    written, so a quarantined function becomes visible again as soon as it or a function it calls changes.
    """
    with db.transaction():
        stored = _stored(function_id)
        if stored is None:
            return None
        current, restarted = _current(stored)
        if restarted:
            _write(current)
        return current


def record(
    function_id: int,
    *,
    running_source: Optional[str],
    arguments: Mapping[str, Any],
    error: Any = None,
) -> Optional[Trust]:
    """Record one reuse: a pass when ``error`` is ``None``, else a failure that quarantines the function.

    ``running_source`` is the source of the code that ran; when it is not the stored source (the function
    changed after it was loaded) nothing is recorded and ``None`` is returned.
    """
    with db.transaction():
        stored = _stored(function_id)
        if stored is None:
            return None
        if running_source is not None and sha256(running_source) != stored.source_hash:
            return None
        current, _ = _current(stored)
        if error is None:
            digest = input_hash(arguments)
            new_input = digest not in current.input_hashes
            hashes = current.input_hashes
            if new_input:
                hashes = (hashes + (digest,))[-MAX_INPUT_HASHES:]
            passes = current.passes + 1
            distinct = current.distinct_inputs + int(new_input)
            state = current.state
            need_passes, need_inputs = PROMOTION[stored.effect_class]
            if state == PROBATION and passes >= need_passes and distinct >= need_inputs:
                state = TRUSTED
            updated = replace(
                current,
                state=state,
                passes=passes,
                distinct_inputs=distinct,
                clean_uses=current.clean_uses + 1,
                input_hashes=hashes,
            )
        else:
            updated = replace(
                current,
                state=QUARANTINED,
                failures=current.failures + 1,
                clean_uses=0,
                last_failure=failure_reason(error),
            )
            if current.state != QUARANTINED:
                logger.warning(
                    "Quarantined the stored function %r: %s",
                    stored.name,
                    updated.last_failure,
                )
        _write(updated)
        return updated


def reset(function_ids: Iterable[int]) -> None:
    """Put functions back on probation with their counts cleared (an overwrite or a patch)."""
    ids = [(int(fid),) for fid in function_ids]
    if ids:
        db.executemany("DELETE FROM function_trust WHERE function_id = ?", ids)


# ---------------------------------------------------------------------------
# Observing calls
# ---------------------------------------------------------------------------


def bind_arguments(fn: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict:
    """The call's arguments by parameter name; positional ones as ``_0``, ``_1``... when they cannot be bound."""
    import inspect

    if fn is not None:
        try:
            return dict(inspect.signature(fn).bind_partial(*args, **kwargs).arguments)
        except (TypeError, ValueError):
            pass
    bound = {f"_{i}": value for i, value in enumerate(args)}
    bound.update(kwargs)
    return bound


class CallObserver:
    """Records each call of one loaded stored function (never raises into the call)."""

    def __init__(self, function_manager: Any, func_data: Mapping[str, Any]):
        self._fm = function_manager
        self._function_id = int(func_data["function_id"])
        self._name = str(func_data.get("name"))
        self._source = func_data.get("implementation")
        self._depends_on = list(func_data.get("depends_on") or [])

    @classmethod
    def for_function(
        cls,
        function_manager: Any,
        func_data: Mapping[str, Any],
    ) -> Optional["CallObserver"]:
        """An observer for a stored function while the switch is on; ``None`` otherwise."""
        if not enabled() or func_data.get("is_primitive"):
            return None
        if func_data.get("function_id") is None or not func_data.get("implementation"):
            return None
        return cls(function_manager, func_data)

    def before(self, fn: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict:
        """The call's bound arguments."""
        try:
            return bind_arguments(fn, args, kwargs)
        except Exception:  # noqa: BLE001 - observing must never break a call
            return dict(kwargs)

    def after(self, arguments: Mapping[str, Any], error: Any = None) -> None:
        """Record the call: returned (``error`` is ``None``) or raised. A call stopped by steering is not
        evidence either way."""
        from .steering import ExecutionStopped

        if isinstance(error, ExecutionStopped):
            return
        try:
            record(
                self._function_id,
                running_source=self._source,
                arguments=arguments,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001 - observing must never break a call
            logger.warning("Trust evidence for %r was lost: %s", self._name, exc)


__all__ = [
    "CHANGES",
    "CallObserver",
    "PROBATION",
    "PROMOTION",
    "QUARANTINED",
    "READ_ONLY",
    "STATES",
    "TRUSTED",
    "Trust",
    "bind_arguments",
    "enabled",
    "failure_reason",
    "input_hash",
    "record",
    "reset",
    "trust",
]
