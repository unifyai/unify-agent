"""Function cases (``UNIFY_FUNCTION_CASES``): recorded calls of a stored function, replayed before it changes.

A stored function is overwritten or patched with nothing to say whether it still does what it did. In past
runs a function that removed the tracks released *before* a year was patched to remove those released *at or
after* it, under the same name, and every caller written against the old behaviour silently changed with it.
The calls cannot be re-run against the real environment to find out (they may change it, and a function must
not run twice), so this switch records them and replays the record:

- **record**: every call of a stored function through the sandbox boundary, a proxy,
  ``FunctionManager.execute_function`` or the actor's ``execute_function`` tool is a *case*: the function's
  source hash, its arguments (bounded, JSON-safe), what it returned (a digest and a bounded repr) or raised
  (``Type: message``), and the trace of the environment calls it made (``primitives.<namespace>.<method>``
  of a registered environment, recorded at the one wrapper every such call goes through) with what each
  returned or raised. A stored function called inside another is a case of its own, and its environment calls
  are in the caller's trace too, so either can be replayed alone. A transcript session id is kept as a
  pointer when one is current; nothing of the transcript is copied. At most :data:`MAX_PASSING` cases that
  returned and :data:`MAX_FAILING` that raised are kept per function, the most recent, one per distinct
  input (a later call on the same input replaces the earlier case in place). A call that a steering correction
  reached, a call the caller got wrong (the arguments do not fit), and a call of code loaded before the stored
  source changed are not recorded;
- **replay**: when ``add_functions(overwrite=True)`` or ``patch_function`` would store a different source,
  each case that returned is run against the new source in a scratch namespace whose ``primitives`` serve the
  recorded trace: each environment call must be the next one recorded (same method, same arguments once bound
  to the method's signature) and gets the recorded answer; any other call, or one too many or too few, is a
  divergence at that call. Then the return value must have the recorded digest. Nothing reaches the
  environment, the network or a model: ``query_llm`` and the namespaces nobody recorded raise inside the
  replay, which makes the case *inconclusive*, as does a function that reads the clock or randomness, imports
  a module that is not plainly deterministic, or needs packages, and a replay that takes longer than
  :data:`REPLAY_TIMEOUT_S`;
- **policy**: a case that diverges, returns something else or now raises refuses the change, naming the case,
  how it differs, and the two ways on: store the new behaviour under a new name (and say in the old entry's
  docstring that it is superseded, or delete it), or retire the case with a reason
  (``FunctionManager_retire_case``). Inconclusive cases do not block and are reported with the result; cases
  that raised are re-run only when they made no environment call, and whether they now return is reported;
- **visibility**: search and filter results carry a compact ``cases`` field (up to two cases, about
  :data:`SUMMARY_LIMIT` characters per function), so the storage review sees what an entry is known to do;
- **redaction**: an environment may answer with a secret (a login's access token, a stored password). Before
  anything is kept, each string held under a credential-named key or parameter
  (:func:`~unify.function_manager.store_trust.credential_key`) in the function's arguments, an environment
  call's arguments or answer, or the result, is replaced by a placeholder ``<redacted:...>``, a keyed
  digest under a salt drawn per case; the same secret met elsewhere in the case (passed on as an argument
  of another name, returned, inside a longer string) is replaced by the same placeholder. Digests are taken
  over the redacted data, and a replay redacts what the new source does in the same way with the case's
  salt, so a function that only passes a secret on replays unchanged; one that computes on the secret
  itself (its length, a slice) sees the placeholder and may diverge. A value met before anything marks it
  as a credential stays as recorded. Cases recorded before redaction (no salt) replay unredacted.

With the switch off nothing is recorded, replayed or shown, no row is written and every call path is the
shipped one.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import contextvars
import copy
import functools
import hashlib
import hmac
import inspect
import json
import logging
import math
import re
import secrets
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional

from unify import db

logger = logging.getLogger(__name__)

MAX_PASSING = 3
"""Cases that returned kept per function (the most recent, one per input)."""

MAX_FAILING = 3
"""Cases that raised kept per function (the most recent, one per input)."""

MAX_RETIRED = 10
"""Retired cases kept per function, with their reasons; older ones are deleted."""

MAX_TRACE_CALLS = 200
"""Environment calls recorded per case; a longer trace is kept cut short and replays as inconclusive past it."""

MAX_VALUE_CHARS = 32_000
"""Characters of JSON one recorded value may take; a longer environment answer is kept as a digest only."""

MAX_TRACE_CHARS = 200_000
"""Characters of JSON one case's trace may take."""

REPR_LIMIT = 200
"""Characters of a value's repr shown in a case."""

ERROR_LIMIT = 300
"""Characters of a ``Type: message`` kept."""

REPLAY_TIMEOUT_S = 2.0
"""Seconds one replay may take before it is abandoned as inconclusive."""

SUMMARY_LIMIT = 400
"""Characters of the ``cases`` field a search or filter result carries per function."""

MAX_REPORTED = 3
"""Cases a refusal or a report names."""

PASS = "pass"
FAIL = "fail"
ACTIVE = "active"
RETIRED = "retired"

PURE_MODULES = frozenset(
    {
        "abc",
        "asyncio",
        "base64",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "html",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "pprint",
        "re",
        "statistics",
        "string",
        "textwrap",
        "types",
        "typing",
        "unicodedata",
    },
)
"""Modules a replayed function may import: deterministic, and touching nothing outside the process."""

NONDETERMINISTIC_NAMES = frozenset({"time", "random", "uuid", "secrets"})
NONDETERMINISTIC_ATTRIBUTES = frozenset({"now", "today", "utcnow", "urandom"})
UNSTUBBED_NAMES = frozenset(
    {
        "open",
        "os",
        "sys",
        "subprocess",
        "socket",
        "requests",
        "httpx",
        "urllib",
        "aiohttp",
        "shutil",
        "pathlib",
        "Path",
        "query_llm",
        "list_llms",
        "unillm",
        "input",
        "__import__",
        "eval",
        "exec",
        "compile",
        "globals",
        "importlib",
        "threading",
        "multiprocessing",
    },
)
"""Names whose use makes a replay inconclusive: replay serves only the recorded environment calls."""

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")

REDACTED = re.compile(r"<redacted:[0-9a-f]{16}>")
"""A placeholder that stands for a credential value in a recorded case."""

MIN_SECRET_CHARS = 4
"""A credential value at least this long is also replaced where it recurs as a whole string."""

MIN_INNER_SECRET_CHARS = 8
"""A credential value at least this long is also replaced inside longer strings."""

# The environment-call traces of the stored-function calls running in this context, outermost first.
_ACTIVE: contextvars.ContextVar[tuple["_Trace", ...]] = contextvars.ContextVar(
    "unify_function_case_traces",
    default=(),
)
# True inside a replay: stored functions loaded there run bare (no usage, trust or case is recorded).
_REPLAYING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unify_function_case_replaying",
    default=False,
)
# True while a function runs somewhere other than the session's world (a verifier's check): nothing recorded.
_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unify_function_case_quiet",
    default=False,
)


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_FUNCTION_CASES", False))


def replaying() -> bool:
    """Whether the current call runs inside a replay."""
    return _REPLAYING.get()


def recording() -> bool:
    """Whether an environment call made now belongs to a recorded case (cheap; false while the switch is off)."""
    return bool(_ACTIVE.get())


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Record nothing inside the block (a run against a verifier's world, not the session's)."""
    token = _QUIET.set(True)
    traces = _ACTIVE.set(())
    try:
        yield
    finally:
        _ACTIVE.reset(traces)
        _QUIET.reset(token)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _scrub(text: str) -> str:
    return _ADDRESS.sub("", text)


def _short(value: Any, limit: int = REPR_LIMIT) -> str:
    try:
        text = _scrub(repr(value))
    except Exception:  # noqa: BLE001 - a repr that raises
        text = f"<{type(value).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def plain(value: Any, _depth: int = 0) -> tuple[Any, bool]:
    """``value`` as JSON data, and whether that is exact (it was JSON data already).

    Lists, dicts with string keys, strings, numbers, booleans and ``None`` are exact. Tuples become lists,
    other keys strings, pydantic models their dump and anything else its ``str`` (as
    ``FunctionManager.execute_function`` reports a result), none of them exactly.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value, True
    if isinstance(value, float):
        return (value, True) if math.isfinite(value) else (repr(value), False)
    if _depth > 20:
        return _scrub(str(value)), False
    if isinstance(value, (list, tuple)):
        items = [plain(item, _depth + 1) for item in value]
        return [item for item, _ in items], isinstance(value, list) and all(
            exact for _, exact in items
        )
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        exact = True
        for key, item in value.items():
            data, item_exact = plain(item, _depth + 1)
            exact = exact and item_exact and isinstance(key, str)
            out[str(key)] = data
        return out, exact
    if isinstance(value, (set, frozenset)):
        items = sorted((plain(item, _depth + 1)[0] for item in value), key=_canonical)
        return items, False
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return plain(model_dump(), _depth + 1)[0], False
        except Exception:  # noqa: BLE001 - fall back to the text
            pass
    try:
        return _scrub(str(value)), False
    except Exception:  # noqa: BLE001 - a __str__ that raises
        return f"<{type(value).__name__}>", False


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _error_text(error: Any) -> str:
    from .store_trust import failure_reason

    text = failure_reason(error)
    return text if len(text) <= ERROR_LIMIT else text[: ERROR_LIMIT - 3] + "..."


def _bound(call: Optional[Callable[..., Any]], args: Any, kwargs: Mapping[str, Any]):
    """The call's arguments by parameter name (defaults filled in, ``**kwargs`` flattened)."""
    if call is not None:
        try:
            signature = inspect.signature(call)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            out: Dict[str, Any] = {}
            for name, value in bound.arguments.items():
                kind = signature.parameters[name].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    out.update(value)
                elif kind is inspect.Parameter.VAR_POSITIONAL:
                    out[name] = list(value)
                else:
                    out[name] = value
            return out
        except (TypeError, ValueError):
            pass
    out = {f"_{i}": value for i, value in enumerate(args)}
    out.update(kwargs)
    return out


def _shown_arguments(arguments: Mapping[str, Any], limit: int = REPR_LIMIT) -> str:
    """``k=v, ...`` for display; a value whose parameter names a credential is not shown."""
    from .store_trust import _CREDENTIAL_NAME

    text = ", ".join(
        f"{key}={'<withheld>' if _CREDENTIAL_NAME.search(str(key)) else _short(value, limit)}"
        for key, value in arguments.items()
    )
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _Redactor:
    """Replaces credential values in one case's recorded data with placeholders stable under its salt.

    A string under a credential-named key or parameter becomes ``<redacted:digest>``, the digest an HMAC of
    the value under :attr:`salt`; the values seen so far are remembered, so where one recurs (a whole string
    of at least :data:`MIN_SECRET_CHARS`, or inside a string when it is at least
    :data:`MIN_INNER_SECRET_CHARS`) it is replaced by the same placeholder. A placeholder is left as it is,
    so redacting what a replay passes on gives what the recording kept. Without a salt (a case recorded
    before redaction) nothing is replaced.
    """

    def __init__(self, salt: Optional[str]) -> None:
        self.salt = salt or None
        self._secrets: Dict[str, str] = {}

    @classmethod
    def fresh(cls) -> "_Redactor":
        return cls(secrets.token_hex(16))

    def placeholder(self, value: str) -> str:
        if self.salt is None or REDACTED.fullmatch(value):
            return value
        known = self._secrets.get(value)
        if known is not None:
            return known
        digest = hmac.new(
            self.salt.encode("utf-8"),
            value.encode("utf-8", "replace"),
            hashlib.sha256,
        ).hexdigest()[:16]
        out = f"<redacted:{digest}>"
        if len(value) >= MIN_SECRET_CHARS:
            self._secrets[value] = out
        return out

    def _learn(self, data: Any, hidden: bool = False) -> None:
        if isinstance(data, dict):
            for key, item in data.items():
                self._learn(item, hidden or _credential_key(key))
        elif isinstance(data, list):
            for item in data:
                self._learn(item, hidden)
        elif hidden and isinstance(data, str) and data:
            self.placeholder(data)

    def _apply(self, data: Any, hidden: bool = False) -> Any:
        if isinstance(data, dict):
            return {
                key: self._apply(item, hidden or _credential_key(key))
                for key, item in data.items()
            }
        if isinstance(data, list):
            return [self._apply(item, hidden) for item in data]
        if isinstance(data, str) and data:
            return self.placeholder(data) if hidden else self.scrub(data)
        return data

    def redact(self, data: Any) -> Any:
        """JSON ``data`` with its credential values replaced (all of them learned first)."""
        if self.salt is None:
            return data
        self._learn(data)
        return self._apply(data)

    def scrub(self, text: str) -> str:
        """``text`` with every credential value seen so far replaced by its placeholder."""
        if self.salt is None or not self._secrets or not isinstance(text, str):
            return text
        whole = self._secrets.get(text)
        if whole is not None:
            return whole
        inner = sorted(
            (
                s
                for s in self._secrets
                if len(s) >= MIN_INNER_SECRET_CHARS and s in text
            ),
            key=len,
            reverse=True,
        )
        if not inner:
            return text
        pieces = re.split(f"({REDACTED.pattern})", text)
        for index in range(0, len(pieces), 2):  # the odd pieces are placeholders
            for secret in inner:
                pieces[index] = pieces[index].replace(secret, self._secrets[secret])
        return "".join(pieces)

    def shown(self, arguments: Mapping[str, Any]) -> str:
        """:func:`_shown_arguments` with the credential values seen so far replaced."""
        return _shown_arguments(
            {
                key: self.scrub(value) if isinstance(value, str) else value
                for key, value in arguments.items()
            },
        )

    def short(self, value: Any, limit: int = REPR_LIMIT) -> str:
        """:func:`_short` of ``value`` with the credential values seen so far replaced."""
        return _short(self.scrub(_short(value, 10**9)), limit)


@functools.lru_cache(maxsize=4096)
def _credential_name(name: str) -> bool:
    from .store_trust import credential_key

    return credential_key(name)


def _credential_key(name: Any) -> bool:
    return _credential_name(str(name))


@functools.lru_cache(maxsize=256)
def _names_environment_globals(source: str, names: frozenset) -> bool:
    """Whether ``source`` reads one of the environment's extra globals, whose calls are not recorded."""
    if not names:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(tree))


# ---------------------------------------------------------------------------
# Recording environment calls
# ---------------------------------------------------------------------------


class _Trace:
    """The environment calls one stored-function call made, in the order they started.

    ``complete`` is false when some call may be missing: the trace was cut short at its bounds, or calls were
    made where the recorder cannot see them.
    """

    __slots__ = ("calls", "chars", "cut", "partial", "closed", "redactor")

    def __init__(self, redactor: Optional[_Redactor] = None) -> None:
        self.redactor = redactor if redactor is not None else _Redactor(None)
        self.calls: List[dict] = []
        self.chars = 0
        self.cut = False
        self.partial = False
        self.closed = False

    @property
    def complete(self) -> bool:
        return not (self.cut or self.partial)

    def open(self, entry: dict) -> bool:
        """Reserve the next place for a call that is starting; whether it was taken."""
        if self.closed or self.cut:
            return False
        if len(self.calls) >= MAX_TRACE_CALLS:
            self.cut = True
            return False
        self.calls.append(entry)
        return True

    def grow(self, chars: int) -> bool:
        """Account for a finished call's size; ``False`` (and cut short) when it no longer fits."""
        if self.chars + chars > MAX_TRACE_CHARS:
            self.cut = True
            return False
        self.chars += chars
        return True


def _call_key(
    namespace: str,
    method: str,
    call: Optional[Callable[..., Any]],
    args: Any,
    kwargs: Mapping[str, Any],
    redactor: _Redactor,
) -> tuple[str, str]:
    """The digest of an environment call's bound arguments, redacted, and how they read."""
    arguments = _bound(call, args, kwargs)
    data = redactor.redact(plain(arguments)[0])
    return _sha256(_canonical(data)), redactor.shown(arguments)


class _PrimitiveCall:
    """One environment call: its place in every active trace is taken when it starts (so calls run
    concurrently keep the order they were made in), and filled in when it ends."""

    def __init__(
        self,
        namespace: str,
        method: str,
        call: Callable[..., Any],
        args: Any,
        kwargs: Mapping[str, Any],
        *,
        is_async: bool,
    ) -> None:
        # Traces that share a redactor (a stored function called inside
        # another shares its caller's) share one entry.
        groups: Dict[int, tuple[_Redactor, List[_Trace]]] = {}
        for trace in _ACTIVE.get():
            groups.setdefault(id(trace.redactor), (trace.redactor, []))[1].append(trace)
        self.entries: List[tuple[_Redactor, dict, tuple]] = []
        for redactor, traces in groups.values():
            digest, shown = _call_key(namespace, method, call, args, kwargs, redactor)
            entry: dict = {
                "call": f"{namespace}.{method}",
                "args": digest,
                "shown": shown,
                "async": is_async,
                "replayable": False,
            }
            taken = tuple(t for t in traces if t.open(entry))
            self.entries.append((redactor, entry, taken))

    def finish(self, *, result: Any = None, error: Optional[BaseException] = None):
        for redactor, entry, traces in self.entries:
            self._finish(redactor, entry, traces, result=result, error=error)

    @staticmethod
    def _finish(
        redactor: _Redactor,
        entry: dict,
        traces: tuple,
        *,
        result: Any,
        error: Optional[BaseException],
    ) -> None:
        try:
            if error is not None:
                if not isinstance(error, Exception):
                    for trace in traces:
                        trace.partial = True
                    return
                entry["error"] = {
                    "type": type(error).__name__,
                    "message": redactor.scrub(str(error))[:ERROR_LIMIT],
                }
                entry["replayable"] = True
                chars = len(_canonical(entry))
            else:
                data, exact = plain(result)
                data = redactor.redact(data)
                text = _canonical(data)
                chars = len(text)
                if exact and chars <= MAX_VALUE_CHARS:
                    entry["result"] = data
                    entry["replayable"] = True
                else:
                    entry["result_shown"] = redactor.short(result)
                    chars = len(entry["result_shown"])
            for trace in traces:
                if not trace.grow(chars + len(entry["shown"]) + 100):
                    entry.pop("result", None)
                    entry["replayable"] = False
        except Exception as exc:  # noqa: BLE001 - recording must never break a call
            logger.debug("An environment call was not recorded: %s", exc)
            entry["replayable"] = False
            for trace in traces:
                trace.partial = True


def observe_primitive(
    namespace: str,
    method: str,
    call: Callable[..., Any],
    args: Any,
    kwargs: Mapping[str, Any],
) -> Any:
    """Run a synchronous environment call and add it to every case recording in this context."""
    try:
        pending = _PrimitiveCall(namespace, method, call, args, kwargs, is_async=False)
    except Exception:  # noqa: BLE001 - recording must never break a call
        _mark_incomplete()
        return call(*args, **kwargs)
    try:
        value = call(*args, **kwargs)
    except BaseException as exc:
        pending.finish(error=exc)
        raise
    pending.finish(result=value)
    return value


async def observe_primitive_async(
    namespace: str,
    method: str,
    call: Callable[..., Any],
    args: Any,
    kwargs: Mapping[str, Any],
) -> Any:
    """Await an asynchronous environment call and add it to every case recording in this context."""
    try:
        pending = _PrimitiveCall(namespace, method, call, args, kwargs, is_async=True)
    except Exception:  # noqa: BLE001 - recording must never break a call
        _mark_incomplete()
        return await call(*args, **kwargs)
    try:
        value = await call(*args, **kwargs)
    except BaseException as exc:
        pending.finish(error=exc)
        raise
    pending.finish(result=value)
    return value


def _mark_incomplete() -> None:
    for trace in _ACTIVE.get():
        trace.partial = True


# ---------------------------------------------------------------------------
# Recording stored-function calls
# ---------------------------------------------------------------------------


class Pending:
    """One stored-function call being recorded."""

    def __init__(self, recorder: "CaseRecorder", args: Any, kwargs: Mapping[str, Any]):
        from .store_trust import caller_fault, input_hash, source_signature

        signature = source_signature(recorder.source, recorder.name)
        self.caller_fault = caller_fault(signature, recorder.name, args, kwargs)
        if signature is not None:
            try:
                named = dict(signature.bind_partial(*args, **kwargs).arguments)
            except TypeError:
                named = _bound(None, args, kwargs)
        else:
            named = _bound(None, args, kwargs)
        self.args_hash = input_hash(named)
        # A stored function called inside another shares its caller's redactor,
        # so a secret the caller has seen is replaced in this case too.
        self.redactor = (
            next(
                (t.redactor for t in reversed(_ACTIVE.get()) if t.redactor.salt),
                None,
            )
            or _Redactor.fresh()
        )
        self.redactor.redact(plain(named)[0])  # learn the credential arguments
        self.args_shown = self.redactor.shown(named)
        raw, exact = plain({"args": list(args), "kwargs": dict(kwargs)})
        call_data = {
            "args": [
                self.redactor.redact(item)
                for item in self._redacted_positional(signature, raw["args"])
            ],
            "kwargs": self.redactor.redact(raw["kwargs"]),
        }
        text = _canonical(call_data)
        self.call = (
            {**call_data, "salt": self.redactor.salt}
            if exact and len(text) <= MAX_VALUE_CHARS
            else None
        )
        self.trace = _Trace(self.redactor)
        from .primitives.environment import environment_globals

        if _names_environment_globals(
            recorder.source,
            frozenset(environment_globals()),
        ):
            # Calls made through such a global bypass the recorder, so the
            # trace may miss some: a replay that runs past it is inconclusive.
            self.trace.partial = True

    def _redacted_positional(self, signature: Any, args: List[Any]) -> List[Any]:
        """Positional ``args`` with the ones bound to a credential-named parameter replaced."""
        if signature is None:
            return args
        names = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return [
            (
                self.redactor.placeholder(value)
                if index < len(names)
                and isinstance(value, str)
                and value
                and _credential_key(names[index])
                else value
            )
            for index, value in enumerate(args)
        ]


class CaseRecorder:
    """Records the calls of one loaded stored function as cases (never raises into the call)."""

    def __init__(self, func_data: Mapping[str, Any]):
        self.function_id = int(func_data["function_id"])
        self.name = str(func_data.get("name"))
        self.source = str(func_data.get("implementation") or "")

    @classmethod
    def for_function(cls, func_data: Mapping[str, Any]) -> Optional["CaseRecorder"]:
        """A recorder for a stored function while the switch is on; ``None`` otherwise."""
        if not enabled() or not isinstance(func_data, Mapping):
            return None
        if func_data.get("is_primitive") or func_data.get("function_id") is None:
            return None
        if not func_data.get("implementation"):
            return None
        return cls(func_data)

    def begin(self, args: Any, kwargs: Mapping[str, Any]) -> Optional[Pending]:
        """Start recording a call (its arguments are read now, before the function can change them)."""
        if _REPLAYING.get() or _QUIET.get():
            return None
        try:
            return Pending(self, tuple(args), dict(kwargs or {}))
        except Exception as exc:  # noqa: BLE001 - recording must never break a call
            logger.debug("A call of %r was not recorded: %s", self.name, exc)
            return None

    def end(
        self,
        pending: Optional[Pending],
        *,
        result: Any = None,
        error: Any = None,
    ) -> None:
        """Store the call as a case: returned (``error`` is ``None``) or raised."""
        if pending is None:
            return
        pending.trace.closed = True
        try:
            from .steering import ExecutionStopped, active_session

            if isinstance(error, ExecutionStopped):
                return
            steering = active_session()
            if steering is not None and getattr(steering, "messages", None):
                return
            if error is not None and pending.caller_fault is not None:
                return
            _store(self, pending, result=result, error=error)
        except Exception as exc:  # noqa: BLE001 - recording must never break a call
            logger.warning("A case of %r was not recorded: %s", self.name, exc)


@contextlib.contextmanager
def tracing(pending: Optional[Pending]) -> Iterator[None]:
    """Add the environment calls made inside the block to ``pending``'s trace."""
    if pending is None:
        yield
        return
    token = _ACTIVE.set(_ACTIVE.get() + (pending.trace,))
    try:
        yield
    finally:
        try:
            _ACTIVE.reset(token)
        except ValueError:  # reset from another context: drop this trace there
            _ACTIVE.set(tuple(t for t in _ACTIVE.get() if t is not pending.trace))


def _session_pointer() -> Optional[str]:
    try:
        from unify import transcripts

        current = transcripts._CURRENT.get()
    except Exception:  # noqa: BLE001 - no transcript, no pointer
        return None
    return str(getattr(current, "id", "") or "") or None


def _store(
    recorder: CaseRecorder,
    pending: Pending,
    *,
    result: Any,
    error: Any,
) -> None:
    from .store_trust import sha256

    kind = FAIL if error else PASS
    redactor = pending.redactor
    if kind == PASS:
        data, exact = plain(result)
        data = redactor.redact(data)
        result_json: Optional[str] = db.dumps(
            {
                "digest": _sha256(_canonical(data)),
                "exact": exact,
                "shown": _short(data) if exact else redactor.short(result),
            },
        )
        error_text = None
    else:
        result_json = None
        error_text = redactor.scrub(_error_text(error))
    trace = pending.trace
    values = (
        sha256(recorder.source),
        db.dumps(pending.call) if pending.call is not None else None,
        pending.args_shown,
        result_json,
        error_text,
        db.dumps(trace.calls),
        int(trace.complete),
        _session_pointer(),
        db.now_iso(),
    )
    with db.transaction():
        stored = db.query_one(
            "SELECT implementation FROM functions WHERE function_id = ?",
            (recorder.function_id,),
        )
        if stored is None or stored["implementation"] != recorder.source:
            return  # deleted, or the code that ran is not the stored version
        existing = db.query_one(
            "SELECT case_id FROM function_cases WHERE function_id = ? AND kind = ?"
            " AND status = ? AND args_hash = ?",
            (recorder.function_id, kind, ACTIVE, pending.args_hash),
        )
        if existing is not None:
            db.execute(
                "UPDATE function_cases SET source_hash = ?, call = ?, args_shown = ?,"
                " result = ?, error = ?, trace = ?, trace_complete = ?, session = ?,"
                " outcome = NULL, recorded_at = ? WHERE case_id = ?",
                (*values, int(existing["case_id"])),
            )
        else:
            db.execute(
                "INSERT INTO function_cases (function_id, kind, status, args_hash,"
                " source_hash, call, args_shown, result, error, trace, trace_complete,"
                " session, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (recorder.function_id, kind, ACTIVE, pending.args_hash, *values),
            )
        _prune(
            recorder.function_id,
            kind,
            ACTIVE,
            MAX_PASSING if kind == PASS else MAX_FAILING,
        )


def _prune(function_id: int, kind: Optional[str], status: str, keep: int) -> None:
    where = "function_id = ? AND status = ?"
    params: list = [int(function_id), status]
    if kind is not None:
        where += " AND kind = ?"
        params.append(kind)
    db.execute(
        "DELETE FROM function_cases WHERE case_id IN (SELECT case_id FROM"
        f" function_cases WHERE {where} ORDER BY recorded_at DESC, case_id DESC"
        " LIMIT -1 OFFSET ?)",
        (*params, int(keep)),
    )


# ---------------------------------------------------------------------------
# Reading cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    case_id: int
    function_id: int
    kind: str
    status: str
    source_hash: str
    call: Optional[dict]
    args_shown: str
    result: Optional[dict]
    error: Optional[str]
    trace: tuple
    trace_complete: bool
    session: Optional[str]
    outcome: Optional[Any]
    retired_why: Optional[str]
    recorded_at: str
    salt: Optional[str] = None
    """The salt the case's credential values were redacted under; ``None`` for a case recorded before."""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Case":
        call = db.loads(row["call"])
        salt = call.pop("salt", None) if isinstance(call, dict) else None
        return cls(
            case_id=int(row["case_id"]),
            function_id=int(row["function_id"]),
            kind=row["kind"],
            status=row["status"],
            source_hash=row["source_hash"],
            call=call,
            salt=salt,
            args_shown=row["args_shown"] or "",
            result=db.loads(row["result"]),
            error=row["error"],
            trace=tuple(db.loads(row["trace"]) or ()),
            trace_complete=bool(row["trace_complete"]),
            session=row["session"],
            outcome=db.loads(row["outcome"]),
            retired_why=row["retired_why"],
            recorded_at=row["recorded_at"],
        )

    def compact(self, name: str) -> str:
        """One line: the call and what it did."""
        head = f"#{self.case_id} {name}({self.args_shown})"
        if self.kind == PASS:
            tail = f"-> {(self.result or {}).get('shown', '?')}"
        else:
            tail = f"raised {self.error}"
        calls = len(self.trace)
        env = f" [{calls} env call{'s' if calls != 1 else ''}]" if calls else ""
        return f"{head} {tail}{env}"


def cases(function_id: int, *, status: Optional[str] = ACTIVE) -> List[Case]:
    """The function's cases, oldest first (only ``status`` ones unless it is ``None``)."""
    sql = "SELECT * FROM function_cases WHERE function_id = ?"
    params: list = [int(function_id)]
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    return [Case.from_row(row) for row in db.query(sql + " ORDER BY case_id", params)]


def summaries(rows: Iterable[Mapping[str, Any]]) -> Dict[int, str]:
    """A compact ``cases`` text per stored function among ``rows`` that has active cases.

    Up to two cases each, the latest that returned and the latest that raised, cut to
    :data:`SUMMARY_LIMIT` characters.
    """
    names = {
        int(row["function_id"]): str(row.get("name"))
        for row in rows
        if isinstance(row, Mapping)
        and not row.get("is_primitive")
        and row.get("function_id") is not None
    }
    if not names:
        return {}
    found = db.query(
        "SELECT * FROM function_cases WHERE status = ? AND function_id IN"
        f" ({', '.join('?' for _ in names)}) ORDER BY recorded_at DESC, case_id DESC",
        [ACTIVE, *names],
    )
    picked: Dict[int, Dict[str, Case]] = {}
    for row in found:
        case = Case.from_row(row)
        picked.setdefault(case.function_id, {}).setdefault(case.kind, case)
    out: Dict[int, str] = {}
    for function_id, by_kind in picked.items():
        lines = [
            by_kind[kind].compact(names[function_id])
            for kind in (PASS, FAIL)
            if kind in by_kind
        ]
        text = "; ".join(lines)
        out[function_id] = (
            text if len(text) <= SUMMARY_LIMIT else text[: SUMMARY_LIMIT - 3] + "..."
        )
    return out


def with_summaries(rows: List[Any]) -> List[Any]:
    """``rows`` with a ``cases`` field on each function row that has cases (in place); as given when off."""
    if not enabled():
        return rows
    try:
        found = summaries(row for row in rows if isinstance(row, dict))
    except Exception as exc:  # noqa: BLE001 - a read aid must never break a search
        logger.warning("Function cases were left out of a search: %s", exc)
        return rows
    for row in rows:
        if isinstance(row, dict) and row.get("function_id") is not None:
            text = (
                found.get(int(row["function_id"]))
                if not row.get("is_primitive")
                else None
            )
            if text:
                row["cases"] = text
    return rows


def retire(function_id: int, case_id: int, why: str) -> Optional[Case]:
    """Retire one active case of the function, keeping ``why``; ``None`` if it has no such case."""
    with db.transaction():
        row = db.query_one(
            "SELECT * FROM function_cases WHERE case_id = ? AND function_id = ? AND status = ?",
            (int(case_id), int(function_id), ACTIVE),
        )
        if row is None:
            return None
        db.execute(
            "UPDATE function_cases SET status = ?, retired_why = ?, retired_at = ? WHERE case_id = ?",
            (RETIRED, str(why).strip()[:ERROR_LIMIT], db.now_iso(), int(case_id)),
        )
        _prune(function_id, None, RETIRED, MAX_RETIRED)
    return Case.from_row(row)


def mark_outcome(session: str, outcome: Any) -> int:
    """Attach a checked outcome to every case recorded in transcript session ``session``; returns how many.

    Nothing calls this yet: an outcome is posted under the actor's own session id, not the transcript's.
    """
    if not enabled() or not session:
        return 0
    cursor = db.execute(
        "UPDATE function_cases SET outcome = ? WHERE session = ?",
        (db.dumps(outcome), str(session)),
    )
    return int(cursor.rowcount or 0)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


PRESERVED = "preserved"
DIVERGED = "diverged"
INCONCLUSIVE = "inconclusive"
NOW_PASSES = "now passes"
STILL_FAILS = "still fails"


@dataclass(frozen=True)
class Replay:
    """What one case did against the new source."""

    case: Case
    status: str
    detail: str = ""


class _Stop(BaseException):
    """Ends a replay from inside the function (a BaseException, so its own ``except Exception`` cannot hide it)."""


class _World:
    """Serves one case's recorded environment calls, in order, and remembers how the replay ended."""

    def __init__(self, case: Case) -> None:
        from .primitives.environment import environment_namespaces

        self.case = case
        self.calls = case.trace
        # What the new source passes on is redacted as the recording was.
        self.redactor = _Redactor(case.salt)
        self.position = 0
        self.verdict: Optional[tuple[str, str]] = None
        self.cancelled = False
        self.namespaces = environment_namespaces()

    def stop(self, status: str, detail: str) -> _Stop:
        if self.verdict is None:
            self.verdict = (status, detail)
        return _Stop(detail)

    def dispatch(
        self,
        namespace: str,
        method: str,
        call: Any,
        args: Any,
        kwargs: Mapping[str, Any],
    ) -> Any:
        if self.cancelled:
            raise self.stop(INCONCLUSIVE, "the replay timed out")
        k = self.position
        digest, shown = _call_key(namespace, method, call, args, kwargs, self.redactor)
        now = f"primitives.{namespace}.{method}({shown})"
        if k >= len(self.calls):
            if not self.case.trace_complete:
                raise self.stop(
                    INCONCLUSIVE,
                    f"the recorded trace was cut short after {len(self.calls)} environment call(s)",
                )
            raise self.stop(
                DIVERGED,
                f"makes an extra environment call {k + 1}, {now}, after the "
                f"{len(self.calls)} recorded",
            )
        recorded = self.calls[k]
        if (
            recorded.get("call") != f"{namespace}.{method}"
            or recorded.get("args") != digest
        ):
            raise self.stop(
                DIVERGED,
                f"diverges at environment call {k + 1}: recorded "
                f"primitives.{recorded.get('call')}({recorded.get('shown', '')}), now {now}",
            )
        self.position += 1
        error = recorded.get("error")
        if error:
            raise _rebuilt(error)
        if not recorded.get("replayable"):
            raise self.stop(
                INCONCLUSIVE,
                f"the answer to environment call {k + 1} ({now}) was not recorded exactly",
            )
        return copy.deepcopy(recorded.get("result"))


def _rebuilt(error: Mapping[str, Any]) -> Exception:
    """The recorded exception, as a builtin of that name or a class of the same name."""
    name = str(error.get("type") or "Exception")
    message = str(error.get("message") or "")
    cls = getattr(builtins, name, None)
    if isinstance(cls, type) and issubclass(cls, Exception):
        try:
            return cls(message)
        except Exception:  # noqa: BLE001 - a builtin that wants other arguments
            pass
    return type(name, (Exception,), {})(message)


class _Refusing:
    """A name replay does not serve: touching it makes the replay inconclusive."""

    def __init__(self, world: _World, what: str) -> None:
        object.__setattr__(self, "_world", world)
        object.__setattr__(self, "_what", what)

    def _refuse(self) -> _Stop:
        world = object.__getattribute__(self, "_world")
        what = object.__getattribute__(self, "_what")
        return world.stop(INCONCLUSIVE, f"it uses {what}, which replay does not serve")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise self._refuse()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise self._refuse()


class _ReplayNamespace:
    """``primitives.<namespace>`` in a replay: its declared methods, answered from the trace."""

    def __init__(self, world: _World, namespace: Any) -> None:
        object.__setattr__(self, "_world", world)
        object.__setattr__(self, "_namespace", namespace)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        world = object.__getattribute__(self, "_world")
        namespace = object.__getattribute__(self, "_namespace")
        method = namespace.method(name)
        if method is None:
            raise AttributeError(
                f"primitives.{namespace.name} has no method {name!r}; its methods are "
                f"{', '.join(namespace.method_names())}",
            )
        call = method.call
        if inspect.iscoroutinefunction(call):

            async def replayed(*args: Any, **kwargs: Any) -> Any:
                return world.dispatch(namespace.name, name, call, args, kwargs)

        else:

            def replayed(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
                return world.dispatch(namespace.name, name, call, args, kwargs)

        replayed.__name__ = name
        return replayed


class _ReplayPrimitives:
    """The ``primitives`` a replay hands the function: registered environments only, from the trace."""

    def __init__(self, world: _World) -> None:
        object.__setattr__(self, "_world", world)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        world = object.__getattribute__(self, "_world")
        namespace = world.namespaces.get(name)
        if namespace is None:
            raise world.stop(
                INCONCLUSIVE,
                f"it uses primitives.{name}, whose calls are not recorded",
            )
        return _ReplayNamespace(world, namespace)


def _replay_globals(world: _World) -> Dict[str, Any]:
    from .primitives.environment import environment_globals

    extra: Dict[str, Any] = {
        name: _Refusing(world, name) for name in ("query_llm", "list_llms", "unillm")
    }
    for name in environment_globals():
        extra[name] = _Refusing(world, name)
    return extra


def _unreplayable(
    fm: Any,
    *,
    source: str,
    depends_on: Iterable[str],
    dependencies: Iterable[str],
) -> Optional[str]:
    """Why the new source (or a stored function it calls) cannot be replayed faithfully; ``None`` if it can."""
    if list(dependencies or ()):
        return "it needs third-party packages, which replay does not install"
    sources = [source]
    queue = [d for d in depends_on or () if isinstance(d, str) and "." not in d]
    seen: set[str] = set()
    while queue:
        dep = queue.pop(0)
        if dep in seen:
            continue
        seen.add(dep)
        row = fm._get_function_data_by_name(name=dep)
        if row is None:
            continue
        if row.get("dependencies"):
            return f"the stored function it calls, {dep}, needs third-party packages"
        sources.append(str(row.get("implementation") or ""))
        queue.extend(
            d
            for d in row.get("depends_on") or ()
            if isinstance(d, str) and "." not in d
        )
    for text in sources:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return "its source does not parse"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for module in modules:
                    if module.split(".")[0] not in PURE_MODULES:
                        return f"it imports {module or 'a relative module'}, which replay does not stub"
            elif isinstance(node, ast.Name):
                if node.id in NONDETERMINISTIC_NAMES:
                    return f"it reads {node.id}, which differs from run to run"
                if node.id in UNSTUBBED_NAMES:
                    return f"it uses {node.id}, which replay does not serve"
            elif isinstance(node, ast.Attribute):
                if node.attr in NONDETERMINISTIC_ATTRIBUTES:
                    return f"it reads .{node.attr}(), which differs from run to run"
    return None


def _run_bounded(fn: Callable[..., Any], call: Mapping[str, Any], world: _World):
    """Run ``fn`` on the recorded arguments in a thread of its own, for at most :data:`REPLAY_TIMEOUT_S`."""
    box: Dict[str, Any] = {}
    args = list(call.get("args") or [])
    kwargs = dict(call.get("kwargs") or {})

    def target() -> None:
        token = _REPLAYING.set(True)
        try:

            async def main() -> Any:
                value = fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))
                if inspect.isawaitable(value):
                    value = await value
                return value

            box["value"] = asyncio.run(main())
        except BaseException as exc:  # noqa: BLE001 - every ending is a verdict
            box["error"] = exc
        finally:
            _REPLAYING.reset(token)

    thread = threading.Thread(target=target, name="unify-case-replay", daemon=True)
    thread.start()
    thread.join(REPLAY_TIMEOUT_S)
    if thread.is_alive():
        world.cancelled = True
        return "timeout", None
    if "error" in box:
        return "raised", box["error"]
    return "returned", box.get("value")


_REPLAY_ARTEFACTS = (NameError, ImportError)
"""Exceptions a replay may raise because its scratch namespace is not the session's: inconclusive."""


def _replay_one(
    fm: Any,
    *,
    name: str,
    source: str,
    depends_on: List[str],
    case: Case,
) -> Replay:
    if case.call is None:
        return Replay(case, INCONCLUSIVE, "its arguments were not recorded exactly")
    world = _World(case)
    token = _REPLAYING.set(True)
    try:
        candidate = fm._verify_candidate(
            name=name,
            source=source,
            depends_on=depends_on,
        )
        fn = candidate.load(_ReplayPrimitives(world), _replay_globals(world))
    except BaseException as exc:  # noqa: BLE001 - a source that does not load here
        if world.verdict is not None:
            return Replay(case, *world.verdict)
        return Replay(
            case,
            INCONCLUSIVE,
            f"it does not load for replay: {_error_text(exc)}",
        )
    finally:
        _REPLAYING.reset(token)
    how, value = _run_bounded(fn, case.call, world)
    if world.verdict is not None:
        return Replay(case, *world.verdict)
    if how == "timeout":
        return Replay(case, INCONCLUSIVE, f"the replay took over {REPLAY_TIMEOUT_S:g}s")
    redactor = world.redactor
    if case.kind == FAIL:
        if how == "raised":
            return Replay(case, STILL_FAILS, redactor.scrub(_error_text(value)))
        return Replay(
            case,
            NOW_PASSES,
            f"now returns {_short(redactor.redact(plain(value)[0]))}",
        )
    if how == "raised":
        if isinstance(value, _REPLAY_ARTEFACTS) or not isinstance(value, Exception):
            return Replay(case, INCONCLUSIVE, f"the replay raised {_error_text(value)}")
        return Replay(
            case,
            DIVERGED,
            f"returned {(case.result or {}).get('shown', '?')} before; now raises "
            f"{redactor.scrub(_error_text(value))}",
        )
    if world.position < len(world.calls):
        return Replay(
            case,
            DIVERGED,
            f"makes {world.position} of the {len(world.calls)} recorded environment calls "
            f"and then returns; the next recorded one is "
            f"primitives.{world.calls[world.position].get('call')}"
            f"({world.calls[world.position].get('shown', '')})",
        )
    data, exact = plain(value)
    data = redactor.redact(data)
    recorded = case.result or {}
    if _sha256(_canonical(data)) == recorded.get("digest"):
        return Replay(case, PRESERVED)
    shown = _short(data) if exact else redactor.short(value)
    if exact and recorded.get("exact"):
        return Replay(
            case,
            DIVERGED,
            f"returned {recorded.get('shown', '?')} before; now returns {shown}",
        )
    return Replay(
        case,
        INCONCLUSIVE,
        f"returned {recorded.get('shown', '?')} before and {shown} now, which may be the same value",
    )


def replay(
    fm: Any,
    *,
    name: str,
    function_id: int,
    source: str,
    depends_on: List[str],
    dependencies: Iterable[str] = (),
) -> List[Replay]:
    """Replay the function's active cases against ``source``: every case that returned, and the cases that
    raised without making an environment call."""
    found = cases(function_id)
    if not found:
        return []
    why = _unreplayable(
        fm,
        source=source,
        depends_on=depends_on,
        dependencies=dependencies,
    )
    out: List[Replay] = []
    for case in found:
        if case.kind == FAIL and case.trace:
            continue
        if why is not None:
            out.append(Replay(case, INCONCLUSIVE, why))
            continue
        try:
            out.append(
                _replay_one(
                    fm,
                    name=name,
                    source=source,
                    depends_on=depends_on,
                    case=case,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a replay fault never blocks a change
            out.append(
                Replay(case, INCONCLUSIVE, f"the replay failed: {_error_text(exc)}"),
            )
    return out


def _listed(name: str, replays: List[Replay]) -> str:
    lines = [
        f"- case {r.case.case_id}, {name}({r.case.args_shown}): {r.detail}"
        for r in replays[:MAX_REPORTED]
    ]
    if len(replays) > MAX_REPORTED:
        lines.append(f"- and {len(replays) - MAX_REPORTED} more")
    return "\n".join(lines)


def refusal(name: str, replays: List[Replay]) -> Optional[str]:
    """The error that refuses the change when a case that returned now behaves differently; else ``None``."""
    diverged = [r for r in replays if r.status == DIVERGED]
    if not diverged:
        return None
    return (
        f"'{name}' was not changed: the new source does something else on "
        f"{len(diverged)} recorded call(s) that worked before:\n"
        f"{_listed(name, diverged)}\n"
        f"If the behaviour is meant to change, store the new behaviour under a new "
        f"name and mark '{name}' as superseded (say so in its docstring, or delete "
        f"it), because callers written against it still expect the old behaviour. If "
        f"a recorded case is itself wrong, retire it with "
        f"FunctionManager_retire_case(function_name, case_id, why) and try again."
    )


def report(name: str, replays: List[Replay]) -> str:
    """What the replay of an accepted change found, for the result; empty when no case was replayed."""
    if not replays:
        return ""
    preserved = sum(1 for r in replays if r.status == PRESERVED)
    parts = [f"{preserved} recorded call(s) replayed unchanged"]
    unsure = [r for r in replays if r.status == INCONCLUSIVE]
    if unsure:
        parts.append(
            f"{len(unsure)} could not be replayed faithfully:\n{_listed(name, unsure)}",
        )
    fixed = [r for r in replays if r.status == NOW_PASSES]
    if fixed:
        parts.append(
            f"{len(fixed)} that raised before now return:\n{_listed(name, fixed)}",
        )
    still = [r for r in replays if r.status == STILL_FAILS]
    if still:
        parts.append(
            f"{len(still)} that raised before still raise:\n{_listed(name, still)}",
        )
    return "cases: " + "; ".join(parts)


__all__ = [
    "Case",
    "CaseRecorder",
    "MAX_FAILING",
    "MAX_PASSING",
    "Pending",
    "Replay",
    "cases",
    "enabled",
    "mark_outcome",
    "observe_primitive",
    "observe_primitive_async",
    "plain",
    "quiet",
    "recording",
    "refusal",
    "replay",
    "replaying",
    "report",
    "retire",
    "summaries",
    "tracing",
    "with_summaries",
]
