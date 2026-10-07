"""One observer seam for the calls an actor makes into its environment.

Every ``primitives.<ns>.<method>`` call passes through one wrapper
(``environment._documented``), in process and, for a cell in the sandboxed
worker, when the harness serves the worker's ``{"op": "call"}`` request.
Features that must see those calls subscribe here instead of each editing
that wrapper: an evidence ledger records requests and responses, speculation
answers a call that is not a read without running it.

An observer is pushed for a scope with :func:`observing` (a ``ContextVar``:
a session, a request, a cell; concurrent tasks keep their own). For each call
every active observer's ``before()`` runs in the order they were pushed; the
first that returns :class:`Intercepted` supplies the result and the call does
not run. Otherwise the call runs as it would with no observer. Then every
observer's ``after()`` runs, and the call's own error, if any, is re-raised.
An exception in ``before()`` propagates (nothing runs: speculation fails
closed); one in ``after()`` is logged and the call's outcome stands. With no
observer pushed the wrapper takes the shipped path untouched.

An environment's raw globals (``environment_globals()``, e.g. AppWorld's
``apis``) bypass the wrapper. While a feature that uses the seam is on
(``UNIFY_SPECULATE``, read with a default so the seam ships before it
exists), the sandbox gets each such global wrapped in
a transparent proxy whose attribute-path calls (``apis.spotify.login(...)``)
take the same before/after path as ``EnvCall(namespace="apis",
method="spotify.login", effect="", via="global")``. With it off the
globals are the registered objects themselves.

Results are passed by reference: an observer that keeps one must copy or
serialise it.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import logging
import time
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

from unify.common.tool_errors import ToolInputError

logger = logging.getLogger(__name__)

__all__ = [
    "EnvCall",
    "EnvObserver",
    "Intercepted",
    "RawGlobalRefused",
    "complete_required",
    "current",
    "dispatch",
    "dispatch_async",
    "observed_globals",
    "observing",
    "proxy_enabled",
    "proxy_target",
]


@dataclass(frozen=True)
class EnvCall:
    """One environment call, as observers see it."""

    namespace: str
    """The primitives namespace (``"appworld"``), or the global's name for a raw-global call."""
    method: str
    """The declared method name, or the dotted attribute path of a raw-global call (``"spotify.login"``)."""
    effect: str
    """The declared effect (``read``/``write``/``destructive``); ``""`` when undeclared (every raw-global
    call). Consumers treat ``""`` as not-a-read."""
    args: tuple
    kwargs: Dict[str, Any]
    """As passed; observers must not mutate ``args`` or ``kwargs``."""
    via: str
    """``"primitives"`` or ``"global"``."""


@dataclass(frozen=True)
class Intercepted:
    """Returned by ``before()``: ``result`` is returned to the caller instead of running the call."""

    result: Any


class EnvObserver(Protocol):
    complete: bool
    """True: this observer must see every environment call. While one is active, a raw-global access the
    proxy cannot dispatch is refused."""

    def before(self, call: EnvCall) -> Optional[Intercepted]: ...

    def after(
        self,
        call: EnvCall,
        *,
        result: Any,
        error: Optional[BaseException],
        intercepted: bool,
        started: float,
        elapsed_s: float,
    ) -> None:
        """``intercepted``: ``result`` is the intercepting observer's, not observed evidence. ``started`` is
        ``time.monotonic()`` when the call (or its interception) began, after every ``before()``;
        ``elapsed_s`` is how long it took."""


_OBSERVERS: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "unify_env_observers",
    default=(),
)


@contextlib.contextmanager
def observing(observer: EnvObserver) -> Iterator[None]:
    """Push ``observer`` for the current context (and tasks created in it) until the block ends."""
    token = _OBSERVERS.set(_OBSERVERS.get() + (observer,))
    try:
        yield
    finally:
        _OBSERVERS.reset(token)


def current() -> tuple:
    """The observers active here, in the order they were pushed."""
    return _OBSERVERS.get()


def complete_required() -> bool:
    """Whether an active observer must see every environment call."""
    return any(getattr(o, "complete", False) for o in _OBSERVERS.get())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _before(observers: Sequence[Any], call: EnvCall) -> Optional[Intercepted]:
    """Every observer's ``before()``, in order; the first interception (an exception propagates)."""
    won: Optional[Intercepted] = None
    for observer in observers:
        out = observer.before(call)
        if out is None:
            continue
        if not isinstance(out, Intercepted):
            raise TypeError(
                f"environment observer {observer!r} returned {type(out).__name__} from before(); "
                "return None or Intercepted(result)",
            )
        if won is None:
            won = out
    return won


def _after(
    observers: Sequence[Any],
    call: EnvCall,
    *,
    result: Any,
    error: Optional[BaseException],
    intercepted: bool,
    started: float,
) -> None:
    elapsed_s = time.monotonic() - started
    for observer in observers:
        try:
            observer.after(
                call,
                result=result,
                error=error,
                intercepted=intercepted,
                started=started,
                elapsed_s=elapsed_s,
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - an observer never changes the call's outcome
            logger.warning(
                "environment observer %r failed in after() for %s.%s: %s: %s",
                observer,
                call.namespace,
                call.method,
                type(exc).__name__,
                exc,
                exc_info=True,
            )


def dispatch(
    observers: Sequence[Any],
    call: EnvCall,
    run: Callable[[], Any],
) -> Any:
    """Run ``run()`` (a synchronous environment call) under ``observers``.

    A synchronous callable that returns an awaitable (a lambda over a
    coroutine function) gets that awaitable back wrapped, so ``after()`` sees
    the awaited value when the caller awaits it.
    """
    hit = _before(observers, call)
    started = time.monotonic()
    if hit is not None:
        _after(
            observers,
            call,
            result=hit.result,
            error=None,
            intercepted=True,
            started=started,
        )
        return hit.result
    try:
        result = run()
    except BaseException as exc:
        _after(
            observers,
            call,
            result=None,
            error=exc,
            intercepted=False,
            started=started,
        )
        raise
    if inspect.isawaitable(result):
        return _finish(observers, call, result, started)
    _after(
        observers,
        call,
        result=result,
        error=None,
        intercepted=False,
        started=started,
    )
    return result


async def _finish(
    observers: Sequence[Any],
    call: EnvCall,
    pending: Awaitable[Any],
    started: float,
) -> Any:
    try:
        result = await pending
    except BaseException as exc:
        _after(
            observers,
            call,
            result=None,
            error=exc,
            intercepted=False,
            started=started,
        )
        raise
    _after(
        observers,
        call,
        result=result,
        error=None,
        intercepted=False,
        started=started,
    )
    return result


async def dispatch_async(
    observers: Sequence[Any],
    call: EnvCall,
    run: Callable[[], Awaitable[Any]],
) -> Any:
    """Await ``run()`` (an asynchronous environment call) under ``observers``.

    ``run`` is called only when no observer intercepts, so an intercepted
    call never creates the environment's coroutine.
    """
    hit = _before(observers, call)
    started = time.monotonic()
    if hit is not None:
        _after(
            observers,
            call,
            result=hit.result,
            error=None,
            intercepted=True,
            started=started,
        )
        return hit.result
    try:
        pending = run()
    except BaseException as exc:
        _after(
            observers,
            call,
            result=None,
            error=exc,
            intercepted=False,
            started=started,
        )
        raise
    return await _finish(observers, call, pending, started)


# ---------------------------------------------------------------------------
# Raw globals
# ---------------------------------------------------------------------------

_OFF_WORDS = frozenset({"", "0", "false", "no", "off", "none"})
_SWITCHES = ("UNIFY_SPECULATE",)


def _on(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in _OFF_WORDS
    return bool(value)


def proxy_enabled(settings: Any = None) -> bool:
    """Whether a feature that uses the seam is on, so raw globals are proxied.

    Read with ``getattr(..., default)``: the seam ships before either switch
    exists. A string switch reads as off when empty or ``off``/``false``/``0``.
    """
    if settings is None:
        from unify.settings import SETTINGS as settings
    return any(_on(getattr(settings, name, "")) for name in _SWITCHES)


class RawGlobalRefused(ToolInputError, AttributeError):
    """A raw-global access the proxy cannot route through the observers, while one must see every call.

    Also an ``AttributeError``, so a probe with a default (``getattr(x, name, None)``) reads it as missing.
    """


#: Values a proxy hands out as they are: data has no calls of its own to
#: observe (an environment object kept inside a container is not wrapped).
_DATA = (
    str,
    bytes,
    bytearray,
    int,
    float,
    complex,
    bool,
    type(None),
    list,
    tuple,
    dict,
    set,
    frozenset,
)


def _suggestion(path: Sequence[str]) -> str:
    """The ``primitives`` method(s) to use instead of a raw-global ``path``."""
    from .environment import environment_namespaces

    namespaces = environment_namespaces()
    named = [str(p) for p in path if isinstance(p, str) and not p.startswith("_")]
    if len(named) >= 2:
        ns = namespaces.get(named[-2])
        if ns is not None and ns.method(named[-1]) is not None:
            return f"call primitives.{ns.name}.{named[-1]} instead"
    if named:
        ns = namespaces.get(named[0])
        if ns is not None:
            listed = ", ".join(f"primitives.{ns.name}.{m}" for m in ns.method_names())
            return f"call one of primitives.{ns.name}'s methods instead ({listed})"
        hits = [
            f"primitives.{ns.name}.{named[-1]}"
            for ns in namespaces.values()
            if ns.method(named[-1]) is not None
        ]
        if hits:
            return f"call {' or '.join(hits)} instead"
    if namespaces:
        listed = ", ".join(f"primitives.{name}" for name in sorted(namespaces))
        return f"call the environment through {listed} instead"
    return "call the environment through its primitives namespaces instead"


def _refuse(name: str, path: Sequence[str], what: str) -> RawGlobalRefused:
    shown = ".".join([name, *path]) if path else name
    return RawGlobalRefused(
        f"`{shown}`: {what} cannot be observed, and an active environment observer must see every "
        f"environment call; {_suggestion(path)}.",
    )


class _Proxy:
    """A raw environment global, or an object reached from it by attribute access."""

    __slots__ = ("_unify_target", "_unify_name", "_unify_path")

    def __init__(self, target: Any, name: str, path: tuple) -> None:
        object.__setattr__(self, "_unify_target", target)
        object.__setattr__(self, "_unify_name", name)
        object.__setattr__(self, "_unify_path", path)

    def __getattribute__(self, attr: str) -> Any:
        # The proxy's own slots are not the target's attributes, and cell code
        # must not reach the unwrapped object through them.
        if attr in _Proxy.__slots__:
            return _Proxy.__getattr__(self, attr)
        return object.__getattribute__(self, attr)

    def __getattr__(self, attr: str) -> Any:
        target, name, path = _parts(self)
        if attr.startswith("_"):
            if complete_required():
                raise _refuse(name, (*path, attr), "a private attribute")
            return getattr(target, attr)
        return _wrap(getattr(target, attr), name, (*path, attr))

    def __getitem__(self, key: Any) -> Any:
        target, name, path = _parts(self)
        return _wrap(
            target[key],
            name,
            (*path, key if isinstance(key, str) else repr(key)),
        )

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_unify_target"), attr, value)

    def __delattr__(self, attr: str) -> None:
        delattr(object.__getattribute__(self, "_unify_target"), attr)

    def __iter__(self) -> Iterator[Any]:
        target, name, path = _parts(self)
        if complete_required():
            raise _refuse(name, path, "iterating over it")
        return iter(target)

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_unify_target"))

    def __bool__(self) -> bool:
        return bool(object.__getattribute__(self, "_unify_target"))

    def __dir__(self) -> list:
        return dir(object.__getattribute__(self, "_unify_target"))

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_unify_target"))

    def __str__(self) -> str:
        return str(object.__getattribute__(self, "_unify_target"))


def _parts(proxy: _Proxy) -> tuple:
    get = object.__getattribute__
    return (
        get(proxy, "_unify_target"),
        get(proxy, "_unify_name"),
        get(proxy, "_unify_path"),
    )


def _call_of(proxy: _Proxy, args: tuple, kwargs: dict) -> EnvCall:
    _, name, path = _parts(proxy)
    return EnvCall(
        namespace=name,
        method=".".join(str(p) for p in path),
        effect="",
        args=args,
        kwargs=kwargs,
        via="global",
    )


class _CallableProxy(_Proxy):
    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target = object.__getattribute__(self, "_unify_target")
        active = _OBSERVERS.get()
        if not active:
            return target(*args, **kwargs)
        return dispatch(
            active,
            _call_of(self, args, kwargs),
            lambda: target(*args, **kwargs),
        )


class _AsyncCallableProxy(_Proxy):
    __slots__ = ()
    # ``inspect.iscoroutinefunction`` (and the worker's static probe) read the
    # proxy as what it wraps.
    _is_coroutine_marker = getattr(inspect, "_is_coroutine_mark", None)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target = object.__getattribute__(self, "_unify_target")
        active = _OBSERVERS.get()
        if not active:
            return await target(*args, **kwargs)
        return await dispatch_async(
            active,
            _call_of(self, args, kwargs),
            lambda: target(*args, **kwargs),
        )


def _is_async(fn: Any) -> bool:
    """Whether calling ``fn`` gives a coroutine, reading a dynamic object only statically.

    An environment object that answers any attribute name (AppWorld's apps
    raise for an unknown one) would take a dynamic probe such as
    ``_is_coroutine_marker`` for a request, so only plain functions are
    probed the usual way.
    """
    marker = getattr(inspect, "_is_coroutine_mark", None)
    seen: set[int] = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        if inspect.isfunction(fn) or inspect.ismethod(fn):
            if inspect.iscoroutinefunction(fn):
                return True
        elif inspect.iscoroutinefunction(
            inspect.getattr_static(type(fn), "__call__", None),
        ):
            return True
        elif (
            marker is not None
            and inspect.getattr_static(fn, "_is_coroutine_marker", None) is marker
        ):
            return True
        fn = inspect.getattr_static(fn, "__wrapped__", None)
    return False


def _wrap(value: Any, name: str, path: tuple) -> Any:
    if isinstance(value, _DATA) or isinstance(value, _Proxy):
        return value
    if callable(value):
        cls = _AsyncCallableProxy if _is_async(value) else _CallableProxy
        return cls(value, name, path)
    return _Proxy(value, name, path)


def proxy_target(value: Any) -> Any:
    """The object a raw-global proxy wraps; ``value`` itself when it is not one."""
    if isinstance(value, _Proxy):
        return object.__getattribute__(value, "_unify_target")
    return value


def observed_globals(
    values: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """An environment's raw globals as the sandbox gets them.

    With a feature that uses the seam on (:func:`proxy_enabled`, or
    ``enabled=True``) each is wrapped in a transparent proxy whose calls reach
    the observers; otherwise they are the registered objects themselves.
    """
    if enabled is None:
        enabled = proxy_enabled()
    if not enabled:
        return dict(values)
    return {name: _wrap(value, name, ()) for name, value in values.items()}
