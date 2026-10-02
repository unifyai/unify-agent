"""Environment namespaces: the callable surface of the environment Unify runs in.

Unify's own managers are fixed ``primitives`` namespaces (``primitives.actor``).
An environment the assistant is deployed into (a benchmark's apps, a company's
internal services) registers its surface the same way at start-up, from the
factories named by ``UNIFY_ENV_NAMESPACES`` (``package.module:factory``,
several separated by commas). A registered namespace is then treated like
``primitives.actor``:

- it is a valid primitive alias and in the default runtime scope, and a
  ``Primitives`` object whose scope holds it resolves ``primitives.<name>`` to
  it. An actor's sandbox holds only the primitives its environments inject,
  so the actor side adds an environment for the registered namespaces;
- each method is a primitive row (signature, docstring, effect label) that
  function search finds and ``execute_function`` runs by its dotted name;
- ``add_functions`` records ``primitives.<name>.<method>`` calls in
  ``depends_on``, and a stored function gets the namespace injected when it is
  loaded later (``construct_sandbox_root("primitives")``);
- the actor's prompt carries a short listing of it.

A factory takes no arguments and returns an :class:`EnvironmentSurface`, or a
mapping with the same fields:

- ``namespaces``: :class:`EnvironmentNamespace` entries, each method with its
  callable, signature, docstring and an effect label (``read``, ``write`` or
  ``destructive``), the way the legacy integration rows carried an
  ``action_class``;
- ``modules``: import names the environment supplies, so a stored function
  that imports one is not asked for a pip dependency (they join
  ``ENVIRONMENT_MODULES``);
- ``globals``: further sandbox globals the environment binds, such as the name
  its own documentation uses for the same surface.

A factory that fails, or a surface that does not validate, raises at start-up:
an environment that cannot register would otherwise run with a surface the
prompts do not describe. With ``UNIFY_ENV_NAMESPACES`` unset nothing is
registered and every code path is the shipped one.
"""

from __future__ import annotations

import importlib
import inspect
import keyword
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

EFFECTS: tuple[str, ...] = ("read", "write", "destructive")
"""Effect labels, from least to most consequential: ``read`` changes nothing,
``write`` creates or changes state, ``destructive`` deletes or overwrites it."""

CLASS_PATH_PREFIX = "environment:"
"""``primitive_class`` of an environment namespace's rows: not an importable
class path, so no row can be confused with a manager class."""

RESERVED_NAMES = frozenset({"actor"})
"""Namespace names Unify's own managers own."""


@dataclass(frozen=True)
class EnvironmentMethod:
    """One callable of an environment namespace."""

    name: str
    call: Callable[..., Any]
    effect: str
    signature: str = "(**kwargs)"
    docstring: str = ""


@dataclass(frozen=True)
class EnvironmentNamespace:
    """``primitives.<name>``: a group of an environment's methods."""

    name: str
    methods: tuple[EnvironmentMethod, ...]
    description: str = ""

    @property
    def class_path(self) -> str:
        """The ``primitive_class`` its rows carry."""
        return f"{CLASS_PATH_PREFIX}{self.name}"

    def method(self, name: str) -> Optional[EnvironmentMethod]:
        for method in self.methods:
            if method.name == name:
                return method
        return None

    def method_names(self) -> list[str]:
        return sorted(method.name for method in self.methods)

    def effect_counts(self) -> Dict[str, int]:
        counts = {effect: 0 for effect in EFFECTS}
        for method in self.methods:
            counts[method.effect] += 1
        return counts


@dataclass(frozen=True)
class EnvironmentSurface:
    """What an environment registers: namespaces, modules and extra globals."""

    namespaces: tuple[EnvironmentNamespace, ...] = ()
    modules: frozenset[str] = frozenset()
    globals: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""


class EnvironmentNamespaceError(RuntimeError):
    """An environment's factory failed or returned a surface that does not validate."""


def method_docstring(namespace: str, method: EnvironmentMethod) -> str:
    """The docstring a method's primitive row and ``help()`` show.

    The summary names the effect, so it survives the compaction that search
    results apply to primitive rows (summary plus ``Parameters``).
    """
    head = f"`primitives.{namespace}.{method.name}{method.signature}` (effect: {method.effect})."
    body = inspect.cleandoc(method.docstring or "")
    if not body:
        return head
    first, _, rest = body.partition("\n\n")
    text = f"{first.strip()} {head}"
    return f"{text}\n\n{rest.strip()}" if rest.strip() else text


class EnvironmentNamespaceObject:
    """What ``primitives.<name>`` resolves to: exactly the declared methods.

    Each method is a thin wrapper around the environment's callable that
    carries the method's documentation, so ``help(primitives.<name>.<m>)``
    reads it; anything undeclared raises ``AttributeError`` naming the
    methods that exist.
    """

    def __init__(self, namespace: EnvironmentNamespace) -> None:
        object.__setattr__(self, "_namespace", namespace)
        object.__setattr__(
            self,
            "_methods",
            {m.name: _documented(namespace.name, m) for m in namespace.methods},
        )

    def __getattr__(self, name: str) -> Any:
        methods = object.__getattribute__(self, "_methods")
        if name in methods:
            return methods[name]
        if name.startswith("__"):
            raise AttributeError(name)
        namespace = object.__getattribute__(self, "_namespace")
        raise AttributeError(
            f"primitives.{namespace.name} has no method {name!r}; its methods are "
            f"{', '.join(sorted(methods))}",
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("environment namespaces are read-only")

    def __dir__(self) -> list[str]:
        return sorted(object.__getattribute__(self, "_methods"))

    def __repr__(self) -> str:
        namespace = object.__getattribute__(self, "_namespace")
        return f"<primitives.{namespace.name}: {len(namespace.methods)} methods>"


def _documented(namespace: str, method: EnvironmentMethod) -> Callable[..., Any]:
    call = method.call
    # UNIFY_FUNCTION_CASES: every environment call goes through here, so this
    # is where a call made inside a stored function joins its recorded case;
    # nothing is recording while the switch is off.
    from unify.function_manager import store_cases

    if inspect.iscoroutinefunction(call):

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if store_cases.recording():
                return await store_cases.observe_primitive_async(
                    namespace,
                    method.name,
                    call,
                    args,
                    kwargs,
                )
            return await call(*args, **kwargs)

    else:

        def wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            if store_cases.recording():
                return store_cases.observe_primitive(
                    namespace,
                    method.name,
                    call,
                    args,
                    kwargs,
                )
            return call(*args, **kwargs)

    wrapper.__name__ = method.name
    wrapper.__qualname__ = f"primitives.{namespace}.{method.name}"
    wrapper.__doc__ = method_docstring(namespace, method)
    wrapper.__module__ = f"{CLASS_PATH_PREFIX}{namespace}"
    return wrapper


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _identifier(value: Any, what: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isidentifier()
        or keyword.iskeyword(value)
        or value.startswith("_")
    ):
        raise EnvironmentNamespaceError(
            f"{what} {value!r} is not a public Python identifier",
        )
    return value


def _coerce_method(item: Any, namespace: str) -> EnvironmentMethod:
    if isinstance(item, Mapping):
        item = EnvironmentMethod(**dict(item))
    if not isinstance(item, EnvironmentMethod):
        raise EnvironmentNamespaceError(
            f"primitives.{namespace}: a method must be an EnvironmentMethod or a mapping, "
            f"not {type(item).__name__}",
        )
    _identifier(item.name, f"primitives.{namespace}: method name")
    if item.effect not in EFFECTS:
        raise EnvironmentNamespaceError(
            f"primitives.{namespace}.{item.name}: effect {item.effect!r} is not one of {EFFECTS}",
        )
    if not callable(item.call):
        raise EnvironmentNamespaceError(
            f"primitives.{namespace}.{item.name}: call is not callable",
        )
    if not isinstance(item.signature, str) or not item.signature.startswith("("):
        raise EnvironmentNamespaceError(
            f"primitives.{namespace}.{item.name}: signature must be a string like '(x: int)'",
        )
    return item


def _coerce_namespace(item: Any) -> EnvironmentNamespace:
    if isinstance(item, Mapping):
        data = dict(item)
        name = data.get("name")
        data["methods"] = tuple(
            _coerce_method(m, str(name)) for m in data.get("methods") or ()
        )
        item = EnvironmentNamespace(**data)
    if not isinstance(item, EnvironmentNamespace):
        raise EnvironmentNamespaceError(
            f"a namespace must be an EnvironmentNamespace or a mapping, not {type(item).__name__}",
        )
    _identifier(item.name, "namespace name")
    if item.name in RESERVED_NAMES:
        raise EnvironmentNamespaceError(
            f"primitives.{item.name} belongs to Unify's own managers",
        )
    methods = tuple(_coerce_method(m, item.name) for m in item.methods)
    if not methods:
        raise EnvironmentNamespaceError(f"primitives.{item.name} has no methods")
    names = [m.name for m in methods]
    if len(set(names)) != len(names):
        raise EnvironmentNamespaceError(
            f"primitives.{item.name} declares a method twice",
        )
    return EnvironmentNamespace(
        name=item.name,
        methods=methods,
        description=str(item.description or ""),
    )


def coerce_surface(value: Any, *, source: str = "") -> EnvironmentSurface:
    """Validate what a factory returned and normalise it to a surface."""
    if isinstance(value, Mapping):
        value = EnvironmentSurface(
            namespaces=tuple(value.get("namespaces") or ()),
            modules=frozenset(value.get("modules") or ()),
            globals=dict(value.get("globals") or {}),
            source=str(value.get("source") or source),
        )
    if not isinstance(value, EnvironmentSurface):
        raise EnvironmentNamespaceError(
            f"{source or 'the factory'} returned {type(value).__name__}, not an EnvironmentSurface",
        )
    namespaces = tuple(_coerce_namespace(n) for n in value.namespaces)
    modules = frozenset(value.modules)
    for module in modules:
        if not isinstance(module, str) or not all(
            part.isidentifier() for part in module.split(".")
        ):
            raise EnvironmentNamespaceError(f"module {module!r} is not a module name")
    extra = dict(value.globals)
    for name in extra:
        _identifier(name, "global")
        if name == "primitives":
            raise EnvironmentNamespaceError(
                "the global 'primitives' belongs to Unify; register namespaces instead",
            )
    return EnvironmentSurface(
        namespaces=namespaces,
        modules=modules,
        globals=extra,
        source=value.source or source,
    )


def _merge(surfaces: Sequence[EnvironmentSurface]) -> EnvironmentSurface:
    namespaces: list[EnvironmentNamespace] = []
    seen: set[str] = set()
    extra: Dict[str, Any] = {}
    for surface in surfaces:
        for namespace in surface.namespaces:
            if namespace.name in seen:
                raise EnvironmentNamespaceError(
                    f"primitives.{namespace.name} is registered twice",
                )
            seen.add(namespace.name)
            namespaces.append(namespace)
        for name, value in surface.globals.items():
            if name in extra and extra[name] is not value:
                raise EnvironmentNamespaceError(
                    f"the global {name!r} is registered twice",
                )
            extra[name] = value
    return EnvironmentSurface(
        namespaces=tuple(namespaces),
        modules=frozenset().union(*(s.modules for s in surfaces)),
        globals=extra,
        source=",".join(s.source for s in surfaces if s.source),
    )


# ---------------------------------------------------------------------------
# The process's registry
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_loaded = False
_registered: list[EnvironmentSurface] = []
_surface: Optional[EnvironmentSurface] = None


def _factory(spec: str) -> Callable[[], Any]:
    module_name, _, attr = spec.strip().partition(":")
    if not module_name or not attr:
        raise EnvironmentNamespaceError(
            f"UNIFY_ENV_NAMESPACES entry {spec!r} is not 'package.module:factory'",
        )
    try:
        target: Any = importlib.import_module(module_name)
        for part in attr.split("."):
            target = getattr(target, part)
    except Exception as exc:
        raise EnvironmentNamespaceError(
            f"UNIFY_ENV_NAMESPACES entry {spec!r} cannot be imported: {type(exc).__name__}: {exc}",
        ) from exc
    if not callable(target):
        raise EnvironmentNamespaceError(
            f"UNIFY_ENV_NAMESPACES entry {spec!r} is not callable",
        )
    return target


def load_from_spec(spec: str) -> list[EnvironmentSurface]:
    """Call every factory named in ``spec`` and validate what each returns."""
    surfaces = []
    for entry in (part.strip() for part in spec.split(",")):
        if not entry:
            continue
        factory = _factory(entry)
        try:
            value = factory()
        except Exception as exc:
            raise EnvironmentNamespaceError(
                f"UNIFY_ENV_NAMESPACES factory {entry!r} failed: {type(exc).__name__}: {exc}",
            ) from exc
        surfaces.append(coerce_surface(value, source=entry))
    return surfaces


def _rebuild() -> None:
    global _surface
    _surface = _merge(_registered) if _registered else None
    # Rows are seeded from the registry once per store and process; a surface
    # registered after that must reach the ``primitives`` table too.
    fm = sys.modules.get("unify.function_manager.function_manager")
    if fm is not None:
        fm._PRIMITIVES_SEEDED_FOR.clear()


def load_environment_namespaces() -> Optional[EnvironmentSurface]:
    """Register the factories named by ``UNIFY_ENV_NAMESPACES``, once per process.

    Called by ``unify.init()`` so a failing factory stops start-up, and lazily
    by every reader below. Returns the registered surface, or ``None``.
    """
    global _loaded
    if _loaded:
        return _surface
    with _lock:
        if _loaded:
            return _surface
        from unify.settings import SETTINGS

        spec = str(SETTINGS.UNIFY_ENV_NAMESPACES or "").strip()
        if spec:
            _registered.extend(load_from_spec(spec))
            _rebuild()
        _loaded = True
        return _surface


def register_environment(surface: Any, *, source: str = "") -> EnvironmentSurface:
    """Register a surface from code (an embedding application, or a test)."""
    checked = coerce_surface(surface, source=source)
    with _lock:
        load_environment_namespaces()
        _registered.append(checked)
        try:
            _rebuild()
        except EnvironmentNamespaceError:
            _registered.pop()
            _rebuild()
            raise
    return checked


def clear_environment_namespaces() -> None:
    """Forget every registered surface; the switch is read again on next use."""
    global _loaded
    with _lock:
        _registered.clear()
        _objects.clear()
        _loaded = False
        _rebuild()


def environment_surface() -> Optional[EnvironmentSurface]:
    """The registered surface, or ``None`` when no environment registered one."""
    return _surface if _loaded else load_environment_namespaces()


def environment_namespaces() -> Dict[str, EnvironmentNamespace]:
    surface = environment_surface()
    return {n.name: n for n in surface.namespaces} if surface else {}


def environment_aliases() -> frozenset[str]:
    surface = environment_surface()
    return frozenset(n.name for n in surface.namespaces) if surface else frozenset()


def environment_namespace(alias: str) -> Optional[EnvironmentNamespace]:
    return environment_namespaces().get(alias)


def namespace_for_class_path(class_path: str) -> Optional[EnvironmentNamespace]:
    if not isinstance(class_path, str) or not class_path.startswith(CLASS_PATH_PREFIX):
        return None
    return environment_namespace(class_path[len(CLASS_PATH_PREFIX) :])


def environment_modules() -> frozenset[str]:
    surface = environment_surface()
    return surface.modules if surface else frozenset()


def environment_globals() -> Dict[str, Any]:
    surface = environment_surface()
    return dict(surface.globals) if surface else {}


_objects: Dict[str, EnvironmentNamespaceObject] = {}


def namespace_object(alias: str) -> Optional[EnvironmentNamespaceObject]:
    """The sandbox object for ``primitives.<alias>`` (one per registered namespace)."""
    namespace = environment_namespace(alias)
    if namespace is None:
        return None
    found = _objects.get(alias)
    if found is None or object.__getattribute__(found, "_namespace") is not namespace:
        found = _objects[alias] = EnvironmentNamespaceObject(namespace)
    return found


__all__ = [
    "CLASS_PATH_PREFIX",
    "EFFECTS",
    "EnvironmentMethod",
    "EnvironmentNamespace",
    "EnvironmentNamespaceError",
    "EnvironmentNamespaceObject",
    "EnvironmentSurface",
    "clear_environment_namespaces",
    "coerce_surface",
    "environment_aliases",
    "environment_globals",
    "environment_modules",
    "environment_namespace",
    "environment_namespaces",
    "environment_surface",
    "load_environment_namespaces",
    "method_docstring",
    "namespace_for_class_path",
    "namespace_object",
    "register_environment",
]
