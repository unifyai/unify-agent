"""The contract of a model-facing call: typed arguments checked at the call, and errors that teach.

Standard library only. The harness imports it as :mod:`unify.common.contract_runtime`; the generated library package
carries it byte for byte as ``memory/_contract.py``, so a cell's ``memory`` calls check and explain themselves with no
harness code in the box.

* :class:`ContractError` and its kinds (:class:`ArgumentError`, :class:`NotFound`, :class:`Ambiguous`,
  :class:`WrongCall`) carry what was given (the call as written), what was wrong, what was expected with one correct
  example call, and the nearest valid targets or the right call. Each kind is also the builtin a caller would catch
  (``TypeError`` or ``LookupError``).
* :func:`checked` makes a function check its arguments against its type hints before it runs, and turns a wrong
  arity, an unknown keyword or a wrong type into an :class:`ArgumentError` that names the parameter, the expected type,
  what was given and the function's example call.
* :func:`nearest` lists the valid names closest to a wrong one.
* :func:`render_pyi` writes the typed stub of a namespace, so the whole contract reads in one place.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import inspect
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "Ambiguous",
    "ArgumentError",
    "ContractError",
    "Ge",
    "Le",
    "MinLen",
    "NotFound",
    "WrongCall",
    "call_text",
    "checked",
    "conforms",
    "example_of",
    "nearest",
    "render",
    "render_pyi",
    "type_text",
]

#: The longest a value is shown in an error before it is cut.
SHOWN_CHARS = 60


# --- errors that teach -------------------------------------------------------------------------------------------


def render(
    call: str,
    problem: str,
    *,
    expected: str | None = None,
    example: str | None = None,
    nearest: Sequence[str] = (),
    hint: str | None = None,
) -> str:
    """The error text: ``<call>: <problem>``, then ``Expected <expected>, e.g. <example>.``, then the nearest valid
    targets and the hint (the right call). With neither *expected* nor *nearest*, the hint ends the first line."""
    first = f"{call}: {problem}".rstrip()
    if expected is None and not nearest:
        return f"{first} {hint}".rstrip() if hint else first
    lines = [first]
    if expected is not None:
        lines.append(f"Expected {expected}" + (f", e.g. {example}." if example else "."))
    tail = []
    if nearest:
        tail.append("Nearest: " + ", ".join(nearest) + ".")
    if hint:
        tail.append(hint)
    if tail:
        lines.append(" ".join(tail))
    return "\n".join(lines)


class ContractError(Exception):
    """A call that broke its contract. :attr:`call` is the call as written; the text is :func:`render`'s."""

    def __init__(
        self,
        call: str,
        problem: str,
        *,
        expected: str | None = None,
        example: str | None = None,
        nearest: Sequence[str] = (),
        hint: str | None = None,
    ) -> None:
        self.call = call
        self.problem = problem
        self.expected = expected
        self.example = example
        self.nearest = tuple(nearest)
        self.hint = hint
        super().__init__(
            render(call, problem, expected=expected, example=example, nearest=self.nearest, hint=hint),
        )


class ArgumentError(ContractError, TypeError):
    """An argument of the wrong type, a missing or unknown one, or too many."""


class NotFound(ContractError, LookupError):
    """A name that matches no target; :attr:`nearest` lists the closest valid ones."""


class Ambiguous(ContractError, LookupError):
    """A name that matches several targets; :attr:`nearest` lists each exact one."""


class WrongCall(ContractError, LookupError):
    """A valid target given to the wrong call; :attr:`hint` is the right call."""


def shown(value: Any) -> str:
    """*value*'s repr, cut to :data:`SHOWN_CHARS`."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a repr that fails still shows the type
        text = f"<{type(value).__name__}>"
    return text if len(text) <= SHOWN_CHARS else text[: SHOWN_CHARS - 1] + "…"


def call_text(name: str, args: Sequence[Any] = (), kwargs: Mapping[str, Any] | None = None) -> str:
    """The call as written, ``name('a', k=1)``, each value cut to :data:`SHOWN_CHARS`."""
    parts = [shown(a) for a in args] + [f"{k}={shown(v)}" for k, v in (kwargs or {}).items()]
    return f"{name}({', '.join(parts)})"


def nearest(word: str, candidates: Iterable[str], n: int = 3) -> list[str]:
    """The valid names closest to *word*: those whose last segment (after ``/``, ``.`` or ``:``) is *word*'s last
    segment first, then the closest by spelling."""
    pool = list(dict.fromkeys(c for c in candidates if isinstance(c, str)))
    w = str(word)

    def last(s: str) -> str:
        # the name after the last "/" or ":", and after the last "." of a dotted id (not a file name)
        for sep in ("/", ":"):
            if sep in s.rstrip(sep):
                s = s.rstrip(sep).rsplit(sep, 1)[-1]
        if "." in s and not s.endswith((".md", ".py", ".json")):
            s = s.rsplit(".", 1)[-1]
        return s

    exact_tail = [c for c in pool if last(c) == last(w) or c.endswith("/" + w) or c.endswith(":" + w)]
    close = difflib.get_close_matches(w, pool, n=n * 2, cutoff=0.5)
    tails = {last(c): c for c in pool}
    close += [tails[t] for t in difflib.get_close_matches(last(w), list(tails), n=n, cutoff=0.6)]
    return list(dict.fromkeys(exact_tail + close))[:n]


# --- types --------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Ge:
    """``Annotated[int, Ge(0)]``: at least *bound*."""

    bound: float

    def holds(self, value: Any) -> bool:
        return value >= self.bound

    def text(self) -> str:
        return f">= {self.bound:g}"


@dataclasses.dataclass(frozen=True)
class Le:
    """``Annotated[float, Le(600)]``: at most *bound*."""

    bound: float

    def holds(self, value: Any) -> bool:
        return value <= self.bound

    def text(self) -> str:
        return f"<= {self.bound:g}"


@dataclasses.dataclass(frozen=True)
class MinLen:
    """``Annotated[str, MinLen(1)]``: at least *size* long."""

    size: int

    def holds(self, value: Any) -> bool:
        return len(value) >= self.size

    def text(self) -> str:
        return "non-empty" if self.size == 1 else f"length >= {self.size}"


def _bounds(meta: Sequence[Any]) -> list[Any]:
    return [m for m in meta if hasattr(m, "holds") and hasattr(m, "text")]


def type_text(tp: Any) -> str:
    """*tp* as a reader writes it: ``str``, ``int >= 0``, ``'function' | 'note'``, ``list[str]``, ``str | None``."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any:
        return "any value"
    if tp is None or tp is type(None):
        return "None"
    if origin is typing.Annotated:
        base, *meta = args
        texts = [m.text() for m in _bounds(meta)]
        return " ".join([type_text(base), *texts]) if texts else type_text(base)
    if origin is typing.Literal:
        return " | ".join(repr(a) for a in args)
    if origin in (typing.Union, types.UnionType):
        return " | ".join(type_text(a) for a in args)
    if origin in (list, Sequence, tuple, dict, Mapping) or origin is not None:
        name = getattr(origin, "__name__", str(origin))
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return f"tuple[{type_text(args[0])}, ...]"
        return f"{name}[{', '.join(type_text(a) for a in args)}]" if args else name
    if isinstance(tp, type):
        return tp.__name__
    return str(tp)


def conforms(value: Any, tp: Any) -> bool:
    """Whether *value* is of type *tp* (the forms :func:`type_text` names; a class is checked with ``isinstance``;
    ``bool`` is never an ``int`` or a ``float``; an ``int`` is a ``float``)."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any or tp is object:
        return True
    if tp is None or tp is type(None):
        return value is None
    if origin is typing.Annotated:
        base, *meta = args
        return conforms(value, base) and all(b.holds(value) for b in _bounds(meta))
    if origin is typing.Literal:
        return any(value == a and type(value) is type(a) for a in args)
    if origin in (typing.Union, types.UnionType):
        return any(conforms(value, a) for a in args)
    if tp is bool:
        return isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if origin in (list, Sequence):
        if not isinstance(value, list if origin is list else Sequence) or isinstance(value, (str, bytes)):
            return False
        return not args or all(conforms(v, args[0]) for v in value)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(conforms(v, args[0]) for v in value)
        return not args or (len(args) == len(value) and all(conforms(v, a) for v, a in zip(value, args)))
    if origin in (dict, Mapping):
        if not isinstance(value, dict if origin is dict else Mapping):
            return False
        return not args or all(conforms(k, args[0]) and conforms(v, args[1]) for k, v in value.items())
    if origin is not None and isinstance(origin, type):
        return isinstance(value, origin)
    if origin is Callable or tp is Callable:
        return callable(value)
    if isinstance(tp, type):
        return isinstance(value, tp)
    return True


# --- checked calls ------------------------------------------------------------------------------------------------


def example_of(fn: Callable[..., Any]) -> str | None:
    """The first example call in *fn*'s docstring: the first line after ``Example:`` (or ``Examples:``)."""
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if line.strip().rstrip(":").lower() in ("example", "examples"):
            for nxt in lines[i + 1 :]:
                text = nxt.strip()
                if text:
                    return text.removeprefix(">>> ")
    return None


def checked(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Make *fn* check its arguments against its type hints before it runs.

    A wrong arity, an unknown keyword or a value of the wrong type raises :class:`ArgumentError` naming the call as
    written, the parameter, the expected type and *fn*'s example call (:func:`example_of`). *name* is the call's
    public name (``memory.get``); it defaults to the function's own name."""
    if fn is None:
        return functools.partial(checked, name=name)
    public = name or fn.__name__
    signature = inspect.signature(fn)
    example = example_of(fn)

    resolved: list[dict[str, Any]] = []

    def hints() -> dict[str, Any]:
        # resolved at the first call: a hint may name a type defined after the function
        if not resolved:
            resolved.append(typing.get_type_hints(fn, include_extras=True))
        return resolved[0]

    def check(args: tuple, kwargs: dict) -> None:
        written = call_text(public, args, kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise ArgumentError(
                written,
                f"{exc}.",
                expected=f"{public}{signature}",
                example=example,
            ) from None
        types_of = hints()
        for param, value in bound.arguments.items():
            tp = types_of.get(param)
            kind = signature.parameters[param].kind
            if tp is None:
                continue
            if kind is inspect.Parameter.VAR_POSITIONAL:
                bad = next((v for v in value if not conforms(v, tp)), _MISSING)
            elif kind is inspect.Parameter.VAR_KEYWORD:
                bad = next((v for v in value.values() if not conforms(v, tp)), _MISSING)
            else:
                bad = value if not conforms(value, tp) else _MISSING
            if bad is not _MISSING:
                raise ArgumentError(
                    written,
                    f"{param} is {shown(bad)} ({type(bad).__name__}).",
                    expected=f"{param}: {type_text(tp)}",
                    example=example,
                )

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def call(*args: Any, **kwargs: Any) -> Any:
            check(args, kwargs)
            return await fn(*args, **kwargs)

    else:

        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            check(args, kwargs)
            return fn(*args, **kwargs)

    call.__contract_checked__ = True  # type: ignore[attr-defined]
    return call


_MISSING = object()


# --- the typed stub -----------------------------------------------------------------------------------------------


def _annotation(tp: Any) -> str:
    if tp is inspect.Parameter.empty:
        return ""
    if isinstance(tp, str):
        return tp
    return typing_repr(tp)


def typing_repr(tp: Any) -> str:
    """*tp* as Python source (``list[str]``, ``Literal['a', 'b']``, ``int | None``)."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any:
        return "Any"
    if tp is None or tp is type(None):
        return "None"
    if origin is typing.Annotated:
        return typing_repr(args[0])
    if origin is typing.Literal:
        return f"Literal[{', '.join(repr(a) for a in args)}]"
    if origin in (typing.Union, types.UnionType):
        return " | ".join(typing_repr(a) for a in args)
    if origin is not None:
        name = getattr(origin, "__name__", str(origin))
        inner = ", ".join("..." if a is Ellipsis else typing_repr(a) for a in args)
        return f"{name}[{inner}]" if args else name
    if isinstance(tp, type):
        return tp.__name__
    return str(tp)


def _signature_text(fn: Callable[..., Any]) -> str:
    try:
        hints = typing.get_type_hints(fn, include_extras=False)
    except Exception:  # noqa: BLE001 - a stub still names the parameters
        hints = {}
    sig = inspect.signature(fn)
    params = []
    for p in sig.parameters.values():
        ann = hints.get(p.name, p.annotation)
        text = p.name if ann is inspect.Parameter.empty else f"{p.name}: {_annotation(ann)}"
        if p.kind is p.VAR_POSITIONAL:
            text = "*" + text
        elif p.kind is p.VAR_KEYWORD:
            text = "**" + text
        if p.default is not p.empty:
            text += f" = {p.default!r}" if ann is not inspect.Parameter.empty else f"={p.default!r}"
        params.append(text)
    # the positional-only and keyword-only markers
    out, kinds = [], [p.kind for p in sig.parameters.values()]
    for i, (text, kind) in enumerate(zip(params, kinds)):
        if kind is inspect.Parameter.KEYWORD_ONLY and (i == 0 or kinds[i - 1] not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        )):
            out.append("*")
        out.append(text)
        if kind is inspect.Parameter.POSITIONAL_ONLY and (i + 1 == len(kinds) or kinds[i + 1] is not kind):
            out.append("/")
    ret = hints.get("return", sig.return_annotation)
    arrow = "" if ret is inspect.Signature.empty else f" -> {_annotation(ret)}"
    return f"({', '.join(out)}){arrow}"


def _doc_block(obj: Any, indent: str) -> list[str]:
    doc = inspect.getdoc(obj)
    if not doc:
        return [f"{indent}..."]
    body = doc.replace('"""', '\\"\\"\\"').splitlines()
    return [f'{indent}"""{body[0]}'] + [f"{indent}{line}" if line else "" for line in body[1:]] + [f'{indent}"""']


def render_pyi(namespace: Mapping[str, Any], *, header: str = "") -> str:
    """The typed stub of *namespace* (name → function, frozen dataclass or class), in the given order, each with its
    docstring: the whole contract of a namespace, readable in one place."""
    lines = []
    if header:
        lines += [f"# {line}" if line else "#" for line in header.splitlines()] + [""]
    lines += ["from dataclasses import dataclass", "from typing import Any, Callable, Literal, Sequence", ""]
    for name, obj in namespace.items():
        if dataclasses.is_dataclass(obj) and isinstance(obj, type):
            frozen = obj.__dataclass_params__.frozen  # type: ignore[attr-defined]
            lines.append("@dataclass(frozen=True)" if frozen else "@dataclass")
            lines.append(f"class {name}:")
            lines += _doc_block(obj, "    ")
            try:
                hints = typing.get_type_hints(obj)
            except Exception:  # noqa: BLE001
                hints = {}
            for f in dataclasses.fields(obj):
                lines.append(f"    {f.name}: {_annotation(hints.get(f.name, f.type))}")
            for attr, member in vars(obj).items():
                if attr.startswith("_") or not callable(member):
                    continue
                lines.append(f"    def {attr}{_signature_text(member)}:")
                lines += _doc_block(member, "        ")
            lines.append("")
        elif inspect.isclass(obj):
            lines.append(f"class {name}:")
            lines += _doc_block(obj, "    ")
            lines.append("")
        elif callable(obj):
            target = inspect.unwrap(obj)
            lines.append(f"def {name}{_signature_text(target)}:")
            lines += _doc_block(target, "    ")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
