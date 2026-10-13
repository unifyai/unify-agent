"""The contract every model-facing call meets, as checks a test applies to a namespace.

* :func:`contract_gaps`: a public call with a type hint on every parameter and on its return, a docstring with its
  arguments, its return, the errors it raises and an example call, checked at the call (``contract_runtime.checked``).
* :func:`error_gaps`: an error that teaches: the call as written, what was wrong, then what was expected with an
  example call, or the right call; nothing unfilled, and no host path.
"""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Callable, Mapping
from typing import Any

from unify.common.contract_runtime import ContractError, example_of

#: Text a model must never read in an error: an unfilled template field, or a path on the host.
_UNFILLED = re.compile(r"\{[a-z_]+\}")
_HOST_PATH = re.compile(r"/(?:home|tmp/claude|root|Users)/")


def contract_gaps(name: str, fn: Callable[..., Any]) -> list[str]:
    """What *fn* (the public call *name*) lacks of the contract; empty when it meets it."""
    gaps: list[str] = []
    target = inspect.unwrap(fn)
    if not getattr(fn, "__contract_checked__", False):
        gaps.append(f"{name}: its arguments are not checked at the call (contract_runtime.checked)")
    try:
        hints = typing.get_type_hints(target)
    except Exception as exc:  # noqa: BLE001
        return [*gaps, f"{name}: its type hints do not resolve ({type(exc).__name__}: {exc})"]
    params = [
        p
        for p in inspect.signature(target).parameters.values()
        if p.name not in ("self", "cls")
    ]
    for p in params:
        if p.name not in hints:
            gaps.append(f"{name}: parameter {p.name!r} has no type hint")
    if "return" not in hints:
        gaps.append(f"{name}: no return type")
    doc = inspect.getdoc(target) or ""
    if not doc:
        gaps.append(f"{name}: no docstring")
    else:
        if params and not re.search(r"(?m)^\s*Args:\s*$", doc):
            gaps.append(f"{name}: the docstring has no Args section")
        if not re.search(r"(?m)^\s*Returns:\s*$", doc):
            gaps.append(f"{name}: the docstring has no Returns section")
        if not re.search(r"(?m)^\s*Raises:\s*$", doc):
            gaps.append(f"{name}: the docstring has no Raises section")
        if example_of(target) is None:
            gaps.append(f"{name}: the docstring has no Example")
        for p in params:
            if params and not re.search(rf"(?m)^\s*\*{{0,2}}{re.escape(p.name)}\b.*:", doc):
                gaps.append(f"{name}: the Args section does not describe {p.name!r}")
    return gaps


def namespace_gaps(namespace: Mapping[str, Callable[..., Any]]) -> list[str]:
    """:func:`contract_gaps` of every call of *namespace*."""
    return [gap for name, fn in namespace.items() for gap in contract_gaps(name, fn)]


def error_gaps(exc: BaseException) -> list[str]:
    """What *exc* lacks of an error that teaches; empty when it has every part."""
    if not isinstance(exc, ContractError):
        return [f"{type(exc).__name__} is not a ContractError: {str(exc)[:200]}"]
    gaps = []
    text = str(exc)
    if not exc.call or "(" not in exc.call:
        gaps.append("no call as written")
    if not exc.problem:
        gaps.append("no statement of what was wrong")
    if not ((exc.expected and exc.example) or exc.hint):
        gaps.append("neither what was expected with an example, nor the right call")
    if _UNFILLED.search(text):
        gaps.append(f"an unfilled template field: {_UNFILLED.search(text).group(0)}")
    if _HOST_PATH.search(text):
        gaps.append("a host path")
    return gaps
