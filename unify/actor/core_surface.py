"""What the actor needs before it starts, and what ``help(...)`` prints in a cell for a harness object.

``execute_code`` is the actor's only model-API tool; every other tool is a call ``tools.<name>(...)`` from a cell
(unify/actor/in_code_tools.py). The harness objects bound in a cell (``query_llm``, an environment's
``primitives``, ...) live in the harness and are reached through the sandboxed worker's proxy
(unify/actor/execution/worker.py); ``help(obj)`` on one of them is answered here.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Any, List, Optional


class ToolSurfaceError(RuntimeError):
    """A configuration the actor cannot run under."""


def require_prerequisites() -> None:
    """Refuse to start an actor the harness cannot confine.

    Raises :class:`ToolSurfaceError` naming what to change. Nothing falls back to running model code in this
    process.
    """
    from unify import sandbox

    try:
        sandbox.require_bwrap()
    except Exception as exc:
        raise ToolSurfaceError(
            "the actor runs every cell in the sandboxed worker, and bubblewrap "
            f"is not available: {exc}",
        ) from None


# ---------------------------------------------------------------------------
# help(): the documentation of a harness object, as text
# ---------------------------------------------------------------------------

# What a contract says about ``_``-prefixed parameters outside their own
# entries: a Returns bullet or a Raises entry about a mode only the harness
# selects (``_return_callable=True``, ``_also_return_metadata=True``), and the
# "When ``_x=False``:" prefix of the bullet that describes the default.
_PRIVATE_BULLET = re.compile(
    r"^(?P<i>[ \t]*)- When ``_\w+=True``[^\n]*\n(?:(?P=i)[ \t]+[^\n]*\n)*",
    re.M,
)
_PRIVATE_DEFAULT = re.compile(r"- When ``_\w+=False``: (?P<first>\w)")
_PRIVATE_RAISES = re.compile(
    r"^(?P<i>[ \t]*)\w+Error\n(?P=i)[ \t]+If ``_\w+[^\n]*\n"
    r"(?:(?P=i)[ \t]{5,}[^\n]*\n)*",
    re.M,
)
_EMPTY_RAISES = re.compile(r"\n[ \t]*Raises\n[ \t]*-+[ \t]*(?=\n[ \t]*\n|\n?\Z)")


def public_doc(doc: Optional[str]) -> str:
    """*doc* without what it says about ``_``-prefixed parameters (their
    entries, and the Returns and Raises text about the modes they select:
    the harness passes them, never a cell), cleaned."""
    text = inspect.cleandoc(doc or "")
    out: List[str] = []
    skipping = False
    skip_indent = 0
    blank = False
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if re.match(r"_[A-Za-z0-9_]* *[:(]", stripped):
            skipping, skip_indent, blank = True, indent, False
            continue
        if skipping:
            if stripped and indent <= skip_indent:
                skipping = False
                if blank:
                    # The blank line before the next section was skipped too.
                    out.append("")
            else:
                blank = blank or not stripped
                continue
        out.append(line)
    text = "\n".join(out).strip() + "\n"
    text = _PRIVATE_BULLET.sub("", text)
    text = _PRIVATE_DEFAULT.sub(lambda m: "- " + m.group("first").upper(), text)
    text = _PRIVATE_RAISES.sub("", text)
    return _EMPTY_RAISES.sub("", text).strip()


class _Hidden:
    """A default value help() does not show (a credential-named parameter's)."""

    def __repr__(self) -> str:
        return "<hidden>"


def _signature(obj: Any) -> str:
    try:
        sig = inspect.signature(obj, eval_str=True)
    except Exception:  # noqa: BLE001 - an annotation that does not evaluate
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return "(...)"
    from unify import sandbox

    params = [
        (
            p.replace(default=_Hidden())
            if p.default is not p.empty and sandbox.is_secret_name(p.name)
            else p
        )
        for p in sig.parameters.values()
        if not p.name.startswith("_")
    ]
    try:
        return str(sig.replace(parameters=params, return_annotation=sig.empty))
    except ValueError:
        return "(...)"


#: What ``help()`` with no argument prints in a cell.
INDEX_HELP = (
    "Objects the harness provides in this sandbox (all awaited):\n\n\n"
    "help(obj) prints the full documentation of one of them or of a method.\n"
)


def help_text(obj: Any, label: str) -> str:
    """What ``help(obj)`` prints in a cell for a harness object named ``label``.

    Signatures and docstrings only: never the value of an attribute, so a
    help call shows nothing a cell could not otherwise call.
    """
    lines: List[str] = []
    is_async = inspect.iscoroutinefunction(obj) or inspect.iscoroutinefunction(
        getattr(obj, "__call__", None),
    )
    if callable(obj) and not inspect.isclass(obj):
        target = obj if inspect.isroutine(obj) else getattr(obj, "__call__", obj)
        prefix = "await " if is_async else ""
        lines.append(f"{prefix}{label}{_signature(target)}")
        doc = inspect.getdoc(obj) or ""
        if doc:
            lines += ["", textwrap.indent(public_doc(doc), "    ")]
        return "\n".join(lines).rstrip() + "\n"
    lines.append(label)
    doc = inspect.getdoc(obj) or ""
    if doc:
        lines += ["", textwrap.indent(public_doc(doc), "    ")]
    methods: List[str] = []
    for attr in sorted(n for n in dir(obj) if not n.startswith("_")):
        try:
            member = getattr(type(obj), attr, None)
            if member is None or isinstance(member, property):
                continue
            value = getattr(obj, attr)
        except Exception:  # noqa: BLE001 - an unreadable member is left out
            continue
        if not callable(value):
            continue
        first = (public_doc(inspect.getdoc(value) or "").split("\n\n")[0]).strip()
        first = " ".join(first.split())
        prefix = "await " if inspect.iscoroutinefunction(value) else ""
        methods.append(f"  {prefix}{label}.{attr}{_signature(value)}")
        if first:
            methods.append(textwrap.indent(textwrap.fill(first, 72), "      "))
    if methods:
        lines += ["", "Methods:", *methods]
    return "\n".join(lines).rstrip() + "\n"
