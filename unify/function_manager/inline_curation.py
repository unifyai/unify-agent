"""Inline curation (``UNIFY_INLINE_CURATION``): the actor stores and repairs skills during the task.

As shipped, the actor writes a function only when the user asks for one; what it works out on its own is left
to the review that follows the task. With this switch the actor's prompt says it may store a unit that ran
and worked during the task, and repair a stored function that failed on the spot, keeping its behaviour on the
inputs it already handled. Unchecked in-task writes have stored junk before (functions named ``_unused`` or
``nope``), so every write the actor makes itself goes through two mechanical guards, which the review's own
writes do not:

- **names**: a stored function's name must describe what it does: a public snake_case identifier of at least
  two words, at least one of which is a real word (three or more letters, not a placeholder such as ``tmp``,
  ``foo`` or ``unused``) -- ``parse_invoice_dates``, ``count_rows_by_status``. A call that adds or overwrites
  a function under any other name is refused whole, naming each such function and the rule;
- **resolution**: the storage check of ``UNIFY_STORE_CHECK=resolve`` (every name and ``primitives.*``
  reference resolves, the function loads) applies to the actor's writes whether or not that switch is on.

With ``UNIFY_FUNCTION_CASES`` on, its replay gate applies to an inline overwrite or patch as to any other.

Values: empty (as shipped), ``on`` (inline writes, and the post-task review still runs and sees what was
written through the store as usual) and ``only`` (inline writes, and no review: neither the one after the task
or each turn, nor ``store_skills``). ``UNIFY_STORE_ADMISSION=never`` cannot give the inline-only arm, because it
withholds the session's write tools too; and with any admission value the session's writes are withheld, so
this switch is ignored there and says so in the log.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import functools
import keyword
import logging
import re
from typing import Any, Callable, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)

ON = "on"
ONLY = "only"

# Words that say nothing about behaviour: a name made only of these (and short words) is a placeholder.
PLACEHOLDER_WORDS = frozenset(
    {
        "attempt",
        "bar",
        "baz",
        "debug",
        "draft",
        "dummy",
        "example",
        "fn",
        "foo",
        "func",
        "function",
        "helper",
        "misc",
        "new",
        "nope",
        "old",
        "placeholder",
        "qux",
        "sample",
        "scratch",
        "stuff",
        "temp",
        "test",
        "thing",
        "tmp",
        "todo",
        "try",
        "unused",
        "util",
        "utils",
        "wip",
        "xxx",
    },
)

_SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")

NAMING_RULE = (
    "a stored function's name must say what it does: snake_case, at least two words, a verb and what it acts "
    "on (e.g. `parse_invoice_dates`, `count_rows_by_status`), with at least one real word of three or more "
    "letters that is not a placeholder such as "
    + ", ".join(f"`{w}`" for w in ("tmp", "test", "foo", "unused", "helper"))
)

# True while the actor's own write runs: the storage check applies whatever UNIFY_STORE_CHECK says.
_INLINE_WRITE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unify_inline_curation_write",
    default=False,
)


def mode() -> str:
    from unify.settings import SETTINGS

    value = str(getattr(SETTINGS, "UNIFY_INLINE_CURATION", "") or "")
    return value if value in (ON, ONLY) else ""


def store_check_forced() -> bool:
    """Whether the current write is the actor's own, so the storage check applies."""
    return _INLINE_WRITE.get()


@contextlib.contextmanager
def inline_write() -> Iterator[None]:
    token = _INLINE_WRITE.set(True)
    try:
        yield
    finally:
        _INLINE_WRITE.reset(token)


def name_problem(name: str) -> Optional[str]:
    """Why *name* does not describe a function's behaviour, or ``None`` when it does."""
    if not name or not name.isidentifier() or keyword.iskeyword(name):
        return "it is not a Python identifier"
    if name.startswith("_"):
        return "it starts with an underscore, which marks a private or unused name"
    if name != name.lower():
        return "it is not lower-case snake_case"
    if "_" not in name:
        return "it is a single word"
    if not _SNAKE.match(name):
        return "it is not lower-case snake_case"
    words = name.split("_")
    if not any(
        len(w) >= 3 and w.isalpha() and w not in PLACEHOLDER_WORDS for w in words
    ):
        return "it is made only of placeholder or short words"
    return None


def _source_names(implementations: Union[str, List[str], None]) -> List[str]:
    """Names of the functions defined at the top of each source; unparsable sources are left to the store."""
    if isinstance(implementations, str):
        implementations = [implementations]
    names: List[str] = []
    for source in implementations or []:
        try:
            tree = ast.parse(str(source))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
    return names


def check_names(implementations: Union[str, List[str], None]) -> None:
    """Refuse the whole write when a function in it has a name that does not describe its behaviour."""
    bad = [(n, p) for n in _source_names(implementations) if (p := name_problem(n))]
    if not bad:
        return
    listed = "; ".join(f"`{n}`: {p}" for n, p in bad)
    raise ValueError(
        f"Nothing was stored. {listed}. In-task writes follow a naming rule: {NAMING_RULE}. Rename the "
        "function after what it does and add it again; if it is not meant to be reused, do not store it.",
    )


def guard_add_functions(fn: Callable[..., Any]) -> Callable[..., Any]:
    """*fn* (``add_functions``) with the naming rule and the storage check, for the actor's own writes."""

    @functools.wraps(fn)
    def add_functions(*args: Any, **kwargs: Any) -> Any:
        check_names(kwargs.get("implementations", args[0] if args else None))
        with inline_write():
            return fn(*args, **kwargs)

    return add_functions


def guard_patch_function(fn: Callable[..., Any]) -> Callable[..., Any]:
    """*fn* (``patch_function``) with the storage check, for the actor's own writes.

    The name already exists and a patch cannot change it, so only the storage check is added.
    """

    @functools.wraps(fn)
    def patch_function(*args: Any, **kwargs: Any) -> Any:
        with inline_write():
            return fn(*args, **kwargs)

    return patch_function
