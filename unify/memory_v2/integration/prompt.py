"""The memory section at the end of the cached system prompt (spec §G2, G6–G9; v2.1 surfacing).

Memory is a Python library the working model uses like any well-kept internal package, so the prompt
holds one short paragraph (where the library is, how to import from it, the README, ``help()``,
``memory.find`` and ``memory.describe``, "check the example before relying on it", and that edits to the
scratch copy are discarded but recorded) and then the **channel catalogue**: one line per channel with its
function and note counts, never one line per function. Per-function detail lives in the generated
``README.md`` and ``.memory/catalog.json`` of the export (:mod:`..catalogue`), read on demand.

The text is a pure function of the exported memory commit, the export's fixed path under ``UNIFY_HOME`` and
the harness's suspect set (which changes only at drift events, between requests): sorted, with no
timestamps, counters or request text. Two requests on the same ``main`` therefore send byte-identical
system prompts, and nothing request-specific ever enters the prefix. There is no size cut: the section
grows by one line per channel, and the catalogue's soft budget is the gate's (G4), which flags hygiene and
never freezes growth.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..catalogue import README, channel_lines

GUIDE = (
    "Memory: a library of Python functions distilled from earlier work, at `{root}` (first on the "
    "import path; import with `from env.<channel> import <function>`). Its functions are candidates to "
    "check, not authority. `{root}/{readme}` lists every function with its signature, one-line summary "
    "and input form; `help(env.<channel>)` and `help(<function>)` show the documentation, each with a "
    "runnable example. For data you hold (a file path, bytes, text or a parsed value), `import memory` "
    "and call `memory.find(value)`: it lists the functions whose recorded inputs have the same shape; "
    "`memory.describe(<function>)` shows one function's documentation. Check a function's example before "
    "relying on it. A function raises MemoryInputError when its input differs from what it was built "
    "from, and the message says what it expected; then do the work directly. The library is a scratch "
    "copy: you may propose an improvement by editing a function there or adding "
    "`{root}/proposals/<name>.md`; the copy is discarded when the request ends, but what you wrote is "
    "recorded for the next consolidation.\n"
)


def render_memory_section(checkout: Path, suspect: Iterable[str] = ()) -> str:
    """The memory section for the export at *checkout*, or ``""`` when the library lists nothing."""
    checkout = Path(checkout)
    lines = channel_lines(checkout, suspect)
    if not lines:
        return ""
    return GUIDE.format(root=checkout, readme=README) + "\nChannels:\n" + lines
