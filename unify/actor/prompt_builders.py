"""The actor's system prompt.

Every agent reads the shared opening first (:func:`unify.common.tool_registry.preamble`: how cells work, the names
bound in them, ``tools.list()`` and ``tools.help(name)``); these sections say what is the actor's own: its role,
the answer, how it works, the clock, the workspace, its environments' own context and the caller's guidelines.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Mapping, Optional

if TYPE_CHECKING:
    from unify.actor.environments.base import BaseEnvironment

_WRAP = 72

#: The role's first sentence; with memory on, the second says where the library is.
_ROLE_FIRST = (
    "You solve the request in this conversation by writing and running\n"
    "Python cells with `execute_code`."
)
_ROLE_FIRST_MEMORY = (
    "You solve the request in this conversation by writing and running\n"
    "Python cells with `execute_code`. A library of functions and notes from\n"
    "earlier work, when one exists, is `memory` in your cells, already bound."
)
#: The one answer rule: the final text reply is the answer, and a cell gives it with ``tools.answer``.
_ANSWER_RULE = (
    "Your answer is your final reply: a message without a tool call. "
    "`tools.answer(answer)` gives it from a cell. When the requester defines a "
    "format for replies (a JSON object, a keyword, a fixed template), each reply "
    "follows that format exactly."
)

_ANSWER_STRUCTURED = (
    "### Answer\n\n"
    "When the request is addressed, answer by calling `tools.answer(answer)`\n"
    "with the answer in the required format."
)

# The bound globals, named once; each one's contract is its ``help(...)``.
_SANDBOX = (
    "Cells run with these globals already bound, no import needed: `tools` "
    "(your harness tools; `tools.list()` lists them), `display(obj)` (rich "
    "output, such as an image, that comes back to you as visual input), "
    "`query_llm(...)` and `list_llms()` (semantic LLM calls from code), "
    "`unillm` (direct LLM use beyond `query_llm`) and "
    "`run_coro_sync(factory)`; `help(name)` documents each."
)
# The query_llm doctrine: the method choice stays; how a call is made is its help.
_QUERY_LLM = textwrap.dedent("""
    Use plain Python for exact steps (lookups, filters, arithmetic,
    reshaping) and `query_llm(...)` inside the code for steps that judge
    meaning (classify, extract, summarise, draft).
""").strip()

_PYTHON_FIRST = textwrap.dedent("""
    ### Python First

    Prefer Python packages over shell CLI tools; installed packages stay for
    every later task. When a task genuinely needs a CLI, run it with
    `tools.sh(command)` or from Python with `subprocess`, and work with its
    output as data.
""").strip()

_EXECUTION_RULES = textwrap.dedent("""
    ### Execution

    1. **Sessions**: bind results to variables and build on them, and print
       only what the next decision needs.
    2. **Async**: the runtime owns the event loop, so code `await`s (and a
       sync facade uses the injected `run_coro_sync(factory)`) rather than
       calling `asyncio.run(...)`; `asyncio.gather` runs independent I/O
       concurrently.
    3. **Evidence**: a step that ran is not a step that worked. The result
       of a mutation or an extraction (a return value, a re-read) shows
       whether it worked; when it contradicts the expected result, fix and
       re-run.
    4. **Provenance**: when a source fails, the reply says so; what you
       recall or make up is not presented as sourced data.
""").strip()

_VERIFY_BEFORE_SCALING = textwrap.dedent("""
    ### Verify Before Scaling

    Run a loop body once and check its result before iterating over many
    items.
""").strip()


def _refill(text: str) -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=_WRAP,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _role() -> str:
    from unify.memory.integration.switch import memory_on
    from unify.settings import SETTINGS

    first = _ROLE_FIRST_MEMORY if memory_on(SETTINGS) else _ROLE_FIRST
    return f"### Role\n\n{first}\n\n{_refill(_ANSWER_RULE)}"


def _sandbox_section() -> str:
    return f"### Sandbox Environment\n\n{_refill(_SANDBOX)}\n\n{_QUERY_LLM}"


def _clock() -> str:
    """The current time, sampled when the prompt is built: the clock a cell's own reading never overrides."""
    from unify.common import prompt_helpers

    return textwrap.dedent(f"""
        ### Current Time

        The current date and time is **{prompt_helpers.now()}**. Use this
        clock for "today" and similar, not a clock read in a cell.
    """).strip()


def _workspace() -> str:
    # The path cells see: the sandbox binds the workspace there, so the host
    # path (which can name the run) never reaches the model.
    from unify.sandbox import WORKSPACE_MOUNT

    return (
        "### Workspace\n\n"
        f"Your working directory is `{WORKSPACE_MOUNT}`; it persists across "
        "tasks. Write files there, with absolute paths."
    )


def _environment_contexts(environments: Mapping[str, "BaseEnvironment"]) -> str:
    """Each environment's own rules, docs and examples."""
    parts = [env.get_prompt_context() for env in environments.values()]
    return "\n\n---\n\n".join(p for p in parts if p and p.strip()).strip()


def build_code_act_prompt(
    *,
    environments: Mapping[str, "BaseEnvironment"],
    structured: bool = False,
    guidelines: Optional[str] = None,
) -> str:
    """The actor's system prompt: a pure function of the configuration and the clock, so its prefix stays stable
    for the run.

    Args:
        environments: The namespaces bound in the actor's cells; each adds its own context.
        structured: A response format is set, so the answer is a ``tools.answer`` call.
        guidelines: The caller's guidelines, the last section.
    """
    parts: list[str] = [_role()]
    if structured:
        parts.append(_ANSWER_STRUCTURED)
    parts += [
        _sandbox_section(),
        _PYTHON_FIRST,
        _EXECUTION_RULES,
        _VERIFY_BEFORE_SCALING,
        _clock(),
        _workspace(),
        _environment_contexts(environments),
    ]
    if guidelines:
        parts.append(
            f"### Guidelines\n\n"
            f"Follow these guidelines throughout this session:\n\n"
            f"{guidelines}",
        )
    return "\n\n".join(p for p in parts if p and p.strip())
