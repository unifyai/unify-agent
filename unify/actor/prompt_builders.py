from __future__ import annotations

import re
import textwrap
from typing import (
    Callable,
    Optional,
    Mapping,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from unify.actor.core_surface import PromptSurface
    from unify.actor.environments.base import BaseEnvironment

# ---------------------------------------------------------------------------
# Static prompt content (inlined rather than wrapped in trivial functions)
# ---------------------------------------------------------------------------


_TOOL_SELECTION = textwrap.dedent("""
    ### Tool Selection: `execute_function` vs `execute_code`

    - One exact function or primitive call is
      `execute_function(function_name="...", call_kwargs={...})`. Reach
      for `execute_code` only for genuine multi-step composition
      (branching, loops, combining intermediate results); a
      `print()`, `await handle.result()`, or temporary variable around a
      single call is boilerplate, not composition.
    - Procedures are **not** primitives — use the GuidanceManager JSON
      tools (`GuidanceManager_search`, `GuidanceManager_add_guidance`, …)
      directly.
""").strip()


# Sandbox Environment lines that exist only where an environment injects
# primitives: an actor without them cannot spawn a sub-actor, so its prompt
# must not offer one.
_PRIMITIVES_GLOBAL_ROW = "| `primitives` | `await primitives.actor.act(...)` spawns a sub-actor; `help(primitives.actor.act)` reads its live docs |\n"


# ---------------------------------------------------------------------------
# UNIFY_REPLY_CHANNEL: the reply rule
# ---------------------------------------------------------------------------
# Where the prompt states that the answer is a reply without a tool call, it
# can say in one more sentence that a cell can send the reply with
# ``reply(text)``. The sentences around it are made consistent with it; the
# requester's own text is never changed. Every text below is built from the
# shipped one, which is returned unchanged while the switch is off.

# UNIFY_REPLY_CHANNEL=code+text: a cell can send the reply.
_REPLY_FROM_CELL = (
    "You can also reply from a cell with `reply(text)`, for example "
    "`reply(answer)` when the answer is in a variable; it ends your turn."
)
_WRAP = 72


def _reply_from_cell() -> bool:
    from unify.common._async_tool import cell_reply

    return cell_reply.enabled()


def _reply_rule_additions() -> list[str]:
    """The sentences that follow the statement of the reply rule."""
    added: list[str] = []
    if _reply_from_cell():
        added.append(_REPLY_FROM_CELL)
    return added


def _refill(text: str, *, indent: str = "") -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=_WRAP,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


_LEAN_ROLE_REPLY_RULE = (
    "Your answer is your final reply: a message without a tool call. When\n"
    "the requester defines a format for replies (a JSON object, a keyword, a\n"
    "fixed template), each reply follows that format exactly."
)


def _lean_role() -> str:
    """The lean profile's role, with the reply rule as the switches word it."""
    added = _reply_rule_additions()
    if not added:
        return _LEAN_ROLE
    governs = "each reply"
    rule = " ".join(
        [
            "Your answer is your final reply: a message without a tool call.",
            *added,
            "When the requester defines a format for replies (a JSON object, "
            f"a keyword, a fixed template), {governs} follows that format "
            "exactly.",
        ],
    )
    return _unified(_LEAN_ROLE, _LEAN_ROLE_REPLY_RULE, _refill(rule))


# UNIFY_BIND_REQUEST=on: a cell reads the current request as ``request``
# (unify/common/_async_tool/bound_request.py), said once, before the table of
# the injected globals.
_BIND_REQUEST_LINE = (
    "The current request is available in cells as `request`: `request.text`\n"
    "is its text and `request.data` the JSON values it contains, in order."
)
_GLOBALS_TABLE = "| Global | What it is |"


def _with_bound_request(section: str) -> str:
    from unify.common._async_tool import bound_request

    if not bound_request.enabled():
        return section
    return section.replace(
        f"\n\n{_GLOBALS_TABLE}",
        f"\n\n{_BIND_REQUEST_LINE}\n\n{_GLOBALS_TABLE}",
        1,
    )


def _build_sandbox_environment_section(*, has_primitives: bool) -> str:
    """The sandbox section; UNIFY_BIND_REQUEST=on says a cell reads the
    request as ``request``, before the table of globals."""
    return _with_bound_request(
        _shipped_sandbox_environment_section(has_primitives=has_primitives),
    )


def _shipped_sandbox_environment_section(*, has_primitives: bool) -> str:
    """One table of the actually injected sandbox globals + the query_llm doctrine.

    The globals table mirrors ``create_execution_globals()``
    (``unify/function_manager/execution_env.py``) plus the per-execution
    ``display`` injection (``unify/actor/execution/session.py``) — if a
    global is added or removed there, update the table. ``primitives``
    and the sub-agent notch of the query_llm dial appear only when
    *has_primitives*, i.e. an environment injects primitives; otherwise
    the sandbox's ``primitives`` exposes nothing. Full contracts live in
    the callables' docstrings behind ``help(...)``; signatures are
    introspected so this block never drifts from the callables.
    """
    import inspect as _inspect

    from unify.common.reasoning import list_llms, query_llm

    query_prefix = "async def " if _inspect.iscoroutinefunction(query_llm) else "def "
    query_signature = (
        f"{query_prefix}{query_llm.__name__}{_inspect.signature(query_llm)}"
    )
    list_signature = f"def {list_llms.__name__}{_inspect.signature(list_llms)}"

    # Dedented before formatting, so an optional block is spliced in at
    # column 0 and leaves no gap when absent.
    template = textwrap.dedent("""
        ### Sandbox Environment

        Python in `execute_code` and stored functions runs with the
        injected globals below. Find stored functions with the
        `FunctionManager_search_functions` JSON tool, then read live docs
        in-sandbox with `help(...)` — do not guess signatures. `help` and
        `dir` are builtins; `import inspect` first for
        `inspect.signature(...)`.

        | Global | What it is |
        |--------|------------|
        {primitives_row}| `display` | `display(obj)` emits rich output — use it over `print(...)` for images; whatever you `display()` comes back as visual input next turn — inspect it directly, no separate vision/observe call |
        | `query_llm` / `list_llms` | Semantic LLM calls from code (doctrine below); full contract `help(query_llm)`, endpoints `list_llms()` |
        | `run_coro_sync` | Drives a coroutine factory from a sync façade under the already-running loop |
        | `unillm` | Advanced direct LLM usage beyond `query_llm` |

        ```python
        {query_signature}
        {list_signature}
        ```

    """)
    # The globals table and signatures; the doctrine short.
    lean = template.format(
        primitives_row=_PRIMITIVES_GLOBAL_ROW if has_primitives else "",
        query_signature=query_signature,
        list_signature=list_signature,
    ).strip()
    doctrine = _LEAN_QUERY_LLM
    if has_primitives:
        doctrine = f"{doctrine} {_LEAN_SUB_ACTOR_DIAL}"
    return f"{lean}\n\n{doctrine}"


def _unified(text: str, old: str, new: str) -> str:
    """``text`` with ``old`` replaced by ``new``; ``old`` must occur exactly once."""
    if text.count(old) != 1:
        raise ValueError(f"expected one occurrence of {old!r}")
    return text.replace(old, new)


def _build_clock_context() -> str:
    """State the assistant's current time so a plan never has to discover it.

    Read through the shared helper at build time, so it is the same clock
    every other prompt shows and it is authoritative for "today" and "this
    week": code run inside ``execute_code`` may execute on a host whose own
    clock is set differently.
    """
    from unify.common import prompt_helpers

    return textwrap.dedent(f"""
        ### Current Time

        The current date and time is **{prompt_helpers.now()}**. This is the
        assistant's clock; resolve "today", "this week" and similar against
        it, and prefer it over any clock read inside `execute_code`, which
        may run on a host set differently.
    """).strip()


def _clock_in_first_message() -> bool:
    """``UNIFY_CLOCK_PLACEMENT=first_message``."""
    from unify.settings import SETTINGS

    return SETTINGS.UNIFY_CLOCK_PLACEMENT == "first_message"


def first_message_clock_line() -> str:
    """``UNIFY_CLOCK_PLACEMENT=first_message``: the line that opens the
    session's first user message, sampled now; empty otherwise.

    The host's reading, not an authority: what the work itself says about
    dates comes first.
    """
    if not _clock_in_first_message():
        return ""
    from unify.common import prompt_helpers

    return (
        f"The host clock reads {prompt_helpers.now()}. Dates stated in the "
        "request or in the files and records you work with take precedence."
    )


def _session_sections() -> list[str]:
    """The per-session sections of the system prompt, sampled now: the
    clock (unless ``UNIFY_CLOCK_PLACEMENT`` moves it to the first user
    message) and the filesystem context."""
    if _clock_in_first_message():
        return [_build_filesystem_context()]
    return [_build_clock_context(), _build_filesystem_context()]


def _build_filesystem_context() -> str:
    from unify.workspace import get_local_root

    resolved = get_local_root()
    return (
        "### Workspace\n\n"
        f"Your working directory is `{resolved}`; it persists across "
        "tasks. Write files there, with absolute paths."
    )


# ---------------------------------------------------------------------------
# Private helpers with real logic
# ---------------------------------------------------------------------------


def _build_code_act_rules_and_examples(
    *,
    environments: Mapping[str, "BaseEnvironment"],
) -> str:
    """
    Builds the reusable environment rules block for CodeAct-style execution.

    Composes the prompt context each environment provides.
    """
    parts: list[str] = []

    # Each environment provides its own rules, docs, and examples.
    for _ns, env in environments.items():
        env_ctx = env.get_prompt_context()
        if env_ctx and env_ctx.strip():
            parts.append(env_ctx)

    return "\n\n---\n\n".join(p for p in parts if p and p.strip()).strip()


# ---------------------------------------------------------------------------
# The lean profile
# ---------------------------------------------------------------------------
# For a non-interactive session: one requester, nobody reading progress
# notifications. The sections describe the session's mechanisms and state
# few rules; the requester's reply format comes first.


_LEAN_ROLE = textwrap.dedent("""
    ### Role

    You solve the request in this conversation by writing and running
    Python, with a library of stored functions and procedures to draw on.

    Your answer is your final reply: a message without a tool call. When
    the requester defines a format for replies (a JSON object, a keyword, a
    fixed template), each reply follows that format exactly.
""").strip()

_LEAN_QUERY_LLM = textwrap.dedent("""
    Use plain Python for exact steps (lookups, filters, arithmetic,
    reshaping) and `query_llm(...)` inside the code for steps that judge
    meaning (classify, extract, summarise, draft). Each `query_llm` call is
    stateless, so its prompt carries all the evidence it needs; pass a
    Pydantic `response_format=` when the code branches on the result.
""").strip()

_LEAN_SUB_ACTOR_DIAL = (
    "A sub-agent (`primitives.actor.act`) suits a sub-task whose plan must "
    "be discovered while it runs."
)


_LEAN_EXECUTION_RULES = textwrap.dedent("""
    ### Execution

    1. **Sessions**: cells share one persistent sandbox for the task, like
       a notebook: bind results to variables and build on them, and print
       only what the next decision needs. `list_sessions()` and
       `inspect_state()` show live sessions and names;
       `state_mode="stateless"` or a named session isolates a cell. A
       `NameError` on a known name usually means the sandbox restarted.
    2. **Async**: the runtime owns the event loop, so code `await`s (and a
       sync facade uses the injected `run_coro_sync(factory)`) rather than
       calling `asyncio.run(...)`; `asyncio.gather` runs independent I/O
       concurrently.
    3. **Structured outputs**: Pydantic models defined in the code need
       `model_rebuild()` on the outermost model.
    4. **Evidence**: a step that ran is not a step that worked. The result
       of a mutation or an extraction (a return value, a re-read) shows
       whether it worked; when it contradicts the expected result, fix and
       re-run.
    5. **Final reply**: when the request is addressed, reply without a tool
       call, in the requester's format when it defines one.
    6. **Provenance**: when a source fails, the reply says so; records
       generated from memory are not presented as sourced data.
""").strip()

_LEAN_CLARIFICATION_RULE = (
    "7. **Clarification**: `request_clarification` asks the requester a\n"
    "   question and waits for the answer."
)

_LEAN_INCREMENTAL_EXECUTION = textwrap.dedent("""
    ### Verify Before Scaling

    Run a loop body once and check its result before iterating over many
    items. `state_mode="read_only"` tries an alternative on the current
    state without changing it.
""").strip()


def _lean_execution_rules(can_clarify: bool) -> str:
    if can_clarify:
        return f"{_LEAN_EXECUTION_RULES}\n{_LEAN_CLARIFICATION_RULE}"
    return _LEAN_EXECUTION_RULES


def _injects_actor_primitives(environments: Mapping[str, "BaseEnvironment"]) -> bool:
    """Whether an environment puts the actor primitive in the sandbox.

    Any environment in the ``primitives`` namespace does, except the one for
    the namespaces an environment registered (``UNIFY_ENV_NAMESPACES``): an
    actor holding only those may not spawn a sub-actor. With the switch off
    this is ``"primitives" in environments``.
    """
    env = environments.get("primitives")
    if env is None:
        return False
    from unify.actor.environments.environment_namespaces import (
        EnvironmentNamespacesEnvironment,
    )

    members = getattr(env, "sub_environments", None) or [env]
    return not all(isinstance(e, EnvironmentNamespacesEnvironment) for e in members)


# ---------------------------------------------------------------------------
# UNIFY_PROMPT_TRIM and UNIFY_STATEFUL_CELLS
# ---------------------------------------------------------------------------
# Both rewrite the static sections (never an environment's own context, the
# guidelines or the per-session sections). Each rewrite applies where its
# text occurs: a section holds only some of them.

# UNIFY_PROMPT_TRIM, no environment in the ``primitives`` namespace: nothing
# in the sandbox is a primitive.
_TRIM_NO_PRIMITIVES = (
    (re.compile(r"\(`primitives\.\*`, `query_llm`, …\)"), "(`query_llm`, …)"),
    (
        re.compile(r"not functions, tools or `primitives\.\*` methods"),
        "not functions or tools",
    ),
    (
        re.compile(
            r"stored function or primitive by name\.",
        ),
        "stored function by name.",
    ),
    (
        re.compile(r" primitive calls,(?P<ws>\s+)filters"),
        lambda m: m.group("ws") + "filters",
    ),
    (re.compile(r"`print\(\)`, `await handle\.result\(\)`, or"), "`print()` or"),
    (
        re.compile(r"Procedures are \*\*not\*\* primitives —"),
        "Procedures are **not** callable —",
    ),
    (
        re.compile(r"function(?P<ws>\s+)or\s+primitive\s+call"),
        lambda m: "function" + ("\n" if "\n" in m.group(0) else " ") + "call",
    ),
)
# ... and no environment at all: nothing is documented in the prompt for
# search to leave out, and search finds stored functions only.
_TRIM_DISCOVERY_SCOPE = re.compile(
    r"\*\*Discovery index scope:\*\*.*?via `execute_function`\.\n\n",
    re.DOTALL,
)
_TRIM_PRIMITIVE_CATALOGUE = (
    re.compile(
        r"Function search covers user-stored functions"
        r"\s+\*\*and\*\* the built-in `primitives\.\*` catalogue — primitive rows come back"
        r"\s+with `is_primitive`, `argspec`, and `docstring`\.",
    ),
    "Function search covers user-stored functions.",
)
# No sub-actor primitive: the actor cannot delegate.
_TRIM_NO_DELEGATE = (
    (
        re.compile(r"no code, search or\s+sub-agent can take them for you"),
        "no code or search\ncan take them for you",
    ),
    (
        re.compile(
            r" Do not delegate a sub-task whose result would be\s+such an action\.",
        ),
        "",
    ),
)
# There are no sub-agents (delegation through primitives.actor is off).
_TRIM_NO_SUB_AGENTS = (
    re.compile(
        r" If you are a sub-agent and your task seems to need one,"
        r"\s+say so in your result instead of calling a function that does not"
        r"\s+exist\.",
    ),
    "",
)
# No list_sessions / inspect_state tools.
_TRIM_NO_SESSION_TOOLS = (
    (
        re.compile(
            r"`list_sessions\(\)` and\s+`inspect_state\(\)` show live sessions and"
            r" names;\s+",
        ),
        "",
    ),
    (
        re.compile(
            r"`list_sessions\(\)` / `inspect_state\(\)` rediscover live sessions"
            r"\s+and names — variables survive",
        ),
        "Variables survive",
    ),
)


# UNIFY_STATEFUL_CELLS: every cell runs in the task's one session, and the
# session tools are not offered (unify/actor/cell_state.py), so the prompt
# names no cell mode and no session tool.
_STATEFUL_CELLS = (
    (
        re.compile(
            r"\s*`list_sessions\(\)` and\s+`inspect_state\(\)` show live sessions and"
            r'\s+names;\s+`state_mode="stateless"` or a\s+named session isolates a'
            r" cell\.",
        ),
        "",
    ),
    (
        re.compile(
            r'\s*`state_mode="stateless"` or a\s+named session isolates a cell\.',
        ),
        "",
    ),
    (
        re.compile(
            r'\s*`state_mode="read_only"` tries an alternative on the current'
            r"\s+state without changing it\.",
        ),
        "",
    ),
    (
        re.compile(
            r"`list_sessions\(\)` / `inspect_state\(\)` rediscover live sessions"
            r"\s+and names — variables survive",
        ),
        "Variables survive",
    ),
    (
        re.compile(
            r'Isolate a cell with\s+`state_mode="stateless"` or a named session;'
            r"\s+fan out",
        ),
        "Fan out",
    ),
    (
        re.compile(
            r"\*\*Read-only for exploration\*\*: branch off known-good state with"
            r'\s+`state_mode="read_only"` to try alternatives without risk\.\s+',
        ),
        "",
    ),
    (
        re.compile(
            r"Functions support execution mode overrides independent of the"
            r" session's\s+`state_mode`:",
        ),
        "Functions called in a cell support execution mode overrides:",
    ),
)


def _section_rewrites(
    environments: Mapping[str, "BaseEnvironment"],
    tools: Optional[Mapping[str, Callable]],
) -> list:
    """The (pattern, replacement) pairs the switches apply to static sections."""
    from unify.actor import cell_state

    rewrites: list = []
    # First: it removes the whole session sentence the trim would shorten.
    if cell_state.enabled():
        rewrites.extend(_STATEFUL_CELLS)
    if "primitives" not in environments:
        rewrites.extend(_TRIM_NO_PRIMITIVES)
        if not environments:
            rewrites.append((_TRIM_DISCOVERY_SCOPE, ""))
        else:
            rewrites.append(_TRIM_PRIMITIVE_CATALOGUE)
    if not _injects_actor_primitives(environments):
        rewrites.extend(_TRIM_NO_DELEGATE)
    # No sub-agents: helpers come from the agent record.
    rewrites.append(_TRIM_NO_SUB_AGENTS)
    if tools is not None and not (
        "list_sessions" in tools and "inspect_state" in tools
    ):
        rewrites.extend(_TRIM_NO_SESSION_TOOLS)
    return rewrites


def _rewrite_sections(parts: list[str], rewrites: list) -> list[str]:
    if not rewrites:
        return parts
    out = []
    for part in parts:
        for pattern, replacement in rewrites:
            part = pattern.sub(replacement, part)
        out.append(part)
    return out


def build_code_act_prompt(
    *,
    environments: Mapping[str, "BaseEnvironment"],
    core: "PromptSurface",
    can_store: bool = False,
    guidelines: Optional[str] = None,
    persist: bool = False,
    library_read_only: bool = False,
    session_sections: bool = True,
) -> str:
    """Build the system prompt for the CodeActActor.

    Assembles prompt sections in a fixed order, skipping sections that
    don't apply to the current configuration. This is intentionally a
    pure prompt builder (no side effects). Gating keys on assistant
    config only — never on task text — so the prompt stays a pure
    function of config and the cache prefix stays stable.

    Parameters
    ----------
    core:
        What the session's sandbox holds (unify/actor/core_surface.py). The
        prompt names its objects in a short index and describes
        ``execute_code`` as the only tool.
    persist:
        When ``True``, the skill-storage notice describes the persistent
        session's schedule; when ``False`` (one-shot act), the review that
        follows the result.
    library_read_only:
        When ``True`` (an admission-gated session), states that the
        libraries cannot be written during the session and that a review
        after it runs only when an external check admits it.
    session_sections:
        When ``False``, the per-session sections (the clock and the
        workspace) are left out, so the prompt is the same for every session
        of one configuration.
    """
    return _build_core_prompt(
        core,
        environments=environments,
        can_store=can_store,
        guidelines=guidelines,
        persist=persist,
        library_read_only=library_read_only,
        session_sections=session_sections,
    )


# ---------------------------------------------------------------------------
# The core tool surface
# ---------------------------------------------------------------------------

_CORE_SANDBOX_SEARCH = (
    "Find stored functions with the\n"
    "`FunctionManager_search_functions` JSON tool, then read live docs\n"
    "in-sandbox with `help(...)`"
)
_CORE_SANDBOX_SEARCH_PYTHON = (
    "Find stored functions with\n"
    "`await functions.search(...)`, then read live docs in-sandbox with\n"
    "`help(...)`"
)


# The lean rules' pointer to the session tools, which the core surface does not have.
_LEAN_RULE_SESSIONS = (
    "`list_sessions()` and\n"
    "   `inspect_state()` show live sessions and names;\n"
    '   `state_mode="stateless"` or a named session isolates a cell.'
)
_LEAN_RULE_SESSIONS_CORE = (
    '`state_mode="stateless"` or a\n   named session isolates a cell.'
)


def _build_core_prompt(
    core: "PromptSurface",
    *,
    environments: Mapping[str, "BaseEnvironment"],
    can_store: bool,
    guidelines: Optional[str],
    persist: bool,
    library_read_only: bool,
    session_sections: bool,
) -> str:
    """The system prompt of a core-surface session: the lean profile's
    sections, less what names JSON tools the session does not have."""
    can_clarify = core.clarification
    parts: list[str] = [_lean_role()]
    tools = core.tools_section()
    parts.append(tools)
    sandbox = _build_sandbox_environment_section(
        has_primitives=_injects_actor_primitives(environments),
    )
    parts.append(_unified(sandbox, _CORE_SANDBOX_SEARCH, _CORE_SANDBOX_SEARCH_PYTHON))
    parts.append(core.index())
    parts.append(core.python_first())
    parts.append(
        _unified(
            _lean_execution_rules(can_clarify),
            _LEAN_RULE_SESSIONS,
            _LEAN_RULE_SESSIONS_CORE,
        ),
    )
    parts.append(_LEAN_INCREMENTAL_EXECUTION)
    if core.functions or core.guidance:
        library = core.library_section()
        parts.append(library)
        if library_read_only:
            parts.append(core.read_only_notice())
    if can_store:
        parts.append(core.storage_notice(persist=persist))
    parts = _rewrite_sections(parts, _section_rewrites(environments, None))
    if session_sections:
        parts.extend(_session_sections())
    rules_and_examples = _build_code_act_rules_and_examples(environments=environments)
    if rules_and_examples:
        parts.append(rules_and_examples)
    if guidelines:
        parts.append(
            f"### Guidelines\n\n"
            f"Follow these guidelines throughout this session:\n\n"
            f"{guidelines}",
        )
    return "\n\n".join(p for p in parts if p and p.strip())
