"""``UNIFY_CODE_PROJECTION=notebook``: the code tool as a notebook cell.

As shipped the model fills ``execute_code``'s ``thought`` (required, "shown
to the user"), ``state_mode``, ``session_id``, ``session_name`` and, in a
sandboxed workspace, ``language``. A cheap model fills every field it is
offered: ``state_mode`` was sent in 3,987 of 3,987 recorded calls, mostly as
``"stateless"``, so each cell started empty and the model typed its data in
again; and the required ``thought`` became the place to announce the next
step while the cell only printed a sentence.

With the switch on, the model sees one field, ``code``. Where a cell runs is
written in the cell itself, as Jupyter magics on its first lines:

========================  ==================================================
``%%bash``                ``language="bash"`` (sandboxed workspace only)
``%pip install PKG ...``  the harness-side install, then the rest of the cell
``%%scratch``             ``state_mode="stateless"``
``%%what_if``             ``state_mode="read_only"`` on this notebook
``%%session NAME``        ``state_mode="stateful", session_name=NAME``
``%sessions``             the session tools' data, as cell output
========================  ==================================================

``%%what_if`` with ``%%session NAME`` is a what-if on that notebook; any of
the location magics may come with ``%%bash`` except ``%%what_if`` (bash
refuses read-only runs). The first comment of the cell, or else its first
line of code, becomes the runtime's ``thought``, so the inspection digest,
active-work metadata and logs keep a per-step line.

Only what the model sees changes. The function behind the tool, every one
of its arguments and hidden channels (steering, notifications,
clarification, parent chat context), the ``ExecutionResult`` it returns,
handle adoption, events and stored-function recording are the runtime's own:
a parsed cell is passed to it as the arguments the legacy projection would
have sent. The model reads a cell's result as a notebook shows it (what it
printed, ``[stderr]``, ``Out: <repr>``, the traceback) instead of the JSON
envelope; the object the product keeps is unchanged. The session JSON tools
are not offered (``%sessions`` gives their data), and the system prompt names
the magics where it named the fields.

A magic the session lacks, an unknown or malformed one, or a magic written
below a cell's first lines is refused with what to write instead; nothing
falls back silently.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import json
import re
import shlex
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, Tuple

from unify.actor.execution.types import (
    ExecutionResult,
    ImagePart,
    TextPart,
    compact_diagnostic_text,
    parts_to_llm_content,
)
from unify.common.tool_errors import ToolInputError

__all__ = [
    "NOTEBOOK",
    "Capabilities",
    "Cell",
    "NotebookCellResult",
    "caption",
    "describe",
    "enabled",
    "language_and_code",
    "parse_cell",
    "project_tools",
    "rewrite_prompt",
]

NOTEBOOK = "notebook"

#: The session tools the magics replace; not offered under the projection.
SESSION_TOOLS = (
    "list_sessions",
    "inspect_state",
    "close_session",
    "close_all_sessions",
)

#: Characters of a first code line kept as the step's caption.
CAPTION_CHARS = 160

_SESSION_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,64}")


def enabled() -> bool:
    """Whether ``UNIFY_CODE_PROJECTION=notebook`` is set."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_CODE_PROJECTION", "") == NOTEBOOK


# ---------------------------------------------------------------------------
# The parser: a cell onto the runtime's arguments
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """What the session can do with a magic."""

    #: ``%%bash``: the sandboxed workspace's shell (``UNIFY_WORKSPACE=sandboxed``).
    bash: bool = False
    #: ``%pip install``: the harness-side install into the workspace environment.
    install: bool = True
    #: ``%%scratch``, ``%%what_if``, ``%%session NAME``, ``%sessions``.
    sessions: bool = True

    def magics(self) -> List[str]:
        """The magics this session takes, as written."""
        out: List[str] = []
        if self.bash:
            out.append("%%bash")
        if self.install:
            out.append("%pip install PKG")
        if self.sessions:
            out += ["%%scratch", "%%what_if", "%%session NAME", "%sessions"]
        return out


@dataclasses.dataclass(frozen=True)
class Cell:
    """A cell as the runtime takes it."""

    #: The code to run: the cell with its magic lines blanked, so line
    #: numbers in a traceback are the cell's own.
    code: str
    language: str = "python"
    state_mode: str = "stateful"
    session_id: Optional[int] = None
    session_name: Optional[str] = None
    #: ``%pip install`` specifiers, installed before the code runs.
    packages: Tuple[str, ...] = ()
    #: ``%sessions``: list the notebooks instead of running code.
    list_sessions: bool = False
    #: The runtime's ``thought``: the first comment, else the first code line.
    thought: str = ""
    #: The magic lines as written, in order.
    magics: Tuple[str, ...] = ()


def _refuse(message: str, suggestion: str, line: str) -> ToolInputError:
    return ToolInputError(message, suggestion=suggestion, received={"code": line})


def _offered(caps: Capabilities) -> str:
    names = caps.magics()
    if not names:
        return "This session takes no magics; a cell is Python."
    return "Magics, on a cell's first lines: " + ", ".join(names) + "."


def caption(code: str) -> str:
    """The first comment of *code*, or else its first line of code; the
    magic lines at its top are skipped."""
    in_header = True
    for raw in code.splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_header and line.startswith("%"):
            continue
        in_header = False
        if line.startswith("#!"):
            continue
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            if text:
                return text[:500]
            continue
        return line[:CAPTION_CHARS]
    return ""


def _magic_misplaced(code: str) -> Optional[Tuple[int, str]]:
    """``(line number, line)`` when *code* does not parse as Python and the
    line the parser stopped at is a magic or a shell line."""
    try:
        compile(
            code,
            "<cell>",
            "exec",
            flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            dont_inherit=True,
        )
        return None
    except SyntaxError as exc:
        lines = code.splitlines()
        lineno = exc.lineno or 0
        if 1 <= lineno <= len(lines):
            line = lines[lineno - 1].strip()
            if line.startswith(("%", "!")):
                return lineno, line
        return None
    except (ValueError, TypeError):  # a null byte; the runtime reports it
        return None


def parse_cell(code: str, caps: Capabilities) -> Cell:
    """The runtime arguments *code* stands for.

    Magics are read from the cell's first lines (blank lines before them
    are skipped), as Jupyter reads a cell magic from the first line. Raises
    :class:`ToolInputError`, naming what to write instead, for a magic the
    session lacks, an unknown or malformed one, a combination the runtime
    cannot run, or a magic or ``!`` shell line below the first lines.
    """
    lines = code.split("\n")
    first = 0
    while first < len(lines) and not lines[first].strip():
        first += 1
    end = first
    while end < len(lines) and lines[end].lstrip().startswith("%"):
        end += 1
    header = [lines[i].strip() for i in range(first, end)]

    language = "python"
    location: Optional[str] = None
    session_name: Optional[str] = None
    packages: List[str] = []
    list_sessions = False
    seen: set[str] = set()

    for line in header:
        word, _, rest = line.partition(" ")
        rest = rest.strip()
        if word in seen and word != "%pip":
            raise _refuse(
                f"`{word}` appears twice at the top of the cell.",
                "Write each magic once.",
                line,
            )
        seen.add(word)
        if word == "%%bash":
            if not caps.bash:
                raise _refuse(
                    "`%%bash` is not available in this session; cells run Python.",
                    "Run the command from Python with `subprocess`, or remove "
                    "the line.",
                    line,
                )
            if rest:
                raise _refuse(
                    f"`%%bash` takes no arguments here, not {rest!r}.",
                    "Write `%%bash` alone on the cell's first line.",
                    line,
                )
            language = "bash"
        elif word in ("%%scratch", "%%what_if", "%%session", "%sessions"):
            if not caps.sessions:
                raise _refuse(
                    f"`{word}` is not available here: every cell runs in one "
                    "namespace.",
                    _offered(caps),
                    line,
                )
            if word == "%sessions":
                if rest:
                    raise _refuse(
                        f"`%sessions` takes no arguments, not {rest!r}.",
                        "Write `%sessions` alone in a cell.",
                        line,
                    )
                list_sessions = True
                continue
            if word == "%%session":
                parts = rest.split()
                if len(parts) != 1 or not _SESSION_NAME.fullmatch(parts[0]):
                    raise _refuse(
                        "`%%session` needs one name of letters, digits, `_`, "
                        f"`.` or `-` (at most 64), not {rest!r}.",
                        "Write e.g. `%%session experiment` on the cell's first "
                        "line; the notebook is created on first use.",
                        line,
                    )
                session_name = parts[0]
                if location == "what_if":
                    continue
                if location is not None:
                    raise _refuse(
                        f"`%%session` cannot be combined with `%%{location}`.",
                        "Use one of them.",
                        line,
                    )
                location = "session"
                continue
            if rest:
                raise _refuse(
                    f"`{word}` takes no arguments, not {rest!r}.",
                    f"Write `{word}` alone on its line.",
                    line,
                )
            name = word[2:]
            if location == "session" and name == "what_if":
                location = "what_if"
                continue
            if location is not None:
                raise _refuse(
                    f"`{word}` cannot be combined with `%%{location}`.",
                    "Use one of them.",
                    line,
                )
            location = name
        elif word == "%pip":
            if not caps.install:
                raise _refuse(
                    "Installing packages is not available in this session.",
                    "Use what is already installed.",
                    line,
                )
            sub, _, specs_text = rest.partition(" ")
            if sub != "install":
                raise _refuse(
                    f"Only `%pip install` is available, not `%pip {sub}`.",
                    "Write `%pip install PKG ...`.",
                    line,
                )
            try:
                specs = shlex.split(specs_text)
            except ValueError as exc:
                raise _refuse(
                    f"`%pip install` could not read its packages: {exc}.",
                    'Write e.g. `%pip install pandas "numpy>=2"`.',
                    line,
                ) from None
            options = [s for s in specs if s.startswith("-")]
            if options:
                raise _refuse(
                    "`%pip install` takes package specifiers only; options "
                    f"such as {options[0]!r} are not taken.",
                    "Write e.g. `%pip install pandas>=2`.",
                    line,
                )
            if not specs:
                raise _refuse(
                    "`%pip install` needs at least one package.",
                    "Write e.g. `%pip install pandas`.",
                    line,
                )
            packages += specs
        else:
            raise _refuse(
                f"`{word}` is not a magic this notebook has.",
                _offered(caps) + " Remove the line, or write it in Python.",
                line,
            )

    body_lines = list(lines)
    for i in range(first, end):
        body_lines[i] = ""
    body = "\n".join(body_lines)
    magics = tuple(header)

    if list_sessions:
        if len(header) > 1 or body.strip():
            raise _refuse(
                "`%sessions` goes in a cell of its own.",
                "Run `%sessions` alone, then the code in the next cell.",
                header[0] if header else "",
            )
        return Cell(code="", list_sessions=True, thought="%sessions", magics=magics)

    if language == "bash" and location == "what_if":
        raise _refuse(
            "A bash cell cannot run as a what-if: its changes to files and the "
            "shell cannot be discarded.",
            "Drop `%%what_if`, or try the step in Python.",
            "%%what_if",
        )

    stripped = body.lstrip("\n")
    first_line = stripped.split("\n", 1)[0].strip() if stripped else ""
    if not header and first_line.startswith("!"):
        raise _refuse(
            "Shell lines (`!cmd`) are not run here.",
            (
                "Start the cell with `%%bash` to run it in the shell."
                if caps.bash
                else "Run the command from Python with `subprocess`."
            ),
            first_line,
        )
    if language == "python" and body.strip():
        misplaced = _magic_misplaced(body)
        if misplaced is not None:
            lineno, line = misplaced
            if line.startswith("!"):
                suggestion = (
                    "Put shell commands in a cell that starts with `%%bash`."
                    if caps.bash
                    else "Run the command from Python with `subprocess`."
                )
                message = (
                    f"Line {lineno} is a shell line (`!cmd`), which is not run here."
                )
            else:
                suggestion = (
                    "Move the magic to the top of the cell, or into a cell of "
                    "its own. " + _offered(caps)
                )
                message = (
                    f"Line {lineno} ({line.split()[0]}) is a magic below the "
                    "cell's first lines; magics are read only at the top."
                )
            raise _refuse(message, suggestion, line)

    state_mode, session_id = "stateful", None
    if location == "scratch":
        state_mode = "stateless"
    elif location == "what_if":
        state_mode = "read_only"
        if session_name is None:
            session_id = 0
    thought = caption(body) or (" ".join(header) if header else "")
    return Cell(
        code=body,
        language=language,
        state_mode=state_mode,
        session_id=session_id,
        session_name=session_name,
        packages=tuple(packages),
        thought=thought,
        magics=magics,
    )


def language_and_code(code: str) -> Tuple[str, str]:
    """The language a recorded cell ran in and its code without the magic
    lines, for readers of the transcript; ``("python", code)`` for a cell
    the parser refuses."""
    try:
        cell = parse_cell(code, Capabilities(bash=True))
    except ToolInputError:
        return "python", code
    return cell.language, cell.code.lstrip("\n")


# ---------------------------------------------------------------------------
# What the model reads back
# ---------------------------------------------------------------------------


def _repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception as exc:  # noqa: BLE001 - a broken __repr__ still shows something
        return f"<{type(value).__name__} (repr failed: {type(exc).__name__})>"


def _holds_handle(value: Any) -> bool:
    """Whether *value* is, or holds, a steerable handle or the sentinel the
    loop puts in its place once adopted."""
    from unify.common._async_tool.tools_data import (
        _HANDLE_SENTINEL,
        _handle_label_sentinel,
    )
    from unify.common.async_tool_loop import SteerableToolHandle

    if isinstance(value, SteerableToolHandle):
        return True
    if isinstance(value, str):
        labelled = re.escape(_handle_label_sentinel("LABEL")).replace(
            "LABEL",
            r"h\d+",
        )
        return value == _HANDLE_SENTINEL or bool(re.fullmatch(labelled, value))
    if isinstance(value, dict):
        return any(_holds_handle(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_handle(v) for v in value)
    return False


def _has_content(stream: List[Any]) -> bool:
    return any(
        (isinstance(p, TextPart) and p.text.strip()) or isinstance(p, ImagePart)
        for p in stream
    )


class NotebookCellResult(ExecutionResult):
    """An :class:`ExecutionResult` the model reads as a notebook cell.

    The fields, and so everything the product reads, are the runtime's; only
    :meth:`to_llm_content` differs: what the cell printed, then ``[stderr]``
    and its text, then ``Out: <repr>`` of the last expression's value, then
    the traceback. A note on the call (``UNIFY_PLACEHOLDER_NOTE``) comes
    first and what steered the block last, when anything did. No session or
    timing metadata. A result that is or holds a steerable handle keeps the
    shipped rendering while the loop adopts it.
    """

    def to_llm_content(self) -> List[dict]:
        if _holds_handle(self.result):
            return super().to_llm_content()
        parts: List[Any] = []

        def text(value: str) -> None:
            if parts and isinstance(parts[-1], TextPart):
                previous = parts[-1].text
                if previous and not previous.endswith("\n"):
                    value = "\n" + value
            parts.append(TextPart(text=value))

        if self.note is not None:
            text(f"[note] {self.note}\n")
        if _has_content(self.stdout):
            parts.extend(self.stdout)
        if _has_content(self.stderr):
            text("[stderr]\n")
            for part in self.stderr:
                if isinstance(part, TextPart) and part.text:
                    parts.append(TextPart(text=compact_diagnostic_text(part.text)))
                else:
                    parts.append(part)
        if self.result is not None:
            text(f"Out: {_repr(self.result)}\n")
        if self.error is not None:
            text(str(self.error).rstrip("\n") + "\n")
        steered = {
            key: value
            for key, value in (self.steering or {}).items()
            if key in ("interjections_received", "patched")
        }
        if steered:
            text(f"[steering] {json.dumps(self.steering, default=str)}\n")
        blocks = parts_to_llm_content(parts)
        if not blocks:
            return [{"type": "text", "text": "(no output)"}]
        last = blocks[-1]
        if last.get("type") == "text":
            last["text"] = last["text"].rstrip("\n")
        return blocks


def _as_cell_result(out: Any) -> Any:
    """*out* as a :class:`NotebookCellResult` where it is a cell's result."""
    if isinstance(out, NotebookCellResult):
        return out
    if isinstance(out, ExecutionResult):
        return NotebookCellResult(**dict(out))
    if isinstance(out, dict) and {"stdout", "stderr"} <= set(out):
        data = dict(out)
        for stream in ("stdout", "stderr"):
            value = data.get(stream)
            if isinstance(value, str):
                data[stream] = [TextPart(text=value)] if value else []
        try:
            return NotebookCellResult(**data)
        except Exception:  # noqa: BLE001 - an unexpected shape stays as it was
            return out
    return out


def _prepend(out: Any, lines: str) -> Any:
    """*out* with *lines* before what the cell printed."""
    if isinstance(out, ExecutionResult):
        return out.model_copy(update={"stdout": [TextPart(text=lines), *out.stdout]})
    return out


# ---------------------------------------------------------------------------
# The tool the model sees
# ---------------------------------------------------------------------------

_CODE_FIELD = (
    "The cell's source. Python by default; a magic on the cell's first line "
    "changes where it runs."
)


def describe(
    caps: Capabilities,
    *,
    steering: bool,
    structured: bool,
    network: str = "",
) -> str:
    """The tool's description: what a cell is, its magics, its output."""
    out = [
        "Run a cell in this task's notebook. As in Jupyter, variables, imports "
        "and definitions persist from cell to cell, top-level `await` works, "
        "and the value of the last expression is shown.",
    ]
    magics: List[str] = []
    if caps.bash:
        magics.append(
            "- `%%bash` on the first line runs the cell in this task's shell "
            "(its directory and variables persist). `Out:` is the exit status.",
        )
    if caps.install:
        magics.append(
            "- `%pip install PKG ...` installs packages into the workspace "
            "environment, where they stay; the rest of the cell then runs.",
        )
    if caps.sessions:
        magics += [
            "- `%%scratch` runs the cell in a fresh namespace that nothing keeps.",
            "- `%%what_if` runs it on a copy of this notebook; its changes are "
            "discarded.",
            "- `%%session NAME` runs it in a separate notebook called NAME "
            "(created on first use).",
            "- `%sessions` lists the notebooks and their variables.",
        ]
    if magics:
        out.append("Magics, on the cell's first lines:\n" + "\n".join(magics))
    if caps.bash:
        out.append(
            "Bash cells, and every subprocess a Python cell starts, run in the "
            "workspace sandbox: only the workspace (the starting directory) and "
            "a private /tmp are writable; credentials, .env files and the "
            "assistant's own state are hidden; credential environment "
            f"variables are removed; {network or 'there is no network'}. A "
            "refusal names the sandbox rule behind it.",
        )
    out.append(
        "Output reads like a notebook cell: what the cell printed, `[stderr]` "
        "and its text, `Out: <repr>` of the last expression, and the traceback "
        "if it failed.",
    )
    if steering:
        out.append(
            "A steerable handle as the cell's last expression is adopted by "
            "the outer loop for steering. A running cell can be corrected: "
            "checkpoints sit between top-level statements, at the top of every "
            "loop body and before every `primitives.*` call; on a correction "
            "the cell suspends and you get a turn with a progress report, where "
            '`steer(call_id=<id>, action="stop")` abandons it and interjecting '
            "again resumes it as written. A checkpoint runs only when the cell "
            "yields, so prefer async calls in work that may need correcting "
            "partway through.",
        )
    out.append(
        (
            "A cell is for computing. You answer by calling `final_response`."
            if structured
            else "A cell is for computing. You answer, and take any action the "
            "requester defines, by replying."
        ),
    )
    return "\n\n".join(out)


def _runtime_language_kwarg(fn: Callable[..., Any]) -> Optional[str]:
    """The keyword the runtime takes a cell's language by, if any."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if "language" in params:
        return "language"
    return None


async def _list_notebooks(
    list_sessions: Optional[Callable[..., Any]],
    inspect_state: Optional[Callable[..., Any]],
) -> str:
    if list_sessions is None or inspect_state is None:
        return "(no notebooks)"
    listed = await list_sessions()
    lines: List[str] = []
    for entry in (listed or {}).get("sessions", []):
        sid = entry.get("session_id")
        name = entry.get("session_name")
        state = await inspect_state(session_id=sid, detail="names")
        variables = ((state or {}).get("state") or {}).get("variables") or []
        if sid == 0:
            label = "this notebook (the default)"
        elif name:
            label = f"%%session {name}"
        else:
            label = f"session {sid} (unnamed)"
        names = ", ".join(str(v) for v in variables) if variables else "(empty)"
        lines.append(f"{label}: {names}")
    return "\n".join(lines) if lines else "(no notebooks)"


def project_tools(
    tools: Mapping[str, Any],
    *,
    caps: Capabilities,
    steering: bool,
    structured: bool,
    parent_context: bool,
    resolve_session_name: Callable[[str], Any],
    session_tools: Mapping[str, Any],
    network: str = "",
) -> Dict[str, Any]:
    """*tools* with ``execute_code`` as a notebook cell and no session tools.

    The returned ``execute_code`` takes ``code`` (and, where sub-agents read
    it, ``include_parent_chat_context``), parses the cell and calls the
    session's own ``execute_code`` with the arguments the cell stands for.
    *session_tools* holds the actor's ``list_sessions`` and ``inspect_state``,
    which ``%sessions`` reads.
    """
    from unify.common.tool_spec import ToolSpec

    out = {k: v for k, v in tools.items() if k not in SESSION_TOOLS}
    tool = out.get("execute_code")
    if tool is None:
        return out
    runtime = tool.fn if isinstance(tool, ToolSpec) else tool
    language_kw = _runtime_language_kwarg(runtime)
    if caps.bash and language_kw is None:
        caps = dataclasses.replace(caps, bash=False)

    def _fn(name: str) -> Optional[Callable[..., Any]]:
        entry = session_tools.get(name)
        if entry is None:
            return None
        return entry.fn if isinstance(entry, ToolSpec) else entry

    list_fn, inspect_fn = _fn("list_sessions"), _fn("inspect_state")

    async def run_cell(code: str, hidden: Dict[str, Any]) -> Any:
        from unify import environment

        cell = parse_cell(code if isinstance(code, str) else "", caps)
        if cell.list_sessions:
            listing = await _list_notebooks(list_fn, inspect_fn)
            return NotebookCellResult(stdout=[TextPart(text=listing)])
        if (
            cell.state_mode == "read_only"
            and cell.session_name is not None
            and resolve_session_name(cell.session_name) is None
        ):
            raise _refuse(
                f"There is no notebook called {cell.session_name!r} to try a "
                "what-if on.",
                "`%sessions` lists the notebooks; `%%session NAME` alone "
                "creates one.",
                "%%what_if",
            )
        installed = ""
        if cell.packages:
            result = await asyncio.to_thread(environment.install, list(cell.packages))
            log = "\n".join(
                s for s in (result.get("stdout"), result.get("stderr")) if s
            ).strip()
            if len(log) > 2000:
                log = "..." + log[-2000:]
            if not result.get("success"):
                return NotebookCellResult(
                    error=(
                        "%pip install failed; the rest of the cell did not run.\n" + log
                    ).rstrip(),
                )
            installed = (
                "[pip] installed "
                + " ".join(cell.packages)
                + (f"\n{log}" if log else "")
            ).rstrip() + "\n"
            if not cell.code.strip():
                return NotebookCellResult(stdout=[TextPart(text=installed)])
        kwargs: Dict[str, Any] = dict(
            state_mode=cell.state_mode,
            session_id=cell.session_id,
            session_name=cell.session_name,
            **hidden,
        )
        if language_kw is not None:
            kwargs[language_kw] = cell.language
        out = await runtime(cell.thought, cell.code, **kwargs)
        out = _as_cell_result(out)
        if installed:
            out = _prepend(out, installed)
        return out

    if parent_context:

        async def execute_code(
            code: Annotated[str, _CODE_FIELD],
            *,
            _notification_up_q: asyncio.Queue[dict] | None = None,
            _clarification_up_q: asyncio.Queue[str] | None = None,
            _clarification_down_q: asyncio.Queue[str] | None = None,
            _interject_queue: asyncio.Queue | None = None,
            _pause_event: asyncio.Event | None = None,
            _parent_chat_context: list[dict] | None = None,
        ) -> Any:
            return await run_cell(
                code,
                dict(
                    _notification_up_q=_notification_up_q,
                    _clarification_up_q=_clarification_up_q,
                    _clarification_down_q=_clarification_down_q,
                    _interject_queue=_interject_queue,
                    _pause_event=_pause_event,
                    _parent_chat_context=_parent_chat_context,
                ),
            )

    else:

        async def execute_code(  # type: ignore[misc]
            code: Annotated[str, _CODE_FIELD],
            *,
            _notification_up_q: asyncio.Queue[dict] | None = None,
            _clarification_up_q: asyncio.Queue[str] | None = None,
            _clarification_down_q: asyncio.Queue[str] | None = None,
            _interject_queue: asyncio.Queue | None = None,
            _pause_event: asyncio.Event | None = None,
        ) -> Any:
            return await run_cell(
                code,
                dict(
                    _notification_up_q=_notification_up_q,
                    _clarification_up_q=_clarification_up_q,
                    _clarification_down_q=_clarification_down_q,
                    _interject_queue=_interject_queue,
                    _pause_event=_pause_event,
                ),
            )

    execute_code.__doc__ = describe(
        caps,
        steering=steering,
        structured=structured,
        network=network,
    )
    execute_code.__qualname__ = "execute_code"
    #: The session's own tool, for callers that need the legacy contract.
    execute_code.runtime = runtime  # type: ignore[attr-defined]
    if isinstance(tool, ToolSpec):
        out["execute_code"] = dataclasses.replace(tool, fn=execute_code)
    else:
        out["execute_code"] = execute_code
    return out


# ---------------------------------------------------------------------------
# The system prompt
# ---------------------------------------------------------------------------

#: Exact sentences of the shipped prompts that name the session fields or
#: tools, and what the projection says instead.
PROMPT_REWRITES: Tuple[Tuple[str, str], ...] = (
    # The execution rules (shipped profile, JSON surface).
    (
        "   `list_sessions()` / `inspect_state()` rediscover live sessions\n"
        "   and names — variables survive context compression, since state\n"
        "   lives in the sandbox, not the transcript. Isolate a cell with\n"
        '   `state_mode="stateless"` or a named session; fan out in parallel\n',
        "   `%sessions` lists the notebooks and their variables — variables\n"
        "   survive context compression, since state lives in the sandbox,\n"
        "   not the transcript. Isolate a cell with a `%%scratch` or\n"
        "   `%%session NAME` first line; fan out in parallel\n",
    ),
    # The same rule on the core surface.
    (
        "   Variables survive context compression, since state lives in the\n"
        "   sandbox, not the transcript. Isolate a cell with\n"
        '   `state_mode="stateless"` or a named session; fan out in parallel\n',
        "   Variables survive context compression, since state lives in the\n"
        "   sandbox, not the transcript. Isolate a cell with a `%%scratch` or\n"
        "   `%%session NAME` first line; fan out in parallel\n",
    ),
    # The lean profile's rule, JSON surface and core surface.
    (
        "`list_sessions()` and\n"
        "   `inspect_state()` show live sessions and names;\n"
        '   `state_mode="stateless"` or a named session isolates a cell.',
        "`%sessions` lists the\n"
        "   notebooks; a `%%scratch` or `%%session NAME` first line isolates a\n"
        "   cell.",
    ),
    (
        '`state_mode="stateless"` or a\n   named session isolates a cell.',
        "A `%%scratch` or `%%session\n   NAME` first line isolates a cell.",
    ),
    # Verify before scaling: the what-if.
    (
        '`state_mode="read_only"` to try alternatives without risk.',
        "a `%%what_if` cell to try alternatives without risk.",
    ),
    (
        '`state_mode="read_only"` tries an alternative on the current\n'
        "state without changing it.",
        "A `%%what_if` cell tries an alternative on the current state\n"
        "without changing it.",
    ),
    # Per-function modes are unchanged; only the session's field is gone.
    (
        "Functions support execution mode overrides independent of the session's\n"
        "`state_mode`:",
        "Functions support execution mode overrides independent of where the\n"
        "cell runs:",
    ),
    # The core surface's tools line.
    (
        '(or bash, with `language="bash"`)',
        "(or bash, with a `%%bash` first line)",
    ),
)


def rewrite_prompt(prompt: str) -> str:
    """*prompt* naming the magics where it named the session fields and tools."""
    for old, new in PROMPT_REWRITES:
        prompt = prompt.replace(old, new)
    return prompt
