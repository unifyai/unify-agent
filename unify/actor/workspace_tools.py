"""The actor's shell and file tools under ``UNIFY_WORKSPACE=sandboxed``.

``execute_code`` gains ``language="bash"``, and ``read_file`` and ``grep`` read
what a sandboxed shell cell could read (unify/sandbox.py). With the switch off
none of this is built and the actor's tools are exactly as shipped.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Callable, Dict, Optional

from unify import sandbox
from unify.actor.execution import file_tools
from unify.common.tool_errors import ToolInputError
from unify.common.tool_spec import ToolSpec, llm_soft_required

__all__ = ["workspace_tools"]

LANGUAGES = ("python", "bash")

_SHELL_LINE = (
    "- **Shell commands** run from Python via ``subprocess`` (or\n"
    "              ``asyncio.create_subprocess_exec``); there is no shell cell."
)


def _network_text() -> str:
    from unify.settings import SETTINGS

    if SETTINGS.UNIFY_WORKSPACE_NETWORK == "proxy":
        port = SETTINGS.UNIFY_WORKSPACE_PROXY_PORT
        return (
            f"the only network is the proxy at http://127.0.0.1:{port} "
            "(HTTP_PROXY and HTTPS_PROXY point at it)"
        )
    return "there is no network"


def _execute_code_doc(original: str) -> str:
    shell = (
        '- **language**: "python" (default) or "bash". A bash cell runs in\n'
        "              a persistent bash session per session_id: the working\n"
        "              directory, variables and functions carry over between\n"
        "              cells; stdout and stderr arrive interleaved in\n"
        "              ``stdout``; ``result`` is the exit status. Bash takes\n"
        '              state_mode "stateful" or "stateless", not "read_only".\n'
        "            - **Sandbox**: bash cells, and every subprocess a Python\n"
        "              cell starts, run in the workspace sandbox. Only the\n"
        "              workspace (the starting directory) and a private /tmp\n"
        "              are writable; credentials, .env files and the\n"
        "              assistant's own state are hidden; credential\n"
        f"              environment variables are removed; {_network_text()}.\n"
        "              A refusal names the sandbox rule behind it."
    )
    doc = original.replace(
        "Execute arbitrary Python code in a specified state mode.",
        "Execute arbitrary Python or bash code in a specified state mode.",
        1,
    )
    if _SHELL_LINE in doc:
        return doc.replace(_SHELL_LINE, shell, 1)
    return doc.rstrip() + "\n\n            " + shell + "\n"


def workspace_tools(execute_code: Callable[..., Any]) -> Dict[str, ToolSpec]:
    """``execute_code`` with a ``language`` argument, ``read_file`` and ``grep``."""

    @llm_soft_required(thought="")
    async def execute_code_in_language(
        thought: Annotated[
            str,
            "A brief, first-person, one-sentence explanation of what this "
            'code does and why you are running it right now (e.g. "Loading '
            'the data and computing the summary the user asked for."). Shown '
            "to the user as the rationale for this step; always provide it.",
        ],
        code: Optional[str] = None,
        *,
        language: str = "python",
        state_mode: str | None = None,
        session_id: int | None = None,
        session_name: str | None = None,
        _notification_up_q: asyncio.Queue[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        _interject_queue: asyncio.Queue | None = None,
        _pause_event: asyncio.Event | None = None,
        _parent_chat_context: list[dict] | None = None,
    ) -> Any:
        if language not in LANGUAGES:
            raise ToolInputError(
                f"Unsupported language: {language!r}",
                suggestion=f"Use one of: {list(LANGUAGES)}",
                received={"language": language},
            )
        return await execute_code(
            thought,
            code,
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            _notification_up_q=_notification_up_q,
            _clarification_up_q=_clarification_up_q,
            _clarification_down_q=_clarification_down_q,
            _interject_queue=_interject_queue,
            _pause_event=_pause_event,
            _parent_chat_context=_parent_chat_context,
            _language=language,
        )

    execute_code_in_language.__name__ = "execute_code"
    execute_code_in_language.__qualname__ = "execute_code"
    execute_code_in_language.__doc__ = _execute_code_doc(execute_code.__doc__ or "")

    async def read_file(
        path: str,
        start: int = 1,
        end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read lines of a text file, numbered.

        Args:
            path: The file; absolute, or relative to the workspace.
            start: The first line to return (1-based).
            end: The last line to return, inclusive. Omit it for up to 2000
                lines from ``start``.

        Returns:
            ``path`` (resolved), ``start``, ``end``, ``total_lines`` and
            ``content``, one ``<line number>\\t<text>`` line per line;
            ``next_start`` when lines before ``end`` did not fit.

        Paths the workspace sandbox hides (the assistant's own state,
        credentials, .env files, /proc, the host's /tmp) are refused with the
        rule that hides them.
        """
        policy = sandbox.build_policy()
        return await asyncio.to_thread(
            file_tools.read_file,
            path,
            start,
            end,
            policy=policy,
        )

    async def grep(
        pattern: str,
        path: str = ".",
        max_hits: int = 100,
    ) -> Dict[str, Any]:
        """
        Search files for lines matching a regular expression.

        Args:
            pattern: The regular expression (ripgrep syntax).
            path: A file or directory; absolute, or relative to the workspace.
                Directories are searched recursively, skipping hidden files
                and files the tree's own .gitignore files exclude.
            max_hits: Stop after this many matching lines (at most 1000).

        Returns:
            ``hits``, each ``<file>:<line number>:<text>``, and ``truncated``
            when the search stopped at ``max_hits`` or its time limit.

        Paths the workspace sandbox hides are refused with the rule that hides
        them, and never searched.
        """
        return await file_tools.grep(
            pattern,
            path,
            max_hits,
            policy=sandbox.build_policy(),
        )

    return {
        "execute_code": ToolSpec(fn=execute_code_in_language),
        "read_file": ToolSpec(fn=read_file, display_label="Reading a file"),
        "grep": ToolSpec(fn=grep, display_label="Searching files"),
    }
