"""The agent loop's own texts that change when tools are called from code (``UNIFY_TOOLS_IN_CODE=on``).

Each function returns today's text while the setting is off, byte for byte.
"""

from __future__ import annotations

#: ``send_response``'s description today: it points to progress notifications, which a session on the agent
#: record does not have.
SEND_RESPONSE_DESCRIPTION = (
    "Submit your structured response for the current "
    "request in the required JSON format. This signals "
    "that you have completed the current work and are "
    "ready for the next instruction. Do not use this "
    "for progress updates — those should be sent via "
    "notifications while work is still ongoing."
)
SEND_RESPONSE_DESCRIPTION_IN_CODE = (
    "Submit your structured response for the current request in the required JSON format; it ends your work on "
    "that request."
)


def _on() -> bool:
    from unify.common import in_code_switch

    return in_code_switch.tools_in_code()


def send_response_description() -> str:
    """The description of a persistent session's response tool."""
    return SEND_RESPONSE_DESCRIPTION_IN_CODE if _on() else SEND_RESPONSE_DESCRIPTION


def unknown_tool_text(name: str) -> str:
    """The answer to a call of a tool the session does not offer; with tools called from code it points to the
    one model-API tool and to ``tools.list()``."""
    if _on():
        return f"Tool '{name}' does not exist; `execute_code` is your one tool, and `tools.list()` lists the rest."
    return (
        f"⚠️ Error: Tool '{name}' is not available. "
        "The tool may have been removed or does not exist. "
        "Please proceed without using this tool."
    )
