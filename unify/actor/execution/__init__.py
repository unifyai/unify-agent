"""Code execution infrastructure for actors.

Re-exports the public surface so consumers can write::

    from unify.actor.execution import (
        ExecutionResult, TextPart, ImagePart, parts_to_text, ...
    )
"""

from .types import (
    ExecutionResult,
    ImagePart,
    OutputPart,
    TextPart,
    _detect_image_mime_from_b64,
    parts_to_llm_content,
    parts_to_text,
)

from .capture import (
    StreamLike,
    StreamRouter,
    _make_display,
    _stderr_parts,
    _stdout_parts,
    capture_sandbox_output,
)

from .callable import execute_callable

from .session import (
    PythonExecutionSession,
    SessionExecutor,
    SessionKey,
    StateMode,
    _CAN_CLARIFY,
    _CURRENT_ENVIRONMENTS,
    _CURRENT_SANDBOX,
    _PARENT_CHAT_CONTEXT,
    _validate_execution_params,
)

__all__ = [
    # types
    "TextPart",
    "ImagePart",
    "OutputPart",
    "ExecutionResult",
    "parts_to_text",
    "parts_to_llm_content",
    "_detect_image_mime_from_b64",
    # capture
    "StreamLike",
    "StreamRouter",
    "_stdout_parts",
    "_stderr_parts",
    "_make_display",
    "capture_sandbox_output",
    # callable
    "execute_callable",
    # session
    "StateMode",
    "SessionKey",
    "_CAN_CLARIFY",
    "_CURRENT_ENVIRONMENTS",
    "_CURRENT_SANDBOX",
    "_PARENT_CHAT_CONTEXT",
    "_validate_execution_params",
    "PythonExecutionSession",
    "SessionExecutor",
]
