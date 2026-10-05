"""Structured output types for sandbox code execution.

Provides TextPart / ImagePart / OutputPart for rich (text + image) output,
ExecutionResult for LLM-formatted execution results, and helper converters.
"""

from __future__ import annotations

import base64
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from pydantic import BaseModel, Field


def _is_diagnostic_line(text: str) -> bool:
    for marker in ("PHASE ", "SKIP ", "SOFT_FAIL ", "START ", "END ", "FAIL "):
        if marker in text:
            return True
    return False


def compact_diagnostic_text(
    text: str,
    *,
    max_keep_lines: int = 12,
) -> str:
    """Summarize dense PHASE/SKIP/SOFT_FAIL trails for live LLM observations.

    Full text remains on the ExecutionResult / EventBus payload; this only
    shapes what ``to_llm_content`` shows the live CodeAct loop.
    """

    if not text or not text.strip():
        return text
    lines = text.splitlines()
    diagnostic = [ln for ln in lines if _is_diagnostic_line(ln)]
    other = [ln for ln in lines if not _is_diagnostic_line(ln)]
    if len(diagnostic) < 4:
        return text

    phase_n = sum(1 for ln in diagnostic if "PHASE " in ln)
    skip_n = sum(1 for ln in diagnostic if "SKIP " in ln)
    soft_n = sum(1 for ln in diagnostic if "SOFT_FAIL " in ln)
    summary = (
        f"[{len(diagnostic)} diagnostic log lines compacted for live context: "
        f"PHASE={phase_n} SKIP={skip_n} SOFT_FAIL={soft_n}; "
        f"full trail retained in EventBus/Job logs]"
    )
    keep_tail = diagnostic[-min(3, len(diagnostic)) :]
    kept_other = other[:max_keep_lines]
    if len(other) > max_keep_lines:
        kept_other.append(
            f"... ({len(other) - max_keep_lines} more non-diagnostic lines)",
        )
    return "\n".join([summary, *keep_tail, *kept_other])


class TextPart(BaseModel):
    """A text output part from sandbox execution."""

    type: Literal["text"] = "text"
    text: str

    def to_llm_content(self) -> dict:
        """Convert to LLM content block format."""
        return {"type": "text", "text": self.text}


class ImagePart(BaseModel):
    """An image output part from sandbox execution (e.g., from display())."""

    type: Literal["image"] = "image"
    mime: str = "image/png"
    data: str  # base64 encoded

    def to_llm_content(self) -> dict:
        """Convert to LLM content block format."""
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{self.mime};base64,{self.data}"},
        }


# Discriminated union - Pydantic auto-parses based on `type` field
OutputPart = Annotated[Union[TextPart, ImagePart], Field(discriminator="type")]


def _detect_image_mime_from_b64(b64_str: str) -> str:
    """Detect an image MIME type by inspecting decoded header bytes.

    Returns "image/jpeg" for JPEG, "image/png" for PNG, or "image/png" as fallback.
    """
    try:
        raw = base64.b64decode(b64_str[:32])
        if raw[:2] == b"\xff\xd8":
            return "image/jpeg"
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
    except Exception:
        pass
    return "image/png"


def parts_to_text(parts: List[Union[TextPart, ImagePart]]) -> str:
    """Convert a list of OutputPart to a plain text string.

    Useful for backward compatibility and simple text extraction.
    Only TextPart parts are included; ImagePart parts are skipped.
    """
    return "".join(p.text for p in parts if isinstance(p, TextPart))


def parts_to_llm_content(parts: List[Union[TextPart, ImagePart]]) -> List[dict]:
    """Convert a list of OutputParts to LLM content blocks, preserving order.

    This function maintains the original interleaving of text and images,
    unlike the legacy approach which collected all images at the end.

    Adjacent TextParts are merged into a single text block for cleaner output.
    """
    if not parts:
        return []

    blocks: List[dict] = []
    pending_text = ""

    for part in parts:
        if isinstance(part, TextPart):
            pending_text += part.text
        elif isinstance(part, ImagePart):
            # Flush any pending text before the image
            if pending_text:
                blocks.append({"type": "text", "text": pending_text})
                pending_text = ""
            blocks.append(part.to_llm_content())

    # Flush any remaining text
    if pending_text:
        blocks.append({"type": "text", "text": pending_text})

    return blocks


class ExecutionResult(BaseModel):
    """Result from sandbox code execution, implementing FormattedToolResult protocol.

    This model gives the sandbox full control over how its output is formatted
    for the LLM, preserving the original interleaving of text and images from
    print() and display() calls.
    """

    stdout: List[Union[TextPart, ImagePart]] = Field(default_factory=list)
    stderr: List[Union[TextPart, ImagePart]] = Field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    state_mode: Optional[str] = None
    session_id: Optional[int] = None
    session_name: Optional[str] = None
    session_created: Optional[bool] = None
    duration_ms: Optional[int] = None
    #: Where the block got to when something steered it mid-flight. Set only
    #: when an interjection actually reached this execution, so an ordinary
    #: run carries no extra weight in the transcript.
    steering: Optional[Dict[str, Any]] = None
    #: A note about the call's arguments (``UNIFY_PLACEHOLDER_NOTE``); set only
    #: when there is one.
    note: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def to_llm_content(self) -> List[dict]:
        """Format this execution result for the LLM, preserving output order.

        Implements the FormattedToolResult protocol, giving the sandbox full
        control over how its output appears in the LLM transcript.
        """
        from unify.settings import SETTINGS

        if getattr(SETTINGS, "UNIFY_PLAIN_CELL_OUTPUT", False) and not _holds_handle(
            self.result,
        ):
            return self._plain_llm_content()
        blocks: List[dict] = []

        # Build metadata section (non-stdout/stderr fields)
        meta: Dict[str, Any] = {}
        if self.result is not None:
            meta["result"] = self.result
        if self.error is not None:
            meta["error"] = self.error
        if self.note is not None:
            meta["note"] = self.note
        if self.state_mode is not None:
            meta["state_mode"] = self.state_mode
        if self.session_id is not None:
            meta["session_id"] = self.session_id
        if self.session_name is not None:
            meta["session_name"] = self.session_name
        if self.session_created is not None:
            meta["session_created"] = self.session_created
        if self.duration_ms is not None:
            meta["duration_ms"] = self.duration_ms
        if self.steering is not None:
            meta["steering"] = self.steering

        # Add metadata block if present
        if meta:
            import json

            meta_text = json.dumps(meta, indent=2, default=str)
            blocks.append({"type": "text", "text": meta_text})

        # Add stdout with preserved ordering (interleaved text/images)
        if self.stdout:
            has_content = any(
                (isinstance(p, TextPart) and p.text.strip()) or isinstance(p, ImagePart)
                for p in self.stdout
            )
            if has_content:
                if blocks:  # Add separator if we have metadata
                    blocks.append({"type": "text", "text": "\n--- stdout ---\n"})
                blocks.extend(parts_to_llm_content(self.stdout))

        # Add stderr with preserved ordering (if non-empty). Compact dense
        # PHASE/SKIP/SOFT_FAIL trails for the live LLM observation; the raw
        # ``stderr`` field on this object is unchanged for EventBus/Job capture.
        if self.stderr:
            has_content = any(
                (isinstance(p, TextPart) and p.text.strip()) or isinstance(p, ImagePart)
                for p in self.stderr
            )
            if has_content:
                compacted: List[Union[TextPart, ImagePart]] = []
                for part in self.stderr:
                    if isinstance(part, TextPart) and part.text:
                        compacted.append(
                            TextPart(text=compact_diagnostic_text(part.text)),
                        )
                    else:
                        compacted.append(part)
                blocks.append({"type": "text", "text": "\n--- stderr ---\n"})
                blocks.extend(parts_to_llm_content(compacted))

        # Ensure we always return at least something
        if not blocks:
            blocks.append({"type": "text", "text": "(no output)"})

        return blocks

    def _plain_llm_content(self) -> List[dict]:
        """``UNIFY_PLAIN_CELL_OUTPUT``: the result as a notebook cell shows it.

        What the cell printed (stdout; then stderr, after a ``[stderr]``
        line), then ``Out: <repr>`` of the last expression's value when it is
        not None, then the traceback; a note on the call's arguments
        (``UNIFY_PLACEHOLDER_NOTE``) first, and what steered the block
        (interjections it received, functions patched) last. No session or
        timing metadata, and no steering counters for a block nothing
        steered. Images keep their place among the printed parts.
        """
        parts: List[Union[TextPart, ImagePart]] = []

        def text(value: str) -> None:
            if parts and isinstance(parts[-1], TextPart):
                previous = parts[-1].text
                if previous and not previous.endswith("\n"):
                    value = "\n" + value
            parts.append(TextPart(text=value))

        def has_content(stream: List[Union[TextPart, ImagePart]]) -> bool:
            return any(
                (isinstance(p, TextPart) and p.text.strip()) or isinstance(p, ImagePart)
                for p in stream
            )

        if self.note is not None:
            text(f"[note] {self.note}\n")
        if has_content(self.stdout):
            parts.extend(self.stdout)
        if has_content(self.stderr):
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
            import json

            text(f"[steering] {json.dumps(self.steering, default=str)}\n")
        blocks = parts_to_llm_content(parts)
        if not blocks:
            return [{"type": "text", "text": "(no output)"}]
        last = blocks[-1]
        if last.get("type") == "text":
            last["text"] = last["text"].rstrip("\n")
        return blocks


def _repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception as exc:  # noqa: BLE001 - a broken __repr__ still shows something
        return f"<{type(value).__name__} (repr failed: {type(exc).__name__})>"


def _holds_handle(value: Any) -> bool:
    """Whether *value* is, or holds, a steerable handle or the sentinel the loop
    puts in its place once adopted: such a result keeps the JSON envelope."""
    import re

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
