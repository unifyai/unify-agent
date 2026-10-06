"""Per-execution stream capture for sandbox stdout/stderr.

Provides ContextVar-based isolation so concurrent sandbox executions each
capture their own stdout/stderr output as structured TextPart/ImagePart lists.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import io
import sys
from typing import Any, Callable, List, Union

from .types import ImagePart, TextPart

# ---------------------------------------------------------------------------
# ContextVars for per-execution stream isolation
# ---------------------------------------------------------------------------
_stdout_parts: contextvars.ContextVar[List[Union[TextPart, ImagePart]]] = (
    contextvars.ContextVar(
        "sandbox_stdout_parts",
    )
)
_stderr_parts: contextvars.ContextVar[List[Union[TextPart, ImagePart]]] = (
    contextvars.ContextVar(
        "sandbox_stderr_parts",
    )
)
_current_stdout: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "sandbox_current_stdout",
)
_current_stderr: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "sandbox_current_stderr",
)
# A cell's stdin: always empty. The process's own stdin is the harness's
# (``unify act --jsonl`` takes its messages there), and a cell runs on the
# event loop's thread, so a cell that read it would wait, with the whole
# process, for a line only the host can send, and then take that line.
_current_stdin: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "sandbox_current_stdin",
)


# ---------------------------------------------------------------------------
# Stream capture classes
# ---------------------------------------------------------------------------
class StreamLike:
    """Captures output to a parts list, supporting text and images."""

    def __init__(
        self,
        parts_var: contextvars.ContextVar[List[Union[TextPart, ImagePart]]],
    ):
        self._parts_var = parts_var

    def write(self, obj: str) -> int:
        parts = self._parts_var.get()
        # Merge consecutive text writes into a single TextPart
        if parts and isinstance(parts[-1], TextPart):
            # TextPart is immutable (Pydantic), so we need to replace it
            last = parts[-1]
            parts[-1] = TextPart(text=last.text + obj)
        else:
            parts.append(TextPart(text=obj))
        return len(obj)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class StreamRouter:
    """Routes writes to the current context's stream, falls back to original stream.

    Uses __getattr__ to forward ALL unknown attributes/methods to the current stream,
    ensuring compatibility with Jupyter's introspection (e.g., _ipython_* methods),
    and any future stream methods we haven't explicitly handled.
    """

    def __init__(
        self,
        context_var: contextvars.ContextVar[Any],
        fallback: Any,
    ):
        # Use object.__setattr__ to avoid triggering our __getattr__
        object.__setattr__(self, "_context_var", context_var)
        object.__setattr__(self, "_fallback", fallback)

    def _get_stream(self) -> Any:
        try:
            return self._context_var.get()
        except LookupError:
            return self._fallback

    def write(self, s: str) -> int:
        return self._get_stream().write(s)

    def flush(self) -> None:
        return self._get_stream().flush()

    def __iter__(self) -> Any:
        # ``for line in sys.stdin``: iteration is looked up on the type, so
        # __getattr__ does not forward it.
        return iter(self._get_stream())

    def __getattr__(self, name: str) -> Any:
        """Forward any unknown attribute to the current stream."""
        return getattr(self._get_stream(), name)


# ---------------------------------------------------------------------------
# Lazy StreamRouter installation (installed on first sandbox use)
# ---------------------------------------------------------------------------
# We install the StreamRouter lazily (on first sandbox use) rather than at
# module load to avoid conflicts with pytest and other test frameworks that
# replace sys.stdout after imports. By installing on first use, we capture
# whatever stdout is current at that moment (e.g., pytest's capture) as our
# fallback, ensuring proper output routing.
_stream_router_installed = False
_original_stdout: Any = None
_original_stderr: Any = None


def _ensure_stream_router_installed() -> None:
    """Install StreamRouters for sys.stdout/stderr if not already installed.

    This is called at the start of each sandbox execution. We check if
    sys.stdout is actually a StreamRouter (not just a flag) because pytest
    and other frameworks may replace sys.stdout between tests.
    """
    global _stream_router_installed, _original_stdout, _original_stderr

    # stdin is routed on its own: something may replace one stream and not
    # the other. A process with no stdin (None) has nothing a cell could
    # wait on, and keeps it.
    if sys.stdin is not None and not isinstance(sys.stdin, StreamRouter):
        sys.stdin = StreamRouter(_current_stdin, sys.stdin)  # type: ignore[assignment]

    # Check if sys.stdout is still our StreamRouter (pytest may have replaced it)
    if isinstance(sys.stdout, StreamRouter):
        return  # Already installed

    # Install StreamRouter, capturing current stdout as fallback
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = StreamRouter(_current_stdout, _original_stdout)  # type: ignore[assignment]
    sys.stderr = StreamRouter(_current_stderr, _original_stderr)  # type: ignore[assignment]
    _stream_router_installed = True


# ---------------------------------------------------------------------------
# Display function for rich output (images, etc.)
# ---------------------------------------------------------------------------
_IMAGE_BASE64_LIMIT = 5_242_880  # Per-image base64 limit (5 MB)
_SAFETY_MAX_EDGE = 8_000  # Hard reject cap if policy somehow exceeds this


def _make_display(
    parts_var: contextvars.ContextVar[List[Union[TextPart, ImagePart]]],
) -> Callable[[Any], None]:
    """Create a display function that adds images to output parts."""

    def display(obj: Any) -> None:
        try:
            from PIL import Image
        except ImportError:
            Image = None  # type: ignore[misc, assignment]

        parts = parts_var.get()

        if Image is not None and isinstance(obj, Image.Image):
            b64_data, mime = _encode_image_for_llm(obj)
            parts.append(ImagePart(mime=mime, data=b64_data))
        elif isinstance(obj, str):
            parts.append(TextPart(text=obj + "\n"))
        else:
            parts.append(TextPart(text=str(obj) + "\n"))

    return display


def _encode_image_for_llm(img: Any) -> tuple[str, str]:
    """Encode a PIL Image to base64, ensuring it stays within API limits.

    First fits the image to the model-aware observation space (aspect targets
    and provider max-edge caps), then applies PNG encoding with JPEG fallback
    if the base64 payload exceeds the size limit.
    """
    from unify.common.observation_scaling import (
        _cap_dimensions,
        fit_image_to_observation_space,
    )

    img = fit_image_to_observation_space(img)

    w, h = img.size
    capped_w, capped_h = _cap_dimensions(w, h, _SAFETY_MAX_EDGE)
    if (capped_w, capped_h) != (w, h):
        from PIL import Image as _Image

        img = img.resize((capped_w, capped_h), _Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode("ascii")
    if len(b64_data) <= _IMAGE_BASE64_LIMIT:
        return b64_data, "image/png"

    rgb_img = img.convert("RGB") if img.mode != "RGB" else img
    for quality in (85, 60, 40, 20):
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=quality)
        b64_data = base64.b64encode(buf.getvalue()).decode("ascii")
        if len(b64_data) <= _IMAGE_BASE64_LIMIT:
            return b64_data, "image/jpeg"

    raise ValueError(
        f"Image too large for the LLM API even after JPEG compression "
        f"(quality={quality}): {len(b64_data):,} bytes base64, "
        f"limit is {_IMAGE_BASE64_LIMIT:,} bytes (5 MB). "
        f"Resize the image to smaller dimensions before calling display().",
    )


# ---------------------------------------------------------------------------
# Context manager for sandbox output capture
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def capture_sandbox_output():
    """Context manager that sets up stream capture for a sandbox execution.

    Yields (stdout_parts, stderr_parts, display_fn) tuple.
    All ContextVars are properly reset on exit.

    The StreamRouter is installed lazily on first use (not at module load)
    to avoid conflicts with pytest and other test frameworks that replace
    sys.stdout after imports.
    """
    # Ensure StreamRouter is installed (lazy, once per process)
    _ensure_stream_router_installed()

    stdout_parts: List[Union[TextPart, ImagePart]] = []
    stderr_parts: List[Union[TextPart, ImagePart]] = []

    # Set up ContextVars
    stdout_token = _stdout_parts.set(stdout_parts)
    stderr_token = _stderr_parts.set(stderr_parts)

    # Create StreamLike instances for this execution
    stdout_stream = StreamLike(_stdout_parts)
    stderr_stream = StreamLike(_stderr_parts)

    stdout_stream_token = _current_stdout.set(stdout_stream)
    stderr_stream_token = _current_stderr.set(stderr_stream)
    # Reading it returns end of input at once: ``input()`` raises EOFError,
    # ``help()`` ends, ``sys.stdin.read()`` is "".
    stdin_token = _current_stdin.set(io.StringIO(""))

    display_fn = _make_display(_stdout_parts)

    try:
        yield stdout_parts, stderr_parts, display_fn
    finally:
        _stdout_parts.reset(stdout_token)
        _stderr_parts.reset(stderr_token)
        _current_stdout.reset(stdout_stream_token)
        _current_stderr.reset(stderr_stream_token)
        _current_stdin.reset(stdin_token)
