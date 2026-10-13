from __future__ import annotations

import asyncio
import threading
from typing import Any, List

from unify.common.hierarchical_logger import ICONS

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


class SimulatedLineage:
    """Helpers for building nested labels and previews for simulated flows."""

    PREVIEW_LIMIT = 120

    @staticmethod
    def parent_lineage() -> List[str]:
        try:
            # Local import to avoid import cycles at module import time
            from unify.common._async_tool.loop_config import (
                TOOL_LOOP_LINEAGE,
            )  # noqa: WPS433
        except Exception:
            return []
        try:
            val = TOOL_LOOP_LINEAGE.get([])
            return list(val) if isinstance(val, list) else []
        except Exception:
            return []

    @staticmethod
    def has_outer() -> bool:
        try:
            return bool(SimulatedLineage.parent_lineage())
        except Exception:
            return False

    @staticmethod
    def make_label(segment: str) -> str:
        """Compose a nested label like '<outer(xxxx)>->Segment(abcd)'."""
        from secrets import token_hex  # noqa: WPS433

        try:
            parts = SimulatedLineage.parent_lineage()  # already suffixed
        except Exception:
            parts = []
        suffixed = f"{segment}({token_hex(2)})"
        return "->".join([*parts, suffixed]) if parts else suffixed

    @staticmethod
    def question_label(parent_label: str) -> str:
        """Build a concise child label 'Question(<parent>)(abcd)'."""
        from secrets import token_hex  # noqa: WPS433

        return f"Question({parent_label})({token_hex(2)})"

    # --- New helpers for consistent session suffix reuse ---------------------
    @staticmethod
    def extract_suffix(label: str) -> "str | None":
        """
        Return the trailing '(xxxx)' hex suffix from a label when present.
        """
        s = str(label or "").strip()
        if not s.endswith(")"):
            return None
        try:
            open_idx = s.rfind("(")
            if open_idx == -1:
                return None
            return s[open_idx + 1 : -1] or None
        except Exception:
            return None

    @staticmethod
    def make_label_with_suffix(segment: str, suffix: str) -> str:
        """
        Compose '<outer(xxxx)>->Segment(abcd)' using the provided suffix.

        Parent segments from TOOL_LOOP_LINEAGE are already suffixed;
        only the leaf segment gets the suffix.
        """
        from secrets import token_hex  # noqa: WPS433

        try:
            parts = SimulatedLineage.parent_lineage()  # already suffixed
        except Exception:
            parts = []
        suf = str(suffix or "").strip()
        suffixed = f"{segment}({suf})" if suf else f"{segment}({token_hex(2)})"
        return "->".join([*parts, suffixed]) if parts else suffixed

    @staticmethod
    def preview(text: str, limit: int = PREVIEW_LIMIT) -> str:
        s = str(text or "")
        return s if len(s) <= int(limit) else f"{s[:int(limit)]}…"


class SimulatedLog:
    """Small wrapper for consistent iconised request/steering logs."""

    _ICONS = {
        "ask": ICONS["clarification"],
        "update": "📝",
        "execute": "🎬",
        "act": "🎬",
        "interject": ICONS["interjection"],
        "pause": ICONS["pause"],
        "resume": ICONS["resume"],
        "stop": ICONS["stop_requested"],
        # simulated-only convenience
        "clar_req": ICONS["clarification"],
        "clar_ans": ICONS["interjection"],
        "notification": ICONS["notification"],
        # session lifecycle
        "session_start": ICONS["session_start"],
        "session_end": ICONS["session_end"],
    }
    _VERBS = {
        "ask": "Ask requested",
        "update": "Update requested",
        "execute": "Execute requested",
        "act": "Act requested",
        "interject": "Interject requested",
        "pause": "Pause requested",
        "resume": "Resume requested",
        "stop": "Stop requested",
        # simulated-only convenience
        "clar_req": "Clarification requested",
        "clar_ans": "Clarification answer received",
        "notification": "Notification",
        # session lifecycle
        "session_start": "Session started",
        "session_end": "Session ended",
    }

    @staticmethod
    def log_request(kind: str, label: str, text: str = "") -> None:
        try:
            from unify.logger import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            icon = SimulatedLog._ICONS.get(kind, ICONS.get("info", "ℹ️"))
            verb = SimulatedLog._VERBS.get(kind, "Requested")
            suffix = ""
            if kind in {"ask", "update", "act", "interject"}:
                prev = SimulatedLineage.preview(text)
                if prev:
                    suffix = f": {prev}"
            LOGGER.info(f"{icon} [{label}] {verb}{suffix}")
        except Exception:
            # Never let logging break control flow
            pass

    @staticmethod
    def log_clarification_request(label: str, question: str) -> None:
        """Emit a standardised clarification-request log line."""
        try:
            from unify.logger import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            q = SimulatedLineage.preview(question)
            LOGGER.info(
                f"{ICONS['clarification']} [{label}] Clarification requested: {q}",
            )
        except Exception:
            pass

    @staticmethod
    def log_clarification_answer(label: str, answer: str) -> None:
        """Emit a standardised clarification-answer log line."""
        try:
            from unify.logger import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            a = SimulatedLineage.preview(answer)
            LOGGER.info(
                f"{ICONS['interjection']} [{label}] Clarification answer received: {a}",
            )
        except Exception:
            pass

    @staticmethod
    def log_notification(label: str, message: str) -> None:
        """Emit a standardised notification log line."""
        try:
            from unify.logger import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            m = SimulatedLineage.preview(message)
            LOGGER.info(f"{ICONS['notification']} [{label}] Notification: {m}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Tool-call style logging helpers (gated by parent lineage)
# ─────────────────────────────────────────────────────────────────────────────
def maybe_tool_log_scheduled(segment: str, method: str, args: dict):
    """
    Emit a standardized 'ToolCall Scheduled' log line when there is no parent lineage.
    Returns (label, call_id, t0) on success; otherwise None.
    """
    try:
        if SimulatedLineage.has_outer():
            return None
        from unify.logger import LOGGER  # noqa: WPS433
        import time as _time  # noqa: WPS433

        label = SimulatedLineage.make_label(segment)
        cid = SimulatedLineage.extract_suffix(label) or ""
        try:
            LOGGER.info("%s ToolCall Scheduled", ICONS["info"])
        except Exception:
            pass
        return label, cid, _time.perf_counter()
    except Exception:
        return None


def maybe_tool_log_scheduled_with_label(label: str, method: str, args: dict):
    """
    Emit 'ToolCall Scheduled' using a precomputed label when there is no parent lineage.
    Returns (label, call_id, t0) on success; otherwise None.
    """
    try:
        if SimulatedLineage.has_outer():
            return None
        from unify.logger import LOGGER  # noqa: WPS433
        import json as _json  # noqa: WPS433
        import time as _time  # noqa: WPS433

        cid = SimulatedLineage.extract_suffix(label) or ""
        try:
            LOGGER.info(
                f"{ICONS['info']} [{label}] ToolCall Scheduled | args={_json.dumps(args)}",
            )
        except Exception:
            pass
        return label, cid, _time.perf_counter()
    except Exception:
        return None


def maybe_tool_log_completed(
    label: str,
    cid: str,
    method: str,
    result: dict,
    t0: float,
) -> None:
    """
    Emit a standardized 'ToolCall Completed' log line when there is no parent lineage.
    """
    try:
        if SimulatedLineage.has_outer():
            return
        from unify.logger import LOGGER  # noqa: WPS433
        import json as _json  # noqa: WPS433
        import time as _time  # noqa: WPS433

        dt = _time.perf_counter() - float(t0)
        try:
            LOGGER.info(
                f"{ICONS['completed']} [{label}] ToolCall Completed in {dt:.2f}s | result={_json.dumps(result)}",
            )
        except Exception:
            pass
    except Exception:
        pass


async def simulated_llm_roundtrip(
    llm: Any,
    *,
    label: str,
    prompt: str,
    response_format: Any = None,
) -> Any:
    """Unified 'LLM simulating' roundtrip with console logging.

    LLM I/O debugging is now handled by hooks installed on the unify client.

    Parameters
    ----------
    llm : Any
        The LLM client to use for generation.
    label : str
        Human-readable label for logging.
    prompt : str
        The prompt to send to the LLM.
    response_format : Type[BaseModel] | None
        Optional Pydantic model for structured output. When provided,
        the LLM's response_format is set before generation and reset after.
    """
    try:
        from unify.logger import LOGGER  # noqa: WPS433
    except Exception:
        LOGGER = None  # type: ignore

    import time as _time  # noqa: WPS433

    try:
        if LOGGER is not None:
            LOGGER.info(f"{ICONS['llm_thinking']} [{label}] LLM simulating…")
    except Exception:
        pass
    t0 = _time.perf_counter()

    # Set response_format if provided
    if response_format is not None:
        try:
            llm.set_response_format(response_format)
        except Exception:
            pass
    try:
        answer = await llm.generate(prompt)
    finally:
        if response_format is not None:
            try:
                llm.reset_response_format()
            except Exception:
                pass
    dt_ms = int((_time.perf_counter() - t0) * 1000)

    try:
        if LOGGER is not None:
            if SimulatedLineage.has_outer():
                LOGGER.info(f"{ICONS['completed']} [{label}] LLM replied in {dt_ms} ms")
            else:
                _ans_preview = str(answer)
                if len(_ans_preview) > 800:
                    _ans_preview = _ans_preview[:800] + "…"
                LOGGER.info(
                    f"{ICONS['completed']} [{label}] LLM replied in {dt_ms} ms:\n{_ans_preview}",
                )
    except Exception:
        pass

    # If a response_format is requested, mirror real Unify behavior:
    # return a validated Pydantic model instance (not a JSON string) when possible.
    if response_format is not None:
        try:
            from pydantic import BaseModel as _BaseModel  # noqa: WPS433
        except Exception:
            _BaseModel = None  # type: ignore[assignment]

        try:
            if _BaseModel is not None and isinstance(answer, _BaseModel):
                return answer
        except Exception:
            pass

        # dict-like payloads: validate directly
        if isinstance(answer, dict):
            try:
                return response_format.model_validate(answer)  # type: ignore[attr-defined]
            except Exception:
                return answer

        # string payloads: validate JSON; tolerate NDJSON by selecting last valid line
        if isinstance(answer, str):
            try:
                return response_format.model_validate_json(answer)  # type: ignore[attr-defined]
            except Exception:
                best = None
                for ln in [s.strip() for s in answer.splitlines() if s.strip()]:
                    try:
                        best = response_format.model_validate_json(ln)  # type: ignore[attr-defined]
                    except Exception:
                        continue
                if best is not None:
                    return best

    return answer


class SimulatedHandleMixin:
    """Lightweight mixin to standardise steering logs for simulated handles.

    Provides an optional **completion gate**: when ``hold_completion=True`` is
    passed to :meth:`_init_completion_gate`, the handle's ``result()`` blocks
    (and ``done()`` returns ``False``) until :meth:`trigger_completion` is
    called externally.  This enables deterministic test control over handle
    lifetimes without relying on timing.
    """

    # Derived classes are expected to set: self._log_label : str

    # ── Pause state proxy ────────────────────────────────────────────────

    @property
    def _pause_event(self):
        """Proxy for pause state compatibility with ``get_handle_paused_state``
        (``unify/legacy/handle_state.py``, read by the legacy conversation manager).

        Simulated manager handles track pause state via a ``_paused`` boolean
        rather than a real ``threading.Event``.  This property exposes that
        boolean through the ``is_set()`` interface that
        ``get_handle_paused_state`` expects, following the async-tool-loop
        convention (set = running, cleared = paused).

        If a subclass stores a real ``threading.Event`` via the setter (e.g.
        ``SimulatedActorHandle``), that object is returned directly.

        Returns ``None`` when the handle has neither a stored event nor a
        ``_paused`` attribute, causing ``get_handle_paused_state`` to return
        ``None`` (unknown).
        """
        # A subclass wrote a real Event — return it as-is.
        real = self.__dict__.get("_real_pause_event")
        if real is not None:
            return real

        if not hasattr(self, "_paused"):
            return None

        handle = self

        class _Proxy:
            def is_set(self) -> bool:
                return not handle._paused

        return _Proxy()

    @_pause_event.setter
    def _pause_event(self, value):
        """Allow subclasses to store a real ``threading.Event``."""
        self.__dict__["_real_pause_event"] = value

    # ── Completion gate ──────────────────────────────────────────────────
    _completion_gate: "threading.Event | None" = None

    def _init_completion_gate(self, hold_completion: bool = False) -> None:
        """Set up the optional completion gate.

        Args:
            hold_completion: When ``True``, the gate starts *closed* and
                ``result()`` / ``done()`` will block until
                :meth:`trigger_completion` is called.  When ``False``
                (the default), no gate is created and behaviour is
                unchanged from the pre-gate era.
        """
        import threading as _threading  # noqa: WPS433

        if hold_completion:
            self._completion_gate = _threading.Event()
        else:
            self._completion_gate = None

    async def _await_completion_gate(self) -> None:
        """Block (async-friendly) until the completion gate is open."""
        gate = self._completion_gate
        if gate is None:
            return
        while not gate.is_set():
            await asyncio.sleep(0.05)

    def _open_completion_gate(self) -> None:
        """Open the gate, unblocking any waiters."""
        gate = self._completion_gate
        if gate is not None:
            gate.set()

    @property
    def _gate_open(self) -> bool:
        """``True`` when the gate is open (or absent)."""
        return self._completion_gate is None or self._completion_gate.is_set()

    def trigger_completion(self, result: str | None = None) -> None:
        """Release the completion gate so ``result()`` can return.

        Subclasses (e.g. ``SimulatedActorHandle``) may extend this to
        also finalise internal state.
        """
        self._open_completion_gate()

    # ── Clarifications ───────────────────────────────────────────────────

    async def next_clarification(self) -> dict:
        """Block until cancelled — default for handles that don't use clarifications.

        Handles that support clarifications (``_needs_clar=True``) override
        this to block on their ``_clar_up_q`` instead.  The watcher's
        ``asyncio.wait_for(..., timeout=30)`` handles the timeout naturally.
        """
        await asyncio.Event().wait()
        return {}  # unreachable; satisfies return type

    # ── Notifications ────────────────────────────────────────────────────

    async def next_notification(self) -> dict:
        """Block until cancelled — simulated handles don't emit notifications.

        Callers (e.g. ``actor_watch_notifications``) wrap this in
        ``asyncio.wait_for(..., timeout=N)`` which raises ``TimeoutError``
        and re-checks ``handle.done()``.  This matches the real
        ``SteerableToolLoopHandle`` behaviour where ``next_notification``
        blocks on an ``asyncio.Queue.get()`` that may never receive an item.
        """
        await asyncio.Event().wait()
        return {}  # unreachable; satisfies return type

    # ── Steering logs ────────────────────────────────────────────────────

    def _log_interject(self, message: str) -> None:
        try:
            SimulatedLog.log_request(
                "interject",
                getattr(self, "_log_label", "handle"),
                str(message),
            )
        except Exception:
            pass

    def _log_pause(self) -> None:
        try:
            SimulatedLog.log_request("pause", getattr(self, "_log_label", "handle"))
        except Exception:
            pass

    def _log_resume(self) -> None:
        try:
            SimulatedLog.log_request("resume", getattr(self, "_log_label", "handle"))
        except Exception:
            pass

    def _log_stop(self, reason: str | None) -> None:
        try:
            from unify.logger import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            suffix = f" – reason: {reason}" if reason else ""
            LOGGER.info(
                f"{ICONS['stop_requested']} [{getattr(self, '_log_label', 'handle')}] Stop requested{suffix}",
            )
        except Exception:
            pass
