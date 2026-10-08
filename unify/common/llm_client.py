from __future__ import annotations

import copy
import re
import secrets
from pathlib import Path
from typing import Any, Literal

import unillm
from pydantic import BaseModel

from unify.logger import LOGGER
from unify.common.hierarchical_logger import ICONS
from unify.session_details import SESSION_DETAILS
from unify.settings import SETTINGS

_THINKING_ICON = ICONS["llm_thinking"]


class PendingThinkingLog:
    """Manages the combined 'LLM thinking… → /path' log for LLM calls.

    Callers set a thinking suffix (the parenthesised metadata) before
    ``generate()``.  The pending callback emits the combined one-liner.
    If ``UNILLM_LOG_DIR`` is unset the callback never fires, so
    ``emit_fallback`` produces a plain thinking line instead.
    """

    def __init__(self, origin: str) -> None:
        self._origin = origin
        self._suffix: str = ""
        self._emitted: bool = False

    def set_thinking_context(self, suffix: str) -> None:
        self._suffix = suffix
        self._emitted = False

    def on_pending_path(self, path: Path) -> None:
        self._emitted = True
        LOGGER.info(
            f"{_THINKING_ICON} [{self._origin}] LLM thinking…{self._suffix} → {path}",
        )

    def emit_fallback(self) -> None:
        if not self._emitted:
            self._emitted = True
            LOGGER.info(
                f"{_THINKING_ICON} [{self._origin}] LLM thinking…{self._suffix}",
            )


def resolve_default_model() -> tuple[str, str | None]:
    """Resolve the session's default LLM as (model, reasoning_effort).

    The per-assistant default (from the assistant record, via SESSION_DETAILS) takes
    priority over the deployment-wide UNIFY_MODEL. A returned effort of None
    means no effort override and per-call-site effort levels apply.
    """
    session_model = SESSION_DETAILS.assistant.default_model
    if session_model:
        return (
            session_model,
            SESSION_DETAILS.assistant.default_reasoning_effort or None,
        )
    effort = SETTINGS.UNIFY_REASONING_EFFORT.strip() or None
    return SETTINGS.UNIFY_MODEL, effort


def resolve_slow_brain_model() -> tuple[str, str | None]:
    """Resolve the ConversationManager slow-brain LLM as (model, effort).

    Priority:
    1. Per-assistant slow brain (assistant record / SESSION_DETAILS), including effort
    2. ``UNIFY_CONVERSATION_SLOW_BRAIN_MODEL`` when set (non-empty)
    3. The global shared model (``UNIFY_MODEL``)

    Independent of the actor ``default_model``. A returned effort of None means
    per-call-site effort levels apply.
    """
    session_model = SESSION_DETAILS.assistant.slow_brain_model
    if session_model:
        return (
            session_model,
            SESSION_DETAILS.assistant.slow_brain_reasoning_effort or None,
        )
    slow_model = SETTINGS.conversation.SLOW_BRAIN_MODEL.strip()
    if slow_model:
        effort = SETTINGS.conversation.SLOW_BRAIN_REASONING_EFFORT.strip() or None
        return slow_model, effort
    return SETTINGS.UNIFY_MODEL, None


LLMPurpose = Literal["planning"]
_PURPOSE_MARK = "#purpose="


def tag_origin_with_purpose(
    origin: str | None,
    purpose: LLMPurpose | None,
) -> str | None:
    """Encode ``purpose`` in the client's ``origin`` tag.

    ``origin`` is client-side metadata: it reaches ``LLMEvent.origin`` and the
    log filename, and never the request payload, so tagging it neither
    changes what the model sees nor perturbs the response cache.
    """
    if purpose is None:
        return origin
    base = (origin or "").split(_PURPOSE_MARK, 1)[0]
    return f"{base}{_PURPOSE_MARK}{purpose}"


def purpose_from_origin(origin: str | None) -> LLMPurpose | None:
    """Recover the purpose tag from an ``origin`` string, if one is present."""
    if not origin or _PURPOSE_MARK not in origin:
        return None
    value = origin.rsplit(_PURPOSE_MARK, 1)[1].strip()
    if value == "planning":
        return value  # type: ignore[return-value]
    return None


# ── UNIFY_REQUEST_METADATA_HEADERS ──────────────────────────────────────────
# Four HTTP headers on every model call, so a proxy can attribute it. The
# values are random or counted by the harness and never request content; the
# request body is untouched (headers are not part of it).

SESSION_HEADER = "X-Unify-Session"
PARENT_HEADER = "X-Unify-Parent"
REQUEST_HEADER = "X-Unify-Request"
CALL_KIND_HEADER = "X-Unify-Call-Kind"
MSG_COUNT_HEADER = "X-Unify-Msg-Count"

#: Every header value matches this.
HEADER_VALUE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_OUTSIDE_HEADER_CHARSET = re.compile(r"[^A-Za-z0-9_.:-]")


def request_metadata_headers_enabled() -> bool:
    """Whether ``UNIFY_REQUEST_METADATA_HEADERS=on``."""
    return getattr(SETTINGS, "UNIFY_REQUEST_METADATA_HEADERS", "") == "on"


def _header_value(text: Any) -> str:
    """*text* in the header charset, at most 64 characters, never empty."""
    value = _OUTSIDE_HEADER_CHARSET.sub("_", str(text or ""))[:64]
    return value or "none"


class _RequestMetadata:
    """One client's header state: its session id, its parent's, and the
    requester messages counted so far (see :func:`count_requester_message`)."""

    __slots__ = ("session", "parent", "requests")

    def __init__(self, parent: "_RequestMetadata | None" = None) -> None:
        self.session = secrets.token_hex(8)
        self.parent = parent.session if parent is not None else None
        self.requests = parent.requests if parent is not None else 0


def _install_request_metadata(
    client: "unillm.AsyncUnify | unillm.Unify",
    parent: "unillm.AsyncUnify | unillm.Unify | None" = None,
) -> None:
    """With ``UNIFY_REQUEST_METADATA_HEADERS=on``, add the headers to every
    call *client* makes; off, *client* is left untouched.

    The headers are merged into the call's ``extra_headers`` where unillm
    has assembled the request (``_generate``), so ``X-Unify-Msg-Count``
    counts the messages sent, and every retry (the tool-choice fallback's
    included) carries them. A fork is a new client (unillm's ``copy()``
    copies no instance attribute), so it gets its own session id.
    """
    if not request_metadata_headers_enabled():
        return
    state = _RequestMetadata(getattr(parent, "_unify_request_metadata", None))
    client._unify_request_metadata = state
    send = client._generate

    def _generate(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs["messages"] if "messages" in kwargs else args[0]
        headers = dict(kwargs.get("extra_headers") or {})
        headers.update(_request_metadata_headers(client, state, messages))
        kwargs["extra_headers"] = headers
        return send(*args, **kwargs)

    client._generate = _generate


def _request_metadata_headers(
    client: Any,
    state: _RequestMetadata,
    messages: Any,
) -> dict[str, str]:
    headers = {
        SESSION_HEADER: state.session,
        REQUEST_HEADER: str(state.requests),
        CALL_KIND_HEADER: _header_value(getattr(client, "origin", None)),
        MSG_COUNT_HEADER: str(len(messages or ())),
    }
    if state.parent is not None:
        headers[PARENT_HEADER] = state.parent
    return headers


def count_requester_message(client: Any) -> None:
    """A loop answering a requester took one more requester message.

    ``X-Unify-Request`` on *client*'s later calls is the count: 1 from the
    request the loop starts with, then one more for each later requester
    message (the next request of a persistent session, or a message sent
    while a request runs). A client without the headers is left alone.
    """
    state = getattr(client, "_unify_request_metadata", None)
    if state is not None:
        state.requests += 1


def _build_llm_client(
    model: str,
    *,
    async_client: bool,
    stateful: bool,
    origin: str | None,
    default_effort: str | None,
    kwargs: dict[str, Any],
    purpose: LLMPurpose | None = None,
) -> "unillm.AsyncUnify | unillm.Unify":
    origin = tag_origin_with_purpose(origin, purpose)
    config: dict[str, Any] = {
        "reasoning_effort": "high",
        "stateful": stateful,
        "origin": origin,
    }
    # Bound one turn's output so a degenerate generation cannot run to the
    # provider ceiling. Callers may still override per client.
    if SETTINGS.UNIFY_MAX_OUTPUT_TOKENS > 0:
        config["max_completion_tokens"] = SETTINGS.UNIFY_MAX_OUTPUT_TOKENS
    config.update(kwargs)
    if default_effort is not None:
        config["reasoning_effort"] = default_effort

    if async_client:
        client = unillm.AsyncUnify(model, **config)
    else:
        client = unillm.Unify(model, **config)

    if origin:
        pending_log = PendingThinkingLog(origin)
        client.set_on_log_file_pending(pending_log.on_pending_path)
        client._pending_thinking_log = pending_log

    if SETTINGS.UNIFY_TOOL_CHOICE_FALLBACK:
        from unify.common.tool_choice_fallback import install_tool_choice_fallback

        install_tool_choice_fallback(client)

    _install_request_metadata(client)
    return client


def new_llm_client(
    model: str | None = None,
    *,
    async_client: bool = True,
    stateful: bool = False,
    origin: str | None = None,
    purpose: LLMPurpose | None = None,
    **kwargs: Any,
) -> "unillm.AsyncUnify | unillm.Unify":
    """
    Create a configured Unify client.

    If model is not specified, uses the assistant's default model when one is
    set (which also pins its reasoning effort, overriding the call site), and
    otherwise UNIFY_MODEL from settings.
    Defaults to high reasoning_effort where applicable. Callers that want a
    different setting (e.g. fast-path helpers at "low", or max-effort actor
    profiles) pass ``reasoning_effort`` explicitly.
    ``purpose`` tags what the tokens buy — ``planning`` (an actor deciding
    what to do) — so per-purpose accounting can read it back from
    ``LLMEvent.origin`` via :func:`purpose_from_origin`.
    Caching is controlled by the UNILLM_CACHE env var (owned by unillm).
    Returns an AsyncUnify client by default, or a synchronous Unify client when
    async_client=False.
    """
    default_effort: str | None = None
    if model is None:
        model, default_effort = resolve_default_model()

    return _build_llm_client(
        model,
        async_client=async_client,
        stateful=stateful,
        origin=origin,
        default_effort=default_effort,
        kwargs=kwargs,
        purpose=purpose,
    )


def fork_llm_client(
    parent: "unillm.AsyncUnify | unillm.Unify",
    *,
    origin: str,
    purpose: LLMPurpose | None = None,
    messages: list[dict] | None = None,
) -> "unillm.AsyncUnify | unillm.Unify":
    """A client that continues *parent*'s conversation under another origin.

    Built with unillm's ``copy()``: the same model, system prompt, output
    ceiling and prompt-caching targets. ``copy()`` restores the reasoning
    effort *parent* was built with, so the effort it runs with now is set
    again, and a cache affinity key (when the installed unillm has one) is
    carried over so the fork reaches the replica holding the parent's
    prefix. The transcript is *messages*, deep-copied, when given, else a
    copy of the parent's. Nothing done on the fork reaches the parent.
    """
    fork = parent.copy()
    if messages is not None:
        fork._messages = copy.deepcopy(list(messages))
    effort = getattr(parent, "reasoning_effort", None)
    if effort is not None:
        fork.set_reasoning_effort(effort)
    affinity = getattr(parent, "cache_affinity", None)
    if affinity is not None and hasattr(fork, "set_cache_affinity"):
        fork.set_cache_affinity(affinity)
    origin = tag_origin_with_purpose(origin, purpose)
    fork.set_origin(origin)
    if origin:
        pending_log = PendingThinkingLog(origin)
        fork.set_on_log_file_pending(pending_log.on_pending_path)
        fork._pending_thinking_log = pending_log
    if SETTINGS.UNIFY_TOOL_CHOICE_FALLBACK:
        from unify.common.tool_choice_fallback import install_tool_choice_fallback

        install_tool_choice_fallback(fork)
    _install_request_metadata(fork, parent)
    return fork


def new_slow_brain_llm_client(
    model: str | None = None,
    *,
    async_client: bool = True,
    stateful: bool = False,
    origin: str | None = None,
    **kwargs: Any,
) -> "unillm.AsyncUnify | unillm.Unify":
    """Create an LLM client for ConversationManager slow-brain call sites.

    When ``model`` is omitted, resolves via :func:`resolve_slow_brain_model`
    (assistant slow brain → slow-brain setting → global ``UNIFY_MODEL``).
    Defaults to high reasoning effort; an assistant- or setting-level effort
    override pins the client effort the same way as :func:`new_llm_client`.
    """
    default_effort: str | None = None
    if model is None:
        model, default_effort = resolve_slow_brain_model()

    return _build_llm_client(
        model,
        async_client=async_client,
        stateful=stateful,
        origin=origin,
        default_effort=default_effort,
        kwargs=kwargs,
    )


def _make_openai_strict_json_schema_compatible(node: Any) -> None:
    """Mutate a JSON schema in-place to satisfy OpenAI strict requirements.

    OpenAI's strict JSON-schema mode requires that:
    - For any schema object with explicit `properties`, the `required` array
      must exist and include *every* property key.
    - `additionalProperties` must be false (to forbid extra keys).

    Pydantic excludes fields with default values from `required` (because they're
    optional at validation time). This helper normalizes the schema to the strict
    subset that OpenAI enforces.
    """
    if isinstance(node, dict):
        props = node.get("properties")
        if node.get("type") == "object" and isinstance(props, dict):
            node["additionalProperties"] = False
            node["required"] = list(props.keys())
        for v in node.values():
            _make_openai_strict_json_schema_compatible(v)
        return

    if isinstance(node, list):
        for v in node:
            _make_openai_strict_json_schema_compatible(v)


def pydantic_to_json_schema_response_format(
    response_model: type[BaseModel],
    *,
    name: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Build an OpenAI-style `response_format` dict from a Pydantic model.

    This returns the JSON-schema response format shape used by OpenAI:

        {"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}

    When `strict=True`, the schema is post-processed to satisfy OpenAI's strict
    constraints (see `_make_openai_strict_json_schema_compatible`).
    """
    schema = response_model.model_json_schema()
    if strict:
        schema = copy.deepcopy(schema)
        _make_openai_strict_json_schema_compatible(schema)
    else:
        # Keep the existing behaviour of forbidding unknown keys where possible.
        schema.setdefault("additionalProperties", False)
        for def_schema in schema.get("$defs", {}).values():
            if isinstance(def_schema, dict):
                def_schema.setdefault("additionalProperties", False)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or response_model.__name__,
            "schema": schema,
            "strict": strict,
        },
    }
