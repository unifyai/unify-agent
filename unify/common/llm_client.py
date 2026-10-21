from __future__ import annotations

import contextlib
import contextvars
import copy
import re
import secrets
from pathlib import Path
from typing import Any, Literal

import unillm

from unify.logger import LOGGER
from unify.common.hierarchical_logger import ICONS
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
    """The run's model and reasoning effort, ``(UNIFY_MODEL, UNIFY_REASONING_EFFORT)``.

    An effort of None means no effort override: each call site's own level applies.
    """
    effort = SETTINGS.UNIFY_REASONING_EFFORT.strip() or None
    return SETTINGS.UNIFY_MODEL, effort


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


# ── Request metadata headers ─────────────────────────────────────────────────
# Four HTTP headers on every model call, so a proxy can attribute it (for
# example, measure the cache hits of each session's first call and of the
# calls after a compaction). The values are random or counted by the harness
# and never request content; the request body is untouched (headers are not
# part of it), so the provider's prompt cache is not affected. With no proxy
# in front that strips them, the headers reach the model provider.

SESSION_HEADER = "X-Unify-Session"
PARENT_HEADER = "X-Unify-Parent"
REQUEST_HEADER = "X-Unify-Request"
CALL_KIND_HEADER = "X-Unify-Call-Kind"
MSG_COUNT_HEADER = "X-Unify-Msg-Count"

_OUTSIDE_HEADER_CHARSET = re.compile(r"[^A-Za-z0-9_.:-]")

#: ``X-Unify-Call-Kind`` of every model call a cell's code makes (its
#: ``query_llm`` and the ``unillm`` clients it builds,
#: unify/common/cell_models.py), whatever ``origin`` the call names: a proxy
#: tells these calls from the harness's own by this value, which cell code
#: cannot set.
CELL_CALL_KIND = "cell"
_CALL_KIND: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "unify_call_kind",
    default=None,
)


@contextlib.contextmanager
def call_kind(kind: str):
    """Model calls made inside the block (in this context) carry *kind* as
    ``X-Unify-Call-Kind`` in place of their client's origin."""
    token = _CALL_KIND.set(_header_value(kind))
    try:
        yield
    finally:
        _CALL_KIND.reset(token)


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
    """Add the headers to every call *client* makes.

    The headers are merged into the call's ``extra_headers`` where unillm
    has assembled the request (``_generate``), so ``X-Unify-Msg-Count``
    counts the messages sent, and every retry carries them. A fork is a new
    client (unillm's ``copy()`` copies no instance attribute), so it gets its
    own session id.
    """
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


# ── UNIFY_MEMORY_METERING: the pass and entries a call serves ─────────────────
# Ids only, never content: a writer, curator or helper call made inside a writer-service pass carries the pass's
# id and the attention-ledger entries handed to it, so the proxy's journal can attribute the call (the per-pass,
# per-entry and per-item views, unify/memory/metering.py). The fork sets its episode (and segment) on its own
# request headers (unify/memory/integration/fork.py).
PASS_HEADER = "X-Unify-Pass-Id"
ENTRIES_HEADER = "X-Unify-Entries"
#: A pass id or an entry id: 16 lower-case hex digits; the entries header carries at most this many, joined by ".".
_ATTRIBUTION_ID = re.compile(r"^[0-9a-f]{16}$")
MAX_HEADER_ENTRIES = 64
_ATTRIBUTION: contextvars.ContextVar["Attribution | None"] = contextvars.ContextVar(
    "unify_call_attribution",
    default=None,
)


class Attribution:
    """The pass (and the entries) the calls made in this context serve; the entries list may grow during the pass
    (an entry the writer takes)."""

    __slots__ = ("pass_id", "entries")

    def __init__(self, pass_id: str, entries: list[str] | None = None) -> None:
        self.pass_id = pass_id
        self.entries = list(entries or ())

    def headers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if _ATTRIBUTION_ID.match(self.pass_id or ""):
            out[PASS_HEADER] = self.pass_id
        ids = [e for e in self.entries if _ATTRIBUTION_ID.match(e or "")][:MAX_HEADER_ENTRIES]
        if ids:
            out[ENTRIES_HEADER] = ".".join(ids)
        return out


@contextlib.contextmanager
def attribution(pass_id: str, entries: list[str] | None = None):
    """Model calls made inside the block (this context, and the tasks it starts) carry *pass_id* and *entries* as
    ``X-Unify-Pass-Id`` and ``X-Unify-Entries``."""
    token = _ATTRIBUTION.set(Attribution(pass_id, entries))
    try:
        yield _ATTRIBUTION.get()
    finally:
        _ATTRIBUTION.reset(token)


def add_attributed_entry(entry_id: str) -> None:
    """An entry the pass in this context took on: its later calls name it too."""
    current = _ATTRIBUTION.get()
    if current is not None and entry_id not in current.entries:
        current.entries.append(entry_id)


def _request_metadata_headers(
    client: Any,
    state: _RequestMetadata,
    messages: Any,
) -> dict[str, str]:
    headers = {
        SESSION_HEADER: state.session,
        REQUEST_HEADER: str(state.requests),
        CALL_KIND_HEADER: (
            _CALL_KIND.get() or _header_value(getattr(client, "origin", None))
        ),
        MSG_COUNT_HEADER: str(len(messages or ())),
    }
    if state.parent is not None:
        headers[PARENT_HEADER] = state.parent
    current = _ATTRIBUTION.get()
    if current is not None:
        headers.update(current.headers())
    return headers


def count_requester_message(client: Any) -> None:
    """A loop answering a requester took one more requester message.

    ``X-Unify-Request`` on *client*'s later calls is the count: 1 from the
    request the loop starts with, then one more for each later requester
    message (the next request of a persistent session, or a message sent
    while a request runs).
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

    # Every call is sent as a stream and read to its end, so bytes arrive while
    # the model is still generating and a proxy that cancels a silent call
    # does not cancel a long one; unillm reassembles each reply into the
    # completion an unstreamed call returns, and ``copy()`` (a fork) keeps the
    # setting.
    client.set_stream_transport(True)

    if origin:
        pending_log = PendingThinkingLog(origin)
        client.set_on_log_file_pending(pending_log.on_pending_path)
        client._pending_thinking_log = pending_log

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

    If model is not specified, uses UNIFY_MODEL, and UNIFY_REASONING_EFFORT
    (when set) overrides the call site's effort.
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
    _install_request_metadata(fork, parent)
    return fork


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


