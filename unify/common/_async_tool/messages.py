"""
Contains classes and helpers for manipulating and managing messages in an async tool loop.
"""

import copy
import hashlib
import json
import unillm
from typing import Callable, Optional, Any
from .utils import maybe_await
from ...logger import LOGGER
from ...common.hierarchical_logger import DEFAULT_ICON
from contextlib import suppress, contextmanager
from .tools_utils import create_tool_call_message

# Sent-watermark invariant: a message is immutable once it has been included
# in any dispatched LLM request; everything from the watermark index onward
# is still free to mutate. Provider prefix caching matches serialized
# requests byte-for-byte from position 0, so any edit below the watermark
# invalidates the cached prefix for every subsequent request.


def _message_index(client, msg: dict) -> Optional[int]:
    """Identity-index of *msg* within ``client.messages``, or ``None`` if absent.

    Identity (``is``), not equality — two structurally-identical dicts must
    not be confused, and message dicts are mutated in place over their
    lifetime so equality is not stable anyway.
    """
    msgs = getattr(client, "messages", None) or []
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i] is msg:
            return i
    return None


def is_mutable(client, msg: dict) -> bool:
    """True when *msg* has not yet been included in any dispatched request.

    Fails closed: a message absent from the transcript (e.g. a swapped-out
    canonical log during a concurrent dispatch) is treated as immutable, so
    callers route its content through the tail-append paths that reach the
    model instead of writing into a dict the transcript will never contain.
    """
    idx = _message_index(client, msg)
    if idx is None:
        return False
    watermark = getattr(client, "_sent_watermark", 0)
    return idx >= watermark


def loop_user_notice(content: Any, **extra: Any) -> dict:
    """Build a ``role="user"`` message the loop itself authors — status,
    threshold, quota, or context-continuation notices — never a genuine
    user turn.

    This is the only place that stamps ``_loop_authored``, the marker
    ``is_loop_authored_message`` checks; every loop-authored user-role
    message must be built here rather than as an inline dict literal so
    the marker cannot be forgotten at a call site. ``extra`` accepts the
    purpose-specific markers (``_progress_msg``, ``_clarify_msg``,
    ``_lifecycle_msg``, ``_ctx_header``) callers need for their own
    coalescing/filtering — ``_loop_authored`` is stamped regardless, so
    the boundary check never depends on which of those was passed.

    A genuine user interjection (``_interjection``) is built at its own
    call site, never through here — that asymmetry is what makes it a
    real turn boundary.
    """
    return {"role": "user", "content": content, "_loop_authored": True, **extra}


def is_loop_authored_message(msg: dict) -> bool:
    """True for a ``role="user"`` message the loop itself appended, never
    a genuine new user turn.

    Every consumer that must tell "the user said something" apart from
    "the loop said something" (e.g. a boundary check that must not treat
    a loop-authored notice as the start of a new request) uses this
    predicate.
    """
    return bool(msg.get("_loop_authored"))


def extract_substantive_text(content: Any) -> Optional[str]:
    """Normalize assistant content to the text a user would read, for
    deciding whether a turn carries a substantive answer.

    Handles a plain string and a multimodal content-block list (only
    ``"text"`` blocks contribute). Returns ``None`` when the result is
    empty or whitespace-only in either shape, so callers use ``is None``
    rather than truthiness — which would pass a whitespace-only string
    and misreport a non-empty block list whose every block is blank. A
    block list with substantive text yields the extracted text, not the
    raw list, since every consumer treats the answer as plain text.
    Shared by the final-answer walk-back and the parent context snapshot
    filter so both apply one definition of "substantive".
    """
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        joined = "".join(texts)
        return joined if joined.strip() else None
    return None


def _hash_msgs_slice(msgs: list) -> str:
    try:
        blob = json.dumps(msgs, sort_keys=True, default=str)
    except Exception:
        blob = repr(msgs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Diagnostic switch: loop behaviour is identical either way. Off by default
# because the below-watermark hashing costs real per-dispatch CPU on long
# transcripts; the test suite turns it on so CI catches unsanctioned mutation.
_INVARIANT_CHECKS_ENV = "UNIFY_TRANSCRIPT_INVARIANT_CHECKS"


def _invariant_checks_enabled() -> bool:
    import os

    return os.environ.get(_INVARIANT_CHECKS_ENV) == "1"


def _call_id_has_reply(client, call_id: Optional[str]) -> bool:
    """True if some tool-role message in the transcript already answers *call_id*."""
    if not call_id:
        return False
    for m in getattr(client, "messages", None) or []:
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id:
            return True
    return False


def _rebaseline_watermark_hash(client) -> None:
    """Recompute the stored watermark hash after a sanctioned escape-hatch splice.

    A ``bypass_watermark`` splice deliberately shifts content at indices the
    previous hash covered (legality beats cache); without re-baselining, the
    next dispatch's integrity check would read that shift as an unsanctioned
    mutation and raise.
    """
    if not _invariant_checks_enabled():
        return
    watermark = getattr(client, "_sent_watermark", 0)
    with suppress(Exception):
        client._sent_watermark_hash = _hash_msgs_slice(client.messages[:watermark])


_REVIEW_COMPACTION_MARKER = "[compacted after skill review:"
_REVIEW_COMPACTION_MIN_CHARS = 800
_REVIEW_COMPACTION_HEAD_CHARS = 300


_REASONING_PAYLOAD_KEYS = ("provider_specific_fields", "reasoning_details", "reasoning")


def strip_reasoning_payloads(msg: dict) -> int:
    """Drop provider reasoning machinery from one message, in place.

    Encrypted reasoning blobs and summaries let a provider continue an
    in-flight chain of thought; once that turn is over they are re-billed
    bulk, often the largest component of a long-lived transcript. The
    visible ``content`` is never touched. Returns the serialized
    characters removed (approximate, for accounting).
    """
    saved = 0
    for key in _REASONING_PAYLOAD_KEYS:
        if key in msg and msg[key] is not None:
            try:
                saved += len(json.dumps(msg[key], default=str))
            except (TypeError, ValueError):
                saved += 0
            msg.pop(key, None)
    return saved


def compact_reviewed_messages(client, reviewed_message_count: int) -> int:
    """Shed the bulk of an already-reviewed transcript span, in place.

    Once a storage review has consolidated a stretch of the transcript into
    stored functions, guidance and claims, that stretch's raw machinery is
    dead weight that every later dispatch and review re-pays. Within the
    first ``reviewed_message_count`` messages this pass:

    * replaces bulky *tool* result contents with a head slice plus an
      omission marker, and
    * strips provider reasoning payloads from assistant messages — a
      completed turn's chain of thought is not needed to continue.

    Message identity, ordering and tool_call pairing are untouched, so
    nothing holding a reference to a message dict ever sees it disappear.
    User-facing words — requests, requirements, the assistant's visible
    replies — stay verbatim. Placeholder/progress replies, small contents,
    image-bearing parts and already-compacted messages are left alone.

    Mutating below the sent watermark is sanctioned here like an
    escape-hatch splice: the watermark hash is re-baselined afterwards,
    trading provider prefix cache for a permanently smaller transcript.

    Returns the number of characters removed.
    """
    saved = 0
    messages = list(getattr(client, "messages", None) or [])
    span = messages[: max(0, min(reviewed_message_count, len(messages)))]
    for msg in span:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            saved += strip_reasoning_payloads(msg)
            continue
        if msg.get("role") != "tool":
            continue
        if is_non_final_tool_reply(msg):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            if any(
                not (isinstance(part, dict) and part.get("type") == "text")
                for part in content
            ):
                continue
            text = "\n".join(str(part.get("text") or "") for part in content)
        else:
            continue
        if len(text) < _REVIEW_COMPACTION_MIN_CHARS:
            continue
        if _REVIEW_COMPACTION_MARKER in text or "[img:" in text:
            continue
        stub = (
            f"{text[:_REVIEW_COMPACTION_HEAD_CHARS]}\n… "
            f"{_REVIEW_COMPACTION_MARKER} "
            f"{len(text) - _REVIEW_COMPACTION_HEAD_CHARS} chars omitted]"
        )
        msg["content"] = stub
        saved += len(text) - len(stub)
    if saved:
        _rebaseline_watermark_hash(client)
    return saved


async def emit_completion_pair(
    result: str,
    call_id: str,
    msg_dispatcher: Any,
) -> dict:
    """
    Append a synthetic assistant→tool pair carrying *result* for *call_id*
    at the tail of the transcript instead of splicing it into an
    already-dispatched (below-watermark) position. This is the sole
    below-watermark delivery path for late tool results and, via
    ``insert_tool_message_after_assistant``'s gate, any other reply that
    would otherwise land below the mark.
    """
    status_call_id = f"{call_id}_completed"
    status_tool_name = f"check_status_{call_id}"

    assistant_stub = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": status_call_id,
                "type": "function",
                "function": {
                    "name": status_tool_name,
                    "arguments": "{}",
                },
            },
        ],
    }
    tool_msg = create_tool_call_message(
        name=status_tool_name,
        call_id=status_call_id,
        content=result,
    )

    await msg_dispatcher.append_msgs([assistant_stub, tool_msg])
    return tool_msg


@contextmanager
def _preserve_canonical_messages(client, canonical_msgs):
    """Make ``client.messages`` return *canonical_msgs* for the duration of the block.

    A class-level property cannot be shadowed by an instance attribute, so
    the property itself is temporarily patched to consult
    `_canonical_messages` first.
    """
    prop_class = None
    orig_prop = None
    try:
        client._canonical_messages = canonical_msgs
        for klass in type(client).__mro__:
            if "messages" in klass.__dict__:
                prop_class = klass
                orig_prop = klass.__dict__["messages"]
                break
        if prop_class is not None and orig_prop is not None:

            def _patched_getter(self, _orig=orig_prop):
                cm = getattr(self, "_canonical_messages", None)
                return cm if cm is not None else _orig.fget(self)

            prop_class.messages = property(_patched_getter)
    except Exception:
        pass
    try:
        yield
    finally:
        if prop_class is not None and orig_prop is not None:
            with suppress(Exception):
                prop_class.messages = orig_prop
        with suppress(Exception):
            del client._canonical_messages


# TODO: Some of these helpers should not be placed here, but in utils.py or their own files


def is_non_final_tool_reply(msg: dict) -> bool:
    """Return True when a tool message is a placeholder/progress reply, not a final result.

    - Clarification wrappers (name starts with "clarification_request_") are
      non-final. Live clarifications are delivered as "[clarification <call_id>]"
      user-role tail messages by ToolsData.record_clarification; only persisted
      transcripts still carry the wrapper shape.
    - Any tool message whose content parses to a dict with the top-level key
      "_placeholder" is non-final (pending/progress/nested-start placeholders).
    """
    try:
        if msg.get("role") != "tool":
            return False
        name = str(msg.get("name") or "")
        if name.startswith("clarification_request_"):
            return True
        content = msg.get("content")
        if isinstance(content, str):
            try:
                import json as _json

                parsed = _json.loads(content)
                if isinstance(parsed, dict) and "_placeholder" in parsed:
                    return True
            except Exception:
                pass
    except Exception:
        return False
    return False


# Assistant messages whose tool_calls lack a final reply before the next
# assistant message.
def find_unreplied_assistant_entries(client: unillm.AsyncUnify) -> list[dict]:
    findings: list[dict] = []
    try:
        for i, m in enumerate(client.messages):
            if m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            if not tcs:
                continue
            ids = [tc.get("id") for tc in tcs if isinstance(tc, dict)]
            if not ids:
                continue
            responded: set[str] = set()
            j = i + 1
            while (
                j < len(client.messages)
                and client.messages[j].get("role") != "assistant"
            ):
                mm = client.messages[j]
                if mm.get("role") == "tool":
                    tcid = mm.get("tool_call_id")
                    if tcid in ids and not is_non_final_tool_reply(mm):
                        responded.add(tcid)
                j += 1
            missing = [c for c in ids if c not in responded]
            if missing:
                findings.append(
                    {
                        "assistant_index": i,
                        "assistant_msg": m,
                        "missing": missing,
                    },
                )
    except Exception:
        pass
    return findings


async def generate_with_preprocess(
    client: unillm.AsyncUnify,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]],
    **gen_kwargs,
):
    # Sent watermark: everything below this index has been (or is about to
    # be) included in a dispatched request and must never be mutated again.
    # Set here — the one place both llm_task dispatch sites funnel through —
    # on the pre-copy length, since the deep copy taken below is what gets
    # serialized, and advanced *before* the request goes out, so nothing can
    # edit what an in-flight request carries.
    prev_watermark = getattr(client, "_sent_watermark", 0)
    prev_hash = getattr(client, "_sent_watermark_hash", None)
    _checks_on = _invariant_checks_enabled()
    if _checks_on and prev_hash is not None:
        # The below-watermark slice must be byte-identical to what was hashed
        # at the last dispatch, unless a sanctioned escape-hatch splice
        # re-baselined it in between (_rebaseline_watermark_hash).
        assert _hash_msgs_slice(client.messages[:prev_watermark]) == prev_hash, (
            "Append-only transcript invariant violated: a message below "
            "the sent watermark was mutated between dispatches."
        )
    pre_copy_len = len(client.messages)
    client._sent_watermark = max(prev_watermark, pre_copy_len)
    if _checks_on:
        client._sent_watermark_hash = _hash_msgs_slice(
            client.messages[: client._sent_watermark],
        )

    # Stamp the in-flight window so ``handle.ask()``, which snapshots
    # ``client.messages``, can report "waiting on an LLM response since T"
    # instead of dead-ending silently. Cleared in ``finally`` so cancellation
    # (interjection pre-emption) never leaves a stale stamp behind.
    import time as _time

    client._llm_inflight_since = _time.time()
    try:
        return await _generate_with_preprocess_inner(
            client,
            preprocess_msgs,
            **gen_kwargs,
        )
    except BaseException:
        # A dispatch that ends without a response — cancelled because a tool
        # result or steering event superseded it, a read-only cache miss, a
        # provider error — adds nothing to the transcript, so it is undone
        # as if it had never been sent. Were its tail left frozen, the next
        # request would depend on whether the superseding event landed just
        # before this dispatch (a result is written into its placeholder) or
        # just after it (the result arrives as a check_status pair): a few
        # milliseconds of tool latency that a cached replay cannot reproduce.
        # The price is whatever prefix the provider cached for that
        # unanswered tail.
        client._sent_watermark = prev_watermark
        client._sent_watermark_hash = prev_hash
        raise
    finally:
        client._llm_inflight_since = None


_NOT_RECORDED = object()


def _record_sent_request(
    client: Any,
    patched: Optional[list[dict]],
    gen_kwargs: dict,
) -> Any:
    """Under UNIFY_CACHE_DISCIPLINE, keep what this dispatch sends.

    *patched* is the preprocessed list about to be sent; ``None`` means the
    client's own transcript, to which ``generate`` prepends the system
    prompt when no system message is present. Returns what to restore if
    the dispatch ends without a response (``_NOT_RECORDED`` when off).
    """
    from . import cache_discipline

    if not cache_discipline.records_requests():
        return _NOT_RECORDED
    if patched is None:
        patched = list(getattr(client, "messages", None) or [])
        system = getattr(client, "system_message", None)
        if system is not None and not any(
            isinstance(m, dict) and m.get("role") == "system" for m in patched
        ):
            patched = [{"role": "system", "content": system}] + patched
    return cache_discipline.record_sent_request(client, patched, gen_kwargs)


def _restore_sent_request(client: Any, previous: Any) -> None:
    if previous is _NOT_RECORDED:
        return
    from . import cache_discipline

    cache_discipline.restore_sent_request(client, previous)


async def _generate_with_preprocess_inner(
    client: unillm.AsyncUnify,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]],
    **gen_kwargs,
):
    if preprocess_msgs is None:
        _previous = _record_sent_request(client, None, gen_kwargs)
        try:
            return await maybe_await(client.generate(**gen_kwargs))
        except BaseException:
            _restore_sent_request(client, _previous)
            raise

    original_msgs = client.messages  # reference to canonical log
    msgs_copy = copy.deepcopy(original_msgs)

    try:
        patched = preprocess_msgs(msgs_copy) or msgs_copy
    except Exception as exc:  # resilience – don't fail the loop
        LOGGER.error(
            f"{DEFAULT_ICON} preprocess_msgs raised {exc!r}; using original messages.",
        )
        patched = msgs_copy

    sys_txt = getattr(client, "system_message", "") or ""
    sys_patched = sys_txt

    # The client's generate() skips prepending system_message when ANY system
    # message already exists in messages[], so a preprocessor that adds one
    # (e.g. for provider compatibility) would silently drop the original
    # system prompt. Prepend it explicitly unless it is already first.
    if sys_txt:
        first_is_original_system = (
            patched
            and patched[0].get("role") == "system"
            and patched[0].get("content") == sys_txt
        )
        if not first_is_original_system:
            patched = [{"role": "system", "content": sys_txt}] + patched

    start_len = len(patched)

    # The real ``AsyncUnify`` keeps its transcript in the private ``_messages``
    # attribute that ``.generate`` reads, while lightweight doubles expose only
    # a public ``messages`` list; patch whichever one ``generate`` consumes.
    # Swapping ``_messages`` would make the public ``messages`` property return
    # the patched list too, racing external code polling ``client.messages``,
    # so _preserve_canonical_messages keeps the canonical log visible meanwhile.
    target_attr = "_messages" if hasattr(client, "_messages") else "messages"
    original_system_message = getattr(client, "system_message", None)
    with suppress(Exception):
        if original_system_message is not None:
            setattr(client, "system_message", sys_patched)

    original_container = getattr(client, target_attr)

    preserve_ctx = (
        _preserve_canonical_messages(client, original_container)
        if target_attr == "_messages"
        else suppress()
    )

    _previous = _record_sent_request(client, patched, gen_kwargs)
    with preserve_ctx:
        setattr(client, target_attr, patched)
        try:
            try:
                result = await maybe_await(client.generate(**gen_kwargs))
            except BaseException:
                _restore_sent_request(client, _previous)
                raise

            # Copy whatever the LLM produced back into the canonical log.
            current_msgs = getattr(client, target_attr)
            if len(current_msgs) > start_len:
                original_msgs.extend(copy.deepcopy(current_msgs[start_len:]))

            return result
        finally:
            setattr(client, target_attr, original_container)
            with suppress(Exception):
                if original_system_message is not None:
                    setattr(client, "system_message", original_system_message)


def _normalise_kwargs_for_bound_method(
    bound_method,
    incoming_kw: dict,
) -> tuple[dict, dict]:
    """Normalise kwargs for a bound method: expand nested kwargs, drop noise keys,
    map common aliases when there is a single public param, and coerce string
    values to annotated types.

    Returns ``(accepted, unknown)``: the kwargs the method takes, and the ones
    it has no parameter for. ``unknown`` is always empty for a method that
    accepts ``**kwargs``. Nothing is dropped silently here; each caller decides
    what an unknown name means for it.
    """
    import inspect as _inspect

    params = _inspect.signature(bound_method).parameters
    has_varkw = any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values())

    kw = dict(incoming_kw or {})

    # 1) Expand nested {"kwargs": {...}}
    if "kwargs" in kw and isinstance(kw["kwargs"], dict):
        nested_kw = kw.pop("kwargs")
        for k, v in nested_kw.items():
            kw.setdefault(k, v)

    # 2) Drop common placeholder noise keys when empty
    for _noise in ("a", "kw"):
        if _noise in kw and (kw[_noise] is None or kw[_noise] == ""):
            kw.pop(_noise, None)

    # 3) If exactly one public param, accept common aliases
    public_params = [n for n in params if n != "self"]
    if len(public_params) == 1 and public_params[0] not in kw:
        for alias in (
            "content",
            "message",
            "text",
            "prompt",
            "guidance",
            "instruction",
            "question",
            "query",
        ):
            if alias in kw:
                kw[public_params[0]] = kw.pop(alias)
                break

    # 4) Set aside the keys the method has no parameter for, unless it
    #    accepts **kwargs
    unknown = {} if has_varkw else {k: kw.pop(k) for k in list(kw) if k not in params}

    # 5) Coerce string values to annotated int/float/bool/dict types
    #    (best-effort): LLMs often pass every argument as a string.
    #    Annotations may be real types or strings (under
    #    `from __future__ import annotations`), so both forms are checked.
    import json as _json

    for param_name, param in params.items():
        if param_name not in kw or param_name == "self":
            continue
        annotation = param.annotation
        if annotation is _inspect.Parameter.empty:
            continue
        val = kw[param_name]
        try:
            ann_str = annotation if isinstance(annotation, str) else ""
            origin = getattr(annotation, "__origin__", None)

            is_int = annotation is int or ann_str == "int"
            is_float = annotation is float or ann_str == "float"
            is_bool = annotation is bool or ann_str == "bool"
            is_dict = (
                annotation is dict
                or ann_str == "dict"
                or ann_str.startswith("Dict[")
                or (origin is not None and origin is dict)
            )

            if is_int and isinstance(val, str):
                kw[param_name] = int(val)
            elif is_float and isinstance(val, str):
                kw[param_name] = float(val)
            elif is_bool and isinstance(val, str):
                kw[param_name] = val.lower() in ("true", "1", "yes")
            elif is_dict and isinstance(val, str):
                kw[param_name] = _json.loads(val)
        except (ValueError, _json.JSONDecodeError):
            pass

    return kw, unknown


def apply_llm_soft_required_defaults(bound_method, kwargs: dict) -> dict:
    """Backfill arguments advertised as required but optional at runtime.

    See :func:`unify.common.tool_spec.llm_soft_required`. For each parameter a
    tool declared as soft-required, if the model omitted it from *kwargs* the
    configured default is filled in. Parameters not declared soft-required are
    left untouched, so a genuine omission of a functional argument still raises
    the usual error the model can self-correct against.

    Mutates and returns *kwargs* for convenience.
    """
    from unify.common.tool_spec import LLM_SOFT_REQUIRED_DEFAULTS_ATTR

    defaults = getattr(bound_method, LLM_SOFT_REQUIRED_DEFAULTS_ATTR, None)
    if not defaults:
        return kwargs

    import inspect as _inspect

    params = _inspect.signature(bound_method).parameters
    for name, default in defaults.items():
        if name in params and name not in kwargs:
            kwargs[name] = default
    return kwargs


async def forward_handle_call(
    handle: Any,
    method_name: str,
    kwargs: dict | None,
    *,
    call_args: list | tuple | None = None,
    fallback_positional_keys: list[str] | tuple[str, ...] = (),
):
    """Invoke a steering method on a handle with tolerant kwargs handling.

    Filters/normalises kwargs against the bound method's signature; if the
    method rejects them, retries positionally with the first available key
    from fallback_positional_keys (e.g. reason/content), and finally with no
    arguments at all.

    Keys the method has no parameter for are dropped, where a base tool call
    carrying one is refused. The loop assembles these kwargs itself, to the
    base steering contract (a stop reason, an interjection's continuation
    context), and forwards them to handles whose methods may take fewer; no
    model wrote them, so there is nobody to tell. The one payload a model
    does write, steer's ``action="call"``, is bound against the method's
    signature before it is forwarded here.
    """
    try:
        bound = getattr(handle, method_name)
    except Exception:
        return None

    try:
        args = list(call_args or [])
        normalised, _not_taken = _normalise_kwargs_for_bound_method(
            bound,
            kwargs or {},
        )
        return await maybe_await(bound(*args, **normalised))
    except TypeError:
        # Fallbacks: positional-only, then kwargs-only, then single-key
        # positional extraction via fallback_positional_keys.
        try:
            args2 = list(call_args or [])
            return await maybe_await(bound(*args2))  # type: ignore[misc]
        except Exception:
            pass
        try:
            return await maybe_await(bound(**(normalised if isinstance(normalised, dict) else {})))  # type: ignore[misc]
        except Exception:
            pass
        for k in fallback_positional_keys:
            if kwargs and k in kwargs:
                try:
                    rest_kwargs = (
                        dict(normalised) if isinstance(normalised, dict) else {}
                    )
                except Exception:
                    rest_kwargs = {}
                try:
                    # Never pass the alias key twice if it also matched a parameter.
                    rest_kwargs.pop(k, None)
                except Exception:
                    pass
                try:
                    return await maybe_await(bound(kwargs.get(k), **rest_kwargs))  # type: ignore[misc]
                except Exception:
                    pass
        try:
            return await maybe_await(bound())  # type: ignore[misc]
        except Exception:
            return None
    except Exception:
        # Steering failures must never crash the loop.
        return None


# The static steering/inspection surface (steer/wait/ask_about_completed_tool)
# is acknowledged rather than re-executed during backfill: the underlying
# async work is gone on restart whichever action was requested.
def _is_helper_tool(name: str) -> bool:
    return name in ("wait", "steer", "ask_about_completed_tool")


def build_helper_ack_content(name: str, args_json: Any) -> str:
    ack_content = "Acknowledged."
    try:
        payload = (
            json.loads(args_json or "{}")
            if isinstance(args_json, str)
            else (args_json or {})
        )
    except Exception:
        payload = {}

    if name == "wait":
        ack_content = "Waiting acknowledged. Keeping current tool calls in flight."
    elif name == "steer":
        action = str(payload.get("action") or "").strip().lower()
        steer_payload = payload.get("payload")
        if action == "stop":
            ack_content = "Stop request acknowledged. If the underlying call is still running, it will be stopped."
        elif action == "pause":
            ack_content = "Pause request acknowledged. If the underlying call is still running, it will be paused."
        elif action == "resume":
            ack_content = "Resume request acknowledged. If the underlying call was paused, it will be resumed."
        elif action == "clarify":
            ack_content = (
                f"Clarification answer received: {steer_payload!r}. Waiting for the original tool to proceed."
                if steer_payload is not None
                else "Clarification helper acknowledged. Waiting for the original tool to proceed."
            )
        elif action == "interject":
            ack_content = (
                f"Guidance forwarded to the running tool: {steer_payload!r}."
                if steer_payload
                else "Interjection acknowledged and forwarded to the running tool."
            )
        elif action == "ask":
            ack_content = "Ask request acknowledged and forwarded to the running tool."
        elif action == "call":
            method = payload.get("method")
            ack_content = (
                f"Call to {method!r} acknowledged and forwarded to the running tool."
                if method
                else "Custom method call acknowledged and forwarded to the running tool."
            )
        else:
            ack_content = "Steering request acknowledged."
    elif name == "ask_about_completed_tool":
        ack_content = "Follow-up question acknowledged and forwarded for retrospective inspection."
    else:
        ack_content = (
            f"Operation {name!r} acknowledged and forwarded to the running tool."
        )
    return ack_content


# Prune a `wait` tool call from an assistant message; if it was the only tool
# call and there is no content, drop the assistant message from the transcript.
#
# Below the sent watermark, *asst_msg* was already included in a dispatched
# request, so popping or editing its tool_calls entry would mutate cached
# bytes. The stale wait is instead left untouched and acknowledged via a tool
# reply spliced directly after asst_msg, bypassing the watermark gate: a
# one-time prefix break, accepted because a wait tool_calls entry with no
# reply anywhere is a permanently illegal transcript, not just an uncached one.
async def prune_wait_tool_call(
    asst_msg: dict,
    call_id: str,
    *,
    client: unillm.AsyncUnify | None = None,
    assistant_meta: Optional[dict] = None,
    msg_dispatcher: Any = None,
) -> None:
    if client is not None and not is_mutable(client, asst_msg):
        if assistant_meta is None or msg_dispatcher is None:
            # A missing param must never fall through to the pop/in-place
            # edit below — that is exactly the mutation this branch prevents.
            # Logged as well as raised because every caller wraps this in a
            # broad suppress/except-pass that would swallow the raise.
            _msg = (
                "prune_wait_tool_call: asst_msg is already below the sent "
                "watermark, so popping or editing its tool_calls in place "
                "would mutate already-dispatched bytes — but assistant_meta "
                "and msg_dispatcher (needed to route the ack through the "
                "escape hatch) were not both provided."
            )
            LOGGER.error(f"{DEFAULT_ICON} {_msg}")
            raise ValueError(_msg)
        await acknowledge_helper_call(
            asst_msg,
            call_id,
            "wait",
            "{}",
            assistant_meta=assistant_meta,
            client=client,
            msg_dispatcher=msg_dispatcher,
            bypass_watermark=True,
        )
        return

    try:
        tool_calls = asst_msg.get("tool_calls") or []
        remaining = [c for c in tool_calls if c.get("id") != call_id]
        content_present = bool((asst_msg.get("content") or "").strip())
        if not remaining:
            if not content_present:
                if client is not None:
                    try:
                        if client.messages and client.messages[-1] is asst_msg:
                            client.messages.pop()
                        else:
                            idx_in_log = client.messages.index(asst_msg)
                            client.messages.pop(idx_in_log)
                    except Exception:
                        pass
                else:
                    asst_msg.pop("tool_calls", None)
            else:
                asst_msg.pop("tool_calls", None)
        else:
            asst_msg["tool_calls"] = remaining
    except Exception:
        pass


async def insert_tool_message_after_assistant(
    assistant_meta: dict,
    parent_msg: dict,
    tool_msg,
    client,
    msg_dispatcher,
    *,
    skip_event_bus: bool = False,
    bypass_watermark: bool = False,
) -> None:
    """
    Append *tool_msg* and move it directly after *parent_msg*, while
    updating the per-assistant `results_count` bookkeeping.

    If *skip_event_bus* is True, the message is appended to the client
    transcript but not published to the EventBus — used for placeholder
    messages that are updated in place later.

    If the insertion position falls below the client's sent watermark,
    splicing there would shift every already-dispatched message that
    follows and break the provider's cached prefix from that point on, so
    the message is instead delivered as a check_status pair appended at
    the tail — *unless* the transcript would otherwise become illegal:
    when ``tool_msg``'s call_id has no reply anywhere yet, this insertion
    is the first-ever reply, and redirecting it would permanently orphan
    the original ``tool_calls`` entry (a check_status pair answers a
    different, synthesized call_id). That case always splices, whether or
    not the caller passed *bypass_watermark* — legality beats cache,
    enforced here rather than trusted to every call site. A caller with
    its own reason to force the splice (the backfill/restore escape hatch)
    passes *bypass_watermark* explicitly.

    A sanctioned below-watermark splice re-baselines the stored watermark
    hash immediately, so the next dispatch's integrity check (when enabled)
    reads it as the new legitimate state rather than a violation.
    """
    call_id = tool_msg.get("tool_call_id") if isinstance(tool_msg, dict) else None

    if (
        not bypass_watermark
        and client is not None
        and call_id is not None
        and not _call_id_has_reply(client, call_id)
    ):
        bypass_watermark = True

    watermark = getattr(client, "_sent_watermark", 0) if client is not None else 0
    parent_idx = _message_index(client, parent_msg) if client is not None else None
    existing_meta = assistant_meta.get(id(parent_msg))
    results_count = existing_meta["results_count"] if existing_meta else 0
    insert_pos = (parent_idx + 1 + results_count) if parent_idx is not None else None
    below_watermark = insert_pos is not None and insert_pos < watermark

    if below_watermark and not bypass_watermark:
        content = (
            tool_msg.get("content") if isinstance(tool_msg, dict) else str(tool_msg)
        )
        await emit_completion_pair(
            content,
            call_id or "unknown",
            msg_dispatcher,
        )
        return

    # Only now mark the parent handled — a reply diverted to check_status
    # above must not suppress the preflight repair that looks for
    # unanswered tool_calls entries.
    meta = assistant_meta.setdefault(id(parent_msg), {"results_count": 0})
    await msg_dispatcher.append_msgs([tool_msg], skip_event_bus=skip_event_bus)
    final_insert_pos = _message_index(client, parent_msg) + 1 + meta["results_count"]
    client.messages.insert(final_insert_pos, client.messages.pop())
    meta["results_count"] += 1

    if below_watermark:
        _rebaseline_watermark_hash(client)


async def acknowledge_helper_call(
    asst_msg: dict,
    call_id: str,
    name: str,
    args_json: Any,
    *,
    assistant_meta,
    client,
    msg_dispatcher,
    bypass_watermark: bool = False,
) -> None:
    tool_msg = create_tool_call_message(
        name=name,
        call_id=call_id,
        content=build_helper_ack_content(name, args_json),
    )
    await insert_tool_message_after_assistant(
        assistant_meta,
        asst_msg,
        tool_msg,
        client,
        msg_dispatcher,
        bypass_watermark=bypass_watermark,
    )


# Ensure placeholder tool messages exist for pending tasks — those spawned by
# assistant_msg when given, otherwise all of them. Returns the call_ids for
# which a placeholder was created.
async def ensure_placeholders_for_pending(
    assistant_msg: Optional[dict] = None,
    *,
    tools_data,
    assistant_meta,
    client,
    msg_dispatcher,
    time_ctx=None,
) -> list[str]:
    created: list[str] = []
    # Sorted by call_idx so placeholders follow the tool_calls array order and
    # the "at tail" check in process_completed_task is independent of set
    # iteration order.
    for task in sorted(
        list(tools_data.pending),
        key=lambda t: getattr(tools_data.info.get(t), "call_idx", 0),
    ):
        _inf = tools_data.info.get(task)
        if not _inf:
            continue
        if assistant_msg is not None and _inf.assistant_msg is not assistant_msg:
            continue
        # Reuse any existing tool reply message in the transcript for this call_id
        try:
            if _inf.tool_reply_msg is None:
                existing = None
                msgs = client.messages or []
                for m in msgs:
                    try:
                        if m.get("role") == "tool" and str(
                            m.get("tool_call_id"),
                        ) == str(_inf.call_id):
                            existing = m
                            break
                    except Exception:
                        continue
                if existing is not None:
                    _inf.tool_reply_msg = existing
        except Exception:
            pass
        if _inf.tool_reply_msg or _inf.clarify_placeholder:
            continue

        # Self-describing so a permanently-frozen stub (below-watermark by
        # the time the result arrives) still reads truthfully: the result
        # never rewrites this message, it always arrives as a check_status
        # pair appended below. "meta:"-prefixed keys annotate without
        # changing what the placeholder *is* (still "pending").
        ph_content: dict = {
            "_placeholder": "pending",
            "meta:status": "async — result arrives as a check_status message below",
        }
        if time_ctx is not None:
            with suppress(Exception):
                ph_content["meta:started"] = time_ctx.offset_at(_inf.scheduled_time)

        placeholder = create_tool_call_message(
            name=_inf.name,
            call_id=_inf.call_id,
            content=json.dumps(ph_content, indent=4),
        )
        # The first-ever reply to a call_id must sit immediately after its
        # assistant message — an API legality requirement, not a caching
        # nicety — so this always bypasses the watermark gate.
        await insert_tool_message_after_assistant(
            assistant_meta,
            _inf.assistant_msg,
            placeholder,
            client,
            msg_dispatcher,
            skip_event_bus=True,  # Don't publish placeholders; publish when final
            bypass_watermark=True,
        )
        _inf.tool_reply_msg = placeholder
        created.append(_inf.call_id)

    return created


# Schedule a subset of tool_calls on a past assistant message and insert
# placeholders immediately. Skips already-scheduled/finished ids.
async def schedule_missing_for_message(
    asst_msg: dict,
    only_ids: set[str],
    *,
    tools_data,
    context_state,
    propagate_chat_context,
    assistant_meta,
    client,
    msg_dispatcher,
    initial_paused: bool = False,
) -> list[str]:
    scheduled: list[str] = []
    try:
        tool_calls = asst_msg.get("tool_calls") or []
        for idx, call in enumerate(tool_calls):
            cid = call.get("id")
            if cid not in only_ids:
                continue

            if any(task_info.call_id == cid for task_info in tools_data.info.values()):
                continue
            if cid in tools_data.completed_results:
                continue

            name = call["function"]["name"]
            args_json = call["function"].get("arguments", "{}")

            if _is_helper_tool(name):
                # `wait` must not clutter the transcript.
                if name == "wait":
                    try:
                        await prune_wait_tool_call(
                            asst_msg,
                            cid,
                            client=client,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=msg_dispatcher,
                        )
                    except Exception:
                        pass
                    scheduled.append(cid)
                    continue

                # Other helpers are acknowledged, not executed, during backfill.
                # This is the backfill/restore escape hatch: the call_id has no
                # reply anywhere yet, so the ack must splice adjacently
                # regardless of watermark; legality beats cache.
                try:
                    await acknowledge_helper_call(
                        asst_msg,
                        cid,
                        name,
                        args_json,
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=msg_dispatcher,
                        bypass_watermark=True,
                    )
                except Exception:
                    pass
                scheduled.append(cid)
                continue

            if name not in tools_data.normalized:
                scheduled.append(cid)
                continue

            await tools_data.schedule_base_tool_call(
                asst_msg,
                name=name,
                args_json=args_json,
                call_id=cid,
                call_idx=idx,
                context_state=context_state,
                propagate_chat_context=propagate_chat_context,
                assistant_meta=assistant_meta,
                msg_dispatcher=msg_dispatcher,
                initial_paused=initial_paused,
            )
            scheduled.append(cid)
    except Exception:
        pass
    with suppress(Exception):
        await ensure_placeholders_for_pending(
            assistant_msg=asst_msg,
            tools_data=tools_data,
            assistant_meta=assistant_meta,
            client=client,
            msg_dispatcher=msg_dispatcher,
        )
    return scheduled
