"""Keep a tool loop's requests a growing, byte-stable prefix (``UNIFY_CACHE_DISCIPLINE``).

Providers cache a prompt by its exact leading bytes: tools, system prompt,
then messages in order. A request reuses the cache only as far as it matches
the previous one byte for byte, so anything that changes early in the
request -- the tool list, an old message -- makes every later token cold.
With the switch on, the loop keeps what it advertises and what it has sent
fixed:

* **One tool list per session.** The advertised tools are computed once, in
  a deterministic order, and sent unchanged on every call. What a phase does
  not allow (a discovery gate, a context-full turn, a read-only library) is
  refused at call time with the rule that masks it, instead of being removed
  from the list ("mask, don't remove").
* **Sent messages are never edited.** Reasoning payloads are not shed when a
  persistent session parks, and a storage review's compaction note no
  longer shortens the turns it covered.
* **Compression is a fork.** The summary is asked for with the last request
  sent, unchanged, plus one appended instruction, so it is served from the
  cache; the session then continues from the summary under the same system
  prompt and tool list.
* **One cache per prefix.** A client whose unillm takes a cache affinity
  key gets one derived from its model, system prompt and fixed tool list
  (``UNIFY_CACHE_AFFINITY_SCOPE``: or one per session, or one per run), so
  a new session reaches the replica an earlier session with the same prefix
  cached it on; a fork shares its parent's key. Without the key the client
  is left as it is.
* **Measured.** Each call logs how many of its input tokens the provider
  served from the cache, and the running share for the session. A reply
  that does not report cached tokens is counted as unknown, not as zero.

The helpers here hold that policy so the loop itself only asks two questions
per turn: what to advertise, and whether a call is allowed. With the switch
off none of them is consulted.
"""

from __future__ import annotations

import copy
import contextvars
import hashlib
import json
import uuid
from typing import Any, Iterable, Optional

from ...logger import LOGGER

# The tool names a turn allows, set by the loop around each dispatch while the
# switch is on. A completion mutator that reads the request's tool list to
# decide what phase the turn is in (the actor's discovery mutator) reads this
# instead: under the switch the request always carries the whole list.
_TURN_AVAILABLE_TOOLS: contextvars.ContextVar[Optional[frozenset[str]]] = (
    contextvars.ContextVar("unify_turn_available_tools", default=None)
)


def enabled() -> bool:
    """Whether ``UNIFY_CACHE_DISCIPLINE`` is on."""
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", False))


def turn_available_tools() -> Optional[frozenset[str]]:
    """The tool names the current turn allows, or ``None`` outside the switch."""
    return _TURN_AVAILABLE_TOOLS.get()


def set_turn_available_tools(names: Iterable[str]) -> contextvars.Token:
    return _TURN_AVAILABLE_TOOLS.set(frozenset(names))


def reset_turn_available_tools(token: contextvars.Token) -> None:
    _TURN_AVAILABLE_TOOLS.reset(token)


def schema_name(schema: Any) -> Optional[str]:
    """The function name an OpenAI-style tool schema declares."""
    if not isinstance(schema, dict):
        return None
    function = schema.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = schema.get("name")
    return name if isinstance(name, str) else None


def schema_names(schemas: Iterable[Any]) -> list[str]:
    return [n for n in (schema_name(s) for s in schemas or []) if n]


def build_session_schema(
    *,
    base_schemas: dict[str, dict],
    compress_schema: Optional[dict],
    turn_schemas: list[dict],
) -> list[dict]:
    """The tool list a session advertises on every call.

    ``base_schemas`` holds every tool the caller passed, whether or not this
    turn's policy shows it, and is emitted in name order so the list does not
    depend on dict insertion. ``compress_context`` follows whenever
    compression is enabled, eager turns included. The loop-owned tools this
    turn assembled (the response tool, multi-request tools, ``wait``,
    ``steer``, ``ask_about_completed_tool``) are the same on every turn and
    keep the order the loop gives them.
    """
    fixed = [base_schemas[name] for name in sorted(base_schemas)]
    if compress_schema is not None:
        fixed.append(compress_schema)
    taken = set(base_schemas) | ({"compress_context"} if compress_schema else set())
    for schema in turn_schemas:
        name = schema_name(schema)
        if name and name not in taken:
            fixed.append(schema)
            taken.add(name)
    return fixed


def policy_mask_rules(policy_result: Any) -> tuple[dict[str, str], Optional[str]]:
    """The rules a ``tool_policy`` result gives for the tools it withholds.

    A policy may return ``(mode, tools, opts)`` with ``opts["mask_rules"]``
    (tool name -> why it is withheld) and ``opts["mask_rule"]`` (why every
    other withheld tool is). Both are optional; the loop reads nothing else
    from them, so a policy that sets them behaves as before with the switch
    off.
    """
    if not isinstance(policy_result, (tuple, list)) or len(policy_result) < 3:
        return {}, None
    opts = policy_result[2]
    if not isinstance(opts, dict):
        return {}, None
    rules = opts.get("mask_rules")
    rules = (
        {str(k): str(v) for k, v in rules.items() if v}
        if isinstance(rules, dict)
        else {}
    )
    default = opts.get("mask_rule")
    return rules, (str(default) if default else None)


def masked_tool_refusal(
    name: str,
    *,
    advertised: Iterable[str],
    available: Iterable[str],
    rule: Optional[str],
) -> str:
    """The tool reply for a call to a tool this turn does not allow.

    It names the rule that masks the tool and what can be called instead, so
    the model can act on it rather than retry.
    """
    advertised = set(advertised)
    now = sorted(set(available))
    if name not in advertised:
        why = f"`{name}` is not in this session's tool list."
    elif rule:
        why = f"`{name}` is not available right now: {rule.rstrip('.')}."
    else:
        why = f"`{name}` is not available on this turn."
    listed = ", ".join(f"`{n}`" for n in now) if now else "none"
    return (
        f"⚠️ {why} The tool list stays the same for the whole session, so a "
        "tool that is not allowed yet is refused rather than removed. "
        f"Available now: {listed}."
    )


# ── the last request a client sent ─────────────────────────────────────────

_LAST_SENT = "_unify_last_sent_request"


def review_fork_enabled() -> bool:
    """Whether ``UNIFY_REVIEW_FORK`` is on."""
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_REVIEW_FORK", False))


def records_requests() -> bool:
    """Whether dispatches keep a copy of what they send (for a later fork)."""
    return enabled() or review_fork_enabled()


def record_sent_request(client: Any, messages: list, gen_kwargs: dict) -> Any:
    """Keep what one dispatch sends: its messages, tools and tool choice.

    Only these three are kept -- never the call's other keyword arguments,
    which may carry credentials. Returns the record it replaced, so a
    dispatch that ends without a response can put it back.
    """
    previous = getattr(client, _LAST_SENT, None)
    setattr(
        client,
        _LAST_SENT,
        {
            "messages": copy.deepcopy(list(messages)),
            "tools": copy.deepcopy(gen_kwargs.get("tools")),
            "tool_choice": copy.deepcopy(gen_kwargs.get("tool_choice")),
        },
    )
    return previous


def restore_sent_request(client: Any, previous: Any) -> None:
    setattr(client, _LAST_SENT, previous)


def last_sent_request(client: Any) -> Optional[dict]:
    """The last request *client* sent through the tool loop, if recorded."""
    record = getattr(client, _LAST_SENT, None)
    return record if isinstance(record, dict) else None


def is_forced_tool_choice(tool_choice: Any) -> bool:
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() in ("required", "any")
    return isinstance(tool_choice, dict)


COMPRESSION_FORK_INSTRUCTION = (
    "Context compression: the conversation above is about to be replaced by "
    "the summary you write now, and you will continue the task from that "
    "summary alone. Write it as plain text; do not call any tool. Include: "
    "the task and every requirement, correction or preference the user gave, "
    "verbatim where the wording matters; what has been done and what was "
    "found, with the exact names, ids, paths and values the rest of the task "
    "needs (stored functions and guidance used included); what failed and "
    "why; the current state; and the next steps. Leave out tool output that "
    "is no longer needed."
)


def completion_text(content: Any) -> str:
    """The visible text of a reply's content (a string or content parts)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


# ── cache affinity and the cache-hit metric ────────────────────────────────


def affinity_scope() -> str:
    """``UNIFY_CACHE_AFFINITY_SCOPE``: ``prefix``, ``session`` or ``run``."""
    from unify.settings import SETTINGS

    scope = str(getattr(SETTINGS, "UNIFY_CACHE_AFFINITY_SCOPE", "") or "prefix")
    return scope if scope in ("prefix", "session", "run") else "prefix"


_RUN_AFFINITY: Optional[str] = None


def run_affinity_key() -> str:
    """The key every session of this process shares under the ``run`` scope."""
    global _RUN_AFFINITY
    if _RUN_AFFINITY is None:
        _RUN_AFFINITY = uuid.uuid4().hex
    return _RUN_AFFINITY


def prefix_affinity_key(
    model: Any,
    system_message: Any,
    tools: Optional[list],
) -> str:
    """A key for the prefix every request of a session starts with.

    Providers cache tools, system prompt and messages in that order, so two
    sessions with the same model, system prompt and tool list share their
    leading tokens whatever their first user message is. The key is a hash
    of those three, serialised canonically, so it is the same in every
    process and for every session that shares them -- and it names nothing
    about the conversation that follows.
    """
    blob = json.dumps(
        {"model": model, "system": system_message, "tools": tools or []},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _client_attr(client: Any, name: str) -> Any:
    try:
        return getattr(client, name, None)
    except Exception:  # a client property that cannot answer
        return None


def ensure_cache_affinity(
    client: Any,
    tools: Optional[list] = None,
) -> Optional[str]:
    """Give *client* a cache affinity key when its unillm takes one.

    The key follows ``UNIFY_CACHE_AFFINITY_SCOPE``: under ``prefix`` it is
    :func:`prefix_affinity_key` of the client's model and system prompt and
    *tools* (the session's fixed list), so sessions that share that prefix
    are sent to the same replica; under ``session`` a new random key; under
    ``run`` the process's key. A key the client already has (a fork's,
    inherited from its parent) is kept. Returns the key, or ``None`` for a
    unillm without the feature.
    """
    if not hasattr(client, "set_cache_affinity"):
        return None
    key = getattr(client, "cache_affinity", None)
    if key is not None:
        return key
    scope = affinity_scope()
    if scope == "session":
        key = uuid.uuid4().hex
    elif scope == "run":
        key = run_affinity_key()
    else:
        key = prefix_affinity_key(
            _client_attr(client, "endpoint"),
            _client_attr(client, "system_message"),
            tools,
        )
    client.set_cache_affinity(key)
    LOGGER.info(f"🗄️ cache affinity: {scope} key {key[:12]}")
    return key


_CACHE_STATS = "_unify_cache_stats"


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def cache_usage(completion: Any) -> tuple[Optional[int], Optional[int]]:
    """``(input tokens, of which cached)`` from a reply's usage; ``None`` if unreported."""
    usage = _field(completion, "usage")
    prompt = _field(usage, "prompt_tokens")
    cached = _field(_field(usage, "prompt_tokens_details"), "cached_tokens")
    prompt = int(prompt) if isinstance(prompt, (int, float)) else None
    cached = int(cached) if isinstance(cached, (int, float)) else None
    return prompt, cached


def cache_stats(client: Any) -> Optional[dict]:
    """The session's running cache totals, or ``None`` before its first call."""
    stats = getattr(client, _CACHE_STATS, None)
    return dict(stats) if isinstance(stats, dict) else None


def _share(cached: int, total: int) -> str:
    return f"{100 * cached / total:.1f}%" if total else "n/a"


def log_cache_use(client: Any, completion: Any, *, label: str = "") -> None:
    """Add one reply's cache use to the session's totals and log both."""
    prompt, cached = cache_usage(completion)
    stats = getattr(client, _CACHE_STATS, None)
    if not isinstance(stats, dict):
        stats = {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "unknown": 0}
    stats["calls"] += 1
    if prompt is None or cached is None:
        stats["unknown"] += 1
        this_call = "cached tokens not reported"
    else:
        stats["input_tokens"] += prompt
        stats["cached_tokens"] += cached
        this_call = f"{cached}/{prompt} input tokens cached ({_share(cached, prompt)})"
    setattr(client, _CACHE_STATS, stats)
    known = stats["calls"] - stats["unknown"]
    unknown = f", {stats['unknown']} unreported" if stats["unknown"] else ""
    LOGGER.info(
        f"🗄️ [{label}] cache: {this_call}; session "
        f"{stats['cached_tokens']}/{stats['input_tokens']} "
        f"({_share(stats['cached_tokens'], stats['input_tokens'])}) over "
        f"{known} call(s){unknown}",
    )
