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

The helpers here hold that policy so the loop itself only asks two questions
per turn: what to advertise, and whether a call is allowed. With the switch
off none of them is consulted.
"""

from __future__ import annotations

import copy
import contextvars
from typing import Any, Iterable, Optional

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


def records_requests() -> bool:
    """Whether dispatches keep a copy of what they send (for a later fork)."""
    return enabled()


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
