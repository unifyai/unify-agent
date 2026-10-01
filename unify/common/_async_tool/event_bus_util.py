from typing import Union, Dict, List
from .loop_config import LoopConfig
from unify import transcripts


async def to_event_bus(
    messages: Union[Dict, List[Dict]],
    loop_cfg: LoopConfig,
    origin: str | None = None,
    kind: str | None = None,
) -> None:
    """
    Emit *messages* to the shared EventBus.

    Every ``ToolLoop`` event carries the raw chat *message*, the *public
    method* that spawned the loop, and a *kind* discriminator so downstream
    subscribers can filter/render without re-inspecting ad-hoc flags.

    Parameters
    ----------
    kind:
        Explicit :class:`ToolLoopKind` value.  When *None* (the common
        case), the kind is auto-classified from the message dict.
    """
    from ...events.event_bus import (
        Event,
        EVENT_BUS,
    )
    from ...events.types.tool_loop import ToolLoopPayload, classify_tool_loop_message

    if isinstance(messages, dict):
        messages = [messages]
    # Everything a loop publishes passes through here, in order, so this is
    # where its transcript session (if any) records it.
    transcripts.observe(messages, loop_cfg)
    _lookup = loop_cfg.tool_alias_lookup
    for message in messages:
        _aliases = None
        if _lookup and isinstance(message, dict):
            _tool_calls = message.get("tool_calls") or []
            _sparse = {}
            for tc in _tool_calls:
                name = (tc.get("function") or {}).get("name", "")
                if name in _lookup:
                    val = _lookup[name]
                    _sparse[name] = val(tc) if callable(val) else val
            _aliases = _sparse or None

        _kind = kind if kind is not None else classify_tool_loop_message(message)

        payload = ToolLoopPayload(
            kind=_kind,
            message=message,
            method=loop_cfg.loop_id,
            hierarchy=list(loop_cfg.lineage),
            hierarchy_label=loop_cfg.label,
            origin=origin,
            tool_aliases=_aliases,
        )
        await EVENT_BUS.publish(
            Event(type="ToolLoop", payload=payload),
        )
