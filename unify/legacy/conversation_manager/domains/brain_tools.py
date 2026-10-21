from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )


class ConversationManagerBrainTools:
    """
    Read-only tools for the Main CM Brain.

    These tools are intentionally conservative: they expose inspection surfaces
    that help the brain reason about current state without mutating it.
    """

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm

    def cm_list_in_flight_actions(self) -> list[dict[str, Any]]:
        """List in-flight actions with minimal identifying information."""
        out: list[dict[str, Any]] = []
        for handle_id, data in (self._cm.in_flight_actions or {}).items():
            out.append(
                {
                    "handle_id": handle_id,
                    "query": data.get("query"),
                    "num_handle_actions": len(data.get("handle_actions") or []),
                },
            )
        return out

    def cm_list_notifications(
        self,
        *,
        pinned_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List current notifications (optionally pinned only)."""
        notifs = list(self._cm.notifications_bar.notifications or [])
        if pinned_only:
            notifs = [n for n in notifs if getattr(n, "pinned", False)]
        out: list[dict[str, Any]] = []
        for n in notifs:
            d = asdict(n)
            ts = d.get("timestamp")
            if hasattr(ts, "isoformat"):
                d["timestamp"] = ts.isoformat()
            out.append(d)
        return out

    def as_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Return the tools dict for start_async_tool_loop."""
        return {
            "cm_list_in_flight_actions": self.cm_list_in_flight_actions,
            "cm_list_notifications": self.cm_list_notifications,
        }
