"""
unify/common/hierarchical_logger.py
===================================

The emoji prefixes of the package's log lines: ``ICONS`` by event, and
``DEFAULT_ICON`` for any other line. No other module hardcodes an emoji in a
log call.
"""

from __future__ import annotations

DEFAULT_ICON = "⬥"

ICONS = {
    "llm_thinking": "🧠",
    "llm_response": "🤖",
    "llm_error": "❌",
    "system_message": "📋",
    "request": "➡️",
    "stop_requested": "🛑",
    "early_exit": "⏹️",
    "interjection": "💬",
    "completed": "✅",
    "pause": "⏸️",
    "resume": "▶️",
    "summarize": "📑",
}
