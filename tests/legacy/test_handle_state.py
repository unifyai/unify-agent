"""Legacy, unused, unsupported: ``get_handle_paused_state``, moved with the
function from ``tests/async_tool_loop/test_utils.py`` (8 Oct 2026)."""

import asyncio
from unittest.mock import MagicMock

from unify.legacy.handle_state import get_handle_paused_state


class TestGetHandlePausedState:
    """Tests for get_handle_paused_state helper function."""

    def test_returns_true_when_paused(self):
        """Returns True when handle's _pause_event is cleared (paused)."""
        mock_handle = MagicMock()
        mock_handle._pause_event = asyncio.Event()
        mock_handle._pause_event.clear()  # Paused state

        result = get_handle_paused_state(mock_handle)
        assert result is True

    def test_returns_false_when_running(self):
        """Returns False when handle's _pause_event is set (running)."""
        mock_handle = MagicMock()
        mock_handle._pause_event = asyncio.Event()
        mock_handle._pause_event.set()  # Running state

        result = get_handle_paused_state(mock_handle)
        assert result is False

    def test_returns_none_when_no_pause_event(self):
        """Returns None when handle has no _pause_event attribute."""
        mock_handle = MagicMock(spec=[])  # No attributes

        result = get_handle_paused_state(mock_handle)
        assert result is None

    def test_returns_none_when_pause_event_is_none(self):
        """Returns None when handle._pause_event is None."""
        mock_handle = MagicMock(spec=["_pause_event"])
        mock_handle._pause_event = None

        result = get_handle_paused_state(mock_handle)
        assert result is None

    def test_returns_none_when_pause_event_has_no_is_set(self):
        """Returns None when _pause_event doesn't have is_set method."""
        mock_handle = MagicMock(spec=["_pause_event"])
        mock_handle._pause_event = "not an event"

        result = get_handle_paused_state(mock_handle)
        assert result is None

    def test_handles_exception_gracefully(self):
        """Returns None when accessing _pause_event raises an exception."""
        mock_handle = MagicMock()
        type(mock_handle)._pause_event = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("test error")),
        )

        result = get_handle_paused_state(mock_handle)
        assert result is None

    def test_with_mocked_is_set(self):
        """Works with mocked is_set return values."""
        # Test with is_set returning True (running)
        mock_handle = MagicMock()
        mock_handle._pause_event = MagicMock()
        mock_handle._pause_event.is_set.return_value = True

        result = get_handle_paused_state(mock_handle)
        assert result is False  # Running (not paused)

        # Test with is_set returning False (paused)
        mock_handle._pause_event.is_set.return_value = False

        result = get_handle_paused_state(mock_handle)
        assert result is True  # Paused

    def test_with_none_handle(self):
        """Returns None when handle is None."""
        result = get_handle_paused_state(None)
        assert result is None

    def test_with_pause_event_proxy(self):
        """Works with proxy objects that expose is_set()."""

        class _PauseStateProxy:
            """Minimal proxy exposing is_set()."""

            def __init__(self, paused: bool):
                self._paused = paused

            def is_set(self) -> bool:
                return not self._paused  # Event set = running

        # Test paused state
        mock_handle = MagicMock(spec=["_pause_event"])
        mock_handle._pause_event = _PauseStateProxy(paused=True)
        assert get_handle_paused_state(mock_handle) is True

        # Test running state
        mock_handle._pause_event = _PauseStateProxy(paused=False)
        assert get_handle_paused_state(mock_handle) is False
