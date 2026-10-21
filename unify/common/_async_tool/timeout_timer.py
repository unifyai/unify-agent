import time
from typing import Optional


class TimeoutTimer:
    """The loop's two limits: an activity timeout and a step limit (a count of messages)."""

    def __init__(
        self,
        timeout: Optional[int],
        max_steps: Optional[int],
        client,
        message_count_offset: int = 0,
    ):
        self._timeout = timeout
        self._client = client
        self._max_steps = max_steps
        self._message_count_offset = message_count_offset
        # Messages before the current request, which max_steps does not
        # count once start_request() has been called (a persistent loop's
        # next request).
        self._request_base = 0
        self.reset()

    def start_request(self) -> None:
        """Count max_steps from here: a persistent loop's next request."""
        self._request_base = self._message_count_offset + len(self._client.messages)


    def remaining_time(self) -> Optional[float]:
        if self._timeout is None:
            return None

        return self._timeout - (time.perf_counter() - self.last_activity_ts)

    def reset(self):
        """Refresh the rolling timeout."""
        self.last_activity_ts = time.perf_counter()

    def has_exceeded_time(self) -> bool:
        """Whether the activity timeout has passed."""
        if self._timeout is None:
            return False
        return time.perf_counter() - self.last_activity_ts > self._timeout

    def has_exceeded_msgs(self) -> bool:
        """Whether the step limit is reached."""
        if self._max_steps is None:
            return False
        logical_message_count = (
            self._message_count_offset + len(self._client.messages) - self._request_base
        )
        return logical_message_count >= self._max_steps
