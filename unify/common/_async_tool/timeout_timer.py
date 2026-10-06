import asyncio
import time
from typing import Optional


class TimeoutTimer:
    def __init__(
        self,
        timeout: Optional[int],
        max_steps: Optional[int],
        raise_on_limit: bool,
        client,
        message_count_offset: int = 0,
    ):
        self._timeout = timeout
        self._client = client
        self._max_steps = max_steps
        self._raise_on_limit = raise_on_limit
        self._message_count_offset = message_count_offset
        # Messages before the current request, which max_steps does not
        # count once start_request() has been called (UNIFY_STEP_CAP_REPLY).
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
        """
        Return whether we exceeded the timeout threshold, raises Exception if raise_on_limit is set
        """
        if self._timeout is None:
            return False

        ret = time.perf_counter() - self.last_activity_ts > self._timeout
        if self._raise_on_limit and ret:
            raise asyncio.TimeoutError(
                f"Loop exceeded {self._timeout}s wall-clock limit",
            )
        return ret

    def has_exceeded_msgs(self) -> bool:
        """
        Return whether we exceeded the messages threshold, raises Exception if raise_on_limit is set
        """
        if self._max_steps is None:
            return False

        logical_message_count = (
            self._message_count_offset + len(self._client.messages) - self._request_base
        )
        ret = logical_message_count >= self._max_steps
        if self._raise_on_limit and ret:
            raise RuntimeError(
                f"Conversation exceeded max_steps={self._max_steps} "
                f"(len(client.messages)={logical_message_count})",
            )
        return ret
