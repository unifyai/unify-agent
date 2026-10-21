from .loop_config import LoopConfig
from .timeout_timer import TimeoutTimer
import unillm

from unify import transcripts


class LoopMessageDispatcher:
    """Appends messages to a tool loop's client, records them in the loop's
    transcript and resets the loop's activity timer."""

    def __init__(
        self,
        client: unillm.AsyncUnify,
        cfg: LoopConfig,
        timer: TimeoutTimer,
    ):
        self._client = client
        self._cfg = cfg
        self._timer = timer
        # One dispatcher per loop run, created inside the loop's task: the
        # point at which the loop is bound to its client's transcript session.
        transcripts.attach(client, cfg)

    async def append_msgs(self, msgs: list[dict]) -> None:
        self._client.append_messages(msgs)
        transcripts.observe(msgs, self._cfg)
        self._timer.reset()
