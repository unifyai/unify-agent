import json
import uuid
from typing import ClassVar
from datetime import datetime
from dataclasses import dataclass, asdict, field

from pydantic import BaseModel

from unify.common.prompt_helpers import now as prompt_now


def custom_dict_factory(kv):
    d = {}
    for k, v in kv:
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, BaseModel):
            d[k] = v.model_dump()
        else:
            d[k] = v
    return d


def _now_datetime() -> datetime:
    """Wrapper for prompt_now that returns datetime for dataclass default_factory."""
    return prompt_now(as_string=False)


@dataclass(kw_only=True)
class Event:
    timestamp: datetime = field(default_factory=_now_datetime)
    suppress_slow_brain_wake: bool = False

    _registry: ClassVar[dict[str, "Event"]] = {}
    loggable: ClassVar[bool] = True
    content_logged: ClassVar[bool] = False
    prominent: ClassVar[bool] = False
    topic: ClassVar[str | None] = None

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {
            "event_name": self.__class__.__name__,
            "payload": asdict(self, dict_factory=custom_dict_factory),
        }

    def to_bus_event(self):
        from unify.events.event_bus import Event as BusEvent

        payload = self.to_dict()["payload"]
        return BusEvent(
            calling_id="",
            type="Comms",
            timestamp=self.timestamp.isoformat(),
            payload=payload,
            payload_cls=self.__class__.__name__,
        )

    @classmethod
    def from_dict(cls, data) -> "Event":
        import dataclasses

        target_cls = cls._registry.get(data["event_name"])
        if not target_cls:
            raise Exception(f"Class {data['event_name']} is not registered.")
        kwargs = data["payload"].copy()
        timestamp = kwargs.pop("timestamp")

        # Filter to only fields the target dataclass accepts
        valid_fields = {f.name for f in dataclasses.fields(target_cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        return target_cls(
            **filtered_kwargs,
            timestamp=datetime.fromisoformat(timestamp),
        )

    @classmethod
    def from_json(cls, json_data):
        data = json.loads(json_data)
        return cls.from_dict(data)

    def __init_subclass__(cls):
        if cls.__name__ not in Event._registry:
            Event._registry[cls.__name__] = cls
        return cls


@dataclass
class UnifyMessageReceived(Event):
    """The user sent a message in the in-app chat.

    ``attachments`` are workspace paths of files sent with the message.
    """

    topic: ClassVar[str | None] = "app:comms:unify_message_message"
    content_logged: ClassVar[bool] = True

    content: str
    attachments: list[str] = field(default_factory=list)


@dataclass
class UnifyMessageSent(Event):
    """The assistant sent a message in the in-app chat.

    ``attachments`` are workspace paths of files sent with the message.
    """

    topic: ClassVar[str | None] = "app:comms:unify_message_sent"
    content_logged: ClassVar[bool] = True

    content: str
    attachments: list[str] = field(default_factory=list)


@dataclass
class ActionStopRequested(Event):
    """The user requested stop of an in-flight act by calling_id."""

    topic: ClassVar[str | None] = "app:comms:action_stop"

    calling_id: str
    reason: str = ""
    source: str = ""


@dataclass
class InitializationComplete(Event):
    """Published when ConversationManager has fully initialized all managers."""

    loggable: ClassVar[bool] = False


@dataclass
class OpenSlowBrainTurn(Event):
    """Follow-on slow-brain turn because the prior turn did not call wait."""

    loggable: ClassVar[bool] = False
    origin_run_id: str = ""
    previous_tools: list[str] = field(default_factory=list)


@dataclass
class Error(Event):
    prominent: ClassVar[bool] = True

    message: str


@dataclass
class NotificationInjectedEvent(Event):
    """A notification injected into the ConversationManager's notification bar."""

    content: str
    source: str
    interjection_id: str = field(default_factory=lambda: str(uuid.uuid4().hex[:12]))
    pinned: bool = False


@dataclass
class ActorHandleResponse(Event):
    """The answer to a steering query (``ask``) on an Actor handle."""

    handle_id: int
    action_name: str
    query: str
    response: str
    call_id: str


@dataclass
class ActorResult(Event):
    """The final result (or failure) of an Actor action."""

    handle_id: int
    success: bool
    result: dict | str | None = None
    error: str | None = None
    action_type: str = ""


@dataclass
class ActorClarificationRequest(Event):
    """A question an in-flight Actor action asks the user."""

    handle_id: int
    query: str
    call_id: str


@dataclass
class ActorNotification(Event):
    """Event to forward a notification from an Actor handle.

    Notifications carry status updates emitted explicitly by the actor via
    ``send_notification``.  The ``completed`` flag
    distinguishes a completion announcement from an in-progress update so
    downstream consumers can render them with unambiguous prefixes.
    """

    handle_id: int
    response: str
    completed: bool = False
    kind: str = ""


@dataclass
class ActorSessionResponse(Event):
    """Event signalling that a persistent actor session has completed a turn.

    Unlike ``ActorNotification``, a session response means the actor has
    finished its current work and is **waiting for the next instruction**
    (via ``interject``).  The ``content`` field carries the actor's output
    for this turn.
    """

    handle_id: int
    content: str


@dataclass
class ActorHandleStarted(Event):
    action_name: str
    handle_id: int
    query: str
    response_format: dict | None = None


@dataclass
class DirectMessageEvent(Event):
    """
    Send a message directly to the user, bypassing the Main CM Brain's
    decision-making.

    Used by ConversationManagerHandle.ask for questions and acknowledgments
    that should be delivered verbatim without LLM processing.
    """

    content: str
    source: str = "system"
