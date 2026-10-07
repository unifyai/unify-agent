from datetime import datetime
from dataclasses import dataclass


@dataclass
class Notification:
    type: str
    content: str
    timestamp: datetime
    pinned: bool = False


class NotificationBar:
    def __init__(self):
        self.notifications = []

    def push_notif(
        self,
        type,
        notif_content,
        timestamp: datetime,
        pinned=False,
    ):
        self.notifications.append(
            Notification(type, notif_content, timestamp, pinned),
        )
