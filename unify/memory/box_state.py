"""Inside a confined box: the episode's state as an item under test reads it.

Standard library only, and no other module of this package is imported: the harness copies this file into a box
read-only (``/case/lib/box_state.py``) and the box's run script imports it (:mod:`.writer_tests`).

:class:`State` is what an item that takes the episode's state is given: ``.request`` (the first message),
``.messages`` (every user or environment message so far, in order), ``.last`` (the latest one) and ``.data`` (the
JSON objects and arrays those messages contain, in order).
"""

import json


class State:
    """The episode's state as a procedure reads it (module docstring)."""

    def __init__(self, messages, data=None):
        self.messages = [m if isinstance(m, str) else json.dumps(m, default=str) for m in (messages or [])]
        self.request = self.messages[0] if self.messages else ""
        self.last = self.messages[-1] if self.messages else ""
        self.data = list(data or [])

    def __repr__(self):
        return "State(%d message%s, %d JSON value%s)" % (
            len(self.messages),
            "" if len(self.messages) == 1 else "s",
            len(self.data),
            "" if len(self.data) == 1 else "s",
        )
