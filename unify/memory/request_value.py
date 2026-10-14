"""The request as the capture contract hands it to a reader: its text, as a ``str``, with ``.text`` and ``.data``.

A reader (a captured procedure's ``read_input``, a landed ``read_<name>``) is called with one argument: a ``str``
subclass holding the request's text, so a reader written for text reads it as before, that carries what the bound
``request`` carries (``worker_child.Request``): ``.text`` is the text and ``.data`` the JSON objects and arrays the
text contains, in order of appearance (``worker_child.json_values``, the one definition; the host's
``current_request`` gives the same list). A reader that reads its values from ``.data`` does not depend on the lines,
headings or prose around them, which change from one request of a kind to the next.

Every confined child that calls a reader defines the class from :data:`REQUEST_CLASS` and builds it from the text and
the data the host read from it (:func:`request_data`); the box never parses the text for it.
"""

from __future__ import annotations

#: The class as a confined child defines it (standard library only; ``data`` is the list the host read).
REQUEST_CLASS = '''
class Request(str):
    """The request: its text (a str), with .text and .data (the JSON objects and arrays it contains, in order)."""

    def __new__(cls, text, data=None):
        obj = super().__new__(cls, text)
        obj._data = data
        return obj

    @property
    def text(self):
        return str.__str__(self)

    @property
    def data(self):
        return self._data
'''


def request_data(text: str) -> list:
    """The JSON objects and arrays *text* contains, in order: the bound request's ``data``
    (``worker_child.json_values``)."""
    from unify.actor.execution.worker_child import json_values

    return json_values(str(text))
