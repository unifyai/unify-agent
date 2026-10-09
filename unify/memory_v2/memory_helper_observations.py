"""The ``memory`` helper's addition under ``UNIFY_MEMORY_V2_OBSERVATIONS=on``, as source text.

:data:`SOURCE` is appended to the export's ``memory.py`` (:mod:`.memory_helper`) only while the switch is on
(:func:`.catalogue.helper_bytes`), so with it off the helper is the frozen file byte for byte. It says where a
function whose input form is ``observation`` finds the current ones: the counterpart's messages are the entries
from ``user`` in the team record the sandbox already holds (``record.read()``, :mod:`unify.agents.binding`):
``limit=1`` gives the current one, a large ``limit`` all of them oldest first (the default keeps the newest 50). Stored functions parse those entries' ``text`` as they parse the recorded
observations they were built on (both forms measured offline: identical outcomes). It adds one line to the
catalogue's input forms and one to such a function's description; no data, no file, no call to the harness. It
runs inside the helper's module (its names: ``Any``, ``_Text``, ``_entry``, ``_forms_line``, ``describe``).
"""

SOURCE = r'''
# --- UNIFY_MEMORY_V2_OBSERVATIONS=on (appended by the harness) ------------------------------------------

__doc__ = (__doc__ or "") + """
Where a function's ``observation`` input comes from: the counterpart's messages are the team record's entries
from ``user``. ``record.read(author="user", limit=1)`` is the current one (the newest);
``record.read(author="user", limit=10**6)`` is all of them, oldest first (the request, then each reply; the
default limit keeps only the newest 50). Pass an entry's ``text``.
"""

_OBSERVATIONS_LINE = (
    "An `observation` input is a message the counterpart sent, an entry from `user` in the team record: "
    "`record.read(author=\"user\", limit=1)` is the current one, `record.read(author=\"user\", limit=10**6)` "
    "all of them, oldest first (the request, then each reply); pass an entry's `text`.\n"
)

_frozen_forms_line = _forms_line
_frozen_describe = describe


def _forms_line(cat: dict) -> str:  # noqa: F811 - extends the helper's own
    line = _frozen_forms_line(cat)
    return line + _OBSERVATIONS_LINE if line else line


def describe(fn_or_name: Any) -> str:  # noqa: F811 - extends the helper's own
    text = _frozen_describe(fn_or_name)
    if _entry(fn_or_name).get("input") == "observation":
        text = _Text(
            text
            + "The current one: `record.read(author=\"user\", limit=1)[0][\"text\"]` (all of them, oldest "
            + "first: `limit=10**6`).\n",
        )
    return text
'''
