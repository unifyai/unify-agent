"""Which prompt texts an agent reads: today's, or the texts for tools called from code.

``UNIFY_TOOLS_IN_CODE=on`` gives every agent one model-API tool, ``execute_code``, and every other tool as a call
``tools.<name>(...)`` in its cells. The prompt texts that describe tools then name them in that form and leave each
tool's details to ``tools.help(name)``. With ``off`` (the default) every text is byte-identical to today's.

The setting is read from :data:`unify.settings.SETTINGS` when it defines it, else from the environment, at each
call, so a text is chosen when it is rendered.
"""

from __future__ import annotations

import os

#: The setting's name, in the environment and in the settings.
SETTING = "UNIFY_TOOLS_IN_CODE"


def tools_in_code() -> bool:
    """True when ``UNIFY_TOOLS_IN_CODE`` is ``on``."""
    value = None
    try:
        from unify.settings import SETTINGS

        value = getattr(SETTINGS, SETTING, None)
    except ImportError:
        value = None
    if value is None:
        value = os.environ.get(SETTING, "")
    return str(value).strip().lower() == "on"
