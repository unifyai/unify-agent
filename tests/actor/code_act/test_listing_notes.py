"""Symbolic: the shortlist under ``UNIFY_CORE_BIND_LISTED`` binds what it lists.

The listed functions are bound in the first cell's session and the
shortlist's header says how to call them. (The notes this module tested
beside it, ``UNIFY_LISTING_PROVENANCE``, ``UNIFY_LESSON_STATUS`` and
``UNIFY_LISTING_USAGE``, were removed at the code freeze.) Embeddings come
from the concept fake of ``shortlist_world``; nothing leaves the process.
"""

from __future__ import annotations

import re

from tests.actor.code_act.shortlist_world import (  # noqa: F401 (fixture)
    EARLIER,
    GENERIC,
    GENERIC_TITLE,
    ROTATE,
    ROTATE_AGAIN,
    _in_task,
    _rotate_source,
    computed,
)
from unify.actor import library_shortlist as ls


def test_the_bound_core_list(computed):
    """``UNIFY_CORE_BIND_LISTED``: the listed functions are bound and the header says how to call."""
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    fm = FunctionManager(include_primitives=False)
    gm = GuidanceManager()
    _in_task(ROTATE, lambda: fm.add_functions(implementations=_rotate_source()))
    _in_task(ROTATE, lambda: gm.add_guidance(title=GENERIC_TITLE, content=GENERIC))
    for request in EARLIER:  # each earlier task start ranks
        _in_task(request, lambda: ls.shortlist_block(fm, gm, request))
    bound: list[list[str]] = []

    def bind(names):
        bound.append(list(names))
        return {name: False for name in names}

    block = _in_task(
        ROTATE_AGAIN,
        lambda: ls.shortlist_block(fm, gm, ROTATE_AGAIN, bind=bind),
    )
    assert block.startswith(ls._HEADER_CALL)
    assert ls.CALL_FORM in block.splitlines()[0]
    assert bound == [["rotate_table"]]
    assert ls.shortlisted_names(block) == {
        "functions": ["rotate_table"],
        "guidance": [re.search(r"- guidance (\S+)", block).group(1)],
    }
