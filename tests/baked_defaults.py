"""The settings the code freeze baked in, and a fixture that restores the old ones.

Step 3 of the code freeze (7 Oct 2026) made the chosen configuration the
default: lean-all, Python tool mode and the shared agent record. The
switches still exist, so a test of a path that is no longer the default
pins it explicitly with the ``as_shipped`` fixture (import it, then request
it or ``pytestmark = pytest.mark.usefixtures("as_shipped")``), as the
module's tests ran before the freeze; a test that sets a switch itself
still overrides it. Step 4 removes the switches and the tests of removed
paths; the pins left are on tests of the JSON tool surface and the
steerable-handle and steering paths, which go with that code in step 5,
and this module goes with the last of them.
"""

from __future__ import annotations

import pytest

from unify.settings import SETTINGS

# Every switch whose default the code freeze changed, at its new default
# (step 3 of the freeze).
BAKED_DEFAULTS = {
    # The lean-all recipe (make_cells_ov1.env_for(bench, "lean")).
    "UNIFY_PROMPT_PROFILE": "lean",
    "UNIFY_PROMPT_ACCURACY": True,
    "UNIFY_REPLY_PROTOCOL_NOTE": True,
    "UNIFY_BUILTIN_GUIDANCE": False,
    "UNIFY_DELEGATION": "off",
    "UNIFY_DISCOVERY_GATE": False,
    "UNIFY_LIBRARY_SHORTLIST": True,
    "UNIFY_LIBRARY_SNAPSHOT": True,
    "UNIFY_CACHE_DISCIPLINE": True,
    "UNIFY_TRANSCRIPTS": True,
    "UNIFY_WORKSPACE": "sandboxed",
    "UNIFY_OUTCOME": True,
    "UNIFY_FUNCTION_PATCH": True,
    "UNIFY_FUNCTION_CASES": True,
    "UNIFY_STORE_DEDUPE": "warn",
    "UNIFY_REVIEW_FORK": True,
    "UNIFY_REVIEW_FRAMING": "unified",
    "UNIFY_CURATION_DOCTRINE": "minimal",
    "UNIFY_REVIEW_GATE": True,
    "UNIFY_STORE_INSTANCE_LINT": True,
    "UNIFY_PENDING_REQUIRED": False,
    "UNIFY_BATCH_WAKE": True,
    "UNIFY_LIFECYCLE_NOTICES": False,
    "UNIFY_STORE_CHECK": "resolve",
    "UNIFY_SEARCH_SKIP_UNLOADABLE": True,
    # Python tool mode.
    "UNIFY_WORKSPACE_PYTHON": "worker",
    "UNIFY_TOOL_SURFACE": "core",
    "UNIFY_CORE_BIND_LISTED": True,
    "UNIFY_CORE_CALL_EXAMPLE": True,
    "UNIFY_GUIDANCE_LINKED_NAMES": True,
    "UNIFY_FUNCTION_HELPERS": True,
    "UNIFY_REVIEW_FORK_CORE": True,
    # The shared agent record.
    "UNIFY_AGENTS": "record",
    # Fixes.
    "UNIFY_REVIEW_LAST_REPLY": True,
    "UNIFY_STORE_FROM_SESSION": True,
    "UNIFY_ESCAPE_DRIFT_CHECK": "on",
    "UNIFY_PLACEHOLDER_NOTE": True,
    "UNIFY_PROMPT_TRIM": True,
}

# The default of each before the freeze (upstream's behaviour).
AS_SHIPPED = {
    "UNIFY_PROMPT_PROFILE": "",
    "UNIFY_PROMPT_ACCURACY": False,
    "UNIFY_REPLY_PROTOCOL_NOTE": False,
    "UNIFY_BUILTIN_GUIDANCE": True,
    "UNIFY_DELEGATION": "on",
    "UNIFY_DISCOVERY_GATE": True,
    "UNIFY_LIBRARY_SHORTLIST": False,
    "UNIFY_LIBRARY_SNAPSHOT": False,
    "UNIFY_CACHE_DISCIPLINE": False,
    "UNIFY_TRANSCRIPTS": False,
    "UNIFY_WORKSPACE": "",
    "UNIFY_OUTCOME": False,
    "UNIFY_FUNCTION_PATCH": False,
    "UNIFY_FUNCTION_CASES": False,
    "UNIFY_STORE_DEDUPE": "",
    "UNIFY_REVIEW_FORK": False,
    "UNIFY_REVIEW_FRAMING": "",
    "UNIFY_CURATION_DOCTRINE": "",
    "UNIFY_REVIEW_GATE": False,
    "UNIFY_STORE_INSTANCE_LINT": False,
    "UNIFY_PENDING_REQUIRED": True,
    "UNIFY_BATCH_WAKE": False,
    "UNIFY_LIFECYCLE_NOTICES": True,
    "UNIFY_STORE_CHECK": "",
    "UNIFY_SEARCH_SKIP_UNLOADABLE": False,
    "UNIFY_WORKSPACE_PYTHON": "",
    "UNIFY_TOOL_SURFACE": "",
    "UNIFY_CORE_BIND_LISTED": False,
    "UNIFY_CORE_CALL_EXAMPLE": False,
    "UNIFY_GUIDANCE_LINKED_NAMES": False,
    "UNIFY_FUNCTION_HELPERS": False,
    "UNIFY_REVIEW_FORK_CORE": False,
    "UNIFY_AGENTS": "",
    "UNIFY_REVIEW_LAST_REPLY": False,
    "UNIFY_STORE_FROM_SESSION": False,
    "UNIFY_ESCAPE_DRIFT_CHECK": "",
    "UNIFY_PLACEHOLDER_NOTE": False,
    "UNIFY_PROMPT_TRIM": False,
}
assert set(AS_SHIPPED) == set(BAKED_DEFAULTS)


def _env_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@pytest.fixture
def as_shipped(monkeypatch):
    """Every baked switch at its pre-freeze default, in SETTINGS and in the
    environment (for anything that reads settings afresh, such as a child
    ``unify act`` process).

    A switch the strip has already deleted is skipped: its baked value is
    then the only code path."""
    for name, value in AS_SHIPPED.items():
        if not hasattr(SETTINGS, name):
            continue
        monkeypatch.setattr(SETTINGS, name, value)
        monkeypatch.setenv(name, _env_value(value))
