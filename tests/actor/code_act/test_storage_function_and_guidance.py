"""Storage-loop integration tests for FunctionManager + GuidanceManager.

Tests the storage check loop's discrimination between the two stores:

* ``test_storage_loop_stores_both_function_and_guidance`` — a multi-step
  pipeline with reusable utilities AND a non-trivial composition.
  Expected: function(s) stored in FM, procedural guidance stored in GM.

* ``test_storage_loop_stores_function_without_guidance`` — a single
  well-parameterized utility with no multi-step composition.
  Expected: function stored in FM, NO guidance created in GM.

This complements the existing storage tests in ``test_can_compose_and_store.py``
which only assert on FunctionManager storage.
"""

import asyncio

import pytest

from unify.actor.code_act_actor import CodeActActor
from unify.function_manager.function_manager import FunctionManager

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


# ---------------------------------------------------------------------------
# GuidanceManager mock with real method signatures and docstrings.
#
# We avoid MagicMock because the storage-check loop accesses
# ``type(gm).<method>.__doc__`` to wire tool docstrings, and MagicMock's
# metaclass attribute access doesn't expose stable docstrings.
# ---------------------------------------------------------------------------


class _TrackingGuidanceManager:
    """Minimal GuidanceManager stand-in that records ``add_guidance`` calls."""

    def __init__(self) -> None:
        self.add_calls: list[dict] = []

    def search(self, references=None, k=10):
        """Search for guidance entries by semantic similarity to reference text."""
        return []

    def filter(self, filter=None, offset=0, limit=100):
        """Filter guidance entries using a Python filter expression."""
        return []

    def get_guidance(self, *, guidance_id):
        """Fetch one guidance entry by id with its complete content."""
        raise ValueError(f"No guidance found with guidance_id {guidance_id}.")

    def add_guidance(self, *, title, content, function_ids=None):
        """Add a guidance entry describing a compositional procedure or playbook."""
        self.add_calls.append(
            {"title": title, "content": content, "function_ids": function_ids},
        )
        return {"details": {"guidance_id": len(self.add_calls)}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        """Update an existing guidance entry."""
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        """Delete a guidance entry by ID."""
        return {"deleted": True}

    def reconcile_dependencies(self, *, guidance_ids=None):
        """Refresh structured link debt for related functions."""
        return {"outcome": "checked", "details": {"guidance_ids": guidance_ids or []}}


# ---------------------------------------------------------------------------
# Test: storage loop stores both function(s) AND guidance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_storage_loop_stores_both_function_and_guidance():
    """The storage check stores both functions (FM) and guidance (GM).

    The task produces a single reusable utility function AND demonstrates
    a multi-phase procedure with conditional branching that can only be
    captured as a guidance playbook.

    The scenario is deliberately kept small (one function + one procedure)
    so the 30-step storage loop has budget for both FM and GM operations.

    The storage-check librarian should recognise:
    - The utility function as genuinely reusable → store via FM.
    - The adaptive procedure with quality gates and conditional strategy
      selection as a non-trivial orchestration recipe → store via GM.
    """
    fm = FunctionManager(include_primitives=False)
    gm = _TrackingGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=180,
    )
    try:
        handle = await actor.act(
            "Build a reusable text-normalization function and then demonstrate "
            "an adaptive data-cleaning workflow that uses it.\n\n"
            "## Part 1 — Utility Function\n\n"
            "`normalize_text(value, operations: list[str]) -> str`\n"
            "Chains text operations on a single value. Supported ops:\n"
            "'strip', 'lower', 'upper', 'title', 'digits_only', 'collapse_spaces'.\n"
            "Returns '' for None/non-string input. Operations applied left-to-right.\n"
            "Test it on a few examples to verify.\n\n"
            "## Part 2 — Adaptive Cleaning Workflow\n\n"
            "Using normalize_text, demonstrate this multi-phase workflow on the "
            "dataset below. Do NOT wrap the workflow into a single function — "
            "execute each phase inline with explicit decision logic between steps:\n\n"
            "Phase 1: Compute completeness_rate (fraction of records where both "
            "'name' and 'email' are non-empty).\n"
            "Phase 2: Based on completeness:\n"
            "  - If completeness_rate < 0.8: normalize aggressively — apply "
            "['strip', 'lower', 'collapse_spaces'] to name and email fields.\n"
            "  - If completeness_rate >= 0.8: normalize gently — apply "
            "['strip', 'title'] to name, ['strip', 'lower'] to email.\n"
            "Phase 3: Remove records where BOTH name and email are empty.\n"
            "Phase 4: Group records by normalized email (strip+lower). For each "
            "group with >1 record, merge by keeping the record with the most "
            "non-empty fields and filling gaps from others.\n"
            "Phase 5: Re-compute completeness_rate. If it improved by less "
            "than 5 percentage points, print a warning for manual review.\n\n"
            "Dataset:\n"
            "[\n"
            "  {'name': '  JOHN DOE  ', 'email': 'John@example.COM', 'dept': 'Eng'},\n"
            "  {'name': 'john doe', 'email': 'john@example.com', 'dept': ''},\n"
            "  {'name': '', 'email': '', 'dept': ''},\n"
            "  {'name': 'Jane Smith', 'email': 'jane@example.com', 'dept': 'Mkt'},\n"
            "  {'name': '  jane SMITH', 'email': ' Jane@Example.com ', 'dept': ''},\n"
            "]",
            can_store=True,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=240)
        assert result is not None

        # result() resolves after the task phase; wait for storage to finish.
        deadline = asyncio.get_event_loop().time() + 300
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Storage loop did not complete in time")
            await asyncio.sleep(0.5)

        # The storage check should have stored at least one function.
        stored = fm.filter_functions()
        assert stored, (
            "Expected FunctionManager to contain at least one stored function "
            "for the reusable normalize_text utility."
        )

        # The storage check should have stored guidance about the
        # adaptive cleaning procedure.
        assert gm.add_calls, (
            f"Expected GuidanceManager.add_guidance to be called for the "
            f"adaptive data-cleaning workflow. "
            f"FM has {len(stored)} stored function(s), "
            f"but no guidance was stored."
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test: storage loop stores function but NOT guidance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_storage_loop_stores_function_without_guidance():
    """The storage check stores a function (FM) but does NOT create guidance (GM).

    The task produces a single, well-parameterized utility function
    (phone-number normalization) that is clearly reusable but involves no
    multi-step compositional procedure.  The storage-check librarian should:

    - Recognise the utility as genuinely reusable → store via FM.
    - NOT create a guidance entry, because there is no non-obvious
      multi-step composition to document — the function's own docstring
      fully describes its usage.
    """
    fm = FunctionManager(include_primitives=False)
    gm = _TrackingGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=180,
    )
    try:
        handle = await actor.act(
            "Write a reusable Python function called `normalize_phone` that:\n\n"
            "1. Takes a raw phone string in any common format — digits, spaces, "
            "dashes, dots, parentheses, optional leading '+' or country code.\n"
            "   Examples: '(555) 123-4567', '+1-555-123-4567', '555.123.4567', "
            "'15551234567'\n\n"
            "2. Strips all non-digit characters (except a leading '+').\n"
            "3. For US numbers: accepts 10 digits (adds '+1' prefix) or "
            "11 digits starting with '1' (adds '+' prefix). "
            "Raises ValueError for other lengths.\n"
            "4. Returns the normalized string in E.164 format (e.g. '+15551234567').\n\n"
            "Test it with these inputs and verify the expected outputs:\n"
            "- '(555) 123-4567'   → '+15551234567'\n"
            "- '+1-555-123-4567'  → '+15551234567'\n"
            "- '555.123.4567'     → '+15551234567'\n"
            "- '15551234567'      → '+15551234567'\n"
            "- '123'              → raises ValueError",
            can_store=True,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=240)
        assert result is not None

        # result() resolves after the task phase; wait for storage to finish.
        deadline = asyncio.get_event_loop().time() + 300
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Storage loop did not complete in time")
            await asyncio.sleep(0.5)

        # The storage check should have stored the function.
        stored = fm.filter_functions()
        assert stored, (
            "Expected FunctionManager to contain at least one stored function "
            "for the reusable normalize_phone utility."
        )

        # The storage check should NOT have created guidance — this is a
        # single utility with no multi-step compositional procedure.
        assert not gm.add_calls, (
            f"Expected NO GuidanceManager.add_guidance calls for a single "
            f"utility function, but {len(gm.add_calls)} guidance entries "
            f"were created: {[c['title'] for c in gm.add_calls]}"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass
