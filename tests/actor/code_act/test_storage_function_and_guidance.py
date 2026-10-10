"""Storage-loop integration tests for FunctionManager + GuidanceManager.

Tests the storage check loop's discrimination between the two stores:

* ``test_storage_loop_stores_both_function_and_guidance`` — a multi-step
  pipeline with reusable utilities AND a non-trivial composition.
  Expected: function(s) stored in FM, procedural guidance stored in GM.

* ``test_storage_loop_stores_function_without_guidance`` — a single
  well-parameterized utility with no multi-step composition.
  Expected: function stored in FM; at most a bounded number of guidance
  entries in GM, none copying the task's input data.

This complements the existing storage tests in ``test_can_compose_and_store.py``
which only assert on FunctionManager storage.
"""

import ast
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
async def test_storage_loop_stores_both_function_and_guidance(monkeypatch):
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

    The review gate is answered "review" here, so the review it opens runs
    as shipped. When the session stores ``normalize_text`` itself the
    library is no longer empty, the gate is asked, and the frozen gate can
    skip the review (the 7299344a6 re-record: "review=False ... The reusable
    normalize_text code was already stored successfully"; a known
    limitation of the frozen review, FREEZE-TODO). This test is of the
    review's function/guidance split, not of the gate.
    """
    from unify.actor import review_gate

    gate_asked: list[bool] = []

    async def _review(**_kwargs):
        gate_asked.append(True)
        return review_gate.GateDecision(
            review=True,
            reason="test: the storage review runs",
            decided=True,
        )

    monkeypatch.setattr(review_gate, "decide", _review)

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

        # The gate is asked once at most: only when the library holds an
        # entry (the session stored one itself) at the end of the task.
        assert len(gate_asked) <= 1

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
# Test: storage loop stores the function, and no guidance about its use
# ---------------------------------------------------------------------------

# The phone strings the task below feeds in, which no guidance entry may copy.
# ('123', the invalid input, is left out: as a substring it is too common to
# mean a copy.)
_PHONE_TASK_INPUTS = (
    "(555) 123-4567",
    "+1-555-123-4567",
    "555.123.4567",
    "15551234567",
)
_MAX_GUIDANCE_FOR_ONE_UTILITY = 2


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_storage_loop_stores_function_without_guidance():
    """The storage check stores a function (FM) but does NOT create guidance (GM).

    The task produces a single, well-parameterized utility function
    (phone-number normalization) that is clearly reusable but involves no
    multi-step compositional procedure.  The storage-check librarian should:

    - Recognise the utility as genuinely reusable → store via FM.
    - Not turn the function's own use into guidance: its docstring covers
      that.

    What the product guarantees, and so what is asserted: the function is
    stored (which means the store check loaded it in the sandbox), at most a
    bounded number of guidance entries are written, and none of them copies
    the task's input data. "No guidance at all" is not a guarantee: the
    review prompt asks for a short guidance entry when the trajectory
    corrected a pitfall. In the 7299344a6 re-record the session called
    ``normalize_phone`` right after ``functions.add`` in the same cell
    (``NameError: name 'normalize_phone' is not defined``), recovered through
    ``functions.run``, and the review rightly recorded that as guidance.
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

        # The function is stored. Storing it means the store check resolved
        # and loaded it where it runs (store_check, step 4); its body is not
        # run here, because model-written code never executes in the test
        # process (test_bind_load_confinement), so the test checks that the
        # stored source defines the function.
        stored = fm.filter_functions()
        assert stored, (
            "Expected FunctionManager to contain at least one stored function "
            "for the reusable normalize_phone utility."
        )
        by_name = {f.get("name"): f for f in stored if isinstance(f, dict)}
        assert "normalize_phone" in by_name, sorted(by_name)
        tree = ast.parse(by_name["normalize_phone"].get("implementation") or "")
        assert [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ] == ["normalize_phone"]

        # Guidance is bounded and never carries the task's input data.
        assert len(gm.add_calls) <= _MAX_GUIDANCE_FOR_ONE_UTILITY, (
            f"Expected at most {_MAX_GUIDANCE_FOR_ONE_UTILITY} guidance entries "
            f"for a single utility, but {len(gm.add_calls)} were created: "
            f"{[c['title'] for c in gm.add_calls]}"
        )
        for call in gm.add_calls:
            text = f"{call.get('title') or ''}\n{call.get('content') or ''}"
            copied = [value for value in _PHONE_TASK_INPUTS if value in text]
            assert not copied, (
                f"Guidance {call.get('title')!r} copies the task's input data "
                f"{copied}"
            )
    finally:
        try:
            await actor.close()
        except Exception:
            pass
