"""Storage-loop integration tests for FunctionManager + GuidanceManager.

Tests where the storage review's writes land in the two stores:

* ``test_storage_loop_stores_both_function_and_guidance`` — a multi-step
  pipeline with a reusable utility AND a non-trivial composition. The review
  stores the function in FM and the procedure as guidance in GM.

* ``test_storage_loop_stores_function_without_guidance`` — a single
  well-parameterized utility with no multi-step composition. The review
  stores the function in FM, and no guidance is written.

The model is scripted (``tests/scripted_model.py``): the session's and the
review's turns are the ones a model would take. Which of the two stores a
live librarian chooses is the model's judgement; these tests check what its
choices do: a ``functions.add`` from the review's cell stores (and loads) the
function, a ``guidance.add`` reaches the actor's guidance manager, and
nothing else is written.

This complements the existing storage tests in ``test_can_compose_and_store.py``
which only assert on FunctionManager storage.
"""

import ast
import asyncio

import pytest

from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.actor.code_act_actor import CodeActActor
from unify.function_manager.function_manager import FunctionManager


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


_NORMALIZE_TEXT = (
    "def normalize_text(value, operations):\n"
    '    """Apply text operations to *value*, left to right; \'\' for non-strings."""\n'
    "    if not isinstance(value, str):\n"
    "        return ''\n"
    "    for op in operations:\n"
    "        if op == 'strip':\n"
    "            value = value.strip()\n"
    "        elif op == 'lower':\n"
    "            value = value.lower()\n"
    "        elif op == 'upper':\n"
    "            value = value.upper()\n"
    "        elif op == 'title':\n"
    "            value = value.title()\n"
    "        elif op == 'digits_only':\n"
    "            value = ''.join(ch for ch in value if ch.isdigit())\n"
    "        elif op == 'collapse_spaces':\n"
    "            value = ' '.join(value.split())\n"
    "    return value\n"
)
_PLAYBOOK_TITLE = "Adaptive record cleaning with normalize_text"
_PLAYBOOK = (
    "1. Measure completeness (name and email both non-empty).\n"
    "2. Below 0.8 normalize aggressively (strip, lower, collapse_spaces); "
    "otherwise gently (strip+title for names, strip+lower for emails).\n"
    "3. Drop records with neither name nor email.\n"
    "4. Merge records sharing a normalized email, keeping the fullest.\n"
    "5. Re-measure; under 5 points of improvement, flag for manual review."
)


# ---------------------------------------------------------------------------
# Test: storage loop stores both function(s) AND guidance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_storage_loop_stores_both_function_and_guidance(monkeypatch):
    """The storage review stores both a function (FM) and guidance (GM).

    The task produces a single reusable utility function AND demonstrates
    a multi-phase procedure that can only be captured as a guidance
    playbook. The scripted review stores the utility through
    ``functions.add`` and the procedure through ``guidance.add`` in one
    cell; the function lands in the FunctionManager and the guidance entry
    reaches the actor's guidance manager as the review wrote it.

    The review gate is answered "review" here, so the review it opens runs
    as shipped. This test is of the review's function/guidance split, not
    of the gate.
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

    model = ScriptedModel(
        actor=[
            reply(
                calls=[
                    cell(
                        _NORMALIZE_TEXT
                        + "print(normalize_text('  JOHN   DOE ', "
                        "['strip', 'collapse_spaces', 'title']))",
                    ),
                ],
            ),
            reply("Cleaned 5 records into 2; completeness rose from 0.8 to 1.0."),
        ],
        review=[
            reply(
                calls=[
                    cell(
                        "print(await functions.add("
                        f"implementations=[{_NORMALIZE_TEXT!r}]))\n"
                        f"print(await guidance.add(title={_PLAYBOOK_TITLE!r}, "
                        f"content={_PLAYBOOK!r}))",
                    ),
                ],
            ),
            reply("Stored normalize_text and the cleaning playbook."),
        ],
    )
    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=180,
    )
    try:
        with scripted(model):
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
                "Using normalize_text, demonstrate the five-phase workflow "
                "(completeness, adaptive normalization, removal, merge by email, "
                "re-measure) on the dataset below.\n\n"
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
            assert result == "Cleaned 5 records into 2; completeness rose from 0.8 to 1.0."

            # result() resolves after the task phase; wait for storage to finish.
            deadline = asyncio.get_event_loop().time() + 300
            while not handle.done():
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError("Storage loop did not complete in time")
                await asyncio.sleep(0.5)
    finally:
        try:
            await actor.close()
        except Exception:
            pass

    # The stand-in guidance manager gives no count, so the library is not
    # known to be empty and the gate is asked (once).
    assert gate_asked == [True]

    # The review's functions.add stored the utility ...
    stored = {f.get("name") for f in fm.filter_functions() if isinstance(f, dict)}
    assert stored == {"normalize_text"}, stored

    # ... and its guidance.add reached the guidance manager as written.
    assert gm.add_calls == [
        {"title": _PLAYBOOK_TITLE, "content": _PLAYBOOK, "function_ids": None},
    ]
    assert model.kinds() == ["actor", "actor", "review", "review"]
    model.assert_used_up()


# ---------------------------------------------------------------------------
# Test: storage loop stores the function, and no guidance about its use
# ---------------------------------------------------------------------------

_NORMALIZE_PHONE = (
    "def normalize_phone(raw):\n"
    '    """Normalize a US phone number to E.164 (\'+1\' and ten digits)."""\n'
    "    digits = ''.join(ch for ch in str(raw) if ch.isdigit())\n"
    "    if len(digits) == 10:\n"
    "        return '+1' + digits\n"
    "    if len(digits) == 11 and digits.startswith('1'):\n"
    "        return '+' + digits\n"
    "    raise ValueError(f'not a US phone number: {raw!r}')\n"
)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_storage_loop_stores_function_without_guidance():
    """The storage review stores a function (FM) and writes no guidance (GM).

    The task produces a single, well-parameterized utility function
    (phone-number normalization) that involves no multi-step procedure. The
    review gate is asked (the stand-in guidance manager gives no count) and
    scripted to review; the scripted review stores the function and nothing
    else.

    What is asserted: the function is stored, which means the store check
    loaded it where it runs (store_check, step 4); its body is not run here,
    because model-written code never executes in the test process
    (test_bind_load_confinement), so the test checks that the stored source
    defines the function. And no guidance is written: the harness adds none
    of its own.
    """
    fm = FunctionManager(include_primitives=False)
    gm = _TrackingGuidanceManager()

    model = ScriptedModel(
        actor=[
            reply(
                calls=[
                    cell(
                        _NORMALIZE_PHONE
                        + "for raw in ['(555) 123-4567', '+1-555-123-4567', "
                        "'555.123.4567', '15551234567']:\n"
                        "    print(normalize_phone(raw))\n",
                    ),
                ],
            ),
            reply("normalize_phone gives +15551234567 for all four formats."),
        ],
        gate=[reply('{"review": true, "reason": "A reusable normalizer was written."}')],
        review=[
            reply(
                calls=[
                    cell(
                        "print(await functions.add("
                        f"implementations=[{_NORMALIZE_PHONE!r}]))",
                    ),
                ],
            ),
            reply("Stored normalize_phone; its docstring covers its use."),
        ],
    )
    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=180,
    )
    try:
        with scripted(model):
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
            assert result == "normalize_phone gives +15551234567 for all four formats."

            # result() resolves after the task phase; wait for storage to finish.
            deadline = asyncio.get_event_loop().time() + 300
            while not handle.done():
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError("Storage loop did not complete in time")
                await asyncio.sleep(0.5)
    finally:
        try:
            await actor.close()
        except Exception:
            pass

    stored = fm.filter_functions()
    by_name = {f.get("name"): f for f in stored if isinstance(f, dict)}
    assert list(by_name) == ["normalize_phone"], sorted(by_name)
    tree = ast.parse(by_name["normalize_phone"].get("implementation") or "")
    assert [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ] == ["normalize_phone"]

    # No guidance is written.
    assert gm.add_calls == []
    assert model.kinds() == ["actor", "actor", "gate", "review", "review"]
    model.assert_used_up()
