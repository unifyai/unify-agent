from __future__ import annotations

import json

import pytest

from tests.helpers import _handle_project
from tests.scripted_model import Always, ScriptedModel, reply, scripted
from unify.function_manager.simulated import SimulatedFunctionManager

# The simulated catalogue's model calls are tool-less ``query_llm`` requests;
# the tests below script the catalogue they return and check what the
# manager makes of it (and what it asked for).
_CATALOGUE = {
    "parse_csv": {
        "function_id": 1,
        "argspec": "(text: str) -> list",
        "docstring": "Parse CSV text into rows.",
    },
    "sum_column": {
        "function_id": 2,
        "argspec": "(rows: list, column: str) -> float",
        "docstring": "Sum one column of parsed rows.",
    },
}


def _text(content) -> str:
    """A message's text, whether sent as a string or as content blocks."""
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _prompt(call) -> str:
    """The request's last message: the prompt this call sent."""
    return _text(call.messages[-1]["content"])


# --------------------------------------------------------------------------- #
#  Doc-string inheritance                                                     #
# --------------------------------------------------------------------------- #


def test_docstrings_match_base():
    """
    Public methods in SimulatedFunctionManager should copy the real
    BaseFunctionManager doc-strings one-for-one (via functools.wraps).
    """
    from unify.function_manager.base import BaseFunctionManager
    from unify.function_manager.simulated import SimulatedFunctionManager

    assert (
        BaseFunctionManager.add_functions.__doc__.strip()
        in SimulatedFunctionManager.add_functions.__doc__.strip()
    ), ".add_functions doc-string was not copied correctly"

    assert (
        BaseFunctionManager.list_functions.__doc__.strip()
        in SimulatedFunctionManager.list_functions.__doc__.strip()
    ), ".list_functions doc-string was not copied correctly"


# --------------------------------------------------------------------------- #
#  add_functions                                                              #
# --------------------------------------------------------------------------- #


@_handle_project
def test_add_functions_accepts_single_and_multiple():
    fm = SimulatedFunctionManager()

    # single implementation
    one_src = "def alpha(x):\n    return x\n"
    res_one = fm.add_functions(implementations=one_src)
    assert isinstance(res_one, dict) and res_one
    ((name_one, status_one),) = res_one.items()
    assert name_one == "alpha"
    assert "added" in status_one.lower()

    # multiple implementations
    a_src = "def a():\n    return 1\n"
    b_src = "def b(y):\n    return y\n"
    res_many = fm.add_functions(implementations=[a_src, b_src])
    assert set(res_many.keys()) == {"a", "b"}
    assert all("added" in v.lower() for v in res_many.values())


# --------------------------------------------------------------------------- #
#  list_functions                                                             #
# --------------------------------------------------------------------------- #


@_handle_project
def test_list_functions_with_and_without_implementations():
    with_impl = {
        name: {**meta, "implementation": f"def {name}(*a):\n    return None\n"}
        for name, meta in _CATALOGUE.items()
    }
    model = ScriptedModel(
        query_llm=[reply(json.dumps(_CATALOGUE)), reply(json.dumps(with_impl))],
    )
    fm = SimulatedFunctionManager()

    with scripted(model):
        listing = fm.list_functions()
        listing_with_impl = fm.list_functions(include_implementations=True)

    assert isinstance(listing, dict) and listing
    # pick a representative entry and verify shape without relying on specific names
    first_name, first_entry = next(iter(listing.items()))
    assert isinstance(first_entry, dict)
    for key in ("function_id", "argspec", "docstring"):
        assert key in first_entry
    assert "implementation" not in first_entry

    assert isinstance(listing_with_impl, dict) and listing_with_impl
    # Ensure the same entry now includes an implementation when requested
    assert first_name in listing_with_impl
    assert "implementation" in listing_with_impl[first_name]

    # Implementations are asked for only when requested, and the simulation
    # is stateful: the second request carries the first exchange.
    first, second = model.of("query_llm")
    assert model.kinds() == ["query_llm", "query_llm"]
    assert "implementation" not in _prompt(first)
    assert "implementation" in _prompt(second)
    assert any(
        m["role"] == "assistant" and m.get("content") == json.dumps(_CATALOGUE)
        for m in second.messages
    )
    model.assert_used_up()


# --------------------------------------------------------------------------- #
#  get_precondition                                                           #
# --------------------------------------------------------------------------- #


@_handle_project
def test_get_precondition_returns_none():
    fm = SimulatedFunctionManager()
    assert fm.get_precondition(function_name="does_not_matter") is None


# --------------------------------------------------------------------------- #
#  delete_function                                                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_delete_function_acknowledges():
    fm = SimulatedFunctionManager()
    ack = fm.delete_function(function_id=42, delete_dependents=True)
    assert isinstance(ack, dict) and ack
    ((k, v),) = ack.items()
    assert "id=42" in k
    assert "deleted" in v.lower()


# --------------------------------------------------------------------------- #
#  search_functions                                                           #
# --------------------------------------------------------------------------- #


@_handle_project
def test_search_functions_returns_list_of_dicts():
    record = {
        "name": "price_with_tax",
        "function_id": 1,
        "argspec": "(price: float) -> float",
        "docstring": "Add sales tax to a price.",
    }
    model = ScriptedModel(query_llm=[reply(json.dumps([record]))])
    fm = SimulatedFunctionManager()
    with scripted(model):
        hits = fm.filter_functions(filter="docstring LIKE '%price%'")
    assert isinstance(hits, list) and hits
    first = hits[0]
    assert isinstance(first, dict)
    for key in ("name", "function_id", "argspec", "docstring"):
        assert key in first

    # The filter reaches the model; an unscoped manager adds no scope clause.
    (call,) = model.of("query_llm")
    assert "docstring LIKE '%price%'" in _prompt(call)
    assert "Scope constraint" not in _prompt(call)
    model.assert_used_up()


# --------------------------------------------------------------------------- #
#  search_functions                                             #
# --------------------------------------------------------------------------- #


@_handle_project
def test_search_functions_bounds_and_shape():
    # Keeping to n is the model's job (the simulation does not truncate), so
    # the scripted reply keeps to it; the test checks that n and the query
    # reach the model and that a fenced JSON reply is parsed into records.
    n = 2
    records = [
        {"name": "add", "function_id": 1, "argspec": "(a, b)", "score": 0.9},
        {"name": "add_all", "function_id": 2, "argspec": "(xs)", "score": 0.7},
    ]
    model = ScriptedModel(
        query_llm=[reply(f"Here you go:\n```json\n{json.dumps(records)}\n```")],
    )
    fm = SimulatedFunctionManager()
    with scripted(model):
        sims = fm.search_functions(query="add numbers", n=n)
    assert isinstance(sims, list) and 1 <= len(sims) <= n
    first = sims[0]
    assert isinstance(first, dict)
    for key in ("name", "function_id", "argspec", "score"):
        assert key in first

    (call,) = model.of("query_llm")
    assert f"query='add numbers', n={n}" in _prompt(call)
    model.assert_used_up()


# --------------------------------------------------------------------------- #
#  execute_function                                                           #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_returns_simulated_response():
    """
    SimulatedFunctionManager.execute_function should return a simulated
    response with the expected structure (result, error, stdout, stderr).
    """
    fm = SimulatedFunctionManager()
    response = await fm.execute_function(
        function_name="my_function",
        call_kwargs={"x": 1, "y": 2},
    )
    assert isinstance(response, dict)
    # Check required keys in response
    assert "result" in response
    assert "error" in response
    assert "stdout" in response
    assert "stderr" in response
    # Error should be None for successful simulation
    assert response["error"] is None
    # Result should contain simulated marker and function info
    result = response["result"]
    assert isinstance(result, dict)
    assert result.get("simulated") is True
    assert result.get("function_name") == "my_function"
    assert result.get("call_kwargs") == {"x": 1, "y": 2}


# --------------------------------------------------------------------------- #
#  clear                                                                      #
# --------------------------------------------------------------------------- #


@_handle_project
def test_clear_sync():
    """
    SimulatedFunctionManager.clear should reset the manager (hard-coded completion)
    and remain usable afterwards.
    """
    model = ScriptedModel(query_llm=Always(reply(json.dumps(_CATALOGUE))))
    fm = SimulatedFunctionManager()
    with scripted(model):
        # Do a synchronous operation to create some prior state
        fm.add_functions(implementations="def _tmp():\n    return 1\n")
        fm.list_functions()
        # Clear should not raise and should be quick (no LLM roundtrip requirement)
        fm.clear()
        assert len(model.calls) == 1
        # Post-clear, read-only operations should still work
        post = fm.list_functions()
    assert isinstance(post, dict) and post is not None

    # The reset dropped the simulation's history: the request after clear()
    # carries no more than the first one did.
    before, after = model.of("query_llm")
    assert len(after.messages) == len(before.messages)


# --------------------------------------------------------------------------- #
#  filter_scope                                                                #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_respected_in_list_functions():
    """
    A SimulatedFunctionManager with filter_scope="docstring LIKE '%data%'"
    puts the scope before the model twice: in the simulation's system message
    and as a constraint in the list_functions request. The catalogue it
    returns is the model's, parsed as it came.

    The simulation does not filter the reply itself (keeping to the scope is
    the model's job), so the scripted catalogue keeps to it.
    """
    scope = "docstring LIKE '%data%'"
    in_scope = {
        "load_data": {
            "function_id": 1,
            "argspec": "(path: str) -> list",
            "docstring": "Load data rows from a CSV file.",
        },
        "clean_data": {
            "function_id": 2,
            "argspec": "(rows: list) -> list",
            "docstring": "Drop empty data rows.",
        },
    }
    model = ScriptedModel(query_llm=[reply(json.dumps(in_scope))])
    fm = SimulatedFunctionManager(
        description="A mixed catalogue of data-processing and networking utility functions",
        filter_scope=scope,
    )
    with scripted(model):
        listing = fm.list_functions()
    assert listing == in_scope

    (call,) = model.of("query_llm")
    system = "\n".join(
        _text(m["content"]) for m in call.messages if m["role"] == "system"
    )
    assert f"filter_scope applied: {scope!r}" in system
    assert f"Scope constraint (MUST be satisfied by every returned function): {scope!r}" in _prompt(call)
    model.assert_used_up()


# --------------------------------------------------------------------------- #
#  Setter contract (shared with FunctionManager)                              #
# --------------------------------------------------------------------------- #


def test_filter_scope_setter():
    """filter_scope can be set after construction."""
    fm = SimulatedFunctionManager()
    assert fm.filter_scope is None

    fm.filter_scope = "docstring LIKE '%data%'"
    assert fm.filter_scope == "docstring LIKE '%data%'"

    fm.filter_scope = None
    assert fm.filter_scope is None


def test_exclude_primitive_ids_setter():
    """exclude_primitive_ids can be set after construction."""
    fm = SimulatedFunctionManager()
    assert fm.exclude_primitive_ids is None

    fm.exclude_primitive_ids = frozenset({1, 2, 3})
    assert fm.exclude_primitive_ids == frozenset({1, 2, 3})

    fm.exclude_primitive_ids = None
    assert fm.exclude_primitive_ids is None


def test_exclude_compositional_ids_setter():
    """exclude_compositional_ids can be set after construction."""
    fm = SimulatedFunctionManager()
    assert fm.exclude_compositional_ids is None

    fm.exclude_compositional_ids = frozenset({10, 20})
    assert fm.exclude_compositional_ids == frozenset({10, 20})

    fm.exclude_compositional_ids = None
    assert fm.exclude_compositional_ids is None


def test_setters_do_not_crash_when_called_from_code_act_actor():
    """SimulatedFunctionManager should survive CodeActActor.__init__ setter calls.

    CodeActActor.__init__ collects function_ids from environments and sets
    them on the FM via setters. This must not crash for SimulatedFunctionManager.
    The actor environment inlines its method docs whether or not it is
    filtered, so the actor excludes its tool from search either way.
    """
    from unify.actor.code_act_actor import CodeActActor
    from unify.actor.environments import ActorEnvironment

    fm = SimulatedFunctionManager(description="test")

    # Constructing a CodeActActor with a filtered environment should set
    # exclusions on the FM via setters, not replace it.
    actor = CodeActActor(
        environments=[
            ActorEnvironment(allowed_methods={"primitives.actor.act"}),
        ],
        function_manager=fm,
        timeout=30,
    )

    # The FM instance should be the SAME object (not replaced).
    assert actor.function_manager is fm

    # Exclusion IDs should have been set via the setter.
    assert fm.exclude_primitive_ids is not None
    assert len(fm.exclude_primitive_ids) == 1

    # An unfiltered environment documents its whole surface inline, so the
    # same exclusion is set through the same setter.
    fm_unfiltered = SimulatedFunctionManager(description="test")
    actor_unfiltered = CodeActActor(
        environments=[ActorEnvironment()],
        function_manager=fm_unfiltered,
        timeout=30,
    )
    assert actor_unfiltered.function_manager is fm_unfiltered
    assert fm_unfiltered.exclude_primitive_ids == fm.exclude_primitive_ids


def test_setters_update_system_message():
    """Setting exclusions should update the LLM system message."""
    fm = SimulatedFunctionManager()
    original_sys = fm._llm.system_message

    fm.exclude_primitive_ids = frozenset({42})
    updated_sys = fm._llm.system_message

    assert original_sys != updated_sys
    assert "42" in updated_sys
    assert "excluded" in updated_sys.lower()
