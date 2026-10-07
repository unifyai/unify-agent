from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.function_manager.execution_env import create_base_globals
from unify.function_manager.function_manager import FunctionManager
from unify.settings import SETTINGS


@pytest.fixture(autouse=True)
def _python_in_process(monkeypatch):
    """Python in process (``UNIFY_WORKSPACE_PYTHON`` empty): these tests load
    stored functions into this process and call them. With the sandboxed
    worker a read binds them by source and nothing runs here
    (tests/actor/code_act/test_bind_load_confinement.py)."""
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")


@_handle_project
def test_invalid_parameter_combinations_raise():
    fm = FunctionManager()

    with pytest.raises(ValueError, match="_also_return_metadata"):
        fm.filter_functions(filter=None, _also_return_metadata=True)

    with pytest.raises(ValueError, match="_namespace"):
        fm.filter_functions(filter=None, _return_callable=True)

    with pytest.raises(ValueError, match="_also_return_metadata"):
        fm.search_functions(query="anything", _also_return_metadata=True)

    with pytest.raises(ValueError, match="_namespace"):
        fm.search_functions(query="anything", _return_callable=True)


@_handle_project
@pytest.mark.asyncio
async def test_filter_return_callable_injects_dependency_chain():
    fm = FunctionManager()

    c_src = "async def c(x: int) -> int:\n    return x + 1\n"
    b_src = "async def b(x: int) -> int:\n    return (await c(x=x)) + 10\n"
    a_src = "async def a(x: int) -> int:\n    return (await b(x=x)) + 100\n"

    fm.add_functions(implementations=[a_src, b_src, c_src])

    ns = create_base_globals()
    callables = fm.filter_functions(
        filter="name = 'a'",
        limit=1,
        _return_callable=True,
        _namespace=ns,
    )

    assert len(callables) == 1
    assert "a" in ns and callable(ns["a"])
    assert "b" in ns and callable(ns["b"])
    assert "c" in ns and callable(ns["c"])

    result = await ns["a"](x=1)
    assert result == 112


@_handle_project
@pytest.mark.asyncio
async def test_dependency_injection_supports_indirect_calls_and_returned_functions():
    """
    Stress cases that require dependency tracking beyond direct `foo()` calls:
    - alias assignment: `fn = dep; await fn(...)`
    - returning function references: `return dep`
    """
    fm = FunctionManager()

    leaf_src = "async def leaf(x: int) -> int:\n    return x + 1\n"
    factory_src = "async def factory():\n    return leaf\n"
    # Use alias assignment + call the returned function
    use_src = (
        "async def use(x: int) -> int:\n"
        "    f = factory\n"
        "    fn = await f()\n"
        "    return await fn(x=x)\n"
    )

    fm.add_functions(implementations=[leaf_src, factory_src, use_src])

    ns = create_base_globals()
    callables = fm.filter_functions(
        filter="name = 'use'",
        limit=1,
        _return_callable=True,
        _namespace=ns,
    )
    assert len(callables) == 1
    assert "leaf" in ns and callable(ns["leaf"])
    assert "factory" in ns and callable(ns["factory"])
    assert "use" in ns and callable(ns["use"])

    assert await ns["use"](x=10) == 11


@_handle_project
def test_search_return_callable_also_returns_metadata():
    fm = FunctionManager()
    fm.add_functions(
        implementations="def add_numbers(a: int, b: int) -> int:\n    return a + b\n",
    )

    ns = create_base_globals()
    res = fm.filter_functions(
        filter="name = 'add_numbers'",
        limit=1,
        _return_callable=True,
        _namespace=ns,
        _also_return_metadata=True,
    )

    assert isinstance(res, dict)
    assert set(res.keys()) == {"callables", "metadata"}
    assert isinstance(res["callables"], list)
    assert isinstance(res["metadata"], list)
    assert len(res["callables"]) == 1
    assert len(res["metadata"]) == 1


@_handle_project
def test_fresh_namespace_does_not_shadow_builtin_annotation_names():
    """Callable injection into a fresh ``{}`` namespace (the symbolic-entrypoint
    repair snapshot path) must not shadow builtins referenced in annotations.

    A placeholder class named ``dict`` breaks ``dict[str, Any]`` evaluation at
    def time with ``TypeError: type 'dict' is not subscriptable``.
    """
    fm = FunctionManager()
    fm.add_functions(
        implementations=(
            "async def tick(**params: Any) -> dict[str, Any]:\n" "    return {}\n"
        ),
    )

    ns: dict = {}
    res = fm.filter_functions(
        filter="name = 'tick'",
        limit=1,
        _return_callable=True,
        _namespace=ns,
        _also_return_metadata=True,
    )

    assert len(res["callables"]) == 1
    assert callable(ns["tick"])
    # Builtins must resolve to the real types, never injected placeholders.
    assert ns.get("dict", dict) is dict
    assert ns.get("str", str) is str


@_handle_project
def test_circular_dependency_injection_does_not_loop():
    fm = FunctionManager()

    a_src = "async def a() -> int:\n    return await b()\n"
    b_src = "async def b() -> int:\n    return await a()\n"

    fm.add_functions(implementations=[a_src, b_src])

    ns = create_base_globals()
    callables = fm.filter_functions(
        filter="name = 'a'",
        limit=1,
        _return_callable=True,
        _namespace=ns,
    )

    assert len(callables) == 1
    assert "a" in ns and callable(ns["a"])
    assert "b" in ns and callable(ns["b"])


@_handle_project
@pytest.mark.asyncio
async def test_filter_return_callable_with_dependencies_executes():
    """A function stored with dependencies runs in-process like any other.

    ``packaging`` is one of the runtime's own dependencies, so the callable
    resolves it without an install and the proxy round-trips a result.
    """
    fm = FunctionManager()

    fm.add_functions(
        implementations=(
            "def parse_version(text: str) -> str:\n"
            "    from packaging.version import Version\n"
            "    return str(Version(text))\n"
        ),
        dependencies=["packaging"],
    )
    assert fm.list_functions()["parse_version"]["dependencies"] == ["packaging"]

    ns = create_base_globals()
    callables = fm.filter_functions(
        filter="name = 'parse_version'",
        limit=1,
        _return_callable=True,
        _namespace=ns,
    )

    assert len(callables) == 1
    proxy = callables[0]
    assert getattr(proxy, "__name__", None) == "parse_version"

    result = await proxy(text="1.2.0")
    assert result == "1.2.0"


@pytest.mark.requires_provider_key
@_handle_project
@pytest.mark.asyncio
async def test_search_return_callable_with_metadata():
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(
        implementations=(
            "async def foo(x: int) -> int:\n"
            '    """Add one."""\n'
            "    return x + 1\n"
        ),
    )

    ns = create_base_globals()
    res = fm.search_functions(
        query="add one",
        n=1,
        _return_callable=True,
        _namespace=ns,
        _also_return_metadata=True,
    )

    assert isinstance(res, dict)
    assert set(res.keys()) == {"callables", "metadata"}
    assert len(res["callables"]) == 1
    assert [row["name"] for row in res["metadata"]] == ["foo"]

    fn = res["callables"][0]
    assert callable(fn)
    assert "foo" in ns
    assert await fn(x=1) == 2


@_handle_project
@pytest.mark.asyncio
async def test_dependency_injection_supports_user_defined_forward_ref_string_annotations():
    """
    Validate "user-defined" annotation names expressed as forward-ref strings.

    This mirrors patterns like:
      group_by: 'Optional[GroupBy | str]' = None
      time_period: 'TimePeriod' = 'day'
      -> 'MetricResult'

    The key behavior we care about:
    - exec() succeeds (because annotations are strings)
    - if code later resolves hints (e.g. via typing.get_type_hints), it works
      provided the caller injected the referenced types into the namespace.
    """
    fm = FunctionManager()

    metric_src = (
        "async def metric(\n"
        "    group_by: 'Optional[GroupBy | str]' = None,\n"
        "    start_date: 'Optional[str]' = None,\n"
        "    end_date: 'Optional[str]' = None,\n"
        "    time_period: 'TimePeriod' = 'day',\n"
        "    include_plots: 'bool' = False,\n"
        ") -> 'MetricResult':\n"
        "    hints = typing.get_type_hints(metric, include_extras=True)\n"
        "    group_by_str = str(hints['group_by'])\n"
        "    tp = hints['time_period']\n"
        "    ret = hints['return']\n"
        "    return (group_by_str, tp.__name__, ret.__name__)\n"
    )

    fm.add_functions(implementations=[metric_src])

    class GroupBy:  # user-defined type
        pass

    class TimePeriod:  # user-defined type
        pass

    class MetricResult:  # user-defined type
        pass

    ns = create_base_globals()
    ns["GroupBy"] = GroupBy
    ns["TimePeriod"] = TimePeriod
    ns["MetricResult"] = MetricResult

    callables = fm.filter_functions(
        filter="name = 'metric'",
        limit=1,
        _return_callable=True,
        _namespace=ns,
    )
    assert len(callables) == 1
    assert "metric" in ns and callable(ns["metric"])

    group_by_str, time_period_name, return_name = await ns["metric"]()
    assert "GroupBy" in group_by_str
    assert "str" in group_by_str
    assert "NoneType" in group_by_str
    assert time_period_name == "TimePeriod"
    assert return_name == "MetricResult"


@pytest.mark.requires_provider_key
@_handle_project
@pytest.mark.asyncio
async def test_search_return_callable_forward_ref_annotations_just_work():
    """
    Validate the CodeActActor-style flow:
      fm.search_functions(..., return_callable=True)

    Caller provides only `create_base_globals()` (no manual injection of
    GroupBy/TimePeriod/MetricResult), and the returned callable should still
    be able to resolve forward-ref annotations without NameError.
    """
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(
        implementations=(
            "async def metric(\n"
            "    group_by: 'Optional[GroupBy | str]' = None,\n"
            "    start_date: 'Optional[str]' = None,\n"
            "    end_date: 'Optional[str]' = None,\n"
            "    time_period: 'TimePeriod' = 'day',\n"
            "    include_plots: 'bool' = False,\n"
            ") -> 'MetricResult':\n"
            '    """Metric query."""\n'
            "    hints = typing.get_type_hints(metric, include_extras=True)\n"
            "    group_by_str = str(hints['group_by'])\n"
            "    tp = hints['time_period']\n"
            "    ret = hints['return']\n"
            "    return (group_by_str, tp.__name__, ret.__name__)\n"
        ),
    )

    ns = create_base_globals()
    res = fm.search_functions(
        query="metric query",
        n=1,
        _return_callable=True,
        _namespace=ns,
        _also_return_metadata=True,
    )

    assert isinstance(res, dict)
    assert set(res.keys()) == {"callables", "metadata"}
    assert len(res["callables"]) == 1

    fn = res["callables"][0]
    group_by_str, time_period_name, return_name = await fn()

    # "Just works" = no NameError, and forward-ref symbols are resolvable.
    assert "GroupBy" in group_by_str
    assert "str" in group_by_str
    assert "NoneType" in group_by_str
    assert time_period_name == "TimePeriod"
    assert return_name == "MetricResult"


@_handle_project
@pytest.mark.asyncio
async def test_a_root_is_stored_only_once_its_callee_exists():
    fm = FunctionManager()

    a_src = "async def a(x: int) -> int:\n    return (await b(x=x)) + 1\n"
    b_src = "async def b(x: int) -> int:\n    return x + 10\n"

    # The storage check refuses a root whose callee does not exist yet.
    with pytest.raises(ValueError, match="`b` is not defined"):
        fm.add_functions(implementations=[a_src])

    # Once b exists, a is stored with b in its depends_on, and injected.
    fm.add_functions(implementations=[b_src])
    fm.add_functions(implementations=[a_src])

    ns2 = create_base_globals()
    callables2 = fm.filter_functions(
        filter="name = 'a'",
        limit=1,
        _return_callable=True,
        _namespace=ns2,
    )
    assert "b" in ns2
    assert await callables2[0](x=1) == 12
