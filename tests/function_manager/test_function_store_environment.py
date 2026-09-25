"""
Tests for FunctionStoreEnvironment.

Verifies that stored FunctionManager functions can be promoted into an
environment for prompt injection and sandbox execution, and that they are
automatically excluded from FunctionManager search results.
"""

import pytest

from unify.actor.environments.function_store import FunctionStoreEnvironment
from unify.function_manager.function_manager import FunctionManager
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

_PY_ALPHA = (
    'async def alpha(x: int) -> int:\n    """Double the input."""\n    return x * 2\n'
)
_PY_BETA = (
    'async def beta(y: int) -> int:\n    """Square the input."""\n    return y ** 2\n'
)
_PY_GAMMA = (
    'async def gamma(z: int) -> int:\n    """Triple the input."""\n    return z * 3\n'
)


@pytest.fixture
def fm_factory():
    """Factory that creates FunctionManager instances with context cleanup."""
    managers = []

    def _create(**kwargs):
        kwargs.setdefault("include_primitives", False)
        fm = FunctionManager(**kwargs)
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# 1. Construction validation
# ────────────────────────────────────────────────────────────────────────────


def test_requires_names_or_ids():
    """Must provide at least one of function_names or function_ids."""
    from unittest.mock import MagicMock

    fm = MagicMock()
    with pytest.raises(ValueError, match="At least one"):
        FunctionStoreEnvironment(fm)


# ────────────────────────────────────────────────────────────────────────────
# 2. Metadata resolution
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_resolves_metadata_by_name(fm_factory):
    """Functions are resolved by name from the FunctionManager."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA, _PY_GAMMA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha", "beta"])

    assert len(env._func_metadata) == 2
    names = {r["name"] for r in env._func_metadata}
    assert names == {"alpha", "beta"}


@_handle_project
def test_resolves_metadata_by_id(fm_factory):
    """Functions are resolved by ID from the FunctionManager."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])

    # Get the IDs
    listing = fm.list_functions()
    alpha_id = listing["alpha"]["function_id"]

    env = FunctionStoreEnvironment(fm, function_ids=[alpha_id])

    assert len(env._func_metadata) == 1
    assert env._func_metadata[0]["name"] == "alpha"


@_handle_project
def test_resolves_by_both_names_and_ids_deduplicates(fm_factory):
    """When both names and IDs are provided, results are deduplicated."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])

    listing = fm.list_functions()
    alpha_id = listing["alpha"]["function_id"]

    # Request alpha by both name AND id
    env = FunctionStoreEnvironment(
        fm,
        function_names=["alpha"],
        function_ids=[alpha_id],
    )

    # Should not duplicate
    names = [r["name"] for r in env._func_metadata]
    assert names.count("alpha") == 1


# ────────────────────────────────────────────────────────────────────────────
# 3. get_tools() — ToolMetadata with function_id tagging
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_get_tools_returns_tagged_metadata(fm_factory):
    """get_tools() returns ToolMetadata with function_id and context='compositional'."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha", "beta"])
    tools = env.get_tools()

    assert len(tools) == 2
    for fq_name, meta in tools.items():
        assert fq_name.startswith("functions.")
        assert meta.function_id is not None
        assert meta.function_context == "compositional"
        assert isinstance(meta.function_id, int)


@_handle_project
def test_get_tools_custom_namespace(fm_factory):
    """get_tools() uses the custom namespace in fully-qualified names."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA])

    env = FunctionStoreEnvironment(
        fm,
        function_names=["alpha"],
        namespace="skills",
    )
    tools = env.get_tools()

    assert "skills.alpha" in tools
    assert "functions.alpha" not in tools


@_handle_project
def test_get_tools_includes_docstring_and_signature(fm_factory):
    """get_tools() includes the FM-stored docstring and argspec."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha"])
    tools = env.get_tools()
    meta = tools["functions.alpha"]

    assert meta.docstring is not None
    assert "Double" in meta.docstring
    assert meta.signature is not None


# ────────────────────────────────────────────────────────────────────────────
# 4. get_prompt_context() — prompt generation
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_get_prompt_context_includes_functions(fm_factory):
    """get_prompt_context() includes function signatures and docstrings."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha", "beta"])
    context = env.get_prompt_context()

    assert "functions.alpha" in context
    assert "functions.beta" in context
    assert "Double" in context
    assert "Square" in context


@_handle_project
def test_get_prompt_context_warns_not_to_search(fm_factory):
    """get_prompt_context() tells the LLM not to search FM for these functions."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha"])
    context = env.get_prompt_context()

    assert "excluded from FunctionManager search" in context
    assert "prompt-injected" in context


@_handle_project
def test_get_prompt_context_includes_function_id(fm_factory):
    """get_prompt_context() renders function_id in the header."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha"])
    context = env.get_prompt_context()

    assert "function_id:" in context


@_handle_project
def test_get_prompt_context_renders_metadata_fields(fm_factory):
    """get_prompt_context() renders all LLM-meaningful metadata when present."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha"])

    env._func_metadata[0]["guidance_ids"] = [5, 12]
    env._func_metadata[0]["depends_on"] = ["helper_func", "primitives.actor.act"]
    env._func_metadata[0]["precondition"] = {"url": "https://example.com"}

    context = env.get_prompt_context()

    assert "Related guidance: [5, 12]" in context
    assert "Depends on: helper_func, primitives.actor.act" in context
    assert "Precondition:" in context


# ────────────────────────────────────────────────────────────────────────────
# 5. get_sandbox_instance() — callable resolution
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_get_sandbox_instance_returns_callable_namespace(fm_factory):
    """get_sandbox_instance() returns an object with callable function attributes."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha", "beta"])
    sandbox = env.get_sandbox_instance()

    assert hasattr(sandbox, "alpha")
    assert hasattr(sandbox, "beta")
    assert callable(sandbox.alpha)
    assert callable(sandbox.beta)


@_handle_project
def test_get_sandbox_instance_empty_returns_empty_namespace(fm_factory):
    """get_sandbox_instance() returns empty namespace when no functions matched."""
    fm = fm_factory()

    # Create env with a name that doesn't exist — metadata will be empty
    env = FunctionStoreEnvironment.__new__(FunctionStoreEnvironment)
    env._function_manager = fm
    env._requested_names = ["nonexistent"]
    env._requested_ids = []
    env._func_metadata = []
    env._instance = None
    env._namespace = "functions"
    env._clarification_up_q = None
    env._clarification_down_q = None

    sandbox = env.get_sandbox_instance()
    assert not any(attr for attr in dir(sandbox) if not attr.startswith("_"))


@_handle_project
@pytest.mark.asyncio
async def test_actor_sandbox_can_call_promoted_functions(fm_factory):
    """The actor hides promoted ids from its own FunctionManager's discovery;
    the sandbox must still expose those functions under the namespace."""
    from unify.actor.code_act_actor import CodeActActor

    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA])
    env = FunctionStoreEnvironment(fm, function_names=["alpha"])

    actor = CodeActActor(environments=[env], function_manager=fm)
    assert fm.exclude_compositional_ids == {env._func_metadata[0]["function_id"]}

    out = await actor._session_executor.execute(
        code="print(await functions.alpha(3))",
        state_mode="stateless",
        session_id=None,
    )

    assert out["error"] is None
    assert "6" in str(out["stdout"])


# ────────────────────────────────────────────────────────────────────────────
# 6. Exclusion integration — tagged functions hidden from FM search
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_exclusion_integration_list_functions(fm_factory):
    """Functions promoted to the environment should be excludable from list_functions."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA, _PY_GAMMA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha", "beta"])

    # Collect the function_ids from the environment
    excl_ids = frozenset(
        meta.function_id
        for meta in env.get_tools().values()
        if meta.function_id is not None
    )
    assert len(excl_ids) == 2

    # Create a scoped FM with those exclusions
    fm_excl = fm_factory(exclude_compositional_ids=excl_ids)
    listing = fm_excl.list_functions()

    assert "alpha" not in listing, "alpha should be excluded"
    assert "beta" not in listing, "beta should be excluded"
    assert "gamma" in listing, "gamma should still be visible"


@_handle_project
def test_exclusion_integration_search_functions(fm_factory):
    """Functions promoted to the environment should be excludable from search_functions."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA, _PY_GAMMA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha"])
    excl_ids = frozenset(
        meta.function_id
        for meta in env.get_tools().values()
        if meta.function_id is not None
    )

    fm_excl = fm_factory(exclude_compositional_ids=excl_ids)
    hits = fm_excl.search_functions(query="double the input", n=10)
    names = {h["name"] for h in hits}

    assert "alpha" not in names, "alpha should be excluded from search"


@_handle_project
def test_exclusion_integration_filter_functions(fm_factory):
    """Functions promoted to the environment should be excludable from filter_functions."""
    fm = fm_factory()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA, _PY_GAMMA])

    env = FunctionStoreEnvironment(fm, function_names=["alpha", "beta"])
    excl_ids = frozenset(
        meta.function_id
        for meta in env.get_tools().values()
        if meta.function_id is not None
    )

    fm_excl = fm_factory(exclude_compositional_ids=excl_ids)
    hits = fm_excl.filter_functions()
    names = {h["name"] for h in hits}

    assert "alpha" not in names
    assert "beta" not in names
    assert "gamma" in names
