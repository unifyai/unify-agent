"""
pytest tests for the helper utilities:

* annotation_to_schema           – all supported annotation kinds
* method_to_schema               – schema structure & enum handling
"""

from __future__ import annotations

import functools
from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel

import unify.common.llm_helpers as llmh


# --------------------------------------------------------------------------- #
#  TEST DATA TYPES FOR SCHEMA TESTS                                           #
# --------------------------------------------------------------------------- #
class ColumnType(str, Enum):
    str = "str"
    int = "int"


class Person(BaseModel):
    name: str
    age: int


# Helper function defined at module scope to stabilise type-hint resolution
def _tool_with_optional_mapping(
    references: dict[str, str] | None = None,
    k: int = 10,
) -> None:  # pragma: no cover - schema only
    return None


# --------------------------------------------------------------------------- #
#  annotation_to_schema                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "t, checker",
    [
        (str, lambda s: s == {"type": "string"}),
        (int, lambda s: s == {"type": "integer"}),
        (
            ColumnType,
            lambda s: s["type"] == "string" and set(s["enum"]) == {"str", "int"},
        ),
        (
            Person,
            lambda s: s["type"] == "object" and {"name", "age"} <= set(s["properties"]),
        ),
        (
            dict[str, int],
            lambda s: s["type"] == "object"
            and s["additionalProperties"]["type"] == "integer",
        ),
        (
            list[Person],
            lambda s: s["type"] == "array" and s["items"]["type"] == "object",
        ),
        (Any, lambda s: s == {}),
        (object, lambda s: s == {}),
        (
            dict[str, Any],
            lambda s: s["type"] == "object" and s["additionalProperties"] == {},
        ),
    ],
)
def test_annotation_schema_conversion(t, checker):
    """Every major annotation flavour is converted correctly."""
    assert checker(llmh.annotation_to_schema(t))


def test_any_valued_mapping_does_not_constrain_values_to_strings():
    """``Dict[str, Any]`` must advertise unconstrained values, not strings.

    ``execute_function(call_kwargs: Optional[Dict[str, Any]])`` forwards keyword
    arguments to an arbitrary callee. Compiling ``Any`` to ``{"type": "string"}``
    tells the model every value must be a string, so a callee expecting
    ``max_results: int`` leaves no valid token to emit and the model stalls
    mid-argument.
    """
    from typing import Optional

    schema = llmh.annotation_to_schema(Optional[dict[str, Any]])

    assert schema["type"] == ["object", "null"]
    assert schema["additionalProperties"] == {}

    # A concretely-typed mapping must keep its value constraint.
    typed = llmh.annotation_to_schema(dict[str, int])
    assert typed["additionalProperties"] == {"type": "integer"}


# --------------------------------------------------------------------------- #
#  annotation_to_schema – Annotated[..., "description"] support              #
# --------------------------------------------------------------------------- #
def test_annotated_string_metadata_becomes_property_description():
    """A string in Annotated metadata surfaces as the property ``description``,
    including when nested inside Optional/list, while plain annotations and
    non-string metadata stay bare."""
    from typing import Annotated, Optional

    described = llmh.annotation_to_schema(Annotated[str, "why you are calling"])
    assert described["type"] == "string"
    assert described["description"] == "why you are calling"

    # Description survives the nullable widening of Optional[...].
    optional = llmh.annotation_to_schema(Optional[Annotated[str, "opt rationale"]])
    assert optional["type"] == ["string", "null"]
    assert optional["description"] == "opt rationale"

    # And attaches to the item schema inside a list.
    listed = llmh.annotation_to_schema(list[Annotated[str, "item rationale"]])
    assert listed["type"] == "array"
    assert listed["items"]["description"] == "item rationale"

    # Non-string metadata is ignored (no description leaks in).
    non_str = llmh.annotation_to_schema(Annotated[str, 42])
    assert non_str == {"type": "string"}

    # Plain annotations remain unchanged.
    assert llmh.annotation_to_schema(str) == {"type": "string"}


# --------------------------------------------------------------------------- #
#  method_to_schema – enum round-trip                                         #
# --------------------------------------------------------------------------- #
def _demo_func(a: str, col: ColumnType):
    """Docstring for unit test."""
    return None


def test_schema_includes_enum():
    schema = llmh.method_to_schema(_demo_func, "tool")
    params = schema["function"]["parameters"]["properties"]
    assert params["a"]["type"] == "string"
    # Enum must appear with *exact* allowed literals
    assert params["col"]["enum"] == ["str", "int"]


def test_method_to_schema_marks_tools_non_strict_by_default():
    def current_mode() -> str:
        """Return the current mode."""
        return "text"

    schema = llmh.method_to_schema(current_mode, "cm_get_mode")

    assert schema["strict"] is False
    assert schema["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
    }


async def _wrapped_schema_target(
    thought: str,
    code: str | None = None,
    *,
    session_name: str,
    state_mode: str = "stateless",
    _notification_up_q=None,
):
    """Execute arbitrary Python code in a specified state mode."""
    return None


def test_method_to_schema_preserves_wrapped_execute_code_signature():
    @functools.wraps(_wrapped_schema_target)
    async def wrapped_execute_code(*a, **kw):
        return await _wrapped_schema_target(*a, **kw)

    schema = llmh.method_to_schema(wrapped_execute_code, "execute_code")
    params = schema["function"]["parameters"]["properties"]
    required = schema["function"]["parameters"]["required"]
    desc = schema["function"]["description"]

    assert "thought" in params
    assert "code" in params
    assert "session_name" in params
    assert "state_mode" in params
    assert "_notification_up_q" not in params
    assert "thought" in required
    assert "session_name" in required
    assert "Execute arbitrary Python code in a specified state mode." in desc


# --------------------------------------------------------------------------- #
#  PRIVATE ARGUMENTS ARE NEVER EXPOSED                                        #
# --------------------------------------------------------------------------- #
def test_schema_hides_private_optionals() -> None:
    """
    Parameters whose names begin with an underscore (``_``) must **not**
    appear in the schema presented to the LLM — regardless of whether
    they are required or optional. The leading underscore is the
    convention for "internal plumbing", and tools meant to be LLM-callable
    must not make any underscored parameter required.

    Note on history: an earlier version of this test (2025-11-25,
    49abe0cd70) asserted that *required* private params should stay
    visible, on the grounds that "otherwise the tool would become
    impossible to call". But the production schema generator in
    unify/common/llm_helpers.py:method_to_schema (50812d1661,
    2025-06-03) has consistently stripped all underscored params
    without any required/optional split, and no real tool in the
    codebase declares a required underscored param — the convention
    enforces itself. The earlier test assertion was hidden by the
    discover_test_paths.py matrix bug (2026-01-26, 499de17cc) until
    today's matrix fix surfaced the contradiction. Updated here to
    encode the actual design contract.
    """

    # ── 1. optional private argument should be hidden ─────────────────────
    def sample_tool(a: int, b: int = 0, _secret: str = "x") -> int:
        """
        Sample calculator.

        Args:
            a: first addend.
            b: second addend, defaults to 0.
            _secret: **internal** flag, never shown to the LLM.
        """
        return a + b

    schema = llmh.method_to_schema(sample_tool, "tool")
    props = schema["function"]["parameters"]["properties"]
    required = schema["function"]["parameters"]["required"]
    desc = schema["function"]["description"]

    # public arguments are present …
    assert "a" in props and "b" in props
    # … while the optional private one is not
    assert "_secret" not in props
    # and its doc-line has been pruned
    assert "_secret" not in desc

    # required list unchanged
    assert "a" in required and "b" not in required

    # ── 2. required private argument is ALSO hidden (uniform rule) ────────
    def tool_with_required_private(x: int, _hidden: str) -> str:
        """
        Echo tool.

        Parameters
        ----------
        x : int
            Multiplier.
        _hidden : str
            Mandatory private value — internal plumbing, also stripped
            from the LLM-visible schema. Tools must not depend on the
            LLM providing this; supply it from the call site.
        """
        return _hidden * x

    schema2 = llmh.method_to_schema(tool_with_required_private, "tool")
    props2 = schema2["function"]["parameters"]["properties"]
    required2 = schema2["function"]["parameters"]["required"]
    desc2 = schema2["function"]["description"]

    # _hidden must NOT be exposed (uniform "_ = stripped" rule)
    assert "_hidden" not in props2 and "_hidden" not in required2
    # and its doc-line should also be pruned (consistent with optional case)
    assert "_hidden" not in desc2


# --------------------------------------------------------------------------- #
#  `_parent_chat_context` MUST NEVER BE EXPOSED                                #
# --------------------------------------------------------------------------- #
def test_schema_hides_context_param() -> None:
    """
    The special ``_parent_chat_context`` argument is injected automatically by
    the tool-loop.  It must be hidden from both the schema **and** the
    docstring that is sent to the LLM.
    """

    def tool_with_ctx(a: int, _parent_chat_context: list[dict]):
        """
        Dummy tool.

        Parameters
        ----------
        a : int
            Some value.
        _parent_chat_context : list[dict]
            Internal plumbing, never surfaced.
        """
        return a

    def tool_with_ctx_optional(
        a: int,
        _parent_chat_context: list[dict] | None = None,
    ):
        """
        Dummy tool (optional ctx).

        Args:
            a: Some value.
            _parent_chat_context: Internal plumbing, never surfaced.
        """
        return a

    for fn in (tool_with_ctx, tool_with_ctx_optional):
        schema = llmh.method_to_schema(fn, "tool")
        props = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]
        desc = schema["function"]["description"]

        assert "_parent_chat_context" not in props
        assert "_parent_chat_context" not in required
        # docstring has been scrubbed
        assert "_parent_chat_context" not in desc


# --------------------------------------------------------------------------- #
#  OPTIONAL[Dict[str, str]] COLLAPSES TO OBJECT (NO STRING ALTERNATIVE)       #
# --------------------------------------------------------------------------- #
def test_optional_dict_schema_simplification() -> None:
    """
    Optional[Dict[str, str]] should stay one nullable object schema.
    NoneType was once treated as "string", producing anyOf [object, string] and
    inviting serialized strings; the object form must remain the only non-null
    branch, with null expressible so the model can omit the argument outright.
    """

    schema = llmh.method_to_schema(_tool_with_optional_mapping, "tool")
    params = schema["function"]["parameters"]["properties"]
    refs_schema = params["references"]

    # One nullable object with string values — no second branch to mistake.
    assert "anyOf" not in refs_schema
    assert refs_schema["type"] == ["object", "null"]
    assert refs_schema["additionalProperties"]["type"] == "string"


# --------------------------------------------------------------------------- #
#  BUILTIN dict HANDLING (images: dict | None)                                #
# --------------------------------------------------------------------------- #
def _tool_with_optional_builtin_mapping(
    images: dict | None = None,
) -> None:  # pragma: no cover - schema only
    return None


def test_optional_builtin_dict_schema() -> None:
    """
    Optional[builtin dict] should surface as a nullable object to the LLM.
    Builtin dict could once degrade to "string" in unions, leading the model to
    send serialized strings for images.
    """

    schema = llmh.method_to_schema(_tool_with_optional_builtin_mapping, "tool")
    params = schema["function"]["parameters"]["properties"]
    images_schema = params["images"]

    assert "anyOf" not in images_schema
    assert images_schema["type"] == ["object", "null"]
    # Unknown value types → allow arbitrary properties
    assert images_schema.get("additionalProperties") is True


def test_dict_annotation_schema() -> None:
    s = llmh.annotation_to_schema(dict)
    assert s["type"] == "object"
    assert s.get("additionalProperties") is True


# --------------------------------------------------------------------------- #
#  method_to_schema – docstring MRO fallback                                  #
# --------------------------------------------------------------------------- #
def test_schema_inherits_base_docstring() -> None:
    class _Base:
        def action(self, x: int) -> None:
            """Base doc: perform action."""
            return None

    class _Child(_Base):
        def action(self, x: int) -> None:
            # no docstring → should inherit from base via MRO
            return None

    schema = llmh.method_to_schema(_Child().action, "tool")
    desc = schema["function"]["description"]
    assert "Base doc: perform action." in desc


def test_schema_prefers_child_docstring() -> None:
    class _Base:
        def go(self) -> None:
            """Base doc: go."""
            return None

    class _Child(_Base):
        def go(self) -> None:
            """Child doc: go fast."""
            return None

    schema = llmh.method_to_schema(_Child().go, "tool")
    desc = schema["function"]["description"]
    assert "Child doc: go fast." in desc
    assert "Base doc" not in desc


# --------------------------------------------------------------------------- #
#  INHERITED DOCSTRING + PRUNED WRAPPER SIGNATURE                              #
#  When a thin wrapper inherits a docstring (via __doc__) from a base class    #
#  that documents _-prefixed internal params, those params must be stripped     #
#  from the LLM-facing description even though they don't appear in the        #
#  wrapper's own signature.                                                    #
# --------------------------------------------------------------------------- #
def test_schema_strips_hidden_params_from_inherited_doc() -> None:
    """
    Regression: CodeActActor wraps FunctionManager methods with thin
    closures that have pruned signatures (only public params) but inherit
    the full base-class docstring via ``__doc__`` assignment.

    The ``_``-prefixed internal params documented in the base docstring
    must NOT leak through to the LLM-facing tool description.
    """

    class _BaseFM:
        def search(
            self,
            *,
            query: str,
            n: int = 5,
            _return_callable: bool = False,
            _namespace: dict | None = None,
        ) -> list:
            """
            Search for items by similarity.

            Parameters
            ----------
            query : str
                Natural-language search text.
            n : int, default ``5``
                Max results to return.
            _return_callable : bool, default ``False``
                When ``True``, return callables instead of metadata dicts.
            _namespace : dict | None, default ``None``
                Target namespace dict for callable injection when
                ``_return_callable=True``.

            Returns
            -------
            list
                Up to ``n`` results.
            """
            return []

    # Thin wrapper that includes _-prefixed params in the signature so the
    # stripping machinery can see them, even though the body ignores them.
    async def FunctionManager_search(
        query: str,
        n: int = 5,
        _return_callable: bool = False,
        _namespace: dict | None = None,
    ) -> list:
        return []

    FunctionManager_search.__doc__ = _BaseFM.search.__doc__

    schema = llmh.method_to_schema(FunctionManager_search, "tool")
    props = schema["function"]["parameters"]["properties"]
    desc = schema["function"]["description"]

    # Public params visible in the schema
    assert "query" in props
    assert "n" in props

    # Internal params must NOT appear in the schema
    assert "_return_callable" not in props
    assert "_namespace" not in props

    # Internal param documentation must NOT appear in the description
    assert "_return_callable" not in desc
    assert "_namespace" not in desc
    assert "callable injection" not in desc


def test_schema_strips_hidden_param_references_from_returns() -> None:
    """
    The Returns section may contain conditional branches like
    ``When ``_return_callable=False``: ...`` that reference hidden params.
    These branches must be stripped from the LLM-facing description since
    the LLM cannot control the param they depend on.
    """

    def search(
        query: str,
        n: int = 5,
        _return_callable: bool = False,
        _also_return_metadata: bool = False,
    ) -> list:
        """
        Search for items.

        Parameters
        ----------
        query : str
            Search text.
        n : int, default ``5``
            Max results.
        _return_callable : bool, default ``False``
            When ``True``, return callables instead of metadata.
        _also_return_metadata : bool, default ``False``
            When ``True``, return both callables and metadata.

        Returns
        -------
        list[dict] | list[Callable] | dict
            - When ``_return_callable=False``: list of metadata dicts.
            - When ``_return_callable=True``: list of callables.
            - When ``_also_return_metadata=True``: a dict with both.
        """
        return []

    schema = llmh.method_to_schema(search, "tool")
    desc = schema["function"]["description"]

    # The Returns section should not reference hidden params
    assert "_return_callable" not in desc
    assert "_also_return_metadata" not in desc


def test_schema_strips_hidden_param_references_from_raises() -> None:
    """
    The Raises section may document errors for hidden-param validation
    (e.g. ``If ``_return_callable=True`` but ``_namespace`` is ``None````).
    These entries must be stripped from the LLM-facing description since
    the LLM cannot trigger these errors.
    """

    def search(
        query: str,
        _return_callable: bool = False,
        _namespace: dict | None = None,
    ) -> list:
        """
        Search for items.

        Parameters
        ----------
        query : str
            Search text.
        _return_callable : bool, default ``False``
            When ``True``, return callables.
        _namespace : dict | None, default ``None``
            Target namespace for injection.

        Raises
        ------
        ValueError
            If ``_return_callable=True`` but ``_namespace`` is ``None``.
        ValueError
            If ``_return_callable`` is set without proper context.
        """
        return []

    schema = llmh.method_to_schema(search, "tool")
    desc = schema["function"]["description"]

    # The Raises section should not reference hidden params
    assert "_return_callable" not in desc
    assert "_namespace" not in desc


def test_schema_plain_function() -> None:
    def _plain(a: int) -> None:
        """Plain function doc."""
        return None

    schema = llmh.method_to_schema(_plain, "tool")
    desc = schema["function"]["description"]
    assert desc == "Plain function doc."


# --------------------------------------------------------------------------- #
#  make_request_clarification_tool — docstring & schema                       #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#  functools.wraps + get_type_hints: base class annotations must resolve       #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#  OPTIONAL PARAMETERS STAY EXPRESSIBLE AS ABSENT                             #
# --------------------------------------------------------------------------- #
def _tool_with_optional_scalars(
    session_id: int | None = None,
    session_name: str | None = None,
    ratio: float | None = None,
    flag: bool | None = None,
) -> None:  # pragma: no cover - schema only
    return None


def test_optional_scalars_are_nullable() -> None:
    """An optional scalar must advertise null.

    Collapsed to a bare integer, an optional id gives the model no way to say
    "none", so it sends sentinels (0, -1, 999, 2147483647) that the receiver
    reads as real requests for resources that never existed.
    """
    schema = llmh.method_to_schema(_tool_with_optional_scalars, "tool")
    params = schema["function"]["parameters"]["properties"]

    assert params["session_id"]["type"] == ["integer", "null"]
    assert params["session_name"]["type"] == ["string", "null"]
    assert params["ratio"]["type"] == ["number", "null"]
    assert params["flag"]["type"] == ["boolean", "null"]


def test_optional_list_keeps_its_item_schema() -> None:
    from typing import List, Optional

    schema = llmh.annotation_to_schema(Optional[List[str]])
    assert schema["type"] == ["array", "null"]
    assert schema["items"] == {"type": "string"}


def test_required_scalar_stays_non_nullable() -> None:
    """Widening must apply to Optional only, never to a plain annotation."""
    assert llmh.annotation_to_schema(int) == {"type": "integer"}
